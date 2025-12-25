# models/nlse_model.py
import numpy as np
from scipy.optimize import minimize
from datetime import datetime, timedelta
from .data_handler import DataHandler
from utils.calculations import (
    estimate_params, solve_nlse, reconstruct_prices, 
    optimize_phi0, nlse_step
)

class NLSEPredictor:
    def __init__(self, ticker):
        self.ticker = ticker
        self.parameters_history = []
        self.weeks = None
        self.week_dates = None
        
    def prepare(self, end_date, weeks_history=20):
        """Prepare data with specified history length."""
        start_date = end_date - timedelta(weeks=weeks_history)
        prices = DataHandler.download_close(self.ticker, start_date, end_date)
        
        if prices is None:
            return False
            
        self.weeks, self.week_dates = DataHandler.weekly_windows(prices)
        return len(self.weeks) > 0
    
    def predict_week(self, idx):
        """Predict a week and track all parameters."""
        week = self.weeks[idx]
        
        # Estimate parameters
        params = estimate_params(week)
        sigma, k, omega, beta = params['sigma'], params['k'], params['omega'], params['beta']
        
        # Store parameters
        params_dict = {
            'week_start': self.week_dates[idx].strftime('%Y-%m-%d'),
            'friday_price': week[4],
            'sigma': sigma,
            'k': k,
            'omega': omega,
            'beta': beta,
            'volatility_pct': params['volatility_pct'],
            'price_range': params['price_range'],
            'scale': max(1000, k)
        }
        self.parameters_history.append(params_dict)
        
        # Scale for stability
        scale = max(1000, k)
        
        # Optimize phi0
        phi0 = optimize_phi0(week, sigma, k, omega, beta, scale)
        friday_price = week[4]
        
        # Solve NLSE
        phi = solve_nlse(phi0, 50, sigma, k, omega, beta, scale)
        full = reconstruct_prices(phi, friday_price)
        
        # Extract business days
        business = full[:5]
        params_dict['phi0'] = phi0
        params_dict['prediction'] = business
        
        return business
    
    def evaluate(self, history_weeks):
        """Evaluate model performance on historical data."""
        if len(self.weeks) < history_weeks + 1:
            return None, None, None
        
        preds = []
        acts = []
        
        start_idx = len(self.weeks) - history_weeks - 1
        
        for i in range(start_idx, len(self.weeks) - 1):
            pred = self.predict_week(i)
            act = self.weeks[i+1][:5]
            preds.append(pred)
            acts.append(act)
        
        preds = np.array(preds)
        acts = np.array(acts)
        
        # Calculate metrics
        mape = np.mean(np.abs((acts - preds) / acts)) * 100
        rmse = np.sqrt(np.mean((acts - preds)**2))
        mae = np.mean(np.abs(acts - preds))
        
        return preds, acts, {
            'mape': mape,
            'rmse': rmse,
            'mae': mae,
            'parameters_history': self.parameters_history,
            'start_idx': start_idx
        }
    
    def predict_next_week(self):
        """Predict the next week using the last available data."""
        if self.weeks is None or len(self.weeks) == 0:
            return None
        
        last_week = self.weeks[-1]
        last_params = estimate_params(last_week)
        sigma, k, omega, beta = last_params['sigma'], last_params['k'], last_params['omega'], last_params['beta']
        scale = max(1000, k)
        
        phi0 = optimize_phi0(last_week, sigma, k, omega, beta, scale)
        friday = last_week[4]
        phi = solve_nlse(phi0, 50, sigma, k, omega, beta, scale)
        next_pred = reconstruct_prices(phi, friday)[:5]
        
        return {
            'prediction': next_pred,
            'last_friday': friday,
            'sigma': sigma,
            'k': k,
            'omega': omega,
            'beta': beta,
            'phi0': phi0,
            'scale': scale
        }