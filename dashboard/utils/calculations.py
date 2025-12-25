# utils/calculations.py
import numpy as np
from scipy.optimize import minimize

def estimate_params(week):
    """Estimate NLSE parameters from weekly data with detailed output."""
    weekday = week[:5]
    price_changes = np.diff(weekday)
    returns = np.diff(weekday) / weekday[:-1]

    sigma = np.std(price_changes)  # volatility
    k = np.mean(weekday)           # mean price level
    omega = -np.mean(returns)      # negative drift (as per paper)
    beta = 0.0475                  # adaptive market potential (paper constant)
    
    # Additional metrics
    price_range = np.max(weekday) - np.min(weekday)
    volatility_pct = (sigma / k) * 100 if k > 0 else 0
    
    return {
        'sigma': sigma,
        'k': k,
        'omega': omega,
        'beta': beta,
        'volatility_pct': volatility_pct,
        'price_range': price_range,
        'week_min': np.min(weekday),
        'week_max': np.max(weekday),
        'week_std': np.std(weekday)
    }

def nlse_step(phi, dphi, h, sigma, k, omega, beta, scale):
    k_scaled = k / scale
    sigma_scaled = sigma / scale
    lam = (omega - 0.5 * sigma_scaled * k_scaled**2)

    # φ'' = β φ^3 – λ φ
    def accel(p): return beta * p**3 - lam * p

    # RK4 on φ and dφ
    k1_p = dphi
    k1_v = accel(phi)

    k2_p = dphi + 0.5 * h * k1_v
    k2_v = accel(phi + 0.5 * h * k1_p)

    k3_p = dphi + 0.5 * h * k2_v
    k3_v = accel(phi + 0.5 * h * k2_p)

    k4_p = dphi + h * k3_v
    k4_v = accel(phi + h * k3_p)

    phi_next = phi + h / 6.0 * (k1_p + 2*k2_p + 2*k3_p + k4_p)
    dphi_next = dphi + h / 6.0 * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    return phi_next, dphi_next

def solve_nlse(phi0, nsteps, sigma, k, omega, beta, scale):
    """Stable NLSE integration."""
    h = 1 / nsteps
    phi = np.zeros(nsteps + 1)
    dphi = np.zeros(nsteps + 1)
    phi[0] = phi0
    dphi[0] = 0.0

    for i in range(nsteps):
        phi[i+1], dphi[i+1] = nlse_step(phi[i], dphi[i], h, sigma, k, omega, beta, scale)
    return phi

def reconstruct_prices(phi, base_price):
    """Convert phi (relative return signal) back to prices."""
    prices = np.zeros(len(phi))
    prices[0] = base_price
    for i in range(1, len(phi)):
        prices[i] = prices[i-1] * (1 + phi[i])
    return prices

def optimize_phi0(week, sigma, k, omega, beta, scale):
    """Optimize initial phi value for best fit."""
    target = week[:5]
    friday_price = week[4]

    def err(phi0):
        phi = solve_nlse(phi0, 50, sigma, k, omega, beta, scale)
        pred = reconstruct_prices(phi, friday_price)
        return np.mean((pred[:5] - target)**2)

    res = minimize(lambda x: err(x[0]), x0=[0.0], bounds=[(-0.1, 0.1)])
    return res.x[0]