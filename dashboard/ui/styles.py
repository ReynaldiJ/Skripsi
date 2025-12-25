# ui/styles.py
def get_custom_css():
    """Return custom CSS for the app."""
    return """
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E3A8A;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #3B82F6;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            border-left: 5px solid #3B82F6;
        }
    </style>
    """

def get_sidebar_content():
    """Return sidebar content."""
    return {
        'title': "📈 NLSE Stock Predictor",
        'about': """
        **About the Model:**
        
        Nonlinear Schrödinger Equation (NLSE)
        - Quantum finance approach
        - Weekly predictions
        - Parameter optimization
        - Shows fitting to historical data
        """
    }

def get_overview_content():
    """Return overview tab content."""
    return """
    ## Welcome to NLSE Stock Predictor
    
    ### Model Features:
    
    **1. Historical Data Analysis**
    - Download stock prices
    - Visualize trends
    - Export to CSV
    
    **2. NLSE Prediction**
    - Parameter estimation (σ, K, ω, β, φ₀)
    - Weekly price forecasts
    - Backtesting with error metrics
    - **Chart showing model fitting**
    
    **3. How it Works:**
    - Uses Nonlinear Schrödinger Equation
    - Shows actual vs predicted with forecast
    - Displays error metrics (MAPE, RMSE, MAE)
    
    **Supported Tickers:**
    - Indonesian: BBCA.JK, BRMS.JK, AALI.JK
    - Any Yahoo Finance symbol
    """