# config.py
import warnings
warnings.filterwarnings('ignore')

# App configuration
APP_CONFIG = {
    'page_title': "NLSE Stock Predictor",
    'page_icon': "📈",
    'layout': "wide"
}

# Model defaults
MODEL_DEFAULTS = {
    'default_ticker': '^JKSE',
    'min_weeks': 4,
    'max_weeks': 20,
    'default_weeks': 8,
    'scale_factor': 1000,
    'nsteps': 50
}

# Chart settings
CHART_CONFIG = {
    'height': 800,
    'template': 'plotly_white',
    'vertical_spacing': 0.15
}