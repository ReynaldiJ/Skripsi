# app.py
import streamlit as st
from datetime import date

# Import configurations and modules
from config import APP_CONFIG, MODEL_DEFAULTS
from ui.styles import get_custom_css, get_sidebar_content, get_overview_content
from ui.components import (
    render_sidebar, render_overview_tab, 
    render_historical_tab, render_prediction_tab
)
from models.data_handler import DataHandler
from models.nlse_model import NLSEPredictor

def main():
    # Set page configuration
    st.set_page_config(**APP_CONFIG)
    
    # Apply custom CSS
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    
    # Initialize session state with defaults
    if 'default_ticker' not in st.session_state:
        st.session_state.default_ticker = MODEL_DEFAULTS['default_ticker']
    if 'default_weeks' not in st.session_state:
        st.session_state.default_weeks = MODEL_DEFAULTS['default_weeks']
    if 'min_weeks' not in st.session_state:
        st.session_state.min_weeks = MODEL_DEFAULTS['min_weeks']
    if 'max_weeks' not in st.session_state:
        st.session_state.max_weeks = MODEL_DEFAULTS['max_weeks']
    
    # Set UI content
    st.session_state.sidebar_content = get_sidebar_content()
    st.session_state.overview_content = get_overview_content()
    
    # Render sidebar
    render_sidebar()
    
    # Main title
    st.markdown('<h1 class="main-header">NLSE Stock Price Predictor</h1>', unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "📅 Historical Data", "🔮 NLSE Prediction"])
    
    # Tab 1: Overview
    with tab1:
        render_overview_tab()
    
    # Tab 2: Historical Data
    with tab2:
        render_historical_tab(DataHandler)
    
    # Tab 3: NLSE Prediction
    with tab3:
        render_prediction_tab(NLSEPredictor)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #6B7280; padding: 20px;'>
        <p>NLSE Stock Prediction Model • Shows fitting chart</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()