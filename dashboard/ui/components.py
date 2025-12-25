# ui/components.py
import streamlit as st
import pandas as pd
from datetime import date, timedelta
import plotly.graph_objects as go

def render_sidebar():
    """Render the sidebar."""
    with st.sidebar:
        sidebar_content = st.session_state.get('sidebar_content', {})
        st.title(sidebar_content.get('title', "📈 NLSE Stock Predictor"))
        st.markdown("---")
        
        default_ticker = st.text_input("Default Ticker", 
                                     value=st.session_state.get('default_ticker', 'AAPL'))
        st.session_state.default_ticker = default_ticker
        
        st.markdown("---")
        st.markdown("**Model Settings**")
        
        default_weeks = st.slider("Weeks to Backtest", 
                                 min_value=st.session_state.get('min_weeks', 4),
                                 max_value=st.session_state.get('max_weeks', 20),
                                 value=st.session_state.get('default_weeks', 8))
        st.session_state.default_weeks = default_weeks
        
        st.markdown("---")
        st.markdown(sidebar_content.get('about', ""))

def render_overview_tab():
    """Render the overview tab."""
    st.markdown(st.session_state.get('overview_content', ""))

def render_historical_tab(data_handler):
    """Render the historical data tab."""
    st.markdown('<h2 class="sub-header">Historical Stock Data</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ticker = st.text_input("Ticker Symbol", 
                              value=st.session_state.get('default_ticker', 'AAPL'), 
                              key="hist_ticker")
        col_start, col_end = st.columns(2)
        with col_start:
            start_date = st.date_input("Start Date", 
                                      value=date.today() - timedelta(days=180))
        with col_end:
            end_date = st.date_input("End Date", value=date.today())
    
    with col2:
        st.write("")
        if st.button("📥 Fetch Data", use_container_width=True):
            with st.spinner("Downloading..."):
                try:
                    prices = data_handler.download_close(ticker, start_date, end_date)
                    
                    if prices is not None:
                        st.session_state.hist_data = {
                            'ticker': ticker,
                            'dates': [d.strftime('%Y-%m-%d') for d in prices.index],
                            'prices': prices.squeeze().astype(float).tolist()

                        }
                        st.success(f"Downloaded {len(prices)} data points")
                    else:
                        st.error("Failed to download data")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    if 'hist_data' in st.session_state:
        data = st.session_state.hist_data
        
        if data and len(data['prices']) > 0:
            df = pd.DataFrame({
                'Date': data['dates'],
                'Price': data['prices']
            })
            
            # Create and display chart
            from utils.visualizations import create_historical_chart
            fig = create_historical_chart(df, data['ticker'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("First Date", df['Date'].iloc[0])
            with col2:
                st.metric("Last Date", df['Date'].iloc[-1])
            with col3:
                st.metric("Days", len(df))
            with col4:
                st.metric("Last Price", f"{df['Price'].iloc[-1]:.2f}")
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"{data['ticker']}_historical.csv",
                mime="text/csv"
            )

def render_prediction_tab(nlse_model):
    """Render the prediction tab."""
    st.markdown('<h2 class="sub-header">NLSE Model Prediction</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        pred_ticker = st.text_input("Ticker", 
                                   value=st.session_state.get('default_ticker', 'AAPL'), 
                                   key="pred_ticker")
    
    with col2:
        weeks_to_test = st.number_input("Weeks to Backtest", 
                                       min_value=st.session_state.get('min_weeks', 4), 
                                       max_value=st.session_state.get('max_weeks', 20), 
                                       value=st.session_state.get('default_weeks', 8),
                                       step=1,
                                       help="Number of historical weeks for backtesting")
    
    with col3:
        st.write("")
        st.write("")
        run_button = st.button("🚀 Run Model", type="primary", use_container_width=True)
    
    if run_button:
        with st.spinner(f"Running NLSE model for {pred_ticker}..."):
            predictor = nlse_model(pred_ticker)
            end_date = date.today()
            
            # Prepare data
            if predictor.prepare(end_date, weeks_history=weeks_to_test + 15):
                if len(predictor.weeks) >= weeks_to_test + 1:
                    # Run evaluation
                    preds, acts, metrics = predictor.evaluate(weeks_to_test)
                    
                    if preds is not None:
                        # Predict next week
                        next_pred_info = predictor.predict_next_week()
                        
                        # Store results
                        st.session_state.prediction_results = {
                            'ticker': pred_ticker,
                            'next_pred': next_pred_info['prediction'],
                            'last_friday': next_pred_info['last_friday'],
                            'predictor': predictor,
                            'metrics': metrics,
                            'start_idx': metrics['start_idx'],
                            'sigma': next_pred_info['sigma'],
                            'k': next_pred_info['k'],
                            'omega': next_pred_info['omega'],
                            'beta': next_pred_info['beta'],
                            'phi0': next_pred_info['phi0'],
                            'scale': next_pred_info['scale'],
                            'weeks_tested': weeks_to_test
                        }
                        
                        st.success(f"✅ Model completed! MAPE: {metrics['mape']:.2f}%")
                    else:
                        st.error("Evaluation failed")
                else:
                    st.warning(f"Need at least {weeks_to_test + 1} weeks of data. Have {len(predictor.weeks)} weeks.")
            else:
                st.error("Failed to download or prepare data")
    
    # Display results if available
    if 'prediction_results' in st.session_state:
        result = st.session_state.prediction_results
        
        # Display results
        display_prediction_results(result)

def build_fit_forecast_csv(predictor, start_idx, next_pred):
    """
    Build a DataFrame with columns:
    [date],[actual],[prediction],[type (fit/forecast)]
    """
    rows = []

    # FIT (historical backtest fit)
    for i in range(start_idx, len(predictor.weeks) - 1):
        week_start = pd.to_datetime(predictor.week_dates[i])
        pred_week = predictor.predict_week(i)
        act_week = predictor.weeks[i + 1][:5]

        for d in range(5):
            rows.append({
                "date": (week_start + pd.Timedelta(days=d)).strftime("%Y-%m-%d"),
                "actual": float(act_week[d]),
                "prediction": float(pred_week[d]),
                "type (fit/forecast)": "fit"
            })

    # FORECAST (next week)
    next_week_start = pd.to_datetime(predictor.week_dates[-1]) + pd.Timedelta(days=7)
    for d in range(5):
        rows.append({
            "date": (next_week_start + pd.Timedelta(days=d)).strftime("%Y-%m-%d"),
            "actual": "",  # empty because unknown
            "prediction": float(next_pred[d]),
            "type (fit/forecast)": "forecast"
        })

    return pd.DataFrame(rows)

def display_prediction_results(result):
    """Display prediction results."""
    # 1. Raw Model Output Table
    st.subheader("📊 Raw Model Output")
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    pred_df = pd.DataFrame({
        'Day': days,
        'Predicted Price': result['next_pred'],
        'Change %': [(p - result['last_friday']) / result['last_friday'] * 100 
                    for p in result['next_pred']]
    })
    st.dataframe(pred_df, use_container_width=True)
    
    # 2. Estimated Parameters
    st.subheader("⚙️ Estimated Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("σ (Volatility)", f"{result['sigma']:.6f}")
        st.metric("φ₀ (Initial Phase)", f"{result['phi0']:.6f}")
    
    with col2:
        st.metric("K (Mean Price)", f"{result['k']:.2f}")
        st.metric("β (Market Potential)", f"{result['beta']:.6f}")
    
    with col3:
        st.metric("ω (Drift Speed)", f"{result['omega']:.6f}")
        st.metric("Scale Factor", f"{result['scale']:.2f}")
    
    # 3. CHART
    st.subheader("📈 Model Fitting Chart")
    
    from utils.visualizations import create_chart
    fig = create_chart(
        result['ticker'],
        result['predictor'],
        result['start_idx'],
        result['weeks_tested'],
        result['next_pred'],
        result['metrics']['mape']
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # --- Download Fit + Forecast CSV ---
    # st.subheader("⬇️ Export Fit & Forecast Data (CSV)")

    ff_df = build_fit_forecast_csv(
        predictor=result["predictor"],
        start_idx=result["start_idx"],
        next_pred=result["next_pred"]
    )

    st.download_button(
        label="📥 Download Fit & Forecast CSV",
        data=ff_df.to_csv(index=False),
        file_name=f"{result['ticker']}_nlse_fit_forecast.csv",
        mime="text/csv",
        # use_container_width=True
    )

    # Chart explanation
    with st.expander("📖 Chart Explanation"):
        st.markdown("""
        **This chart shows Fitting and Model Error distribution:**
        
        **Top Plot (Actual vs Predicted):**
        - **Blue circles**: Actual historical prices
        - **Green stars**: Model predictions (fitted values)
        - **Red dashed line**: Start of forecast period
        - Right side (after red line): Next week's forecast
        
        **Bottom Plot (Error):**
        - **Red line with circles**: Daily prediction error (%)
        - **Orange dashed line**: Mean error across all predictions
        """)
    
    # 4. Performance Metrics
    st.subheader("📊 Model Performance Metrics")
    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric("MAPE", f"{result['metrics']['mape']:.2f}%")
    with metric_cols[1]:
        st.metric("RMSE", f"{result['metrics']['rmse']:.2f}")
    with metric_cols[2]:
        st.metric("MAE", f"{result['metrics']['mae']:.2f}")
    
    # 5. Recent Parameters History
    st.subheader("📋 Recent Parameter History")
    if result['predictor'].parameters_history:
        recent_params = result['predictor'].parameters_history[-5:]
        param_data = []
        
        for i, params in enumerate(recent_params):
            param_data.append({
                'Week': i + 1,
                'Start Date': params['week_start'],
                # 'Friday Price': f"{params['friday_price']:.2f}",
                'Sigma (σ)': f"{params['sigma']:.4f}",
                'K': f"{params['k']:.2f}",
                'ω': f"{params['omega']:.6f}",
                'φ₀': f"{params.get('phi0', 0):.6f}"
            })
        
        st.dataframe(pd.DataFrame(param_data), use_container_width=True)