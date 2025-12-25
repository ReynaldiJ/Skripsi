# utils/visualizations.py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from datetime import timedelta

def create_chart(ticker, predictor, start_idx, history_weeks, next_pred, mape):
    
    # Prepare data for plotting
    preds = []
    acts = []
    x_labels = []
    
    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    
    for i in range(start_idx, len(predictor.weeks) - 1):
        pred = predictor.predict_week(i)
        act = predictor.weeks[i+1][:5]
        preds.append(pred)
        acts.append(act)
        x_labels.append(predictor.week_dates[i].strftime("%Y-%m-%d"))
    
    # Flatten for plotting
    timeline = []
    actual_series = []
    predicted_series = []
    
    for w in range(len(preds)):
        lbl = x_labels[w]
        for d in range(5):
            timeline.append(f"{lbl}-{days[d]}")
            actual_series.append(acts[w][d])
            predicted_series.append(preds[w][d])
    
    # Add future week
    last_date = predictor.week_dates[-1]
    next_label = (last_date + timedelta(days=7)).strftime("%Y-%m-%d")
    for d in range(5):
        timeline.append(f"{next_label}-{days[d]}")
        actual_series.append(None)
        predicted_series.append(next_pred[d])
    
    # Calculate error for second plot
    actual_flat = np.array(acts).flatten()
    pred_flat = np.array(preds).flatten()
    abs_error_pct = np.abs((actual_flat - pred_flat) / actual_flat) * 100
    mean_error = np.mean(abs_error_pct)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f'{ticker} - NLSE Model | {history_weeks} weeks history + forecast | MAPE: {mape:.2f}%',
            'Daily Prediction Error (%)'
        ),
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3]
    )
    
    # Top plot: Actual vs Predicted (green stars and blue circles)
    fig.add_trace(
        go.Scatter(
            x=list(range(len(actual_series))),
            y=predicted_series,
            mode='lines+markers',
            name='Predicted',
            line=dict(color='green', width=2, dash='solid'),
            marker=dict(symbol='star', size=8, color='green'),
            showlegend=True
        ),
        row=1, col=1 
    )
    
    fig.add_trace(
        go.Scatter(
            x=list(range(len(actual_series))),
            y=actual_series,
            mode='lines+markers',
            name='Actual',
            line=dict(color='blue', width=2, dash='solid'),
            marker=dict(symbol='circle', size=8, color='blue'),
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Add vertical line for forecast start (red dashed line)
    forecast_start = len(actual_series) - 5
    fig.add_vline(
        x=forecast_start,
        line=dict(color='red', width=2, dash='dash'),
        annotation_text="Forecast start",
        annotation_position="top right",
        row=1, col=1
    )
    
    # Bottom plot: Error (red dots)
    fig.add_trace(
        go.Scatter(
            x=list(range(len(abs_error_pct))),
            y=abs_error_pct,
            mode='lines+markers',
            name='Absolute % Error',
            line=dict(color='red', width=2),
            marker=dict(symbol='circle', size=6, color='red'),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Add mean error line (orange dashed line)
    fig.add_hline(
        y=mean_error,
        line=dict(color='orange', width=2, dash='dash'),
        annotation_text=f'Mean: {mean_error:.2f}%',
        annotation_position="top left",
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        template='plotly_white',  # Lighter background
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.8)'
        )
    )
    
    # Update axes
    fig.update_xaxes(
        title_text="Trading Days",
        # tickangle=90,  # Rotated x-axis labels
        row=2, col=1
    )
    
    fig.update_yaxes(
        title_text="Price",
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text="Absolute % Error",
        row=2, col=1
    )
    
    return fig

def create_historical_chart(df, ticker):
    """Create historical price chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Price'],
        mode='lines', name='Price',
        line=dict(color='blue', width=2)
    ))
    fig.update_layout(
        height=400,
        template='plotly_dark',
        xaxis_title="Date",
        yaxis_title="Price",
        title=f"{ticker} - Historical Prices"
    )
    return fig