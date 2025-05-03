"""
Visualization utilities for the stock prediction app.
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.io as pio
import base64
import json

def plot_stock_data(data, title=None, symbol_info=None):
    """
    Plot stock data with candlestick and volume charts
    
    Args:
        data (pd.DataFrame): Stock data with OHLCV columns
        title (str, optional): Chart title
        symbol_info (dict, optional): Symbol information
        
    Returns:
        go.Figure: Plotly figure
    """
    # Create subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.02, 
                        row_heights=[0.7, 0.3])
    
    # Default title
    if title is None:
        title = "Stock Price History"
    
    # Add symbol information to title if available
    if symbol_info:
        if 'name' in symbol_info:
            title += f" - {symbol_info['name']}"
        if 'symbol' in symbol_info:
            title += f" ({symbol_info['symbol']})"
    
    # Add candlestick trace
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name="OHLC"
        ),
        row=1, col=1
    )
    
    # Add volume trace
    if 'volume' in data.columns:
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['volume'],
                name="Volume",
                marker_color='rgba(0, 0, 255, 0.5)'
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        yaxis_title="Price",
        yaxis2_title="Volume",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=600,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Candlestick styling
    fig.update_xaxes(
        gridcolor="lightgray",
        showgrid=True
    )
    
    fig.update_yaxes(
        gridcolor="lightgray",
        showgrid=True,
        zeroline=False
    )
    
    return fig

def create_styled_prediction_table(predictions_df, symbol_info=None, date_range=None, model_name=None):
    """
    Create a styled prediction table visualization
    
    Args:
        predictions_df (pd.DataFrame): Prediction data
        symbol_info (dict, optional): Symbol information
        date_range (tuple, optional): Date range for training data (start, end)
        model_name (str, optional): Model name
        
    Returns:
        go.Figure: Plotly figure
    """
    # Create a subplots figure
    fig = make_subplots(rows=1, cols=1)
    
    # Extract OHLC predictions
    ohlc_columns = ['predicted_open', 'predicted_high', 'predicted_low', 'predicted_close']
    ohlc_columns = [col for col in ohlc_columns if col in predictions_df.columns]
    
    # Create table data
    table_data = []
    
    # Create header
    header = ['Date']
    for col in ohlc_columns:
        header.append(col.replace('predicted_', '').title())
    
    table_data.append(header)
    
    # Add rows
    for idx, row in predictions_df.iterrows():
        date_str = idx.strftime('%Y-%m-%d')
        table_row = [date_str]
        
        for col in ohlc_columns:
            value = row[col]
            # Format with 2 decimal places
            table_row.append(f"{value:.2f}")
        
        table_data.append(table_row)
    
    # Create the table
    fig.add_trace(
        go.Table(
            header=dict(
                values=table_data[0],
                fill_color='rgb(40, 167, 69)',  # Green header
                align='center',
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[list(col) for col in zip(*table_data[1:])],
                fill_color='rgb(255, 255, 255)',
                align='right',
                font=dict(size=11)
            )
        )
    )
    
    # Create title with information
    title = "Price Predictions"
    
    if model_name:
        title = f"{model_name} Model: {title}"
    
    if symbol_info:
        if 'name' in symbol_info:
            title += f" - {symbol_info['name']}"
        
        if 'symbol' in symbol_info and 'name' not in symbol_info:
            title += f" - {symbol_info['symbol']}"
    
    if date_range:
        title += f" (Trained on data from {date_range[0]} to {date_range[1]})"
    
    fig.update_layout(
        title=title,
        height=max(150 + len(table_data) * 25, 300),  # Dynamic height based on number of rows
        margin=dict(l=10, r=10, t=60, b=10),
    )
    
    return fig

def plot_predictions(data, predictions_df, title=None, symbol_info=None, date_range=None, model_name=None):
    """
    Plot stock data with predictions
    
    Args:
        data (pd.DataFrame): Historical stock data
        predictions_df (pd.DataFrame): Prediction data with forecast columns
        title (str, optional): Chart title
        symbol_info (dict, optional): Symbol information
        date_range (tuple, optional): Date range for training data (start, end)
        model_name (str, optional): Model name
        
    Returns:
        tuple: (prediction plot figure, prediction table figure)
    """
    # Create subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.02, 
                        row_heights=[0.7, 0.3])
    
    # Default title
    if title is None:
        title = "Stock Price Predictions"
    
    # Add symbol information to title if available
    if symbol_info:
        if 'name' in symbol_info:
            title += f" - {symbol_info['name']}"
        if 'symbol' in symbol_info:
            title += f" ({symbol_info['symbol']})"
    
    # Add model name to title if available
    if model_name:
        title += f" using {model_name} Model"
    
    # Add date range to title if available
    if date_range:
        title += f" (Trained on data from {date_range[0]} to {date_range[1]})"
    
    # Historical data trace
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name="Historical OHLC",
            increasing_line_color='rgba(0, 255, 0, 0.7)', 
            decreasing_line_color='rgba(255, 0, 0, 0.7)'
        ),
        row=1, col=1
    )
    
    # Add prediction traces
    if 'predicted_close' in predictions_df.columns:
        fig.add_trace(
            go.Scatter(
                x=predictions_df.index,
                y=predictions_df['predicted_close'],
                name="Predicted Close",
                line=dict(color='royalblue', width=2),
                mode='lines'
            ),
            row=1, col=1
        )
        
        # Add confidence bounds if available
        if 'upper_bound' in predictions_df.columns and 'lower_bound' in predictions_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=predictions_df.index,
                    y=predictions_df['upper_bound'],
                    name="Upper Bound",
                    line=dict(color='rgba(0, 0, 255, 0.2)', width=0),
                    mode='lines',
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=predictions_df.index,
                    y=predictions_df['lower_bound'],
                    name="Lower Bound",
                    line=dict(color='rgba(0, 0, 255, 0.2)', width=0),
                    fill='tonexty',
                    fillcolor='rgba(0, 0, 255, 0.1)',
                    mode='lines',
                    showlegend=False
                ),
                row=1, col=1
            )
    
    # Add volume trace
    if 'volume' in data.columns:
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['volume'],
                name="Volume",
                marker_color='rgba(0, 0, 255, 0.5)'
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        yaxis_title="Price",
        yaxis2_title="Volume",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=600,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Grid styling
    fig.update_xaxes(
        gridcolor="lightgray",
        showgrid=True
    )
    
    fig.update_yaxes(
        gridcolor="lightgray",
        showgrid=True,
        zeroline=False
    )
    
    # Create prediction table
    table_fig = create_styled_prediction_table(
        predictions_df=predictions_df,
        symbol_info=symbol_info,
        date_range=date_range,
        model_name=model_name
    )
    
    return fig, table_fig

def get_download_link(fig, filename, format="json"):
    """
    Get a download link for a Plotly figure
    
    Args:
        fig (go.Figure): Plotly figure
        filename (str): Download filename
        format (str): Output format
        
    Returns:
        str: Download link for the figure
    """
    if format == "json":
        # Export as JSON
        fig_json = json.dumps(fig.to_dict())
        b64 = base64.b64encode(fig_json.encode()).decode()
        href = f"data:application/json;charset=utf-8;base64,{b64}"
        return href
    else:
        # Formats like PNG are not supported on the server
        return None

def get_html_table_download(fig, filename):
    """
    Get a download link for a table figure as HTML
    
    Args:
        fig (go.Figure): Plotly figure
        filename (str): Download filename
        
    Returns:
        str: Download link for the HTML table
    """
    # Convert figure to HTML
    html = fig.to_html(include_plotlyjs=False)
    
    # Encode as base64
    b64 = base64.b64encode(html.encode()).decode()
    href = f"data:text/html;charset=utf-8;base64,{b64}"
    
    return href