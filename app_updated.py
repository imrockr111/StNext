"""
Stock Market Prediction App with enhanced features using AngelOne SmartAPI
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import os
import json
import base64
import io
import threading
import schedule
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import utility functions
from utils.api_connector import SmartAPIConnector as AngelOneConnector
from utils.api_extended import AngelOneAPI, get_api_instance
from utils.data_processor import preprocess_data
from utils.prediction_models import train_arima_model, train_prophet_model
from utils.visualization import plot_stock_data, plot_predictions, get_download_link, get_html_table_download
from utils.instrument_master import get_instrument_manager, get_instrument_by_token, fetch_instrument_master
from utils.database import (
    initialize_database, get_user, save_stock_data, get_stock_data,
    check_stock_data_availability, get_prediction_by_id, get_user_predictions,
    search_instruments
)
from utils.services import (
    initialize_system, register_new_user, authenticate_user, hash_password,
    login_to_angelone, fetch_stock_data_smart, train_model_and_save,
    update_prediction_accuracy_with_new_data
)
from utils.advanced_models import (
    train_lstm_model, train_gru_model, train_transformer_model
)

# Initialize database and system
initialize_database()
initialize_system()

# Set page configuration
st.set_page_config(
    page_title="Stock Market Prediction App",
    page_icon="📈",
    layout="wide",
)

# Initialize session state variables
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'api_connector' not in st.session_state:
    st.session_state.api_connector = None
if 'extended_api' not in st.session_state:
    st.session_state.extended_api = None
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = None
if 'selected_token' not in st.session_state:
    st.session_state.selected_token = None
if 'error_message' not in st.session_state:
    st.session_state.error_message = None
if 'jwt_token' not in st.session_state:
    st.session_state.jwt_token = None
if 'refresh_token' not in st.session_state:
    st.session_state.refresh_token = None
if 'instrument_manager_loaded' not in st.session_state:
    st.session_state.instrument_manager_loaded = False
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'selected_instrument' not in st.session_state:
    st.session_state.selected_instrument = None
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = None
if 'funds_data' not in st.session_state:
    st.session_state.funds_data = None
if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = None
if 'show_registration' not in st.session_state:
    st.session_state.show_registration = False

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0066cc;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #28a745;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.4rem;
        color: #343a40;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin-bottom: 1rem;
    }
    .error-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        background-color: white;
    }
    .login-card {
        padding: 2rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        margin: 2rem auto;
        max-width: 500px;
        background-color: white;
    }
    .token-display {
        font-family: monospace;
        word-break: break-all;
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 0.3rem;
        overflow-x: auto;
    }
    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e9ecef;
        text-align: center;
        color: #6c757d;
        font-size: 0.9rem;
    }
    .prediction-card {
        padding: 1.2rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        background-color: #f8f9fa;
    }
    .prediction-date {
        font-size: 0.9rem;
        color: #6c757d;
        margin-bottom: 0.5rem;
    }
    .accuracy-high {
        color: #28a745;
        font-weight: bold;
    }
    .accuracy-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .accuracy-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Common functions
def display_error(message):
    """Display error message with consistent styling"""
    st.markdown(f"""
    <div class="error-box">
        <strong>Error:</strong> {message}
    </div>
    """, unsafe_allow_html=True)

def display_success(message):
    """Display success message with consistent styling"""
    st.markdown(f"""
    <div class="success-box">
        <strong>Success:</strong> {message}
    </div>
    """, unsafe_allow_html=True)

def display_info(message):
    """Display info message with consistent styling"""
    st.markdown(f"""
    <div class="info-box">
        <strong>Info:</strong> {message}
    </div>
    """, unsafe_allow_html=True)

def display_warning(message):
    """Display warning message with consistent styling"""
    st.markdown(f"""
    <div class="warning-box">
        <strong>Warning:</strong> {message}
    </div>
    """, unsafe_allow_html=True)

def toggle_registration():
    """Toggle registration form display"""
    st.session_state.show_registration = not st.session_state.show_registration

# User authentication functions
def handle_registration(username, password, confirm_password):
    """Handle user registration"""
    if password != confirm_password:
        return False, "Passwords do not match"
    
    # Register user
    success, message, user_id = register_new_user(
        username=username,
        password=password
    )
    
    return success, message

def handle_login(username, password):
    """Handle user login"""
    # Authenticate user
    success, message, user_data = authenticate_user(
        username=username,
        password=password
    )
    
    if success and user_data:
        # Set session variables
        st.session_state.logged_in = True
        st.session_state.user_id = user_data['user_id']
        st.session_state.username = username
        
        # If API credentials are available, try to connect to AngelOne
        if all([user_data.get(key) for key in ['api_key', 'api_secret', 'client_id', 'mpin', 'totp_secret']]):
            connector, api_success, api_message = login_to_angelone(user_data['user_id'])
            
            if api_success:
                st.session_state.api_connector = connector
                st.session_state.jwt_token = connector.session_token
                st.session_state.refresh_token = connector.refresh_token
                
                # Create extended API connector
                api_instance = get_api_instance()
                api_instance.initialize(
                    api_key=user_data['api_key'],
                    api_secret=user_data['api_secret'],
                    client_id=user_data['client_id'],
                    mpin=user_data['mpin'],
                    totp_secret=user_data['totp_secret']
                )
                st.session_state.extended_api = api_instance
            else:
                st.warning(f"Unable to connect to AngelOne API: {api_message}")
        
        return True, "Login successful"
    
    return success, message

def handle_logout():
    """Handle user logout"""
    # Clear session state
    for key in ['logged_in', 'user_id', 'username', 'api_connector', 'extended_api',
                'stock_data', 'predictions', 'selected_stock', 'selected_token',
                'error_message', 'jwt_token', 'refresh_token', 'user_profile',
                'funds_data', 'portfolio_data']:
        if key in st.session_state:
            st.session_state[key] = None
    
    st.session_state.logged_in = False
    st.session_state.show_registration = False
    
    # Rerun the app
    st.rerun()

def handle_api_credentials_update(api_key, api_secret, client_id, mpin, totp_secret):
    """Handle API credentials update"""
    if not st.session_state.user_id:
        return False, "User not logged in"
    
    # Update user credentials
    from utils.database import update_user_credentials
    success = update_user_credentials(
        user_id=st.session_state.user_id,
        api_key=api_key,
        api_secret=api_secret,
        client_id=client_id,
        mpin=mpin,
        totp_secret=totp_secret
    )
    
    if success:
        # Try to connect with new credentials
        from utils.services import login_to_angelone
        connector, api_success, api_message = login_to_angelone(st.session_state.user_id)
        
        if api_success:
            st.session_state.api_connector = connector
            st.session_state.jwt_token = connector.session_token
            st.session_state.refresh_token = connector.refresh_token
            
            # Create extended API connector
            api_instance = get_api_instance()
            api_instance.initialize(
                api_key=api_key,
                api_secret=api_secret,
                client_id=client_id,
                mpin=mpin,
                totp_secret=totp_secret
            )
            st.session_state.extended_api = api_instance
            
            return True, "API credentials updated and connected successfully"
        else:
            return False, f"API credentials updated but connection failed: {api_message}"
    else:
        return False, "Failed to update API credentials"

# Stock data and prediction functions
def fetch_stock_data(token, exchange, interval, start_date, end_date):
    """Fetch stock data with smart caching"""
    if not st.session_state.api_connector:
        return None, "API connector not available. Please log in first."
    
    try:
        # Fetch stock data
        data = fetch_stock_data_smart(
            api_connector=st.session_state.api_connector,
            token=token,
            exchange=exchange,
            interval=interval,
            start_date=start_date,
            end_date=end_date
        )
        
        if data is not None and not data.empty:
            st.session_state.stock_data = data
            st.session_state.selected_token = token
            instrument = get_instrument_by_token(token)
            stock_name = instrument.get('name', f"Token {token}") if instrument else f"Token {token}"
            st.session_state.selected_stock = stock_name
            return data, "Data fetched successfully"
        else:
            return None, "No data available for the specified token and date range"
    except Exception as e:
        logger.exception("Error fetching stock data")
        return None, f"Error fetching stock data: {str(e)}"

def generate_predictions(model_type, forecast_days=7):
    """Generate predictions for the selected stock"""
    if not st.session_state.stock_data is not None:
        return False, "No stock data available. Please fetch data first."
    
    if not st.session_state.selected_token:
        return False, "No stock selected. Please select a stock first."
    
    try:
        # Train model and save predictions
        success, message, prediction_id = train_model_and_save(
            user_id=st.session_state.user_id,
            token=st.session_state.selected_token,
            exchange="NSE",  # Default, could be made selectable
            model_type=model_type,
            data=st.session_state.stock_data,
            forecast_days=forecast_days
        )
        
        if success and prediction_id:
            # Get prediction from database
            prediction = get_prediction_by_id(prediction_id)
            
            if prediction:
                # Convert prediction data to DataFrame
                prediction_data = prediction['prediction_data']
                predictions_df = pd.DataFrame.from_dict(prediction_data, orient='index')
                
                # Set index to datetime
                predictions_df.index = pd.to_datetime(predictions_df.index)
                
                # Create metrics dictionary
                metrics = json.loads(prediction['metrics'])
                
                # Store in session state
                if 'predictions' not in st.session_state:
                    st.session_state.predictions = {}
                
                model_key = model_type.lower()
                st.session_state.predictions[model_key] = {
                    'predictions': predictions_df,
                    'metrics': metrics
                }
                
                return True, "Predictions generated successfully"
            else:
                return False, "Failed to retrieve prediction from database"
        else:
            return False, message
    except Exception as e:
        logger.exception("Error generating predictions")
        return False, f"Error generating predictions: {str(e)}"

def refresh_user_data():
    """Refresh user profile, funds, and portfolio data"""
    if not st.session_state.extended_api:
        return False, "API connector not available. Please log in first."
    
    try:
        # Fetch user profile
        profile_response = st.session_state.extended_api.get_user_profile()
        
        # Fetch funds and margins
        funds_response = st.session_state.extended_api.get_funds_and_margins()
        
        # Fetch holdings
        holdings_response = st.session_state.extended_api.get_holdings()
        
        # Update session state
        if profile_response and isinstance(profile_response, dict) and profile_response.get("status"):
            st.session_state.user_profile = profile_response.get("data", {})
        
        if funds_response and isinstance(funds_response, dict) and funds_response.get("status"):
            st.session_state.funds_data = funds_response.get("data", {})
        
        if holdings_response and isinstance(holdings_response, dict) and holdings_response.get("status"):
            st.session_state.portfolio_data = holdings_response.get("data", {})
        
        return True, "User data refreshed successfully"
    except Exception as e:
        logger.exception("Error refreshing user data")
        return False, f"Error refreshing user data: {str(e)}"

# Main application
def main():
    # Display header
    st.markdown('<h1 class="main-header">Stock Market Prediction App</h1>', unsafe_allow_html=True)
    
    # Display error message if any
    if st.session_state.error_message:
        display_error(st.session_state.error_message)
        st.session_state.error_message = None
    
    # Login / Registration section
    if not st.session_state.logged_in:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown('<div class="login-card">', unsafe_allow_html=True)
            
            if st.session_state.show_registration:
                st.markdown('<h2 class="sub-header">Register</h2>', unsafe_allow_html=True)
                
                with st.form("registration_form"):
                    username = st.text_input("Username")
                    password = st.text_input("Password", type="password")
                    confirm_password = st.text_input("Confirm Password", type="password")
                    
                    submitted = st.form_submit_button("Register")
                    if submitted:
                        if username and password and confirm_password:
                            success, message = handle_registration(username, password, confirm_password)
                            if success:
                                display_success(message)
                                st.session_state.show_registration = False
                                st.rerun()
                            else:
                                display_error(message)
                        else:
                            display_error("Please fill in all fields")
                
                st.markdown("Already have an account?")
                if st.button("Login"):
                    toggle_registration()
            else:
                st.markdown('<h2 class="sub-header">Login</h2>', unsafe_allow_html=True)
                
                with st.form("login_form"):
                    username = st.text_input("Username")
                    password = st.text_input("Password", type="password")
                    
                    submitted = st.form_submit_button("Login")
                    if submitted:
                        if username and password:
                            with st.spinner("Logging in..."):
                                success, message = handle_login(username, password)
                                if success:
                                    display_success(message)
                                    st.rerun()
                                else:
                                    display_error(message)
                        else:
                            display_error("Please fill in all fields")
                
                st.markdown("Don't have an account?")
                if st.button("Register"):
                    toggle_registration()
            
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Main application tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Dashboard", "Stock Prediction", "Past Predictions", "Instruments", "Account", "API Debug"
        ])
        
        # Dashboard Tab
        with tab1:
            st.markdown('<h2 class="sub-header">Dashboard</h2>', unsafe_allow_html=True)
            
            # If API is connected, display user profile and funds
            if st.session_state.extended_api:
                # Refresh button
                if st.button("Refresh Data"):
                    with st.spinner("Refreshing data..."):
                        success, message = refresh_user_data()
                        if success:
                            display_success(message)
                        else:
                            display_error(message)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<h3 class="section-header">User Profile</h3>', unsafe_allow_html=True)
                    
                    if st.session_state.user_profile:
                        profile = st.session_state.user_profile
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        
                        # Display user profile information
                        if isinstance(profile, dict):
                            st.write(f"**Name:** {profile.get('name', 'N/A')}")
                            st.write(f"**Client ID:** {profile.get('clientcode', 'N/A')}")
                            st.write(f"**Email:** {profile.get('email', 'N/A')}")
                            st.write(f"**Mobile:** {profile.get('mobileno', 'N/A')}")
                            st.write(f"**Account Type:** {profile.get('accounttype', 'N/A')}")
                            st.write(f"**Branch ID:** {profile.get('branchid', 'N/A')}")
                        else:
                            st.write("No profile data available")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.write("No profile data available. Click 'Refresh Data' to fetch profile information.")
                        st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<h3 class="section-header">Funds & Margins</h3>', unsafe_allow_html=True)
                    
                    if st.session_state.funds_data:
                        funds = st.session_state.funds_data
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        
                        # Display funds information
                        if isinstance(funds, dict):
                            # Display available cash balance
                            cash_balance = funds.get('cash', 0)
                            st.metric("Available Cash", f"₹ {cash_balance:,.2f}")
                            
                            # Display margin used
                            margin_used = funds.get('utilizedmargin', 0)
                            st.metric("Margin Used", f"₹ {margin_used:,.2f}")
                            
                            # Display available margin
                            available_margin = funds.get('availablemargin', 0)
                            st.metric("Available Margin", f"₹ {available_margin:,.2f}")
                        else:
                            st.write("No funds data available")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.write("No funds data available. Click 'Refresh Data' to fetch funds information.")
                        st.markdown('</div>', unsafe_allow_html=True)
                
                # Holdings Section
                st.markdown('<h3 class="section-header">Portfolio Holdings</h3>', unsafe_allow_html=True)
                
                if st.session_state.portfolio_data:
                    portfolio = st.session_state.portfolio_data
                    
                    if isinstance(portfolio, list) and len(portfolio) > 0:
                        # Create a DataFrame from portfolio data
                        holdings_data = []
                        
                        for holding in portfolio:
                            holdings_data.append({
                                'Symbol': holding.get('tradingsymbol', 'N/A'),
                                'Exchange': holding.get('exchange', 'N/A'),
                                'Quantity': holding.get('quantity', 0),
                                'Average Price': holding.get('averageprice', 0),
                                'LTP': holding.get('ltp', 0),
                                'Current Value': holding.get('currentvalue', 0),
                                'P&L': holding.get('pl', 0),
                                'P&L %': holding.get('plpercent', 0)
                            })
                        
                        if holdings_data:
                            holdings_df = pd.DataFrame(holdings_data)
                            st.dataframe(holdings_df, use_container_width=True)
                        else:
                            st.markdown('<div class="card">', unsafe_allow_html=True)
                            st.write("No holdings found.")
                            st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.write("No holdings found.")
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.write("No portfolio data available. Click 'Refresh Data' to fetch holdings information.")
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.write("AngelOne API not connected. Please update your API credentials in the Account tab.")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Stock Prediction Tab
        with tab2:
            st.markdown('<h2 class="sub-header">Stock Prediction</h2>', unsafe_allow_html=True)
            
            if not st.session_state.api_connector:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.write("AngelOne API not connected. Please update your API credentials in the Account tab.")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                # Stock search and selection section
                st.markdown('<h3 class="section-header">Search Stocks</h3>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    search_query = st.text_input("Search by Name or Token", 
                                                placeholder="e.g., RELIANCE, TATA, 2885")
                
                with col2:
                    search_exchange = st.selectbox("Exchange", 
                                                  ["NSE", "BSE", "NFO", "CDS", "MCX"])
                
                with col3:
                    st.write("")
                    st.write("")
                    search_button = st.button("Search")
                
                if search_button and search_query:
                    with st.spinner("Searching..."):
                        # Search instruments using database
                        results = search_instruments(search_query, search_exchange)
                        st.session_state.search_results = results
                
                # Display search results if available
                if st.session_state.search_results:
                    if len(st.session_state.search_results) > 0:
                        st.markdown('<h3 class="section-header">Search Results</h3>', unsafe_allow_html=True)
                        
                        # Create a DataFrame from search results
                        results_data = [{
                            'Name': result.get('name', 'N/A'),
                            'Symbol': result.get('symbol', 'N/A'),
                            'Token': result.get('token', 'N/A'),
                            'Exchange': result.get('exchange', 'N/A'),
                            'Type': result.get('instrument_type', 'N/A')
                        } for result in st.session_state.search_results]
                        
                        results_df = pd.DataFrame(results_data)
                        result_selection = st.selectbox(
                            "Select an instrument",
                            range(len(results_df)),
                            format_func=lambda i: f"{results_df.iloc[i]['Name']} - {results_df.iloc[i]['Symbol']} ({results_df.iloc[i]['Token']}) - {results_df.iloc[i]['Exchange']}"
                        )
                        
                        selected_result = st.session_state.search_results[result_selection]
                        
                        # Display instrument details
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown('<div class="card">', unsafe_allow_html=True)
                            st.write(f"**Name:** {selected_result.get('name', 'N/A')}")
                            st.write(f"**Symbol:** {selected_result.get('symbol', 'N/A')}")
                            st.write(f"**Token:** {selected_result.get('token', 'N/A')}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown('<div class="card">', unsafe_allow_html=True)
                            st.write(f"**Exchange:** {selected_result.get('exchange', 'N/A')}")
                            st.write(f"**Type:** {selected_result.get('instrument_type', 'N/A')}")
                            st.write(f"**Lot Size:** {selected_result.get('lot_size', 'N/A')}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Fetch data form
                        st.markdown('<h3 class="section-header">Fetch Historical Data</h3>', unsafe_allow_html=True)
                        
                        with st.form("fetch_data_form"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Use the selected token
                                token = selected_result.get('token', '')
                                
                                # Exchange selection
                                exchange = st.selectbox(
                                    "Exchange",
                                    ["NSE", "BSE", "NFO", "CDS", "MCX"],
                                    index=["NSE", "BSE", "NFO", "CDS", "MCX"].index(selected_result.get('exchange', 'NSE')) if selected_result.get('exchange') in ["NSE", "BSE", "NFO", "CDS", "MCX"] else 0
                                )
                                
                                # Interval selection
                                interval = st.selectbox(
                                    "Time Interval",
                                    [
                                        "ONE_MINUTE", "FIVE_MINUTE", "FIFTEEN_MINUTE",
                                        "THIRTY_MINUTE", "ONE_HOUR", "ONE_DAY"
                                    ]
                                )
                            
                            with col2:
                                # Date selection
                                today = datetime.date.today()
                                from_date = st.date_input(
                                    "From Date",
                                    today - datetime.timedelta(days=365)  # Default to 1 year of data
                                )
                                to_date = st.date_input(
                                    "To Date",
                                    today
                                )
                            
                            fetch_button = st.form_submit_button("Fetch Data")
                            
                            if fetch_button:
                                with st.spinner("Fetching historical data..."):
                                    # Convert dates to strings
                                    from_date_str = from_date.strftime("%Y-%m-%d")
                                    to_date_str = to_date.strftime("%Y-%m-%d")
                                    
                                    # Fetch data
                                    data, message = fetch_stock_data(
                                        token=token,
                                        exchange=exchange,
                                        interval=interval,
                                        start_date=from_date_str,
                                        end_date=to_date_str
                                    )
                                    
                                    if data is not None:
                                        display_success(f"Successfully fetched data for {selected_result.get('name', token)}")
                                    else:
                                        display_error(message)
                    else:
                        st.write("No instruments found matching your search")
                
                # Display fetched data
                if st.session_state.stock_data is not None:
                    st.markdown('<h3 class="section-header">Historical Data</h3>', unsafe_allow_html=True)
                    
                    # Plot the stock data
                    fig = plot_stock_data(st.session_state.stock_data)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show raw data in expandable section
                    with st.expander("View Raw Data"):
                        st.dataframe(st.session_state.stock_data)
                    
                    # Generate predictions section
                    st.markdown('<h3 class="section-header">Generate Predictions</h3>', unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        model_type = st.selectbox(
                            "Select Model",
                            ["LSTM", "GRU", "Transformer", "ARIMA", "Prophet"],
                            index=0,
                            help="LSTM, GRU and Transformer are deep learning models. ARIMA and Prophet are statistical models."
                        )
                    
                    with col2:
                        forecast_days = st.slider(
                            "Forecast Days",
                            min_value=1,
                            max_value=30,
                            value=7
                        )
                    
                    with col3:
                        st.write("")
                        st.write("")
                        predict_button = st.button("Generate Predictions")
                    
                    if predict_button:
                        with st.spinner(f"Generating predictions with {model_type} model..."):
                            success, message = generate_predictions(model_type, forecast_days)
                            
                            if success:
                                display_success(message)
                            else:
                                display_error(message)
                    
                    # Display predictions if available
                    if st.session_state.predictions:
                        st.markdown('<h3 class="section-header">Prediction Results</h3>', unsafe_allow_html=True)
                        
                        # Determine which prediction to show
                        model_key = model_type.lower()
                        
                        if model_key in st.session_state.predictions:
                            prediction_data = st.session_state.predictions[model_key]
                            
                            # Display metrics
                            metrics = prediction_data['metrics']
                            col1, col2, col3 = st.columns(3)
                            col1.metric("RMSE", f"{metrics['rmse']:.4f}")
                            col2.metric("MAE", f"{metrics['mae']:.4f}")
                            col3.metric("MAPE (%)", f"{metrics['mape']:.2f}")
                            
                            # Create symbol info dict
                            symbol_info = None
                            if st.session_state.selected_stock:
                                symbol_info = {
                                    'name': st.session_state.selected_stock,
                                    'token': st.session_state.selected_token,
                                }
                            
                            # Create date range tuple
                            date_range = None
                            if not st.session_state.stock_data.empty:
                                start_date = st.session_state.stock_data.index[0].strftime('%Y-%m-%d')
                                end_date = st.session_state.stock_data.index[-1].strftime('%Y-%m-%d')
                                date_range = (start_date, end_date)
                            
                            # Plot predictions
                            fig, table_fig = plot_predictions(
                                st.session_state.stock_data,
                                prediction_data['predictions'],
                                f"{model_type} Predictions",
                                symbol_info=symbol_info,
                                date_range=date_range,
                                model_name=model_type
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display prediction table
                            st.markdown(f"### {model_type} Model: Predicted OHLC Values (Trading Days Only)")
                            st.plotly_chart(table_fig, use_container_width=True)
                            
                            # Export options
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button(f"📥 Download Chart as JSON", key=f"{model_key}_chart_json"):
                                    json_link = get_download_link(fig, f"{model_key}_chart", "json")
                                    st.markdown(f"[Download Chart JSON]({json_link})", unsafe_allow_html=True)
                            
                            with col2:
                                if st.button(f"📥 Download Table as HTML", key=f"{model_key}_table_html"):
                                    html_link = get_html_table_download(table_fig, f"{model_key}_predictions")
                                    st.markdown(f"[Download HTML Table]({html_link})", unsafe_allow_html=True)
                        else:
                            st.write(f"No {model_type} predictions available. Please generate predictions first.")
        
        # Past Predictions Tab
        with tab3:
            st.markdown('<h2 class="sub-header">Past Predictions</h2>', unsafe_allow_html=True)
            
            if not st.session_state.user_id:
                st.write("Please log in to view your past predictions.")
            else:
                # Refresh button
                if st.button("Refresh Predictions"):
                    st.rerun()
                
                # Fetch user's predictions
                from utils.database import get_user_predictions
                predictions = get_user_predictions(st.session_state.user_id, limit=20)
                
                if predictions and len(predictions) > 0:
                    # Group predictions by token
                    tokens = {}
                    for prediction in predictions:
                        token = prediction['token']
                        if token not in tokens:
                            tokens[token] = []
                        tokens[token].append(prediction)
                    
                    # Create tabs for each token
                    token_tabs = st.tabs(list(tokens.keys()))
                    
                    for i, (token, token_predictions) in enumerate(tokens.items()):
                        with token_tabs[i]:
                            st.markdown(f"<h3 class='section-header'>Predictions for Token {token}</h3>", unsafe_allow_html=True)
                            
                            # Sort predictions by date (most recent first)
                            token_predictions.sort(key=lambda x: x['created_at'], reverse=True)
                            
                            # Display each prediction
                            for prediction in token_predictions:
                                col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                                    st.markdown(f"<div class='prediction-date'>Created: {prediction['created_at']}</div>", unsafe_allow_html=True)
                                    st.write(f"**Model:** {prediction['model_type']}")
                                    st.write(f"**Forecast Period:** {prediction['forecast_start_date']} to {prediction['forecast_end_date']}")
                                    
                                    # Display accuracy if available
                                    if prediction.get('accuracy') is not None:
                                        accuracy = prediction['accuracy']
                                        if accuracy >= 70:
                                            st.markdown(f"**Accuracy:** <span class='accuracy-high'>{accuracy:.2f}%</span>", unsafe_allow_html=True)
                                        elif accuracy >= 50:
                                            st.markdown(f"**Accuracy:** <span class='accuracy-medium'>{accuracy:.2f}%</span>", unsafe_allow_html=True)
                                        else:
                                            st.markdown(f"**Accuracy:** <span class='accuracy-low'>{accuracy:.2f}%</span>", unsafe_allow_html=True)
                                    else:
                                        st.write("**Accuracy:** Not yet calculated")
                                    
                                    st.markdown('</div>', unsafe_allow_html=True)
                                
                                with col2:
                                    st.write("")
                                    st.write("")
                                    
                                    # Button to view prediction details
                                    if st.button("View Details", key=f"view_{prediction['prediction_id']}"):
                                        # Get full prediction data
                                        full_prediction = get_prediction_by_id(prediction['prediction_id'])
                                        
                                        if full_prediction:
                                            # Convert prediction_data to DataFrame
                                            prediction_df = pd.DataFrame.from_dict(full_prediction['prediction_data'], orient='index')
                                            prediction_df.index = pd.to_datetime(prediction_df.index)
                                            
                                            # Plot the prediction
                                            st.markdown(f"### {full_prediction['model_type']} Prediction Details")
                                            
                                            # Create symbol info dict
                                            symbol_info = {
                                                'name': full_prediction.get('name', 'Unknown'),
                                                'token': full_prediction['token'],
                                            }
                                            
                                            # Plot predictions (if stock data is available)
                                            if st.session_state.stock_data is not None and st.session_state.selected_token == full_prediction['token']:
                                                fig, table_fig = plot_predictions(
                                                    st.session_state.stock_data,
                                                    prediction_df,
                                                    f"{full_prediction['model_type']} Predictions",
                                                    symbol_info=symbol_info,
                                                    model_name=full_prediction['model_type']
                                                )
                                                st.plotly_chart(fig, use_container_width=True)
                                            else:
                                                # Only show prediction table
                                                st.write("Historical data not available. Showing prediction table only.")
                                            
                                            # Display prediction table
                                            table_fig = create_styled_prediction_table(
                                                prediction_df,
                                                symbol_info=symbol_info,
                                                model_name=full_prediction['model_type']
                                            )
                                            st.plotly_chart(table_fig, use_container_width=True)
                                            
                                            # Display metrics
                                            metrics = json.loads(full_prediction['metrics'])
                                            st.markdown("### Performance Metrics")
                                            col1, col2, col3 = st.columns(3)
                                            col1.metric("RMSE", f"{metrics['rmse']:.4f}")
                                            col2.metric("MAE", f"{metrics['mae']:.4f}")
                                            col3.metric("MAPE (%)", f"{metrics['mape']:.2f}")
                                            
                                            # Option to update accuracy
                                            st.markdown("### Update Accuracy")
                                            if st.button("Calculate Accuracy", key=f"calc_acc_{prediction['prediction_id']}"):
                                                with st.spinner("Calculating accuracy..."):
                                                    success, message, accuracy = update_prediction_accuracy_with_new_data(
                                                        prediction_id=prediction['prediction_id'],
                                                        token=prediction['token'],
                                                        exchange=prediction['exchange'],
                                                        api_connector=st.session_state.api_connector
                                                    )
                                                    
                                                    if success:
                                                        display_success(f"Accuracy updated: {accuracy:.2f}%")
                                                    else:
                                                        display_error(message)
                                        else:
                                            st.error("Failed to retrieve prediction details")
                                    
                                    # Button to check accuracy
                                    if st.button("Check Accuracy", key=f"check_{prediction['prediction_id']}"):
                                        with st.spinner("Checking accuracy..."):
                                            success, message, accuracy = update_prediction_accuracy_with_new_data(
                                                prediction_id=prediction['prediction_id'],
                                                token=prediction['token'],
                                                exchange=prediction['exchange'],
                                                api_connector=st.session_state.api_connector
                                            )
                                            
                                            if success:
                                                display_success(f"Accuracy calculated: {accuracy:.2f}%")
                                                st.rerun()
                                            else:
                                                display_error(message)
                else:
                    st.write("No predictions found. Generate some predictions first.")
        
        # Instruments Tab
        with tab4:
            st.markdown('<h2 class="sub-header">Instruments</h2>', unsafe_allow_html=True)
            
            # Instrument search and navigation
            st.markdown('<h3 class="section-header">Search Instruments</h3>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                instrument_search_query = st.text_input("Search by Name, Symbol or Token", 
                                            placeholder="e.g., RELIANCE, TATA, 2885", key="instrument_search_query")
            
            with col2:
                instrument_search_exchange = st.selectbox("Exchange", 
                                          ["NSE", "BSE", "NFO", "CDS", "MCX"], key="instrument_search_exchange")
            
            with col3:
                st.write("")
                st.write("")
                instrument_search_button = st.button("Search", key="instrument_search_button")
            
            if instrument_search_button and instrument_search_query:
                with st.spinner("Searching..."):
                    # Search instruments using database with improved search
                    results = search_instruments(instrument_search_query, instrument_search_exchange, limit=50)
                    st.session_state.instrument_search_results = results
            
            # Display search results if available
            if hasattr(st.session_state, 'instrument_search_results') and st.session_state.instrument_search_results:
                results_count = len(st.session_state.instrument_search_results)
                if results_count > 0:
                    st.markdown(f'<h3 class="section-header">Search Results ({results_count} found)</h3>', unsafe_allow_html=True)
                    
                    # Create a DataFrame from search results
                    results_data = [{
                        'Name': result.get('name', 'N/A'),
                        'Symbol': result.get('symbol', 'N/A'),
                        'Token': result.get('token', 'N/A'),
                        'Exchange': result.get('exchange', 'N/A'),
                        'Type': result.get('instrument_type', 'N/A'),
                        'Lot Size': result.get('lot_size', 'N/A'),
                        'Tick Size': result.get('tick_size', 'N/A')
                    } for result in st.session_state.instrument_search_results]
                    
                    # Convert to DataFrame and display
                    results_df = pd.DataFrame(results_data)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Option to select an instrument for detailed view
                    if st.checkbox("View Instrument Details"):
                        result_selection = st.selectbox(
                            "Select an instrument for detailed view",
                            range(len(results_df)),
                            format_func=lambda i: f"{results_df.iloc[i]['Name']} - {results_df.iloc[i]['Symbol']} ({results_df.iloc[i]['Token']}) - {results_df.iloc[i]['Exchange']}"
                        )
                        
                        selected_result = st.session_state.instrument_search_results[result_selection]
                        
                        # Display detailed instrument information in a card
                        st.markdown('<div class="card instrument-detail-card">', unsafe_allow_html=True)
                        st.markdown(f'<h4>{selected_result.get("name", "N/A")} ({selected_result.get("symbol", "N/A")})</h4>', unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Symbol:** {selected_result.get('symbol', 'N/A')}")
                            st.write(f"**Name:** {selected_result.get('name', 'N/A')}")
                            st.write(f"**Token:** {selected_result.get('token', 'N/A')}")
                            st.write(f"**Exchange:** {selected_result.get('exchange', 'N/A')}")
                        
                        with col2:
                            st.write(f"**Type:** {selected_result.get('instrument_type', 'N/A')}")
                            st.write(f"**Lot Size:** {selected_result.get('lot_size', 'N/A')}")
                            st.write(f"**Tick Size:** {selected_result.get('tick_size', 'N/A')}")
                            st.write(f"**Expiry:** {selected_result.get('expiry', 'N/A')}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Add quick actions for the selected instrument
                        st.markdown('<h4 class="section-header">Quick Actions</h4>', unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("Fetch Historical Data", key="fetch_instrument_data"):
                                # Set up session state for the Stock Prediction tab
                                st.session_state.search_results = [selected_result]
                                st.write("Historical data selection ready in Stock Prediction tab.")
                                st.write("Please switch to the Stock Prediction tab to continue.")
                        
                        with col2:
                            if st.button("View Latest Price", key="view_instrument_price"):
                                with st.spinner("Fetching latest price..."):
                                    try:
                                        if st.session_state.extended_api:
                                            token = selected_result.get('token', '')
                                            exchange = selected_result.get('exchange', '')
                                            ltp_data = st.session_state.extended_api.get_ltp_data([token], exchange)
                                            
                                            if ltp_data and ltp_data.get("status"):
                                                st.success("Price data fetched successfully")
                                                st.metric("Latest Price", f"₹ {ltp_data.get('data', {}).get('ltp', 0):,.2f}")
                                                st.write(f"**Last Updated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                                            else:
                                                st.error("Failed to fetch latest price")
                                        else:
                                            st.error("API not connected. Please update API credentials in Account tab.")
                                    except Exception as e:
                                        st.error(f"Error fetching price: {str(e)}")
                else:
                    st.info("No instruments found matching your search criteria. Try using different keywords.")
            
            # Instrument master management
            st.markdown('<h3 class="section-header">Instrument Master Management</h3>', unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Fetch Instrument Master", key="fetch_instrument_master"):
                    with st.spinner("Fetching instrument master..."):
                        try:
                            result = fetch_instrument_master(save_to_db=True)
                            if not result.empty:
                                display_success(f"Fetched {len(result)} instruments")
                            else:
                                display_error("Failed to fetch instruments")
                        except Exception as e:
                            display_error(f"Error: {str(e)}")
            
            with col2:
                # Get count of instruments
                import sqlite3
                conn = sqlite3.connect("stock_prediction.db")
                cursor = conn.cursor()
                
                try:
                    cursor.execute("SELECT COUNT(*) FROM instruments")
                    count = cursor.fetchone()[0]
                    st.metric("Instruments in Database", f"{count:,}")
                except:
                    st.write("Instruments table not found or empty")
                
                conn.close()
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Account Tab
        with tab5:
            st.markdown('<h2 class="sub-header">Account Settings</h2>', unsafe_allow_html=True)
            
            # User Information
            st.markdown('<h3 class="section-header">User Information</h3>', unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write(f"**Username:** {st.session_state.username}")
            st.write(f"**User ID:** {st.session_state.user_id}")
            st.write(f"**API Status:** {'Connected' if st.session_state.api_connector else 'Not Connected'}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # API Credentials Form
            st.markdown('<h3 class="section-header">AngelOne API Credentials</h3>', unsafe_allow_html=True)
            
            with st.form("api_credentials_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    api_key = st.text_input("API Key", type="password")
                    api_secret = st.text_input("API Secret", type="password")
                    client_id = st.text_input("Client ID")
                
                with col2:
                    mpin = st.text_input("MPIN", type="password")
                    totp_secret = st.text_input("TOTP Secret", help="Enter the TOTP secret key from your AngelOne account settings (not the 6-digit code)")
                
                submitted = st.form_submit_button("Update API Credentials")
                
                if submitted:
                    if api_key and api_secret and client_id and mpin and totp_secret:
                        with st.spinner("Updating API credentials..."):
                            success, message = handle_api_credentials_update(
                                api_key=api_key,
                                api_secret=api_secret,
                                client_id=client_id,
                                mpin=mpin,
                                totp_secret=totp_secret
                            )
                            
                            if success:
                                display_success(message)
                                # Refresh user data
                                if st.session_state.extended_api:
                                    refresh_user_data()
                                st.rerun()
                            else:
                                display_error(message)
                    else:
                        display_error("Please fill in all fields")
            
            # Logout button
            if st.button("Logout"):
                handle_logout()
        
        # API Debug Tab
        with tab6:
            st.markdown('<h2 class="sub-header">API Debug</h2>', unsafe_allow_html=True)
            
            # API Status
            st.markdown('<h3 class="section-header">API Status</h3>', unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            if st.session_state.api_connector:
                st.write("✅ API Connected")
                
                # Display token information with MPIN protection
                st.markdown('<h4>Session Tokens</h4>', unsafe_allow_html=True)
                
                # Create a toggle for showing/hiding tokens
                show_tokens = False
                
                # Create a form for MPIN verification
                with st.expander("Show Tokens (MPIN Required)"):
                    mpin_for_token = st.text_input("Enter your MPIN to view tokens", type="password")
                    verify_button = st.button("Verify MPIN")
                    
                    if verify_button:
                        # Get user from database to verify MPIN
                        from utils.database import get_user_by_id
                        user = get_user_by_id(st.session_state.user_id)
                        
                        if user and user.get('mpin') == mpin_for_token:
                            show_tokens = True
                            st.success("MPIN verified successfully")
                        else:
                            st.error("Invalid MPIN. Please try again.")
                    
                    if show_tokens:
                        st.markdown('<div class="token-display">', unsafe_allow_html=True)
                        st.write(f"JWT Token: {st.session_state.jwt_token if st.session_state.jwt_token else 'Not available'}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="token-display">', unsafe_allow_html=True)
                        st.write(f"Refresh Token: {st.session_state.refresh_token if st.session_state.refresh_token else 'Not available'}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="token-display">', unsafe_allow_html=True)
                        st.write("JWT Token: ********** (hidden)")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="token-display">', unsafe_allow_html=True)
                        st.write("Refresh Token: ********** (hidden)")
                        st.markdown('</div>', unsafe_allow_html=True)
                
                # Token refresh button
                if st.button("Refresh Token"):
                    with st.spinner("Refreshing token..."):
                        try:
                            if st.session_state.api_connector.refresh_token_if_needed():
                                # Update session state
                                st.session_state.jwt_token = st.session_state.api_connector.session_token
                                st.session_state.refresh_token = st.session_state.api_connector.refresh_token
                                
                                # Update extended API
                                if st.session_state.extended_api:
                                    api_instance = get_api_instance()
                                    api_instance.initialize(
                                        api_key=st.session_state.api_connector.api_key,
                                        api_secret=st.session_state.api_connector.api_secret,
                                        client_id=st.session_state.api_connector.client_id,
                                        mpin=st.session_state.api_connector.mpin,
                                        totp_secret=st.session_state.api_connector.totp_secret
                                    )
                                    st.session_state.extended_api = api_instance
                                
                                display_success("Token refreshed successfully!")
                                st.rerun()
                            else:
                                display_warning("Token does not need to be refreshed yet")
                        except Exception as e:
                            display_error(f"Error refreshing token: {str(e)}")
                
                # API Test buttons
                st.markdown('<h4>API Tests</h4>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("Test User Profile"):
                        with st.spinner("Fetching user profile..."):
                            try:
                                if st.session_state.extended_api:
                                    response = st.session_state.extended_api.get_user_profile()
                                    if response.get("status"):
                                        st.success("Profile API working")
                                        st.json(response)
                                    else:
                                        st.error("Profile API failed")
                                        st.json(response)
                                else:
                                    st.error("Extended API not initialized")
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                
                with col2:
                    if st.button("Test Funds & Margins"):
                        with st.spinner("Fetching funds and margins..."):
                            try:
                                if st.session_state.extended_api:
                                    response = st.session_state.extended_api.get_funds_and_margins()
                                    if response.get("status"):
                                        st.success("Funds API working")
                                        st.json(response)
                                    else:
                                        st.error("Funds API failed")
                                        st.json(response)
                                else:
                                    st.error("Extended API not initialized")
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                
                with col3:
                    if st.button("Test Holdings"):
                        with st.spinner("Fetching holdings..."):
                            try:
                                if st.session_state.extended_api:
                                    response = st.session_state.extended_api.get_holdings()
                                    if response.get("status"):
                                        st.success("Holdings API working")
                                        st.json(response)
                                    else:
                                        st.error("Holdings API failed")
                                        st.json(response)
                                else:
                                    st.error("Extended API not initialized")
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
            else:
                st.write("❌ API Not Connected")
                st.write("Please update your API credentials in the Account tab to connect to the AngelOne API.")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Database Status
            st.markdown('<h3 class="section-header">Database Status</h3>', unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            try:
                # Check database
                import sqlite3
                conn = sqlite3.connect("stock_prediction.db")
                cursor = conn.cursor()
                
                # Get table counts
                table_counts = {}
                tables = [
                    "users", "user_profiles", "funds_and_margins", "instruments",
                    "stock_data", "trained_models", "predictions", "user_portfolios",
                    "user_preferences"
                ]
                
                for table in tables:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cursor.fetchone()[0]
                        table_counts[table] = count
                    except:
                        table_counts[table] = "Not created"
                
                # Display table counts
                st.markdown('<h4>Database Tables</h4>', unsafe_allow_html=True)
                
                for table, count in table_counts.items():
                    st.write(f"**{table}:** {count}")
                
                conn.close()
                
                # Add database maintenance buttons
                st.markdown('<h4>Database Maintenance</h4>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Fetch Instrument Master"):
                        with st.spinner("Fetching instrument master..."):
                            try:
                                result = fetch_instrument_master(save_to_db=True)
                                if not result.empty:
                                    display_success(f"Fetched {len(result)} instruments")
                                else:
                                    display_error("Failed to fetch instruments")
                            except Exception as e:
                                display_error(f"Error: {str(e)}")
                
                with col2:
                    if st.button("Initialize Database"):
                        with st.spinner("Initializing database..."):
                            try:
                                initialize_database()
                                display_success("Database initialized successfully")
                            except Exception as e:
                                display_error(f"Error: {str(e)}")
            except Exception as e:
                st.error(f"Error accessing database: {str(e)}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    st.markdown('Stock Market Prediction App using AngelOne SmartAPI and Advanced Machine Learning', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()