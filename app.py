import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import requests
import json
import os
from utils.api_connector import AngelOneConnector
from utils.data_processor import preprocess_data
from utils.prediction_models import train_arima_model, train_prophet_model, train_lstm_model
from utils.visualization import plot_stock_data, plot_predictions, get_download_link, get_html_table_download
# Import for instrument search
from utils.instrument_master import get_instrument_manager

# Set page configuration
st.set_page_config(
    page_title="Stock Market Prediction App",
    page_icon="📈",
    layout="wide",
)

# Initialize session state variables if they don't exist
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'api_connector' not in st.session_state:
    st.session_state.api_connector = None
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = None
if 'error_message' not in st.session_state:
    st.session_state.error_message = None
if 'jwt_token' not in st.session_state:
    st.session_state.jwt_token = None
if 'refresh_token' not in st.session_state:
    st.session_state.refresh_token = None


# Define function to handle login
def handle_login():
    try:
        # Clear any previous error messages
        st.session_state.error_message = None

        # Create API connector with user credentials
        connector = AngelOneConnector(api_key=api_key,
                                      api_secret=api_secret,
                                      client_id=client_id,
                                      mpin=mpin,
                                      totp=totp)

        # Attempt to login
        success, message = connector.login()
        if success:
            st.session_state.logged_in = True
            st.session_state.api_connector = connector

            # Store tokens in session state for display
            st.session_state.jwt_token = connector.session_token
            st.session_state.refresh_token = connector.refresh_token

            st.success(message)
            time.sleep(1)
            st.rerun()
        else:
            st.session_state.error_message = message
    except Exception as e:
        st.session_state.error_message = f"Error during login: {str(e)}"


# Define function to fetch stock data
def fetch_stock_data():
    try:
        st.session_state.error_message = None

        if not symbol or not from_date or not to_date:
            st.session_state.error_message = "Please provide all required fields."
            return

        # Convert dates to the required format
        from_date_str = from_date.strftime("%Y-%m-%d")
        to_date_str = to_date.strftime("%Y-%m-%d")

        # Fetch historical data
        data = st.session_state.api_connector.get_historical_data(
            symbol=symbol,
            exchange="NSE",  # Default to NSE, could be made selectable
            interval=interval,
            from_date=from_date_str,
            to_date=to_date_str)

        if data is not None and not data.empty:
            st.session_state.stock_data = data
            st.session_state.selected_stock = symbol
            st.success(f"Successfully fetched data for {symbol}")
            # Show message for a better UI experience
            st.info("Scroll down to view the stock data visualization")
        else:
            st.session_state.error_message = "Failed to fetch data. Please check the symbol and try again."
            st.error(
                st.session_state.error_message +
                " If you recently logged in, the token might need to be refreshed. Try logging out and back in."
            )
    except Exception as e:
        error_msg = f"Error fetching stock data: {str(e)}"
        st.session_state.error_message = error_msg

        # Display user-friendly error message
        if "Token" in str(e) or "token" in str(e):
            st.error(error_msg +
                     " Token may have expired. Try logging out and back in.")
        elif "Symbol" in str(e) or "symbol" in str(e):
            st.error(
                "Symbol not found. Please try a different stock symbol or check if it's listed on NSE."
            )
        else:
            st.error(error_msg + " Please check your inputs and try again.")


# Define function to generate predictions
def generate_predictions():
    try:
        st.session_state.error_message = None

        if st.session_state.stock_data is None:
            st.session_state.error_message = "No stock data available. Please fetch data first."
            return

        # Preprocess the data
        processed_data = preprocess_data(st.session_state.stock_data)

        # Generate predictions using different models
        with st.spinner("Generating predictions with ARIMA model..."):
            arima_predictions, arima_metrics = train_arima_model(
                processed_data, forecast_days)

        with st.spinner("Generating predictions with Prophet model..."):
            prophet_predictions, prophet_metrics = train_prophet_model(
                processed_data, forecast_days)

        with st.spinner("Generating predictions with LSTM model..."):
            lstm_predictions, lstm_metrics = train_lstm_model(
                processed_data, forecast_days)

        # Store predictions and metrics
        st.session_state.predictions = {
            'arima': {
                'predictions': arima_predictions,
                'metrics': arima_metrics
            },
            'prophet': {
                'predictions': prophet_predictions,
                'metrics': prophet_metrics
            },
            'lstm': {
                'predictions': lstm_predictions,
                'metrics': lstm_metrics
            }
        }

        st.success("Predictions generated successfully!")
        st.info("Scroll down to view prediction results and model comparison")
    except Exception as e:
        st.session_state.error_message = f"Error generating predictions: {str(e)}"


# Main application interface
st.title("Stock Market Prediction App")

# Display error message if any
if st.session_state.error_message:
    st.error(st.session_state.error_message)

# Login section
if not st.session_state.logged_in:
    st.subheader("Login to AngelOne")

    with st.form("login_form"):
        col1, col2 = st.columns(2)
        with col1:
            api_key = st.text_input("API Key", type="password", value="gzsmaiQY")
            api_secret = st.text_input("API Secret", type="password", value="faeba78b-de61-431f-818f-229aa1c6eaed")
            client_id = st.text_input("Client ID", value="P819276")
        with col2:
            mpin = st.text_input("MPIN", type="password", value="5956", help="Angel One has replaced Password with MPIN for authentication")
            totp = st.text_input("TOTP Secret Key", value="DBCVLNQT5UMSNMDGCXPXRZG46Q", help="Enter the TOTP secret key from your AngelOne account settings (not the 6-digit code). It should be base32 encoded (A-Z and 2-7 characters only)."
            )

        login_button = st.form_submit_button("Login")
        if login_button:
            # Show a temporary message while processing
            with st.spinner("Attempting to login..."):
                handle_login()
else:
    # Display token information for debugging
    st.subheader("API Tokens (For Debugging)")
    with st.expander("Show API Tokens"):
        st.warning(
            "Only use these tokens for debugging purposes. Do not share these tokens with anyone."
        )

        # Create columns for token display
        col1, col2 = st.columns(2)

        with col1:
            st.text_area("JWT Token",
                         value=st.session_state.jwt_token if 'jwt_token'
                         in st.session_state else "Not available",
                         height=100)

        with col2:
            st.text_area(
                "Refresh Token",
                value=st.session_state.refresh_token
                if 'refresh_token' in st.session_state else "Not available",
                height=100)

        # Add a button to refresh token manually
        if st.button("Refresh Token Manually"):
            try:
                with st.spinner("Attempting to refresh token..."):
                    # First try normal token refresh
                    if st.session_state.api_connector.refresh_token_if_needed(
                    ):
                        # Update displayed tokens
                        st.session_state.jwt_token = st.session_state.api_connector.session_token
                        st.session_state.refresh_token = st.session_state.api_connector.refresh_token
                        st.success("Token refreshed successfully!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.warning(
                            "Standard token refresh failed. Attempting re-login..."
                        )

                        # If that fails, try a complete re-login with current credentials
                        try:
                            # Get existing credentials from the connector
                            api_key = st.session_state.api_connector.api_key
                            api_secret = st.session_state.api_connector.api_secret
                            client_id = st.session_state.api_connector.client_id
                            mpin = st.session_state.api_connector.mpin
                            totp = st.session_state.api_connector.totp

                            # Create a new connector instance
                            connector = AngelOneConnector(
                                api_key=api_key,
                                api_secret=api_secret,
                                client_id=client_id,
                                mpin=mpin,
                                totp=totp)

                            # Attempt to login
                            success, message = connector.login()
                            if success:
                                # Replace the old connector
                                st.session_state.api_connector = connector
                                st.session_state.jwt_token = connector.session_token
                                st.session_state.refresh_token = connector.refresh_token
                                st.success(
                                    "Re-login successful! New tokens generated."
                                )
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"Re-login failed: {message}")
                        except Exception as login_error:
                            st.error(
                                f"Error during re-login: {str(login_error)}")
            except Exception as e:
                st.error(f"Error refreshing token: {str(e)}")

    # Stock data fetching section
    st.subheader("Fetch Stock Data")

    # Container for stock data fetching
    with st.container():

        # Button outside the form to initialize the instrument manager if not already done
        if 'instrument_manager_loaded' not in st.session_state:
            st.session_state.instrument_manager_loaded = False
            st.session_state.search_results = None
            st.session_state.selected_instrument = None

        # Initialize the search box
        search_col1, search_col2 = st.columns([3, 1])

        with search_col1:
            search_query = st.text_input(
                "Search by Name or Token",
                help=
                "Enter stock name (e.g., 'RELIANCE', 'HDFC') or token number",
                key="search_query")

        with search_col2:
            search_exchange = st.selectbox(
                "Filter by Exchange",
                ["ALL", "NSE", "BSE", "NFO", "CDS", "MCX"],
                index=0,
                key="search_exchange")

        # Search button
        if st.button("Search Instruments"):
            with st.spinner("Searching for instruments..."):
                # Get the instrument manager
                try:
                    manager = get_instrument_manager()
                    st.session_state.instrument_manager_loaded = True

                    # Perform the search
                    exchange_filter = None if search_exchange == "ALL" else search_exchange
                    results = manager.search_instruments(
                        search_query, exchange=exchange_filter, limit=15)

                    # Store results in session state
                    st.session_state.search_results = results
                except Exception as e:
                    st.error(f"Error searching instruments: {str(e)}")

        # Display search results if available
        if st.session_state.search_results is not None and not st.session_state.search_results.empty:
            st.subheader("Search Results")

            # Create a list of options for the selectbox
            options = []
            for _, row in st.session_state.search_results.iterrows():
                option_label = f"{row['name']} ({row['symbol']}) - {row['exch_seg']} - Token: {row['token']}"
                options.append((option_label, row))

            # Display as a selectbox
            selected_option = st.selectbox(
                "Select an instrument",
                options=[label for label, _ in options],
                key="selected_instrument_option")

            # Find the selected row
            selected_row = None
            for label, row in options:
                if label == selected_option:
                    selected_row = row
                    break

            if selected_row is not None:
                # Store the selected instrument for the form
                st.session_state.selected_instrument = selected_row

                # Display instrument details
                with st.expander("Instrument Details", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.text(f"Name: {selected_row['name']}")
                        st.text(f"Symbol: {selected_row['symbol']}")
                        st.text(f"Token: {selected_row['token']}")
                    with col2:
                        st.text(f"Exchange: {selected_row['exch_seg']}")
                        st.text(
                            f"Expiry: {selected_row['expiry'] if 'expiry' in selected_row else 'N/A'}"
                        )
                        st.text(
                            f"Lot Size: {selected_row['lotsize'] if 'lotsize' in selected_row else 'N/A'}"
                        )

        # Form for fetching data
        with st.form("fetch_by_token_form"):
            col1, col2 = st.columns(2)
            with col1:
                # Auto-fill token if an instrument is selected
                initial_token = ""
                initial_exchange = "NSE"

                if st.session_state.get('selected_instrument') is not None:
                    # Make sure token is an integer (no decimal places)
                    token_val = st.session_state.selected_instrument['token']
                    if isinstance(token_val, float):
                        token_val = int(token_val)
                    initial_token = str(token_val)
                    initial_exchange = st.session_state.selected_instrument[
                        'exch_seg']

                symbol_token = st.text_input(
                    "Symbol Token (e.g., 3045)",
                    value=initial_token,
                    help="Enter a specific symbol token if you already know it"
                )

                exchange = st.selectbox(
                    "Exchange", ["NSE", "BSE", "NFO", "CDS", "MCX"],
                    index=["NSE", "BSE", "NFO", "CDS", "MCX"
                           ].index(initial_exchange) if initial_exchange
                    in ["NSE", "BSE", "NFO", "CDS", "MCX"] else 0)

                interval = st.selectbox(
                    "Time Interval ",  # Adding a space to make it unique from the other form
                    [
                        "ONE_MINUTE", "FIVE_MINUTE", "FIFTEEN_MINUTE",
                        "THIRTY_MINUTE", "ONE_HOUR", "ONE_DAY"
                    ])
            with col2:
                today = datetime.date.today()
                from_date_token = st.date_input(
                    "From Date ",
                    today - datetime.timedelta(days=30))  # Adding a space
                to_date_token = st.date_input("To Date ",
                                              today)  # Adding a space

            fetch_token_button = st.form_submit_button("Fetch Data by Token")
            if fetch_token_button:
                with st.spinner(
                        "Fetching stock data using token... This might take a moment."
                ):
                    # Call fetch function with token
                    try:
                        st.session_state.error_message = None

                        if not symbol_token or not from_date_token or not to_date_token:
                            st.session_state.error_message = "Please provide all required fields."
                            st.error(st.session_state.error_message)
                            # Can't return here since we're not in a function
                            st.stop()

                        # Convert dates to the required format
                        from_date_str = from_date_token.strftime("%Y-%m-%d")
                        to_date_str = to_date_token.strftime("%Y-%m-%d")

                        # Fetch historical data directly with token
                        data = st.session_state.api_connector.get_historical_data_by_token(
                            token=symbol_token,
                            exchange=exchange,
                            interval=interval,
                            from_date=from_date_str,
                            to_date=to_date_str)

                        if data is not None and not data.empty:
                            st.session_state.stock_data = data

                            # Get name if available
                            stock_name = f"Token {symbol_token}"
                            if st.session_state.get('selected_instrument') is not None:
                                # Use float() to prevent potential errors with token comparison
                                try:
                                    if st.session_state.selected_instrument['token'] == int(float(symbol_token)):
                                        stock_name = f"{st.session_state.selected_instrument['name']} ({symbol_token})"
                                except (ValueError, TypeError):
                                    # Handle any conversion errors
                                    pass

                            st.session_state.selected_stock = stock_name
                            st.success(f"Successfully fetched data for {stock_name}")
                            # Show message for a better UI experience
                            st.info("Scroll down to view the stock data visualization")
                        else:
                            # More detailed error message
                            error_details = "Please check the following:\n"
                            error_details += "1. Make sure the token is correct (e.g., 2885 for Reliance from NSE)\n"
                            error_details += "2. Try logging out and logging back in to refresh the API tokens\n"
                            error_details += "3. Verify that you're using a valid date range\n"
                            error_details += "4. Ensure that the exchange selected matches the token"
                            
                            st.session_state.error_message = "Failed to fetch data. Please check the token and try again."
                            st.error(st.session_state.error_message)
                            st.info(error_details)
                    except Exception as e:
                        error_msg = f"Error fetching stock data by token: {str(e)}"
                        st.session_state.error_message = error_msg

                        # Display user-friendly error message
                        if "Token" in str(e) or "token" in str(e):
                            st.error(
                                error_msg +
                                " Token may have expired. Try logging out and back in."
                            )
                        else:
                            st.error(
                                error_msg +
                                " Please check your inputs and try again.")

    # Show data if available
    if st.session_state.stock_data is not None:
        st.subheader(f"Historical Data for {st.session_state.selected_stock}")

        # Plot the stock data
        fig = plot_stock_data(st.session_state.stock_data)
        st.plotly_chart(fig, use_container_width=True)

        # Display data table
        with st.expander("View Raw Data"):
            st.dataframe(st.session_state.stock_data)

        # Prediction section
        st.subheader("Generate Predictions")

        forecast_days = st.slider("Number of days to forecast", 1, 30, 7)

        if st.button("Generate Predictions"):
            generate_predictions()

        # Display predictions if available
        if st.session_state.predictions:
            st.subheader("Prediction Results")

            # Create tabs for different models
            tab1, tab2, tab3 = st.tabs(
                ["ARIMA Model", "Prophet Model", "Linear Regression Model"])

            with tab1:
                arima_data = st.session_state.predictions['arima']
                st.markdown("### ARIMA Model Predictions")

                # Display metrics
                metrics = arima_data['metrics']
                col1, col2, col3 = st.columns(3)
                col1.metric("RMSE", f"{metrics['rmse']:.4f}")
                col2.metric("MAE", f"{metrics['mae']:.4f}")
                col3.metric("MAPE (%)", f"{metrics['mape']:.2f}")

                # Create symbol info dict
                symbol_info = None
                if st.session_state.selected_stock:
                    symbol_info = {
                        'name': st.session_state.selected_stock,
                        'token': '',  # Add token if available in your data structure
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
                    arima_data['predictions'],
                    "ARIMA Predictions",
                    symbol_info=symbol_info,
                    date_range=date_range,
                    model_name="ARIMA"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display prediction table
                st.markdown("### ARIMA Model: Predicted OHLC Values (Trading Days Only)")
                st.plotly_chart(table_fig, use_container_width=True)
                
                # Add export options
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📥 Download Chart as JSON", key="arima_chart_json"):
                        json_link = get_download_link(fig, "arima_chart", "json")
                        st.markdown(f"[Download Chart JSON]({json_link})", unsafe_allow_html=True)
                        
                with col2:
                    if st.button("📥 Download Table as HTML", key="arima_table_html"):
                        html_link = get_html_table_download(table_fig, "arima_predictions")
                        st.markdown(f"[Download HTML Table]({html_link})", unsafe_allow_html=True)

            with tab2:
                prophet_data = st.session_state.predictions['prophet']
                st.markdown("### Prophet Model Predictions")

                # Display metrics
                metrics = prophet_data['metrics']
                col1, col2, col3 = st.columns(3)
                col1.metric("RMSE", f"{metrics['rmse']:.4f}")
                col2.metric("MAE", f"{metrics['mae']:.4f}")
                col3.metric("MAPE (%)", f"{metrics['mape']:.2f}")

                # Plot predictions
                fig, table_fig = plot_predictions(
                    st.session_state.stock_data,
                    prophet_data['predictions'],
                    "Prophet Predictions",
                    symbol_info=symbol_info,
                    date_range=date_range,
                    model_name="Prophet"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display prediction table
                st.markdown("### Prophet Model: Predicted OHLC Values (Trading Days Only)")
                st.plotly_chart(table_fig, use_container_width=True)
                
                # Add export options
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📥 Download Chart as JSON", key="prophet_chart_json"):
                        json_link = get_download_link(fig, "prophet_chart", "json")
                        st.markdown(f"[Download Chart JSON]({json_link})", unsafe_allow_html=True)
                        
                with col2:
                    if st.button("📥 Download Table as HTML", key="prophet_table_html"):
                        html_link = get_html_table_download(table_fig, "prophet_predictions")
                        st.markdown(f"[Download HTML Table]({html_link})", unsafe_allow_html=True)

            with tab3:
                lstm_data = st.session_state.predictions['lstm']
                st.markdown("### Linear Regression Model Predictions")

                # Display metrics
                metrics = lstm_data['metrics']
                col1, col2, col3 = st.columns(3)
                col1.metric("RMSE", f"{metrics['rmse']:.4f}")
                col2.metric("MAE", f"{metrics['mae']:.4f}")
                col3.metric("MAPE (%)", f"{metrics['mape']:.2f}")

                # Plot predictions
                fig, table_fig = plot_predictions(
                    st.session_state.stock_data,
                    lstm_data['predictions'],
                    "Linear Regression Predictions",
                    symbol_info=symbol_info,
                    date_range=date_range,
                    model_name="Linear Regression"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display prediction table
                st.markdown("### Linear Regression Model: Predicted OHLC Values (Trading Days Only)")
                st.plotly_chart(table_fig, use_container_width=True)
                
                # Add export options
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📥 Download Chart as JSON", key="linreg_chart_json"):
                        json_link = get_download_link(fig, "linreg_chart", "json")
                        st.markdown(f"[Download Chart JSON]({json_link})", unsafe_allow_html=True)
                        
                with col2:
                    if st.button("📥 Download Table as HTML", key="linreg_table_html"):
                        html_link = get_html_table_download(table_fig, "linreg_predictions")
                        st.markdown(f"[Download HTML Table]({html_link})", unsafe_allow_html=True)

            # Model comparison
            st.subheader("Model Comparison")

            comparison_data = {
                'Model': ['ARIMA', 'Prophet', 'Linear Regression'],
                'RMSE': [
                    st.session_state.predictions['arima']['metrics']['rmse'],
                    st.session_state.predictions['prophet']['metrics']['rmse'],
                    st.session_state.predictions['lstm']['metrics']['rmse']
                ],
                'MAE': [
                    st.session_state.predictions['arima']['metrics']['mae'],
                    st.session_state.predictions['prophet']['metrics']['mae'],
                    st.session_state.predictions['lstm']['metrics']['mae']
                ],
                'MAPE (%)': [
                    st.session_state.predictions['arima']['metrics']['mape'],
                    st.session_state.predictions['prophet']['metrics']['mape'],
                    st.session_state.predictions['lstm']['metrics']['mape']
                ]
            }

            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df)

            # Determine best model
            best_model_idx = comparison_df['RMSE'].idxmin()
            best_model = comparison_df.iloc[best_model_idx]['Model']

            st.info(
                f"Based on RMSE, the {best_model} model provides the best predictions for {st.session_state.selected_stock}."
            )

    # Logout button
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.api_connector = None
        st.session_state.stock_data = None
        st.session_state.predictions = {}
        st.session_state.selected_stock = None
        st.rerun()

# Display footer
st.markdown("---")
st.markdown(
    "Stock Market Prediction App using AngelOne SmartAPI and Machine Learning")
