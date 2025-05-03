"""
Services module for background tasks and scheduled operations.
"""

import os
import json
import time
import logging
import threading
import schedule
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hashlib
import sqlite3
import time

import bcrypt
import pyotp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/services.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Database file
DB_FILE = 'stock_prediction.db'

# Global variables
scheduler_thread = None
scheduler_stop_event = threading.Event()
api_instance = None

def initialize_system():
    """Initialize the entire system including database, services, and API"""
    try:
        # Initialize database
        from utils.database import initialize_database
        initialize_database()
        logger.info("Database initialized")
        
        # Initialize services
        initialize_services()
        logger.info("System initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing system: {str(e)}")
        return False

def initialize_services():
    """Initialize services including scheduler and API"""
    global api_instance
    
    # Initialize database
    try:
        from utils.database import initialize_database
        initialize_database()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
    
    # Start scheduler
    start_scheduler()
    
    # Import and initialize API if available
    try:
        from utils.api_extended import get_api_instance
        api_instance = get_api_instance()
        logger.info("API instance initialized")
    except Exception as e:
        logger.error(f"Error initializing API: {str(e)}")
    
    logger.info("Services initialized")
    return True

def start_scheduler():
    """Start scheduler thread for running background tasks"""
    global scheduler_thread, scheduler_stop_event
    
    # Clear stop event
    scheduler_stop_event.clear()
    
    # Start thread if not already running
    if not scheduler_thread or not scheduler_thread.is_alive():
        scheduler_thread = threading.Thread(target=_scheduler_worker)
        scheduler_thread.daemon = True
        scheduler_thread.start()
        logger.info("Scheduler started")

def stop_scheduler_thread():
    """Stop scheduler thread"""
    global scheduler_thread, scheduler_stop_event
    
    # Set stop event
    scheduler_stop_event.set()
    
    # Wait for thread to stop
    if scheduler_thread and scheduler_thread.is_alive():
        scheduler_thread.join(timeout=2)
        logger.info("Scheduler stopped")

def _scheduler_worker():
    """Worker function for scheduler thread"""
    # Schedule tasks
    
    # Daily tasks
    schedule.every().day.at("09:00").do(_refresh_instrument_master)  # Refresh instruments at market open
    schedule.every().day.at("09:15").do(_refresh_user_data)  # Refresh user data at market open
    schedule.every().day.at("16:00").do(_refresh_user_data)  # Refresh user data at market close
    
    # Hourly tasks during market hours
    schedule.every().hour.do(_check_and_refresh_user_data)  # Refresh user data hourly during market hours
    
    # Regular maintenance tasks
    schedule.every(6).hours.do(_check_database_integrity)  # Check database integrity every 6 hours
    schedule.every().day.at("01:00").do(_clean_cache)  # Clean cache daily at 1 AM
    
    # Run scheduler loop
    while not scheduler_stop_event.is_set():
        schedule.run_pending()
        time.sleep(1)

def _refresh_instrument_master():
    """Refresh instrument master data"""
    try:
        # Only run during weekdays
        if datetime.now().weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            return
        
        from utils.instrument_master import fetch_instrument_master
        result = fetch_instrument_master()
        
        if result:
            logger.info("Instrument master refreshed successfully")
        else:
            logger.warning("Failed to refresh instrument master")
    except Exception as e:
        logger.error(f"Error refreshing instrument master: {str(e)}")

def _refresh_user_data():
    """Refresh user data for all users"""
    try:
        # Only run during weekdays
        if datetime.now().weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            return
        
        # Get all active users
        users = _get_active_users()
        
        for user in users:
            _refresh_user_data_for_user(user)
    except Exception as e:
        logger.error(f"Error refreshing user data: {str(e)}")

def _check_and_refresh_user_data():
    """Check if market is open and refresh user data if needed"""
    try:
        # Check if market is open
        if api_instance and hasattr(api_instance, 'connector'):
            if api_instance.connector.is_market_open():
                _refresh_user_data()
            else:
                logger.info("Market is closed, skipping user data refresh")
    except Exception as e:
        logger.error(f"Error checking market status: {str(e)}")

def _refresh_user_data_for_user(user):
    """Refresh user data for a specific user"""
    try:
        # Extract user credentials
        user_id = user.get('user_id')
        api_key = user.get('api_key')
        api_secret = user.get('api_secret')
        client_id = user.get('client_id')
        mpin = user.get('mpin')
        totp_secret = user.get('totp_secret')
        
        # Skip if missing credentials
        if not all([api_key, api_secret, client_id, mpin, totp_secret]):
            logger.warning(f"Missing credentials for user {user_id}, skipping refresh")
            return
        
        # Initialize API if needed
        if api_instance:
            # Check if API is already initialized for this user
            if api_instance.client_id == client_id:
                if not api_instance.initialized:
                    api_instance.initialize()
            else:
                # Initialize with this user's credentials
                api_instance.initialize(
                    api_key=api_key,
                    client_id=client_id,
                    api_secret=api_secret,
                    mpin=mpin,
                    totp_secret=totp_secret
                )
            
            # Refresh user data
            if api_instance.initialized:
                result = api_instance.refresh_data()
                
                if result:
                    logger.info(f"User data refreshed successfully for user {user_id}")
                else:
                    logger.warning(f"Failed to refresh user data for user {user_id}")
            else:
                logger.warning(f"API not initialized, skipping refresh for user {user_id}")
    except Exception as e:
        logger.error(f"Error refreshing user data for user {user_id}: {str(e)}")

def _get_active_users():
    """Get all active users with credentials"""
    try:
        # Connect to database
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Query active users
        cursor.execute("""
        SELECT user_id, api_key, api_secret, client_id, mpin, totp_secret 
        FROM users 
        WHERE api_key IS NOT NULL 
        AND api_secret IS NOT NULL 
        AND client_id IS NOT NULL 
        AND mpin IS NOT NULL 
        AND totp_secret IS NOT NULL
        """)
        
        users = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return users
    except Exception as e:
        logger.error(f"Error getting active users: {str(e)}")
        return []

def _check_database_integrity():
    """Check database integrity and repair if needed"""
    try:
        # Connect to database
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Run integrity check
        cursor.execute("PRAGMA integrity_check")
        result = cursor.fetchone()
        
        if result and result[0] == "ok":
            logger.info("Database integrity check passed")
        else:
            logger.warning("Database integrity check failed, running vacuum")
            cursor.execute("VACUUM")
            conn.commit()
            logger.info("Database vacuum completed")
        
        conn.close()
    except Exception as e:
        logger.error(f"Error checking database integrity: {str(e)}")

def _clean_cache():
    """Clean old cached data"""
    try:
        # Cache directory
        cache_dir = './cache'
        
        # Current time
        now = time.time()
        
        # Clean files older than 7 days
        max_age = 7 * 24 * 60 * 60  # 7 days in seconds
        
        # Check if directory exists
        if not os.path.exists(cache_dir):
            return
        
        # List all files in cache
        for filename in os.listdir(cache_dir):
            file_path = os.path.join(cache_dir, filename)
            
            # Skip directories
            if os.path.isdir(file_path):
                continue
            
            # Check file age
            file_age = now - os.path.getmtime(file_path)
            
            # Delete old files
            if file_age > max_age:
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted old cache file: {filename}")
                except Exception as e:
                    logger.error(f"Error deleting cache file {filename}: {str(e)}")
    except Exception as e:
        logger.error(f"Error cleaning cache: {str(e)}")

# Authentication functions
def hash_password(password):
    """Hash password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password, password_hash):
    """Verify password against hash"""
    return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))

def generate_totp(secret):
    """Generate TOTP code"""
    totp = pyotp.TOTP(secret)
    return totp.now()

def verify_totp(secret, code):
    """Verify TOTP code"""
    totp = pyotp.TOTP(secret)
    return totp.verify(code)

def register_new_user(username, password, api_key=None, api_secret=None, client_id=None, mpin=None, totp_secret=None):
    """Register a new user in the system
    
    Args:
        username (str): Username for the new user
        password (str): Plain text password (will be hashed)
        api_key (str, optional): AngelOne API key
        api_secret (str, optional): AngelOne API secret
        client_id (str, optional): AngelOne client ID
        mpin (str, optional): AngelOne MPIN
        totp_secret (str, optional): AngelOne TOTP secret
        
    Returns:
        tuple: (success (bool), message (str), user_id (int))
    """
    try:
        # Check if username already exists
        from utils.database import get_user
        existing_user = get_user(username)
        
        if existing_user:
            return False, f"Username {username} already exists", None
        
        # Hash password
        password_hash = hash_password(password)
        
        # Register user in database
        from utils.database import register_user as db_register_user
        user_id = db_register_user(username, password_hash, api_key, api_secret, client_id, mpin, totp_secret)
        
        if user_id:
            logger.info(f"User registered successfully: {username}")
            return True, f"User {username} registered successfully", user_id
        else:
            logger.error(f"Failed to register user: {username}")
            return False, "Registration failed due to a database error", None
    except Exception as e:
        logger.error(f"Error registering user: {str(e)}")
        return False, f"Registration failed: {str(e)}", None

def register_user(username, password, api_key=None, api_secret=None, client_id=None, mpin=None, totp_secret=None):
    """Register a new user"""
    try:
        # Hash password
        password_hash = hash_password(password)
        
        # Register user in database
        from utils.database import register_user as db_register_user
        user_id = db_register_user(username, password_hash, api_key, api_secret, client_id, mpin, totp_secret)
        
        if user_id:
            logger.info(f"User registered successfully: {username}")
            return user_id
        else:
            logger.error(f"Failed to register user: {username}")
            return None
    except Exception as e:
        logger.error(f"Error registering user: {str(e)}")
        return None

def authenticate_user(username, password):
    """Authenticate user with username and password
    
    Args:
        username (str): Username
        password (str): Plain text password
        
    Returns:
        tuple: (success (bool), message (str), user_data (dict))
    """
    try:
        # Get user from database
        from utils.database import get_user
        user = get_user(username)
        
        if not user:
            logger.warning(f"User not found: {username}")
            return False, "Invalid username or password", None
        
        # Verify password
        if verify_password(password, user['password_hash']):
            # Update last login
            from utils.database import update_user_last_login
            update_user_last_login(user['user_id'])
            
            logger.info(f"User authenticated successfully: {username}")
            return True, "Authentication successful", user
        else:
            logger.warning(f"Invalid password for user: {username}")
            return False, "Invalid username or password", None
    except Exception as e:
        logger.error(f"Error authenticating user: {str(e)}")
        return False, f"Authentication error: {str(e)}", None

def authenticate_api(user_id, totp_code):
    """Authenticate user with API using TOTP"""
    try:
        # Get user from database
        from utils.database import get_user_by_id
        user = get_user_by_id(user_id)
        
        if not user:
            logger.warning(f"User not found: {user_id}")
            return False
        
        # Check if user has TOTP secret
        if not user['totp_secret']:
            logger.warning(f"User {user_id} does not have TOTP secret")
            return False
        
        # Verify TOTP
        if verify_totp(user['totp_secret'], totp_code):
            logger.info(f"API authenticated successfully for user {user_id}")
            return True
        else:
            logger.warning(f"Invalid TOTP code for user {user_id}")
            return False
    except Exception as e:
        logger.error(f"Error authenticating API: {str(e)}")
        return False

def login_to_angelone(user_id):
    """
    Login to AngelOne API using user credentials
    
    Args:
        user_id (int): User ID to login with
        
    Returns:
        tuple: (connector, success, message)
    """
    try:
        # Get user from database
        from utils.database import get_user_by_id
        user = get_user_by_id(user_id)
        
        if not user:
            logger.warning(f"User not found: {user_id}")
            return None, False, "User not found"
        
        # Check if user has API credentials
        required_fields = ['api_key', 'api_secret', 'client_id', 'mpin', 'totp_secret']
        for field in required_fields:
            if not user.get(field):
                logger.warning(f"User {user_id} missing required API credential: {field}")
                return None, False, f"Missing required API credential: {field}"
        
        # Import API connector
        from utils.api_connector import SmartAPIConnector
        
        # Create connector
        connector = SmartAPIConnector(
            api_key=user['api_key'],
            api_secret=user['api_secret'],
            client_id=user['client_id'],
            mpin=user['mpin'],
            totp_secret=user['totp_secret']
        )
        
        # Login to API
        login_success = connector.login()
        
        if login_success:
            logger.info(f"Login successful for user {user_id}")
            return connector, True, "Login successful"
        else:
            logger.warning(f"Login failed for user {user_id}")
            return connector, False, "API login failed"
    except Exception as e:
        logger.error(f"Error logging in to AngelOne: {str(e)}")
        return None, False, f"Login error: {str(e)}"

def fetch_stock_data_smart(api_connector, token, exchange, interval="ONE_DAY", start_date=None, end_date=None):
    """
    Fetch stock data with smart caching
    
    Args:
        api_connector: API connector instance
        token (str): Stock token
        exchange (str): Exchange (NSE, BSE, etc.)
        interval (str): Time interval (ONE_DAY, ONE_HOUR, etc.)
        start_date: Start date for data fetching
        end_date: End date for data fetching
        
    Returns:
        pd.DataFrame: DataFrame with stock data
    """
    try:
        # Check cache first
        from utils.database import check_stock_data_availability, get_stock_data
        
        # Use default dates if not provided
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        # Convert string dates to datetime if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Check if data is available in cache
        available = check_stock_data_availability(token, exchange, interval, start_date, end_date)
        
        if available:
            # Get data from cache
            logger.info(f"Getting stock data from cache for token {token}")
            data = get_stock_data(token, exchange, interval, start_date, end_date)
            return data
        else:
            # Fetch data from API
            logger.info(f"Fetching stock data from API for token {token}")
            
            # Format dates for API
            start_date_str = start_date.strftime("%Y-%m-%d")
            end_date_str = end_date.strftime("%Y-%m-%d")
            
            # Fetch data
            data = api_connector.get_historical_data(token, exchange, start_date_str, end_date_str, interval)
            
            # Save to cache if data is available
            if data is not None and not data.empty:
                from utils.database import save_stock_data
                save_stock_data(data, token, exchange, interval)
                logger.info(f"Saved stock data to cache for token {token}")
            
            return data
    except Exception as e:
        logger.error(f"Error fetching stock data: {str(e)}")
        return None

def train_model_and_save(data, model_type, stock_token, exchange, forecast_days=7, user_id=None):
    """
    Train a model and save results
    
    Args:
        data (pd.DataFrame): Stock data
        model_type (str): Type of model (arima, prophet, lstm, gru, transformer)
        stock_token (str): Stock token
        exchange (str): Exchange
        forecast_days (int): Number of days to forecast
        user_id (int, optional): User ID
        
    Returns:
        tuple: (success, message, predictions_df)
    """
    try:
        # Import prediction models
        from utils.prediction_models import (
            train_arima_model, train_prophet_model
        )
        from utils.advanced_models import (
            train_lstm_model, train_gru_model, train_transformer_model
        )
        
        # Choose the model
        model_func = None
        if model_type.lower() == 'arima':
            model_func = train_arima_model
        elif model_type.lower() == 'prophet':
            model_func = train_prophet_model
        elif model_type.lower() == 'lstm':
            model_func = train_lstm_model
        elif model_type.lower() == 'gru':
            model_func = train_gru_model
        elif model_type.lower() == 'transformer':
            model_func = train_transformer_model
        else:
            return False, f"Unknown model type: {model_type}", None
        
        # Train the model
        logger.info(f"Training {model_type} model for stock {stock_token}")
        predictions, metrics, model = model_func(data, forecast_days=forecast_days)
        
        if predictions is None:
            return False, f"Failed to train {model_type} model", None
        
        # Save predictions and model to database
        from utils.database import save_prediction, save_trained_model
        
        prediction_id = save_prediction(
            user_id=user_id,
            stock_token=stock_token,
            exchange=exchange,
            model_type=model_type,
            forecast_days=forecast_days,
            predictions=predictions,
            metrics=metrics
        )
        
        model_id = save_trained_model(
            prediction_id=prediction_id,
            model=model,
            model_type=model_type
        )
        
        logger.info(f"Saved {model_type} model and predictions for stock {stock_token}")
        
        return True, f"{model_type.upper()} model trained successfully", predictions
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return False, f"Error training model: {str(e)}", None

def update_prediction_accuracy_with_new_data(prediction_id, actual_data):
    """
    Update prediction accuracy with new actual data
    
    Args:
        prediction_id (int): Prediction ID
        actual_data (pd.DataFrame): Actual stock data
        
    Returns:
        tuple: (success, message, updated_metrics)
    """
    try:
        # Get prediction from database
        from utils.database import get_prediction_by_id
        prediction = get_prediction_by_id(prediction_id)
        
        if not prediction:
            return False, f"Prediction not found with ID: {prediction_id}", None
        
        # Extract prediction data
        predictions_df = prediction['predictions']
        metrics = prediction['metrics']
        model_type = prediction['model_type']
        
        # Import prediction models
        from utils.prediction_models import evaluate_prediction_accuracy
        
        # Evaluate accuracy
        updated_metrics = evaluate_prediction_accuracy(predictions_df, actual_data)
        
        # Update metrics in database
        from utils.database import update_prediction_metrics
        update_prediction_metrics(prediction_id, updated_metrics)
        
        logger.info(f"Updated prediction accuracy for prediction ID {prediction_id}")
        
        return True, "Prediction accuracy updated successfully", updated_metrics
        
    except Exception as e:
        logger.error(f"Error updating prediction accuracy: {str(e)}")
        return False, f"Error updating prediction accuracy: {str(e)}", None

def initialize_api_for_user(user_id):
    """Initialize API for a user"""
    try:
        global api_instance
        
        # Get user from database
        from utils.database import get_user_by_id
        user = get_user_by_id(user_id)
        
        if not user:
            logger.warning(f"User not found: {user_id}")
            return False
        
        # Check if user has API credentials
        if not all([user['api_key'], user['api_secret'], user['client_id'], user['mpin'], user['totp_secret']]):
            logger.warning(f"User {user_id} does not have complete API credentials")
            return False
        
        # Import API instance
        from utils.api_extended import get_api_instance
        api_instance = get_api_instance()
        
        # Initialize API
        result = api_instance.initialize(
            api_key=user['api_key'],
            client_id=user['client_id'],
            api_secret=user['api_secret'],
            mpin=user['mpin'],
            totp_secret=user['totp_secret']
        )
        
        if result:
            logger.info(f"API initialized successfully for user {user_id}")
            return True
        else:
            logger.warning(f"Failed to initialize API for user {user_id}")
            return False
    except Exception as e:
        logger.error(f"Error initializing API for user: {str(e)}")
        return False