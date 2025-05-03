"""
Database module for managing data persistence.

This module handles database operations for:
- User management (login, registration)
- Stock data caching
- Prediction storage
- Instrument data
"""

import os
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

# Database file path
DB_FILE = 'stock_prediction.db'

def get_db_connection():
    """Get a connection to the SQLite database"""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_database():
    """Initialize the database with required tables"""
    conn = get_db_connection()
    
    # Create users table
    conn.execute('''
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        api_key TEXT,
        api_secret TEXT,
        client_id TEXT,
        mpin TEXT,
        totp_secret TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP
    )
    ''')
    
    # Create user_profiles table
    conn.execute('''
    CREATE TABLE IF NOT EXISTS user_profiles (
        profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        profile_data TEXT NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    )
    ''')
    
    # Create funds_and_margins table
    conn.execute('''
    CREATE TABLE IF NOT EXISTS funds_and_margins (
        funds_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        funds_data TEXT NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    )
    ''')
    
    # Create user_portfolios table
    conn.execute('''
    CREATE TABLE IF NOT EXISTS user_portfolios (
        portfolio_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        portfolio_data TEXT NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    )
    ''')
    
    # Create instruments table
    conn.execute('''
    CREATE TABLE IF NOT EXISTS instruments (
        instrument_id INTEGER PRIMARY KEY AUTOINCREMENT,
        token TEXT UNIQUE NOT NULL,
        symbol TEXT,
        name TEXT,
        expiry TEXT,
        strike REAL,
        lot_size INTEGER,
        tick_size REAL,
        exchange TEXT,
        instrument_type TEXT,
        allowed_for_intraday INTEGER DEFAULT 0,
        intraday_margin REAL DEFAULT 0,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create stock_data table
    conn.execute('''
    CREATE TABLE IF NOT EXISTS stock_data (
        data_id INTEGER PRIMARY KEY AUTOINCREMENT,
        token TEXT NOT NULL,
        exchange TEXT NOT NULL,
        interval TEXT NOT NULL,
        start_date TEXT NOT NULL,
        end_date TEXT NOT NULL,
        data_json TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(token, exchange, interval, start_date, end_date)
    )
    ''')
    
    # Create trained_models table
    conn.execute('''
    CREATE TABLE IF NOT EXISTS trained_models (
        model_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        token TEXT NOT NULL,
        exchange TEXT NOT NULL,
        model_type TEXT NOT NULL,
        parameters TEXT,
        metrics TEXT,
        model_path TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    )
    ''')
    
    # Create predictions table
    conn.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        token TEXT NOT NULL,
        exchange TEXT NOT NULL,
        model_type TEXT NOT NULL,
        prediction_date TIMESTAMP NOT NULL,
        forecast_start_date TEXT NOT NULL,
        forecast_end_date TEXT NOT NULL,
        prediction_data TEXT NOT NULL,
        metrics TEXT,
        accuracy REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    )
    ''')
    
    # Create user_preferences table
    conn.execute('''
    CREATE TABLE IF NOT EXISTS user_preferences (
        preference_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER UNIQUE NOT NULL,
        theme TEXT DEFAULT 'light',
        chart_style TEXT DEFAULT 'candles',
        prediction_days INTEGER DEFAULT 7,
        data_period INTEGER DEFAULT 365,
        notification_enabled INTEGER DEFAULT 0,
        notify_email TEXT,
        notify_mobile TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    )
    ''')
    
    conn.commit()
    conn.close()

# User management functions
def register_user(username, password_hash, api_key=None, api_secret=None, client_id=None, mpin=None, totp_secret=None):
    """Register a new user"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            'INSERT INTO users (username, password_hash, api_key, api_secret, client_id, mpin, totp_secret) VALUES (?, ?, ?, ?, ?, ?, ?)',
            (username, password_hash, api_key, api_secret, client_id, mpin, totp_secret)
        )
        
        user_id = cursor.lastrowid
        
        # Initialize user preferences
        cursor.execute(
            'INSERT INTO user_preferences (user_id) VALUES (?)',
            (user_id,)
        )
        
        conn.commit()
        conn.close()
        
        return user_id
    except Exception as e:
        print(f"Error registering user: {str(e)}")
        return None

def get_user(username):
    """Get user by username"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        
        conn.close()
        
        return dict(user) if user else None
    except Exception as e:
        print(f"Error getting user: {str(e)}")
        return None

def get_user_by_id(user_id):
    """Get user by ID"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
        user = cursor.fetchone()
        
        conn.close()
        
        return dict(user) if user else None
    except Exception as e:
        print(f"Error getting user by ID: {str(e)}")
        return None

def update_user_credentials(user_id, api_key=None, api_secret=None, client_id=None, mpin=None, totp_secret=None):
    """Update user API credentials"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get current values
        cursor.execute('SELECT api_key, api_secret, client_id, mpin, totp_secret FROM users WHERE user_id = ?', (user_id,))
        user = cursor.fetchone()
        
        if not user:
            conn.close()
            return False
        
        # Update only provided values
        new_api_key = api_key if api_key is not None else user['api_key']
        new_api_secret = api_secret if api_secret is not None else user['api_secret']
        new_client_id = client_id if client_id is not None else user['client_id']
        new_mpin = mpin if mpin is not None else user['mpin']
        new_totp_secret = totp_secret if totp_secret is not None else user['totp_secret']
        
        cursor.execute(
            'UPDATE users SET api_key = ?, api_secret = ?, client_id = ?, mpin = ?, totp_secret = ? WHERE user_id = ?',
            (new_api_key, new_api_secret, new_client_id, new_mpin, new_totp_secret, user_id)
        )
        
        conn.commit()
        conn.close()
        
        return True
    except Exception as e:
        print(f"Error updating user credentials: {str(e)}")
        return False

def update_user_last_login(user_id):
    """Update user's last login time"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            'UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE user_id = ?',
            (user_id,)
        )
        
        conn.commit()
        conn.close()
        
        return True
    except Exception as e:
        print(f"Error updating last login: {str(e)}")
        return False

# User profile functions
def save_user_profile(user_id, profile_data):
    """Save or update user profile"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Convert profile data to JSON
        profile_json = json.dumps(profile_data)
        
        # Check if profile exists
        cursor.execute('SELECT profile_id FROM user_profiles WHERE user_id = ?', (user_id,))
        profile = cursor.fetchone()
        
        if profile:
            # Update existing profile
            cursor.execute(
                'UPDATE user_profiles SET profile_data = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?',
                (profile_json, user_id)
            )
        else:
            # Insert new profile
            cursor.execute(
                'INSERT INTO user_profiles (user_id, profile_data) VALUES (?, ?)',
                (user_id, profile_json)
            )
        
        conn.commit()
        conn.close()
        
        return True
    except Exception as e:
        print(f"Error saving user profile: {str(e)}")
        return False

def get_user_profile(user_id):
    """Get user profile"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT profile_data FROM user_profiles WHERE user_id = ?', (user_id,))
        profile = cursor.fetchone()
        
        conn.close()
        
        if profile:
            return json.loads(profile['profile_data'])
        else:
            return None
    except Exception as e:
        print(f"Error getting user profile: {str(e)}")
        return None

# Funds and margins functions
def save_funds_and_margins(user_id, funds_data):
    """Save or update funds and margins"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Convert funds data to JSON
        funds_json = json.dumps(funds_data)
        
        # Check if funds data exists
        cursor.execute('SELECT funds_id FROM funds_and_margins WHERE user_id = ?', (user_id,))
        funds = cursor.fetchone()
        
        if funds:
            # Update existing funds data
            cursor.execute(
                'UPDATE funds_and_margins SET funds_data = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?',
                (funds_json, user_id)
            )
        else:
            # Insert new funds data
            cursor.execute(
                'INSERT INTO funds_and_margins (user_id, funds_data) VALUES (?, ?)',
                (user_id, funds_json)
            )
        
        conn.commit()
        conn.close()
        
        return True
    except Exception as e:
        print(f"Error saving funds and margins: {str(e)}")
        return False

def get_funds_and_margins(user_id):
    """Get funds and margins"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT funds_data FROM funds_and_margins WHERE user_id = ?', (user_id,))
        funds = cursor.fetchone()
        
        conn.close()
        
        if funds:
            return json.loads(funds['funds_data'])
        else:
            return None
    except Exception as e:
        print(f"Error getting funds and margins: {str(e)}")
        return None

# Portfolio functions
def save_user_portfolio(user_id, portfolio_data):
    """Save or update user portfolio"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Convert portfolio data to JSON
        portfolio_json = json.dumps(portfolio_data)
        
        # Check if portfolio exists
        cursor.execute('SELECT portfolio_id FROM user_portfolios WHERE user_id = ?', (user_id,))
        portfolio = cursor.fetchone()
        
        if portfolio:
            # Update existing portfolio
            cursor.execute(
                'UPDATE user_portfolios SET portfolio_data = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?',
                (portfolio_json, user_id)
            )
        else:
            # Insert new portfolio
            cursor.execute(
                'INSERT INTO user_portfolios (user_id, portfolio_data) VALUES (?, ?)',
                (user_id, portfolio_json)
            )
        
        conn.commit()
        conn.close()
        
        return True
    except Exception as e:
        print(f"Error saving user portfolio: {str(e)}")
        return False

def get_user_portfolio(user_id):
    """Get user portfolio"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT portfolio_data FROM user_portfolios WHERE user_id = ?', (user_id,))
        portfolio = cursor.fetchone()
        
        conn.close()
        
        if portfolio:
            return json.loads(portfolio['portfolio_data'])
        else:
            return None
    except Exception as e:
        print(f"Error getting user portfolio: {str(e)}")
        return None

# Instrument functions
def save_instruments(instruments, replace=False):
    """Save instruments to database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        for instrument in instruments:
            if replace:
                # Replace existing instrument
                cursor.execute('''
                INSERT OR REPLACE INTO instruments (
                    token, symbol, name, expiry, strike, lot_size, tick_size, 
                    exchange, instrument_type, allowed_for_intraday, intraday_margin, 
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    instrument.get('token'),
                    instrument.get('symbol'),
                    instrument.get('name'),
                    instrument.get('expiry'),
                    instrument.get('strike', 0),
                    instrument.get('lot_size', 0),
                    instrument.get('tick_size', 0),
                    instrument.get('exchange'),
                    instrument.get('instrument_type'),
                    instrument.get('allowed_for_intraday', 0),
                    instrument.get('intraday_margin', 0)
                ))
            else:
                # Update if exists, otherwise insert
                cursor.execute('SELECT instrument_id FROM instruments WHERE token = ?', (instrument.get('token'),))
                if cursor.fetchone():
                    # Update relevant fields only
                    updates = []
                    params = []
                    
                    for field in ['symbol', 'name', 'expiry', 'strike', 'lot_size', 'tick_size', 'exchange', 'instrument_type', 'allowed_for_intraday', 'intraday_margin']:
                        if field in instrument:
                            updates.append(f"{field} = ?")
                            params.append(instrument.get(field))
                    
                    if updates:
                        updates.append("updated_at = CURRENT_TIMESTAMP")
                        params.append(instrument.get('token'))
                        
                        query = f"UPDATE instruments SET {', '.join(updates)} WHERE token = ?"
                        cursor.execute(query, params)
                else:
                    # Insert new instrument
                    cursor.execute('''
                    INSERT INTO instruments (
                        token, symbol, name, expiry, strike, lot_size, tick_size, 
                        exchange, instrument_type, allowed_for_intraday, intraday_margin
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        instrument.get('token'),
                        instrument.get('symbol'),
                        instrument.get('name'),
                        instrument.get('expiry'),
                        instrument.get('strike', 0),
                        instrument.get('lot_size', 0),
                        instrument.get('tick_size', 0),
                        instrument.get('exchange'),
                        instrument.get('instrument_type'),
                        instrument.get('allowed_for_intraday', 0),
                        instrument.get('intraday_margin', 0)
                    ))
        
        conn.commit()
        conn.close()
        
        return True
    except Exception as e:
        print(f"Error saving instruments: {str(e)}")
        return False

def get_instrument_by_token(token):
    """Get instrument by token"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM instruments WHERE token = ?', (token,))
        instrument = cursor.fetchone()
        
        conn.close()
        
        return dict(instrument) if instrument else None
    except Exception as e:
        print(f"Error getting instrument by token: {str(e)}")
        return None

def search_instruments(query, exchange=None, limit=20):
    """Search instruments by name, symbol, or token"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Build query
        sql_query = '''
        SELECT * FROM instruments 
        WHERE (name LIKE ? OR symbol LIKE ? OR token LIKE ?)
        '''
        params = [f"%{query}%", f"%{query}%", f"%{query}%"]
        
        if exchange:
            sql_query += " AND exchange = ?"
            params.append(exchange)
        
        sql_query += " LIMIT ?"
        params.append(limit)
        
        cursor.execute(sql_query, params)
        instruments = cursor.fetchall()
        
        conn.close()
        
        return [dict(instrument) for instrument in instruments]
    except Exception as e:
        print(f"Error searching instruments: {str(e)}")
        return []

# Stock data functions
def save_stock_data(token, exchange, data, interval="ONE_DAY"):
    """Save stock data to database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Convert data to JSON
        if not data.empty:
            # Get date range
            start_date = data.index.min().strftime('%Y-%m-%d')
            end_date = data.index.max().strftime('%Y-%m-%d')
            
            # Convert DataFrame to JSON
            data_json = data.reset_index().to_json(orient='records', date_format='iso')
            
            # Save to database
            cursor.execute('''
            INSERT OR REPLACE INTO stock_data (token, exchange, interval, start_date, end_date, data_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (token, exchange, interval, start_date, end_date, data_json))
            
            conn.commit()
            conn.close()
            
            return True
        else:
            conn.close()
            return False
    except Exception as e:
        print(f"Error saving stock data: {str(e)}")
        return False

def get_stock_data(token, exchange, start_date, end_date, interval="ONE_DAY"):
    """Get stock data from database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Format dates
        if isinstance(start_date, str):
            start_date_str = start_date
        else:
            start_date_str = start_date.strftime('%Y-%m-%d')
        
        if isinstance(end_date, str):
            end_date_str = end_date
        else:
            end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Query for records that include the requested date range
        cursor.execute('''
        SELECT data_json FROM stock_data 
        WHERE token = ? AND exchange = ? AND interval = ? 
        AND start_date <= ? AND end_date >= ?
        ORDER BY created_at DESC LIMIT 1
        ''', (token, exchange, interval, end_date_str, start_date_str))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            # Convert JSON to DataFrame
            df = pd.read_json(result['data_json'], orient='records')
            
            # Set datetime index
            if 'index' in df.columns:
                df['index'] = pd.to_datetime(df['index'])
                df.set_index('index', inplace=True)
            elif 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
            
            # Filter to requested date range
            df = df.loc[start_date_str:end_date_str]
            
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"Error getting stock data: {str(e)}")
        return pd.DataFrame()

def check_stock_data_availability(token, exchange, start_date, end_date, interval="ONE_DAY"):
    """Check if stock data is available in database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Format dates
        if isinstance(start_date, str):
            start_date_str = start_date
        else:
            start_date_str = start_date.strftime('%Y-%m-%d')
        
        if isinstance(end_date, str):
            end_date_str = end_date
        else:
            end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Query for records that include the requested date range
        cursor.execute('''
        SELECT COUNT(*) as count FROM stock_data 
        WHERE token = ? AND exchange = ? AND interval = ? 
        AND start_date <= ? AND end_date >= ?
        ''', (token, exchange, interval, end_date_str, start_date_str))
        
        result = cursor.fetchone()
        conn.close()
        
        return result['count'] > 0
    except Exception as e:
        print(f"Error checking stock data availability: {str(e)}")
        return False

# Trained model functions
def save_trained_model(user_id, token, exchange, model_type, model, parameters=None, metrics=None):
    """Save trained model"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create model directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Generate model path
        model_path = f"models/model_{user_id}_{token}_{exchange}_{model_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Convert parameters and metrics to JSON
        parameters_json = json.dumps(parameters) if parameters else None
        metrics_json = json.dumps(metrics) if metrics else None
        
        # Save to database
        cursor.execute('''
        INSERT INTO trained_models (user_id, token, exchange, model_type, parameters, metrics, model_path)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, token, exchange, model_type, parameters_json, metrics_json, model_path))
        
        model_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        # Save model files (using functions from advanced_models.py)
        from utils.advanced_models import save_model
        save_model(model, model_path)
        
        return model_id
    except Exception as e:
        print(f"Error saving trained model: {str(e)}")
        return None

def get_trained_model(model_id):
    """Get trained model by ID"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM trained_models WHERE model_id = ?', (model_id,))
        model_record = cursor.fetchone()
        
        conn.close()
        
        if model_record:
            model_dict = dict(model_record)
            
            # Load model from path
            from utils.advanced_models import load_model
            model_bundle = load_model(model_dict['model_path'])
            
            if model_bundle:
                return {
                    'model_id': model_dict['model_id'],
                    'user_id': model_dict['user_id'],
                    'token': model_dict['token'],
                    'exchange': model_dict['exchange'],
                    'model_type': model_dict['model_type'],
                    'parameters': json.loads(model_dict['parameters']) if model_dict['parameters'] else None,
                    'metrics': json.loads(model_dict['metrics']) if model_dict['metrics'] else None,
                    'model_path': model_dict['model_path'],
                    'created_at': model_dict['created_at'],
                    'model_bundle': model_bundle
                }
            else:
                return None
        else:
            return None
    except Exception as e:
        print(f"Error getting trained model: {str(e)}")
        return None

# Prediction functions
def save_prediction(user_id, token, exchange, model_type, prediction_date, forecast_start_date, forecast_end_date, prediction_data, metrics=None, accuracy=None):
    """Save prediction"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Convert prediction data and metrics to JSON
        prediction_json = json.dumps(prediction_data)
        metrics_json = json.dumps(metrics) if metrics else None
        
        # Save to database
        cursor.execute('''
        INSERT INTO predictions (user_id, token, exchange, model_type, prediction_date, forecast_start_date, forecast_end_date, prediction_data, metrics, accuracy)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, token, exchange, model_type, prediction_date, forecast_start_date, forecast_end_date, prediction_json, metrics_json, accuracy))
        
        prediction_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        return prediction_id
    except Exception as e:
        print(f"Error saving prediction: {str(e)}")
        return None

def get_prediction_by_id(prediction_id):
    """Get prediction by ID"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM predictions WHERE prediction_id = ?', (prediction_id,))
        prediction = cursor.fetchone()
        
        conn.close()
        
        if prediction:
            prediction_dict = dict(prediction)
            
            # Parse JSON fields
            prediction_dict['prediction_data'] = json.loads(prediction_dict['prediction_data'])
            prediction_dict['metrics'] = json.loads(prediction_dict['metrics']) if prediction_dict['metrics'] else None
            
            return prediction_dict
        else:
            return None
    except Exception as e:
        print(f"Error getting prediction: {str(e)}")
        return None

def get_user_predictions(user_id, token=None, exchange=None, model_type=None, limit=10):
    """Get user predictions"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Build query
        query = 'SELECT * FROM predictions WHERE user_id = ?'
        params = [user_id]
        
        if token:
            query += ' AND token = ?'
            params.append(token)
        
        if exchange:
            query += ' AND exchange = ?'
            params.append(exchange)
        
        if model_type:
            query += ' AND model_type = ?'
            params.append(model_type)
        
        query += ' ORDER BY created_at DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        predictions = cursor.fetchall()
        
        conn.close()
        
        return [dict(pred) for pred in predictions]
    except Exception as e:
        print(f"Error getting user predictions: {str(e)}")
        return []

def update_prediction_accuracy(prediction_id, accuracy):
    """Update prediction accuracy"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('UPDATE predictions SET accuracy = ? WHERE prediction_id = ?', (accuracy, prediction_id))
        
        conn.commit()
        conn.close()
        
        return True
    except Exception as e:
        print(f"Error updating prediction accuracy: {str(e)}")
        return False

# User preferences functions
def save_user_preferences(user_id, preferences):
    """Save user preferences"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if preferences exist
        cursor.execute('SELECT preference_id FROM user_preferences WHERE user_id = ?', (user_id,))
        if cursor.fetchone():
            # Update existing preferences
            set_clause = ', '.join([f'{key} = ?' for key in preferences.keys()])
            set_clause += ', updated_at = CURRENT_TIMESTAMP'
            
            query = f'UPDATE user_preferences SET {set_clause} WHERE user_id = ?'
            params = list(preferences.values())
            params.append(user_id)
            
            cursor.execute(query, params)
        else:
            # Insert new preferences
            columns = ', '.join(['user_id'] + list(preferences.keys()))
            placeholders = ', '.join(['?'] * (len(preferences) + 1))
            
            query = f'INSERT INTO user_preferences ({columns}) VALUES ({placeholders})'
            params = [user_id] + list(preferences.values())
            
            cursor.execute(query, params)
        
        conn.commit()
        conn.close()
        
        return True
    except Exception as e:
        print(f"Error saving user preferences: {str(e)}")
        return False

def get_user_preferences(user_id):
    """Get user preferences"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM user_preferences WHERE user_id = ?', (user_id,))
        preferences = cursor.fetchone()
        
        conn.close()
        
        if preferences:
            preferences_dict = dict(preferences)
            # Remove ID fields
            if 'preference_id' in preferences_dict:
                del preferences_dict['preference_id']
            if 'user_id' in preferences_dict:
                del preferences_dict['user_id']
            
            return preferences_dict
        else:
            return None
    except Exception as e:
        print(f"Error getting user preferences: {str(e)}")
        return None