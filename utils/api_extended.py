"""
Extended API functionality for the stock prediction app.
"""

import os
import json
import time
import sqlite3
import logging
from datetime import datetime, timedelta
import pandas as pd
import requests
import hashlib
import pyotp
from functools import wraps
from threading import Thread, Event

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/api_extended.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = './cache'
os.makedirs(CACHE_DIR, exist_ok=True)

# Database file
DB_FILE = 'stock_prediction.db'

# API connector class
class AngelOneAPI:
    """
    Extended AngelOne API with caching and automated token refresh
    """
    def __init__(self, api_key=None, client_id=None, api_secret=None, mpin=None, totp_secret=None):
        self.api_key = api_key
        self.client_id = client_id
        self.api_secret = api_secret
        self.mpin = mpin
        self.totp_secret = totp_secret
        self.connector = None
        self.initialized = False
        self.last_login = None
        self.refresh_thread = None
        self.refresh_interval = 4 * 60 * 60  # 4 hours in seconds
        self.stop_refresh = Event()
    
    def initialize(self, api_key=None, client_id=None, api_secret=None, mpin=None, totp_secret=None):
        """Initialize API connector with credentials"""
        # Update credentials if provided
        self.api_key = api_key or self.api_key
        self.client_id = client_id or self.client_id
        self.api_secret = api_secret or self.api_secret
        self.mpin = mpin or self.mpin
        self.totp_secret = totp_secret or self.totp_secret
        
        # Check if we have all required credentials
        if not all([self.api_key, self.client_id, self.api_secret, self.mpin, self.totp_secret]):
            logger.error("Missing required credentials")
            return False
        
        try:
            # Import SmartAPIConnector
            from utils.api_connector import SmartAPIConnector
            
            # Create connector
            self.connector = SmartAPIConnector(
                api_key=self.api_key,
                client_id=self.client_id,
                api_secret=self.api_secret,
                mpin=self.mpin,
                totp_secret=self.totp_secret
            )
            
            # Login
            if self.connector.login():
                self.initialized = True
                self.last_login = datetime.now()
                
                # Start refresh thread if not already running
                if not self.refresh_thread or not self.refresh_thread.is_alive():
                    self.stop_refresh.clear()
                    self.refresh_thread = Thread(target=self._token_refresh_worker)
                    self.refresh_thread.daemon = True
                    self.refresh_thread.start()
                
                return True
            else:
                logger.error("Failed to login to AngelOne API")
                return False
        
        except Exception as e:
            logger.error(f"Error initializing AngelOne API: {str(e)}")
            return False
    
    def reinitialize(self):
        """Reinitialize API connector with existing credentials"""
        return self.initialize()
    
    def _token_refresh_worker(self):
        """Background worker to refresh token periodically"""
        while not self.stop_refresh.is_set():
            # Sleep for refresh interval
            for _ in range(self.refresh_interval):
                if self.stop_refresh.is_set():
                    break
                time.sleep(1)
            
            # Refresh token if not stopped
            if not self.stop_refresh.is_set():
                try:
                    logger.info("Refreshing API token...")
                    if self.connector:
                        if self.connector.refresh_token_if_needed():
                            self.last_login = datetime.now()
                            logger.info("API token refreshed successfully")
                        else:
                            logger.warning("Failed to refresh API token, trying to login again")
                            if self.connector.login():
                                self.last_login = datetime.now()
                                logger.info("Re-login successful")
                            else:
                                logger.error("Failed to re-login to AngelOne API")
                except Exception as e:
                    logger.error(f"Error refreshing token: {str(e)}")
    
    def stop(self):
        """Stop the API connector and background refresh"""
        if self.refresh_thread and self.refresh_thread.is_alive():
            self.stop_refresh.set()
            self.refresh_thread.join(timeout=2)
        
        if self.connector:
            self.connector.logout()
            self.connector = None
            self.initialized = False
    
    def require_initialized(func):
        """Decorator to ensure API is initialized before calling a method"""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self.initialized or not self.connector:
                if not self.reinitialize():
                    logger.error(f"API not initialized, cannot call {func.__name__}")
                    if func.__name__ == 'get_historical_data':
                        return pd.DataFrame()
                    elif func.__name__ in ['get_ltp_data', 'get_profile', 'get_funds']:
                        return {}
                    elif func.__name__ in ['get_holdings', 'get_positions', 'get_orders', 'get_trades']:
                        return []
                    else:
                        return None
            return func(self, *args, **kwargs)
        return wrapper
    
    @require_initialized
    def get_historical_data(self, symbol, exchange, from_date, to_date, interval="ONE_DAY"):
        """Get historical data with caching"""
        # Generate cache key
        cache_key = f"{symbol}_{exchange}_{from_date}_{to_date}_{interval}"
        cache_key = hashlib.md5(cache_key.encode()).hexdigest()
        cache_file = os.path.join(CACHE_DIR, f"hist_data_{cache_key}.csv")
        
        # Check if we have cache
        if os.path.exists(cache_file) and self._is_cache_valid(cache_file):
            try:
                # Load from cache
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                logger.info(f"Loaded historical data from cache for {symbol} from {from_date} to {to_date}")
                return df
            except Exception as e:
                logger.error(f"Error loading cache: {str(e)}")
        
        # Not in cache or cache error, fetch from API
        df = self.connector.get_historical_data(symbol, exchange, from_date, to_date, interval)
        
        # Save to cache if we got data
        if not df.empty:
            try:
                df.to_csv(cache_file)
                logger.info(f"Saved historical data to cache for {symbol} from {from_date} to {to_date}")
            except Exception as e:
                logger.error(f"Error saving to cache: {str(e)}")
        
        return df
    
    @require_initialized
    def get_ltp_data(self, symbols, exchange):
        """Get LTP data for symbols"""
        return self.connector.get_ltp_data(symbols, exchange)
    
    @require_initialized
    def get_profile(self):
        """Get user profile with caching"""
        try:
            # Try to get from database first
            try:
                from utils.database import get_user_profile
                
                # Get current user ID
                user_id = self._get_current_user_id()
                if user_id:
                    profile = get_user_profile(user_id)
                    if profile:
                        logger.info("Loaded profile from database")
                        return profile
            except ImportError:
                logger.warning("database module not available")
            
            # Fetch from API
            profile = self.connector.get_profile()
            
            # Save to database if available
            if profile:
                try:
                    from utils.database import save_user_profile
                    
                    user_id = self._get_current_user_id()
                    if user_id:
                        save_user_profile(user_id, profile)
                        logger.info("Saved profile to database")
                except ImportError:
                    logger.warning("database module not available")
            
            return profile
        except Exception as e:
            logger.error(f"Error in get_profile: {str(e)}")
            return {}
    
    @require_initialized
    def get_funds(self):
        """Get funds with caching"""
        try:
            # Try to get from database first
            try:
                from utils.database import get_funds_and_margins
                
                # Get current user ID
                user_id = self._get_current_user_id()
                if user_id:
                    funds = get_funds_and_margins(user_id)
                    if funds:
                        logger.info("Loaded funds from database")
                        return funds
            except ImportError:
                logger.warning("database module not available")
            
            # Fetch from API
            funds = self.connector.get_funds()
            
            # Save to database if available
            if funds:
                try:
                    from utils.database import save_funds_and_margins
                    
                    user_id = self._get_current_user_id()
                    if user_id:
                        save_funds_and_margins(user_id, funds)
                        logger.info("Saved funds to database")
                except ImportError:
                    logger.warning("database module not available")
            
            return funds
        except Exception as e:
            logger.error(f"Error in get_funds: {str(e)}")
            return {}
    
    @require_initialized
    def get_holdings(self):
        """Get holdings"""
        return self.connector.get_holdings()
    
    @require_initialized
    def get_portfolio(self):
        """Get portfolio with caching"""
        try:
            # Try to get from database first
            try:
                from utils.database import get_user_portfolio
                
                # Get current user ID
                user_id = self._get_current_user_id()
                if user_id:
                    portfolio = get_user_portfolio(user_id)
                    if portfolio:
                        logger.info("Loaded portfolio from database")
                        return portfolio
            except ImportError:
                logger.warning("database module not available")
            
            # Fetch from API
            portfolio = self.connector.get_portfolio()
            
            # Save to database if available
            if portfolio:
                try:
                    from utils.database import save_user_portfolio
                    
                    user_id = self._get_current_user_id()
                    if user_id:
                        save_user_portfolio(user_id, portfolio)
                        logger.info("Saved portfolio to database")
                except ImportError:
                    logger.warning("database module not available")
            
            return portfolio
        except Exception as e:
            logger.error(f"Error in get_portfolio: {str(e)}")
            return {
                'holdings': [],
                'total_investment': 0,
                'total_current_value': 0,
                'total_pnl': 0,
                'total_pnl_percent': 0
            }
    
    @require_initialized
    def get_positions(self):
        """Get positions"""
        return self.connector.get_positions()
    
    @require_initialized
    def get_orders(self):
        """Get orders"""
        return self.connector.get_orders()
    
    @require_initialized
    def get_trades(self):
        """Get trades"""
        return self.connector.get_trades()
    
    @require_initialized
    def refresh_data(self):
        """Refresh all user data"""
        try:
            # Refresh profile
            profile = self.connector.get_profile()
            
            # Refresh funds
            funds = self.connector.get_funds()
            
            # Refresh portfolio
            portfolio = self.connector.get_portfolio()
            
            # Save to database if available
            try:
                from utils.database import save_user_profile, save_funds_and_margins, save_user_portfolio
                
                user_id = self._get_current_user_id()
                if user_id:
                    if profile:
                        save_user_profile(user_id, profile)
                        logger.info("Saved profile to database")
                    
                    if funds:
                        save_funds_and_margins(user_id, funds)
                        logger.info("Saved funds to database")
                    
                    if portfolio:
                        save_user_portfolio(user_id, portfolio)
                        logger.info("Saved portfolio to database")
            except ImportError:
                logger.warning("database module not available")
            
            return {
                'profile': profile,
                'funds': funds,
                'portfolio': portfolio
            }
        except Exception as e:
            logger.error(f"Error in refresh_data: {str(e)}")
            return {}
            
    def get_user_profile(self, user_id=None):
        """Get user profile with API or from cache"""
        try:
            profile = self.get_profile()
            
            # Save to database if available and user_id is provided
            if user_id and profile:
                try:
                    from utils.database import save_user_profile
                    save_user_profile(user_id=user_id, profile_data=profile)
                except ImportError:
                    pass
                
            return {
                "status": True,
                "data": profile,
                "message": "Profile fetched successfully"
            }
        except Exception as e:
            logger.error(f"Error fetching profile: {str(e)}")
            return {
                "status": False,
                "data": {},
                "message": f"Error fetching profile: {str(e)}"
            }
            
    def get_funds_and_margins(self, user_id=None):
        """Get funds and margins with API or from cache"""
        try:
            funds = self.get_funds()
            
            # Save to database if available and user_id is provided
            if user_id and funds:
                try:
                    from utils.database import save_funds_and_margins
                    save_funds_and_margins(user_id=user_id, funds_data=funds)
                except ImportError:
                    pass
                
            return {
                "status": True,
                "data": funds,
                "message": "Funds and margins fetched successfully"
            }
        except Exception as e:
            logger.error(f"Error fetching funds and margins: {str(e)}")
            return {
                "status": False,
                "data": {},
                "message": f"Error fetching funds and margins: {str(e)}"
            }
    
    def _is_cache_valid(self, cache_file, max_age=86400):
        """Check if cache file is valid (not too old)"""
        if not os.path.exists(cache_file):
            return False
        
        file_time = os.path.getmtime(cache_file)
        file_age = time.time() - file_time
        
        # Allow longer cache for weekends
        now = datetime.now()
        if now.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            max_age = 7 * 86400  # 7 days for weekend
        
        return file_age < max_age
    
    def _get_current_user_id(self):
        """Get current user ID from database"""
        try:
            # Connect to database
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            
            # Find user by client ID
            cursor.execute("SELECT user_id FROM users WHERE client_id = ?", (self.client_id,))
            result = cursor.fetchone()
            
            conn.close()
            
            if result:
                return result[0]
            else:
                return None
        except Exception as e:
            logger.error(f"Error getting user ID: {str(e)}")
            return None

# Singleton instance
api_instance = AngelOneAPI()

def initialize_api(api_key=None, client_id=None, api_secret=None, mpin=None, totp_secret=None):
    """Initialize API with credentials"""
    global api_instance
    return api_instance.initialize(api_key, client_id, api_secret, mpin, totp_secret)

def get_api_instance():
    """Get API instance"""
    global api_instance
    return api_instance