"""
API connector for AngelOne SmartAPI integration.
"""

import os
import time
from SmartApi.smartConnect import SmartConnect
import pandas as pd
import json
import pyotp
import logging
import requests
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/api_connector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SmartAPIConnector:
    """
    Connector for AngelOne SmartAPI
    
    Args:
        api_key (str): API key for SmartAPI
        client_id (str): Client ID for SmartAPI
        api_secret (str): API secret for SmartAPI
        mpin (str): MPIN for SmartAPI login
        totp_secret (str): TOTP secret key for 2FA
    """
    def __init__(self, api_key, client_id, api_secret, mpin, totp_secret):
        self.api_key = api_key
        self.client_id = client_id
        self.api_secret = api_secret
        self.mpin = mpin
        self.totp_secret = totp_secret
        self.smart_api = None
        self.session_token = None
        self.refresh_token = None
        self.feed_token = None
        self.login_time = None
        self.max_session_time = 6 * 60 * 60  # 6 hours in seconds

    def login(self):
        """
        Login to SmartAPI
        
        Returns:
            bool: True if login successful, False otherwise
        """
        try:
            # Initialize SmartAPI
            self.smart_api = SmartConnect(api_key=self.api_key)
            
            # Generate TOTP
            totp = pyotp.TOTP(self.totp_secret)
            totp_value = totp.now()
            
            # Login to SmartAPI
            data = self.smart_api.generateSession(
                self.client_id,
                self.mpin,
                totp_value
            )
            
            logger.info("Login response: %s", json.dumps(data, indent=2))
            
            # Save tokens
            if data['status']:
                # Handle both old and new response formats
                if 'sessionToken' in data['data']:
                    self.session_token = data['data']['sessionToken']
                elif 'jwtToken' in data['data']:
                    self.session_token = data['data']['jwtToken']
                else:
                    logger.error("Session token not found in response")
                    return False
                    
                self.refresh_token = data['data']['refreshToken']
                
                try:
                    self.feed_token = data['data']['feedToken']
                except KeyError:
                    # Try the old method if 'feedToken' is not in the response
                    try:
                        self.feed_token = self.smart_api.getfeedToken()
                    except Exception as e:
                        logger.warning(f"Could not get feed token: {str(e)}")
                        # Continue without feed token
                
                self.login_time = datetime.now()
                
                # Get user profile (but don't fail if it doesn't work)
                try:
                    self.user_profile = self.smart_api.getProfile()
                except Exception as e:
                    logger.warning(f"Could not get user profile: {str(e)}")
                    # Continue without user profile
                
                logger.info("Login successful for client ID: %s", self.client_id)
                return True
            else:
                logger.error("Login failed: %s", data['message'])
                return False
        
        except Exception as e:
            logger.error("Error in login: %s", str(e))
            return False
    
    def logout(self):
        """
        Logout from SmartAPI
        
        Returns:
            bool: True if logout successful, False otherwise
        """
        try:
            if self.smart_api:
                result = self.smart_api.terminateSession(self.client_id)
                logger.info("Logout result: %s", result)
                return True
            return False
        except Exception as e:
            logger.error("Error in logout: %s", str(e))
            return False
    
    def refresh_token_if_needed(self):
        """
        Refresh session token if it's about to expire
        
        Returns:
            bool: True if token is valid, False otherwise
        """
        try:
            # Check if login time is set
            if not self.login_time:
                logger.info("No login time set, need to login first")
                return False
            
            # Check if session is about to expire (after 5.5 hours)
            time_since_login = (datetime.now() - self.login_time).total_seconds()
            if time_since_login > (5.5 * 60 * 60):
                logger.info("Session token about to expire, refreshing...")
                
                # Try to refresh token
                if self.refresh_token:
                    data = self.smart_api.renewAccessToken(
                        self.refresh_token,
                        self.client_id
                    )
                    
                    if data['status']:
                        # Handle both old and new token formats
                        if 'sessionToken' in data['data']:
                            self.session_token = data['data']['sessionToken']
                        elif 'jwtToken' in data['data']:
                            self.session_token = data['data']['jwtToken']
                        else:
                            logger.error("Session token not found in response")
                            return self.login()  # Try to login again
                            
                        self.login_time = datetime.now()
                        logger.info("Token refreshed successfully")
                        return True
                    else:
                        logger.error("Failed to refresh token: %s", data['message'])
                        return self.login()  # Try to login again
                else:
                    logger.error("No refresh token available")
                    return self.login()  # Try to login again
            
            # Token is still valid
            return True
        
        except Exception as e:
            logger.error("Error in refresh_token_if_needed: %s", str(e))
            return self.login()  # Try to login again
    
    def get_historical_data(self, symbol, exchange, from_date, to_date, interval="ONE_DAY"):
        """
        Get historical data for a symbol
        
        Args:
            symbol (str): Symbol to fetch data for
            exchange (str): Exchange code (NSE, BSE, etc.)
            from_date (str): Start date in format 'YYYY-MM-DD'
            to_date (str): End date in format 'YYYY-MM-DD'
            interval (str, optional): Interval for data (ONE_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE, THIRTY_MINUTE, ONE_HOUR, ONE_DAY). Defaults to "ONE_DAY".
        
        Returns:
            pd.DataFrame: Historical data
        """
        try:
            # Get instrument token
            token = None
            
            # Try to get token using the instrument_master module
            try:
                from utils.instrument_master import get_instrument_by_symbol
                instrument = get_instrument_by_symbol(symbol, exchange)
                if instrument:
                    token = instrument['token']
            except ImportError:
                logger.warning("instrument_master module not available")
                token = None
            
            # If token is found, use it to fetch data
            if token:
                return self.get_historical_data_by_token(token, exchange, from_date, to_date, interval)
            else:
                logger.error("Instrument token not found for symbol: %s", symbol)
                return pd.DataFrame()
        
        except Exception as e:
            logger.error("Error in get_historical_data: %s", str(e))
            return pd.DataFrame()
    
    def get_historical_data_by_token(self, token, exchange, from_date, to_date, interval="ONE_DAY"):
        """
        Get historical data for a token
        
        Args:
            token (str): Instrument token
            exchange (str): Exchange code (NSE, BSE, etc.)
            from_date (str): Start date in format 'YYYY-MM-DD'
            to_date (str): End date in format 'YYYY-MM-DD'
            interval (str, optional): Interval for data (ONE_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE, THIRTY_MINUTE, ONE_HOUR, ONE_DAY). Defaults to "ONE_DAY".
        
        Returns:
            pd.DataFrame: Historical data
        """
        try:
            # Check database first
            try:
                from utils.database import check_stock_data_availability, get_stock_data, save_stock_data
                
                # Check if data is available in database
                if check_stock_data_availability(token, exchange, from_date, to_date, interval):
                    logger.info("Data available in database, fetching from there")
                    return get_stock_data(token, exchange, from_date, to_date, interval)
            except ImportError:
                logger.warning("database module not available")
            
            # Ensure login is valid
            if not self.refresh_token_if_needed():
                if not self.login():
                    logger.error("Failed to login")
                    return pd.DataFrame()
            
            # Parse dates
            from_date_obj = datetime.strptime(from_date, '%Y-%m-%d')
            to_date_obj = datetime.strptime(to_date, '%Y-%m-%d')
            
            # SmartAPI requires times in these formats
            from_date_str = from_date_obj.strftime('%Y-%m-%d %H:%M')
            to_date_str = to_date_obj.strftime('%Y-%m-%d %H:%M')
            
            # SmartAPI has a limit of fetching data for 1 year at a time
            # Split into chunks if needed
            chunks = []
            current_from_date = from_date_obj
            
            while current_from_date < to_date_obj:
                chunk_to_date = min(current_from_date + timedelta(days=365), to_date_obj)
                
                chunk_from_str = current_from_date.strftime('%Y-%m-%d %H:%M')
                chunk_to_str = chunk_to_date.strftime('%Y-%m-%d %H:%M')
                
                logger.info("Fetching data chunk from %s to %s", chunk_from_str, chunk_to_str)
                
                # Fetch data for this chunk
                try:
                    params = {
                        "exchange": exchange,
                        "symboltoken": token,
                        "interval": interval,
                        "fromdate": chunk_from_str,
                        "todate": chunk_to_str
                    }
                    
                    hist_data = self.smart_api.getCandleData(params)
                    
                    if hist_data and hist_data.get('status'):
                        # Parse data
                        data = hist_data.get('data', [])
                        
                        if data:
                            # Create dataframe
                            chunk_df = pd.DataFrame(
                                data,
                                columns=['datetime', 'open', 'high', 'low', 'close', 'volume']
                            )
                            
                            # Convert datetime to pandas datetime
                            chunk_df['datetime'] = pd.to_datetime(chunk_df['datetime'])
                            
                            # Set datetime as index
                            chunk_df.set_index('datetime', inplace=True)
                            
                            chunks.append(chunk_df)
                        else:
                            logger.warning("No data found for chunk %s to %s", chunk_from_str, chunk_to_str)
                    else:
                        logger.error("Error fetching data for chunk %s to %s: %s", 
                                     chunk_from_str, chunk_to_str, 
                                     hist_data.get('message', 'Unknown error'))
                
                except Exception as e:
                    logger.error("Error fetching data chunk: %s", str(e))
                
                # Move to next chunk
                current_from_date = chunk_to_date + timedelta(days=1)
            
            # Combine all chunks
            if chunks:
                result_df = pd.concat(chunks)
                
                # Remove duplicates
                result_df = result_df[~result_df.index.duplicated(keep='first')]
                
                # Sort by datetime
                result_df.sort_index(inplace=True)
                
                # Save to database if available
                try:
                    save_stock_data(token, exchange, result_df, interval)
                    logger.info("Data saved to database")
                except Exception as e:
                    logger.error("Error saving data to database: %s", str(e))
                
                return result_df
            else:
                logger.warning("No data found")
                return pd.DataFrame()
        
        except Exception as e:
            logger.error("Error in get_historical_data_by_token: %s", str(e))
            return pd.DataFrame()
    
    def get_ltp_data(self, symbols, exchange):
        """
        Get LTP (Last Traded Price) data for symbols
        
        Args:
            symbols (list): List of symbols
            exchange (str): Exchange code (NSE, BSE, etc.)
            
        Returns:
            dict: LTP data
        """
        try:
            # Ensure login is valid
            if not self.refresh_token_if_needed():
                if not self.login():
                    logger.error("Failed to login")
                    return {}
            
            # Convert symbols to list if it's a string
            if isinstance(symbols, str):
                symbols = [symbols]
            
            # Get LTP data
            ltp_data = {}
            for symbol in symbols:
                try:
                    # Get token for symbol
                    token = None
                    
                    # Try to get token using the instrument_master module
                    try:
                        from utils.instrument_master import get_instrument_by_symbol
                        instrument = get_instrument_by_symbol(symbol, exchange)
                        if instrument:
                            token = instrument['token']
                    except ImportError:
                        logger.warning("instrument_master module not available")
                        token = None
                    
                    if token:
                        params = {
                            "exchange": exchange,
                            "tradingsymbol": symbol,
                            "symboltoken": token
                        }
                        
                        ltp = self.smart_api.ltpData(params)
                        
                        if ltp and ltp.get('status'):
                            ltp_data[symbol] = ltp.get('data', {})
                        else:
                            logger.error("Error fetching LTP for %s: %s", symbol, ltp.get('message', 'Unknown error'))
                    else:
                        logger.error("Instrument token not found for symbol: %s", symbol)
                
                except Exception as e:
                    logger.error("Error fetching LTP for %s: %s", symbol, str(e))
            
            return ltp_data
        
        except Exception as e:
            logger.error("Error in get_ltp_data: %s", str(e))
            return {}
    
    def get_profile(self):
        """
        Get user profile
        
        Returns:
            dict: User profile data
        """
        try:
            # Ensure login is valid
            if not self.refresh_token_if_needed():
                if not self.login():
                    logger.error("Failed to login")
                    return {}
            
            profile_data = self.smart_api.getProfile()
            
            if profile_data and profile_data.get('status'):
                return profile_data.get('data', {})
            else:
                logger.error("Error fetching profile: %s", profile_data.get('message', 'Unknown error'))
                return {}
        
        except Exception as e:
            logger.error("Error in get_profile: %s", str(e))
            return {}
    
    def get_funds(self):
        """
        Get funds and margin limits
        
        Returns:
            dict: Funds and margin data
        """
        try:
            # Ensure login is valid
            if not self.refresh_token_if_needed():
                if not self.login():
                    logger.error("Failed to login")
                    return {}
            
            funds_data = self.smart_api.rmsLimit()
            
            if funds_data and funds_data.get('status'):
                return funds_data.get('data', {})
            else:
                logger.error("Error fetching funds: %s", funds_data.get('message', 'Unknown error'))
                return {}
        
        except Exception as e:
            logger.error("Error in get_funds: %s", str(e))
            return {}
    
    def get_holdings(self):
        """
        Get holdings
        
        Returns:
            list: Holdings data
        """
        try:
            # Ensure login is valid
            if not self.refresh_token_if_needed():
                if not self.login():
                    logger.error("Failed to login")
                    return []
            
            holdings_data = self.smart_api.holding()
            
            if holdings_data and holdings_data.get('status'):
                return holdings_data.get('data', [])
            else:
                logger.error("Error fetching holdings: %s", holdings_data.get('message', 'Unknown error'))
                return []
        
        except Exception as e:
            logger.error("Error in get_holdings: %s", str(e))
            return []
    
    def get_portfolio(self):
        """
        Get portfolio (holdings with current prices)
        
        Returns:
            dict: Portfolio data
        """
        try:
            # Get holdings
            holdings = self.get_holdings()
            
            if not holdings:
                return {
                    'holdings': [],
                    'total_investment': 0,
                    'total_current_value': 0,
                    'total_pnl': 0,
                    'total_pnl_percent': 0
                }
            
            # Get LTP for all symbols
            symbols = []
            for holding in holdings:
                symbol = holding.get('tradingsymbol')
                if symbol:
                    symbols.append(symbol)
            
            # Skip if no symbols
            if not symbols:
                return {
                    'holdings': holdings,
                    'total_investment': 0,
                    'total_current_value': 0,
                    'total_pnl': 0,
                    'total_pnl_percent': 0
                }
            
            # Get LTP data
            ltp_data = self.get_ltp_data(symbols, 'NSE')  # Assuming NSE for now
            
            # Update holdings with current prices
            total_investment = 0
            total_current_value = 0
            
            for holding in holdings:
                symbol = holding.get('tradingsymbol')
                
                if symbol and symbol in ltp_data:
                    # Extract data
                    quantity = float(holding.get('quantity', 0))
                    average_price = float(holding.get('averageprice', 0))
                    
                    # Calculate investment
                    investment = quantity * average_price
                    total_investment += investment
                    
                    # Get current price
                    current_price = float(ltp_data[symbol].get('ltp', 0))
                    
                    # Calculate current value
                    current_value = quantity * current_price
                    total_current_value += current_value
                    
                    # Calculate P&L
                    pnl = current_value - investment
                    pnl_percent = (pnl / investment) * 100 if investment > 0 else 0
                    
                    # Update holding with these values
                    holding['current_price'] = current_price
                    holding['current_value'] = current_value
                    holding['pnl'] = pnl
                    holding['pnl_percent'] = pnl_percent
            
            # Calculate overall P&L
            total_pnl = total_current_value - total_investment
            total_pnl_percent = (total_pnl / total_investment) * 100 if total_investment > 0 else 0
            
            return {
                'holdings': holdings,
                'total_investment': total_investment,
                'total_current_value': total_current_value,
                'total_pnl': total_pnl,
                'total_pnl_percent': total_pnl_percent
            }
        
        except Exception as e:
            logger.error("Error in get_portfolio: %s", str(e))
            return {
                'holdings': [],
                'total_investment': 0,
                'total_current_value': 0,
                'total_pnl': 0,
                'total_pnl_percent': 0
            }
    
    def get_positions(self):
        """
        Get positions
        
        Returns:
            list: Positions data
        """
        try:
            # Ensure login is valid
            if not self.refresh_token_if_needed():
                if not self.login():
                    logger.error("Failed to login")
                    return []
            
            positions_data = self.smart_api.position()
            
            if positions_data and positions_data.get('status'):
                return positions_data.get('data', [])
            else:
                logger.error("Error fetching positions: %s", positions_data.get('message', 'Unknown error'))
                return []
        
        except Exception as e:
            logger.error("Error in get_positions: %s", str(e))
            return []
    
    def get_orders(self):
        """
        Get orders
        
        Returns:
            list: Orders data
        """
        try:
            # Ensure login is valid
            if not self.refresh_token_if_needed():
                if not self.login():
                    logger.error("Failed to login")
                    return []
            
            orders_data = self.smart_api.orderBook()
            
            if orders_data and orders_data.get('status'):
                return orders_data.get('data', [])
            else:
                logger.error("Error fetching orders: %s", orders_data.get('message', 'Unknown error'))
                return []
        
        except Exception as e:
            logger.error("Error in get_orders: %s", str(e))
            return []
    
    def get_trades(self):
        """
        Get trades
        
        Returns:
            list: Trades data
        """
        try:
            # Ensure login is valid
            if not self.refresh_token_if_needed():
                if not self.login():
                    logger.error("Failed to login")
                    return []
            
            trades_data = self.smart_api.tradeBook()
            
            if trades_data and trades_data.get('status'):
                return trades_data.get('data', [])
            else:
                logger.error("Error fetching trades: %s", trades_data.get('message', 'Unknown error'))
                return []
        
        except Exception as e:
            logger.error("Error in get_trades: %s", str(e))
            return []
            
    def is_market_open(self):
        """
        Check if market is open
        
        Returns:
            bool: True if market is open, False otherwise
        """
        try:
            # Get current time in IST (Indian Standard Time)
            now = datetime.now() + timedelta(hours=5, minutes=30)  # Convert to IST
            
            # Check if it's a weekday (0 = Monday, 6 = Sunday)
            if now.weekday() >= 5:  # Saturday or Sunday
                return False
            
            # Check if time is between 9:15 AM and 3:30 PM IST
            market_open_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close_time = now.replace(hour=15, minute=30, second=0, microsecond=0)
            
            if now < market_open_time or now > market_close_time:
                return False
            
            # Check for holidays
            # This is a simplified approach, would need to integrate with a holiday calendar
            # for a more accurate check
            
            return True
        
        except Exception as e:
            logger.error("Error in is_market_open: %s", str(e))
            return False