"""
Instrument master management utilities.
"""

import pandas as pd
import json
import os
from datetime import datetime
import requests
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/instruments.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = './cache'
INSTRUMENT_CACHE_FILE = os.path.join(CACHE_DIR, 'instrument_master.json')
OPEN_API_URL = "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"

class InstrumentManager:
    """
    Instrument manager for caching and retrieving instruments
    """
    def __init__(self, cache_file=INSTRUMENT_CACHE_FILE):
        """Initialize instrument manager"""
        self.cache_file = cache_file
        self.instruments = {}
        self.instruments_by_token = {}
        self.instruments_by_symbol = {}
        self.last_updated = None
        self.load_instruments()
    
    def load_instruments(self):
        """Load instruments from cache file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    self.instruments = data.get('instruments', {})
                    self.last_updated = data.get('last_updated')
                    
                    # Create lookup dictionaries
                    self.instruments_by_token = {}
                    self.instruments_by_symbol = {}
                    
                    for token, instrument in self.instruments.items():
                        self.instruments_by_token[token] = instrument
                        
                        symbol = instrument.get('symbol')
                        if symbol:
                            self.instruments_by_symbol[symbol] = instrument
                    
                    logger.info(f"Loaded {len(self.instruments)} instruments from cache")
                    return True
            else:
                logger.warning("Instrument cache file not found")
                return False
        except Exception as e:
            logger.error(f"Error loading instruments: {str(e)}")
            return False
    
    def save_instruments(self):
        """Save instruments to cache file"""
        try:
            # Create cache directory if it doesn't exist
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            
            # Save to cache file
            with open(self.cache_file, 'w') as f:
                json.dump({
                    'instruments': self.instruments,
                    'last_updated': self.last_updated
                }, f)
            
            logger.info(f"Saved {len(self.instruments)} instruments to cache")
            return True
        except Exception as e:
            logger.error(f"Error saving instruments: {str(e)}")
            return False
    
    def fetch_instrument_master(self):
        """Fetch instrument master from API"""
        try:
            logger.info("Fetching instrument master from API...")
            response = requests.get(OPEN_API_URL)
            response.raise_for_status()
            
            # Parse JSON
            instruments_data = response.json()
            
            # Process instruments
            for instrument in instruments_data:
                token = str(instrument.get('token', ''))
                
                if token:
                    # Create instrument record
                    self.instruments[token] = {
                        'token': token,
                        'symbol': instrument.get('symbol', ''),
                        'name': instrument.get('name', ''),
                        'expiry': instrument.get('expiry', ''),
                        'strike': instrument.get('strike', 0),
                        'lot_size': instrument.get('lotsize', 0),
                        'tick_size': instrument.get('tick_size', 0),
                        'exchange': instrument.get('exch_seg', ''),
                        'instrument_type': instrument.get('instrumenttype', '')
                    }
                    
                    # Update lookup dictionaries
                    self.instruments_by_token[token] = self.instruments[token]
                    
                    symbol = instrument.get('symbol')
                    if symbol:
                        self.instruments_by_symbol[symbol] = self.instruments[token]
            
            # Update last updated timestamp
            self.last_updated = datetime.now().isoformat()
            
            # Save to cache
            self.save_instruments()
            
            logger.info(f"Fetched {len(self.instruments)} instruments from API")
            
            # Save to database if available
            try:
                from utils.database import save_instruments
                
                # Convert to list format for database
                instruments_list = list(self.instruments.values())
                save_instruments(instruments_list)
                logger.info(f"Saved {len(instruments_list)} instruments to database")
            except ImportError:
                logger.warning("Database module not available, skipping database save")
            
            return True
        except Exception as e:
            logger.error(f"Error fetching instrument master: {str(e)}")
            return False
    
    def get_instrument_by_token(self, token):
        """Get instrument by token"""
        return self.instruments_by_token.get(str(token))
    
    def get_instrument_by_symbol(self, symbol, exchange=None):
        """Get instrument by symbol"""
        instrument = self.instruments_by_symbol.get(symbol)
        
        if instrument and exchange:
            # Check exchange
            if instrument.get('exchange') == exchange:
                return instrument
            else:
                # Search for instrument with matching symbol and exchange
                for token, instr in self.instruments.items():
                    if instr.get('symbol') == symbol and instr.get('exchange') == exchange:
                        return instr
                
                return None
        else:
            return instrument
    
    def search_instruments(self, query, exchange=None, limit=20):
        """Search instruments by name, symbol, or token"""
        results = []
        
        # Minimum 2 characters required for search
        if not query or len(query) < 2:
            return results
            
        query = query.lower()
        
        for token, instrument in self.instruments.items():
            # Check if query matches token, symbol, or name
            symbol = instrument.get('symbol', '').lower()
            name = instrument.get('name', '').lower()
            
            if (
                query in token.lower() or
                query in symbol or
                query in name or
                # Try partial matches for symbols and names
                (len(query) >= 2 and (
                    symbol.startswith(query) or
                    name.startswith(query) or
                    any(word.startswith(query) for word in name.split())
                ))
            ):
                # Check exchange if provided
                if exchange and instrument.get('exchange') != exchange:
                    continue
                
                results.append(instrument)
                
                # Limit results
                if len(results) >= limit:
                    break
        
        return results

# Singleton instance
_instrument_manager = None

def get_instrument_manager():
    """Get instrument manager singleton instance"""
    global _instrument_manager
    
    if _instrument_manager is None:
        _instrument_manager = InstrumentManager()
        
        # If empty, try to fetch instruments
        if not _instrument_manager.instruments:
            _instrument_manager.fetch_instrument_master()
    
    return _instrument_manager

def get_instrument_by_token(token):
    """Get instrument by token"""
    manager = get_instrument_manager()
    return manager.get_instrument_by_token(token)

def get_instrument_by_symbol(symbol, exchange=None):
    """Get instrument by symbol"""
    manager = get_instrument_manager()
    return manager.get_instrument_by_symbol(symbol, exchange)

def search_instruments(query, exchange=None, limit=20):
    """Search instruments by name, symbol, or token"""
    manager = get_instrument_manager()
    return manager.search_instruments(query, exchange, limit)

def fetch_instrument_master(save_to_db=True):
    """
    Fetch instrument master from API
    
    Args:
        save_to_db (bool): Whether to save instruments to database
        
    Returns:
        pd.DataFrame: DataFrame with instruments if successful, empty DataFrame otherwise
    """
    try:
        logger.info("Fetching instrument master from API...")
        response = requests.get(OPEN_API_URL)
        response.raise_for_status()
        
        # Parse JSON
        instruments_data = response.json()
        
        # Create DataFrame
        df = pd.DataFrame(instruments_data)
        
        # Update instrument manager
        manager = get_instrument_manager()
        manager.fetch_instrument_master()
        
        # Save to database if requested
        if save_to_db:
            try:
                from utils.database import save_instruments
                save_instruments(instruments_data)
                logger.info(f"Saved {len(instruments_data)} instruments to database")
            except ImportError:
                logger.warning("Database module not available, skipping database save")
            except Exception as e:
                logger.error(f"Error saving instruments to database: {str(e)}")
        
        return df
    except Exception as e:
        logger.error(f"Error fetching instrument master: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error