"""
Data processing utilities for the stock prediction app.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/data_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def preprocess_data(data):
    """
    Preprocess stock data by calculating technical indicators
    
    Args:
        data (pd.DataFrame): Raw stock data with OHLCV columns
        
    Returns:
        pd.DataFrame: Processed data with technical indicators
    """
    if data is None or data.empty:
        logger.warning("Empty data provided for preprocessing")
        return pd.DataFrame()
    
    # Make a copy to avoid modifying original data
    df = data.copy()
    
    # Ensure expected columns exist
    required_columns = ['open', 'high', 'low', 'close']
    
    # Check column names (case-insensitive)
    df_columns_lower = [col.lower() for col in df.columns]
    for col in required_columns:
        if col not in df_columns_lower:
            logger.error(f"Required column {col} not found in data")
            return pd.DataFrame()
    
    # If column names are capitalized, convert to lowercase
    if 'Open' in df.columns:
        df.columns = [col.lower() for col in df.columns]
    
    # Fill missing values
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    try:
        # Calculate technical indicators
        
        # Moving Averages
        df['ma_5'] = df['close'].rolling(window=5).mean()
        df['ma_10'] = df['close'].rolling(window=10).mean()
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['ma_50'] = df['close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
        df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        
        # Bollinger Bands (20-day, 2 standard deviations)
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        
        # Relative Strength Index (RSI) - 14 periods
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Moving Average Convergence Divergence (MACD)
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Average True Range (ATR) - 14 periods
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = true_range.rolling(window=14).mean()
        
        # Commodity Channel Index (CCI) - 20 periods
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        mean_dev = typical_price.rolling(window=20).apply(
            lambda x: np.sum(np.abs(x - x.mean())) / len(x)
        )
        df['cci_20'] = (typical_price - typical_price.rolling(window=20).mean()) / (0.015 * mean_dev)
        
        # Stochastic Oscillator
        low_min = df['low'].rolling(window=14).min()
        high_max = df['high'].rolling(window=14).max()
        
        df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # Rate of Change (ROC) - 10 periods
        df['roc_10'] = (df['close'] / df['close'].shift(10) - 1) * 100
        
        # Williams %R - 14 periods
        df['williams_r'] = -100 * (high_max - df['close']) / (high_max - low_min)
        
        # On-Balance Volume (OBV)
        df['obv'] = np.where(
            df['close'] > df['close'].shift(1),
            df['volume'],
            np.where(
                df['close'] < df['close'].shift(1),
                -df['volume'],
                0
            )
        ).cumsum()
        
        # Price Rate of Change (ROC) for 1, 5, and 10 days
        df['price_roc_1'] = df['close'].pct_change(periods=1) * 100
        df['price_roc_5'] = df['close'].pct_change(periods=5) * 100
        df['price_roc_10'] = df['close'].pct_change(periods=10) * 100
        
        # Volume Rate of Change
        df['volume_roc_1'] = df['volume'].pct_change(periods=1) * 100
        df['volume_roc_5'] = df['volume'].pct_change(periods=5) * 100
        
        # Price volatility - standard deviation over different periods
        df['volatility_5'] = df['close'].rolling(window=5).std()
        df['volatility_10'] = df['close'].rolling(window=10).std()
        
        # Fill NaN values after calculating indicators
        df.fillna(method='bfill', inplace=True)
        
        # Drop rows with any remaining NaN values
        df.dropna(inplace=True)
        
        logger.info("Data preprocessing completed successfully")
        return df
    
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        return pd.DataFrame()

def create_features_and_targets(data, target_column='close', forecast_periods=[1, 3, 5, 7]):
    """
    Create features and targets for machine learning models
    
    Args:
        data (pd.DataFrame): Preprocessed data with technical indicators
        target_column (str): Target column name
        forecast_periods (list): List of forecast periods
        
    Returns:
        tuple: (features, targets) DataFrames
    """
    if data is None or data.empty:
        logger.warning("Empty data provided for feature creation")
        return pd.DataFrame(), pd.DataFrame()
    
    # Make a copy to avoid modifying original data
    df = data.copy()
    
    # Create target variables for different forecast periods
    targets = pd.DataFrame(index=df.index)
    
    for period in forecast_periods:
        # Future price
        targets[f'{target_column}_next_{period}'] = df[target_column].shift(-period)
        
        # Price change (percent)
        targets[f'{target_column}_change_{period}'] = df[target_column].pct_change(periods=-period) * 100
        
        # Price direction (1 if price goes up, 0 if down or unchanged)
        targets[f'{target_column}_direction_{period}'] = (df[target_column].shift(-period) > df[target_column]).astype(int)
    
    # Remove rows with NaN targets
    valid_idx = targets.dropna().index
    features = df.loc[valid_idx].copy()
    targets = targets.loc[valid_idx].copy()
    
    logger.info(f"Created features and targets with shapes: {features.shape}, {targets.shape}")
    return features, targets

def split_train_test(features, targets, test_size=0.2, random=False):
    """
    Split data into training and testing sets
    
    Args:
        features (pd.DataFrame): Feature data
        targets (pd.DataFrame): Target data
        test_size (float): Proportion of data to use for testing
        random (bool): Whether to randomize the split
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if features.empty or targets.empty:
        logger.warning("Empty data provided for train-test split")
        return None, None, None, None
    
    if random:
        # Random split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=test_size, random_state=42
        )
    else:
        # Time-based split
        split_idx = int(len(features) * (1 - test_size))
        X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
        y_train, y_test = targets.iloc[:split_idx], targets.iloc[split_idx:]
    
    logger.info(f"Split data with shapes: X_train={X_train.shape}, X_test={X_test.shape}")
    return X_train, X_test, y_train, y_test

def normalize_data(X_train, X_test):
    """
    Normalize data using MinMaxScaler
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    if X_train is None or X_test is None:
        logger.warning("Empty data provided for normalization")
        return None, None, None
    
    try:
        from sklearn.preprocessing import MinMaxScaler
        
        # Initialize scaler
        scaler = MinMaxScaler()
        
        # Fit on training data
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        # Transform test data
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        logger.info("Data normalization completed successfully")
        return X_train_scaled, X_test_scaled, scaler
    
    except Exception as e:
        logger.error(f"Error normalizing data: {str(e)}")
        return None, None, None

def create_train_test_data(data, target_column='close', test_size=0.2, window_size=20, forecast_days=7):
    """
    Create training and testing data for time series models
    
    Args:
        data (pd.DataFrame): Input stock data
        target_column (str): Target column to predict
        test_size (float): Proportion of data to use for testing
        window_size (int): Size of lookback window
        forecast_days (int): Number of days to forecast
        
    Returns:
        dict: Dictionary with train/test data and other metadata
    """
    if data is None or data.empty:
        logger.warning("Empty data provided for train/test creation")
        return None
    
    try:
        # Preprocess data
        processed_data = preprocess_data(data)
        
        # Create features and targets
        features, targets = create_features_and_targets(
            processed_data, 
            target_column=target_column, 
            forecast_periods=[forecast_days]
        )
        
        # Select the appropriate target column
        target_column_name = f'{target_column}_next_{forecast_days}'
        y = targets[target_column_name]
        
        # Split data
        X_train, X_test, y_train, y_test = split_train_test(features, y, test_size=test_size)
        
        # Normalize data
        X_train_scaled, X_test_scaled, scaler = normalize_data(X_train, X_test)
        
        # Store in a dictionary
        data_dict = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'scaler': scaler,
            'feature_names': features.columns.tolist(),
            'target_column': target_column,
            'target_name': target_column_name,
            'window_size': window_size,
            'forecast_days': forecast_days,
            'processed_data': processed_data
        }
        
        logger.info("Train-test data creation completed successfully")
        return data_dict
    
    except Exception as e:
        logger.error(f"Error creating train/test data: {str(e)}")
        return None

def inverse_transform(data, scaler, column_idx=None):
    """
    Inverse transform scaled data to original scale
    
    Args:
        data (pd.DataFrame/np.ndarray): Scaled data
        scaler (MinMaxScaler): Scaler used for normalization
        column_idx (int, optional): Index of column to transform if data is 1D array
        
    Returns:
        np.ndarray: Data in original scale
    """
    try:
        # If data is a pandas Series or 1D numpy array
        if isinstance(data, pd.Series) or (isinstance(data, np.ndarray) and data.ndim == 1):
            if column_idx is not None:
                # Create a dummy array with zeros
                dummy = np.zeros((len(data), scaler.scale_.shape[0]))
                # Place the data in the specified column
                dummy[:, column_idx] = data
                # Inverse transform
                return scaler.inverse_transform(dummy)[:, column_idx]
            else:
                # If no column index is provided, use first column
                return inverse_transform(data, scaler, 0)
        
        # If data is a pandas DataFrame
        elif isinstance(data, pd.DataFrame):
            return scaler.inverse_transform(data)
        
        # If data is a 2D numpy array
        elif isinstance(data, np.ndarray) and data.ndim == 2:
            return scaler.inverse_transform(data)
        
        else:
            logger.error(f"Unsupported data type for inverse_transform: {type(data)}")
            return None
    
    except Exception as e:
        logger.error(f"Error in inverse_transform: {str(e)}")
        return None