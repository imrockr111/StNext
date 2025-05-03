"""
Advanced time series models for stock prediction using scikit-learn.
These models offer improved forecasting capabilities over basic models.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
import joblib
import os
import json
import pickle

from utils.prediction_models import is_trading_day, get_next_trading_days, calculate_metrics

def preprocess_data_for_ml(data, feature_columns=None, target_columns=None, window_size=10):
    """
    Preprocess stock data for machine learning models
    
    Args:
        data (pd.DataFrame): Input stock data with OHLCV columns
        feature_columns (list, optional): List of feature column names
        target_columns (list, optional): List of target column names
        window_size (int): Size of the window for feature engineering
    
    Returns:
        dict: Dictionary with processed data and metadata
    """
    if feature_columns is None:
        # Use all available technical indicators
        feature_columns = [col for col in data.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        # Add OHLCV data
        feature_columns = ['open', 'high', 'low', 'close', 'volume'] + feature_columns
    
    if target_columns is None:
        target_columns = ['open', 'high', 'low', 'close']
    
    # Fill missing values
    data = data.copy()
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)
    data.fillna(0, inplace=True)
    
    # Feature engineering with time windows
    df_processed = data.copy()
    
    # Create lag features
    for col in feature_columns:
        for lag in range(1, window_size + 1):
            df_processed[f'{col}_lag_{lag}'] = data[col].shift(lag)
    
    # Create rolling statistics
    for col in feature_columns:
        # Rolling mean
        df_processed[f'{col}_rolling_mean_3'] = data[col].rolling(window=3).mean()
        df_processed[f'{col}_rolling_mean_5'] = data[col].rolling(window=5).mean()
        
        # Rolling std
        df_processed[f'{col}_rolling_std_5'] = data[col].rolling(window=5).std()
        
        # Rolling min/max
        df_processed[f'{col}_rolling_min_5'] = data[col].rolling(window=5).min()
        df_processed[f'{col}_rolling_max_5'] = data[col].rolling(window=5).max()
    
    # Drop rows with NaN values (due to lagging)
    df_processed.dropna(inplace=True)
    
    # Get updated feature columns
    engineered_feature_cols = [col for col in df_processed.columns if col not in target_columns]
    
    # Extract features and targets
    features = df_processed[engineered_feature_cols].values
    targets = df_processed[target_columns].values
    
    # Scale the data
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    scaled_features = feature_scaler.fit_transform(features)
    scaled_targets = target_scaler.fit_transform(targets)
    
    # Split into train and test sets
    train_size = int(len(df_processed) * 0.8)
    X_train = scaled_features[:train_size]
    X_test = scaled_features[train_size:]
    y_train = scaled_targets[:train_size]
    y_test = scaled_targets[train_size:]
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaled_features': scaled_features,
        'scaled_targets': scaled_targets,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'feature_columns': engineered_feature_cols,
        'target_columns': target_columns,
        'df_processed': df_processed
    }

def train_lstm_model(data, forecast_days=7, window_size=10, n_estimators=100, **kwargs):
    """
    Train a Random Forest model as a substitute for LSTM for stock price prediction
    
    Args:
        data (pd.DataFrame): Processed stock data
        forecast_days (int): Number of days to forecast
        window_size (int): Size of window for feature engineering
        n_estimators (int): Number of trees in the random forest
        
    Returns:
        tuple: (predictions DataFrame, metrics dict, trained model)
    """
    # Preprocess data
    processed_data = preprocess_data_for_ml(data, window_size=window_size)
    
    X_train = processed_data['X_train']
    X_test = processed_data['X_test']
    y_train = processed_data['y_train']
    y_test = processed_data['y_test']
    feature_scaler = processed_data['feature_scaler']
    target_scaler = processed_data['target_scaler']
    
    # Create and train the Random Forest model (as substitute for LSTM)
    model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on test data
    test_predictions = model.predict(X_test)
    
    # Inverse transform predictions
    test_predictions = target_scaler.inverse_transform(test_predictions)
    y_test_actual = target_scaler.inverse_transform(y_test)
    
    # Calculate metrics
    metrics = {
        'mse': mean_squared_error(y_test_actual, test_predictions),
        'rmse': np.sqrt(mean_squared_error(y_test_actual, test_predictions)),
        'mae': mean_absolute_error(y_test_actual, test_predictions),
        'mape': np.mean(np.abs((y_test_actual - test_predictions) / y_test_actual)) * 100
    }
    
    # Generate future predictions
    # Get the last data point (most recent)
    last_data_point = processed_data['scaled_features'][-1:]
    
    # Prepare for predictions
    future_features = last_data_point.copy()
    future_predictions = []
    
    # Get the last date from the original data
    last_date = data.index[-1]
    
    # Get next trading days
    forecast_dates = get_next_trading_days(last_date, forecast_days)
    
    # Instead of sequence-based prediction, create a simpler autoregressive model
    for _ in range(forecast_days):
        # Predict next values
        next_pred = model.predict(future_features)
        future_predictions.append(next_pred[0])
        
        # Update features for next prediction (simple approach)
        # This is a basic autoregressive approach - in a real application,
        # more sophisticated feature updating would be needed
        future_features = np.roll(future_features, -1, axis=0)
        future_features[0, :len(next_pred[0])] = next_pred[0]
    
    # Convert predictions to numpy array
    future_predictions = np.array(future_predictions)
    
    # Inverse transform predictions
    future_predictions = target_scaler.inverse_transform(future_predictions)
    
    # Create DataFrame with predictions
    predictions_df = pd.DataFrame(
        future_predictions,
        columns=[f'predicted_{col}' for col in processed_data['target_columns']],
        index=forecast_dates
    )
    
    # Add upper and lower bounds (simple approach: ±10%)
    for col in processed_data['target_columns']:
        predictions_df[f'upper_bound_{col}'] = predictions_df[f'predicted_{col}'] * 1.1
        predictions_df[f'lower_bound_{col}'] = predictions_df[f'predicted_{col}'] * 0.9
    
    # For backward compatibility with the existing code
    predictions_df['upper_bound'] = predictions_df['predicted_close'] * 1.1
    predictions_df['lower_bound'] = predictions_df['predicted_close'] * 0.9
    
    # Create model metadata
    model_metadata = {
        'type': 'RandomForest (LSTM replacement)',
        'window_size': window_size,
        'n_estimators': n_estimators,
        'feature_columns': processed_data['feature_columns'],
        'target_columns': processed_data['target_columns'],
        'metrics': metrics
    }
    
    # Create model bundle for saving
    model_bundle = {
        'model': model,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'metadata': model_metadata
    }
    
    return predictions_df, metrics, model_bundle

def train_gru_model(data, forecast_days=7, window_size=10, n_estimators=100, **kwargs):
    """
    Train a Gradient Boosting model as a substitute for GRU for stock price prediction
    
    Args:
        data (pd.DataFrame): Processed stock data
        forecast_days (int): Number of days to forecast
        window_size (int): Size of window for feature engineering
        n_estimators (int): Number of estimators in gradient boosting
        
    Returns:
        tuple: (predictions DataFrame, metrics dict, trained model)
    """
    # Preprocess data
    processed_data = preprocess_data_for_ml(data, window_size=window_size)
    
    X_train = processed_data['X_train']
    X_test = processed_data['X_test']
    y_train = processed_data['y_train']
    y_test = processed_data['y_test']
    feature_scaler = processed_data['feature_scaler']
    target_scaler = processed_data['target_scaler']
    
    # Create and train the GradientBoosting model (as substitute for GRU)
    model = MultiOutputRegressor(
        GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on test data
    test_predictions = model.predict(X_test)
    
    # Inverse transform predictions
    test_predictions = target_scaler.inverse_transform(test_predictions)
    y_test_actual = target_scaler.inverse_transform(y_test)
    
    # Calculate metrics
    metrics = {
        'mse': mean_squared_error(y_test_actual, test_predictions),
        'rmse': np.sqrt(mean_squared_error(y_test_actual, test_predictions)),
        'mae': mean_absolute_error(y_test_actual, test_predictions),
        'mape': np.mean(np.abs((y_test_actual - test_predictions) / y_test_actual)) * 100
    }
    
    # Generate future predictions
    # Get the last data point (most recent)
    last_data_point = processed_data['scaled_features'][-1:]
    
    # Prepare for predictions
    future_features = last_data_point.copy()
    future_predictions = []
    
    # Get the last date from the original data
    last_date = data.index[-1]
    
    # Get next trading days
    forecast_dates = get_next_trading_days(last_date, forecast_days)
    
    # Autoregressive prediction model for future dates
    for _ in range(forecast_days):
        # Predict next values
        next_pred = model.predict(future_features)
        future_predictions.append(next_pred[0])
        
        # Update features for next prediction (simple approach)
        future_features = np.roll(future_features, -1, axis=0)
        future_features[0, :len(next_pred[0])] = next_pred[0]
    
    # Convert predictions to numpy array
    future_predictions = np.array(future_predictions)
    
    # Inverse transform predictions
    future_predictions = target_scaler.inverse_transform(future_predictions)
    
    # Create DataFrame with predictions
    predictions_df = pd.DataFrame(
        future_predictions,
        columns=[f'predicted_{col}' for col in processed_data['target_columns']],
        index=forecast_dates
    )
    
    # Add upper and lower bounds (simple approach: ±10%)
    for col in processed_data['target_columns']:
        predictions_df[f'upper_bound_{col}'] = predictions_df[f'predicted_{col}'] * 1.1
        predictions_df[f'lower_bound_{col}'] = predictions_df[f'predicted_{col}'] * 0.9
    
    # For backward compatibility with the existing code
    predictions_df['upper_bound'] = predictions_df['predicted_close'] * 1.1
    predictions_df['lower_bound'] = predictions_df['predicted_close'] * 0.9
    
    # Create model metadata
    model_metadata = {
        'type': 'GradientBoosting (GRU replacement)',
        'window_size': window_size,
        'n_estimators': n_estimators,
        'feature_columns': processed_data['feature_columns'],
        'target_columns': processed_data['target_columns'],
        'metrics': metrics
    }
    
    # Create model bundle for saving
    model_bundle = {
        'model': model,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'metadata': model_metadata
    }
    
    return predictions_df, metrics, model_bundle

def train_transformer_model(data, forecast_days=7, window_size=10, n_estimators=100, **kwargs):
    """
    Train an ExtraTrees model as a substitute for Transformer for stock price prediction
    
    Args:
        data (pd.DataFrame): Processed stock data
        forecast_days (int): Number of days to forecast
        window_size (int): Size of window for feature engineering
        n_estimators (int): Number of trees in the ensemble
        
    Returns:
        tuple: (predictions DataFrame, metrics dict, trained model)
    """
    # Preprocess data
    processed_data = preprocess_data_for_ml(data, window_size=window_size)
    
    X_train = processed_data['X_train']
    X_test = processed_data['X_test']
    y_train = processed_data['y_train']
    y_test = processed_data['y_test']
    feature_scaler = processed_data['feature_scaler']
    target_scaler = processed_data['target_scaler']
    
    # Create and train the ExtraTrees model (as substitute for Transformer)
    model = MultiOutputRegressor(
        ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on test data
    test_predictions = model.predict(X_test)
    
    # Inverse transform predictions
    test_predictions = target_scaler.inverse_transform(test_predictions)
    y_test_actual = target_scaler.inverse_transform(y_test)
    
    # Calculate metrics
    metrics = {
        'mse': mean_squared_error(y_test_actual, test_predictions),
        'rmse': np.sqrt(mean_squared_error(y_test_actual, test_predictions)),
        'mae': mean_absolute_error(y_test_actual, test_predictions),
        'mape': np.mean(np.abs((y_test_actual - test_predictions) / y_test_actual)) * 100
    }
    
    # Generate future predictions
    # Get the last data point (most recent)
    last_data_point = processed_data['scaled_features'][-1:]
    
    # Prepare for predictions
    future_features = last_data_point.copy()
    future_predictions = []
    
    # Get the last date from the original data
    last_date = data.index[-1]
    
    # Get next trading days
    forecast_dates = get_next_trading_days(last_date, forecast_days)
    
    # Autoregressive prediction model for future dates
    for _ in range(forecast_days):
        # Predict next values
        next_pred = model.predict(future_features)
        future_predictions.append(next_pred[0])
        
        # Update features for next prediction (simple approach)
        future_features = np.roll(future_features, -1, axis=0)
        future_features[0, :len(next_pred[0])] = next_pred[0]
    
    # Convert predictions to numpy array
    future_predictions = np.array(future_predictions)
    
    # Inverse transform predictions
    future_predictions = target_scaler.inverse_transform(future_predictions)
    
    # Create DataFrame with predictions
    predictions_df = pd.DataFrame(
        future_predictions,
        columns=[f'predicted_{col}' for col in processed_data['target_columns']],
        index=forecast_dates
    )
    
    # Add upper and lower bounds (simple approach: ±10%)
    for col in processed_data['target_columns']:
        predictions_df[f'upper_bound_{col}'] = predictions_df[f'predicted_{col}'] * 1.1
        predictions_df[f'lower_bound_{col}'] = predictions_df[f'predicted_{col}'] * 0.9
    
    # For backward compatibility with the existing code
    predictions_df['upper_bound'] = predictions_df['predicted_close'] * 1.1
    predictions_df['lower_bound'] = predictions_df['predicted_close'] * 0.9
    
    # Create model metadata
    model_metadata = {
        'type': 'ExtraTrees (Transformer replacement)',
        'window_size': window_size,
        'n_estimators': n_estimators,
        'feature_columns': processed_data['feature_columns'],
        'target_columns': processed_data['target_columns'],
        'metrics': metrics
    }
    
    # Create model bundle for saving
    model_bundle = {
        'model': model,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'metadata': model_metadata
    }
    
    return predictions_df, metrics, model_bundle

def save_model(model_bundle, model_path):
    """
    Save a trained model bundle to disk
    
    Args:
        model_bundle (dict): Model bundle containing model, scalers, and metadata
        model_path (str): Path to save the model
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model using pickle
        with open(f"{model_path}_model.pkl", 'wb') as f:
            pickle.dump(model_bundle['model'], f)
        
        # Save scalers
        joblib.dump(model_bundle['feature_scaler'], f"{model_path}_feature_scaler.pkl")
        joblib.dump(model_bundle['target_scaler'], f"{model_path}_target_scaler.pkl")
        
        # Save metadata
        with open(f"{model_path}_metadata.json", 'w') as f:
            # Convert non-serializable parts of metadata
            metadata = model_bundle['metadata'].copy()
            if 'training_history' in metadata:
                # Convert numpy values to Python native types
                history = {}
                for key, values in metadata['training_history'].items():
                    history[key] = [float(val) for val in values]
                metadata['training_history'] = history
            
            # Convert numpy arrays in metrics to Python native types
            if 'metrics' in metadata:
                metrics = {}
                for key, value in metadata['metrics'].items():
                    metrics[key] = float(value) if hasattr(value, 'item') else value
                metadata['metrics'] = metrics
                
            json.dump(metadata, f)
        
        return True
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        return False

def load_model(model_path):
    """
    Load a trained model bundle from disk
    
    Args:
        model_path (str): Path to load the model from
        
    Returns:
        dict: Model bundle if successful, None otherwise
    """
    try:
        # Load model using pickle
        with open(f"{model_path}_model.pkl", 'rb') as f:
            model = pickle.load(f)
        
        # Load scalers
        feature_scaler = joblib.load(f"{model_path}_feature_scaler.pkl")
        target_scaler = joblib.load(f"{model_path}_target_scaler.pkl")
        
        # Load metadata
        with open(f"{model_path}_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Create and return model bundle
        model_bundle = {
            'model': model,
            'feature_scaler': feature_scaler,
            'target_scaler': target_scaler,
            'metadata': metadata
        }
        
        return model_bundle
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def predict_with_model(model_bundle, data, forecast_days=7):
    """
    Make predictions using a loaded model bundle
    
    Args:
        model_bundle (dict): Model bundle containing model, scalers, and metadata
        data (pd.DataFrame): Historical stock data
        forecast_days (int): Number of days to forecast
        
    Returns:
        pd.DataFrame: DataFrame with predictions
    """
    try:
        model = model_bundle['model']
        feature_scaler = model_bundle['feature_scaler']
        target_scaler = model_bundle['target_scaler']
        metadata = model_bundle['metadata']
        
        # Check model type to determine prediction approach
        model_type = metadata.get('type', '')
        is_ml_model = 'Random' in model_type or 'Gradient' in model_type or 'ExtraTrees' in model_type
        
        if is_ml_model:
            # For sklearn models (Random Forest, Gradient Boosting, ExtraTrees)
            window_size = metadata.get('window_size', 10)
            
            # Preprocess data similar to training
            # Note: This is a simplified version of preprocess_data_for_ml
            # We need to create the same features as during training
            
            # Feature columns might include original feature names before engineering
            # Get base feature columns (without _lag_ or _rolling_ suffix)
            base_feature_cols = []
            for col in metadata['feature_columns']:
                if '_lag_' not in col and '_rolling_' not in col:
                    if col not in base_feature_cols:
                        base_feature_cols.append(col)
            
            # Fill missing values
            processed_data = data.copy()
            processed_data.fillna(method='ffill', inplace=True)
            processed_data.fillna(method='bfill', inplace=True)
            processed_data.fillna(0, inplace=True)
            
            # Create all required engineered features
            # Create lag features for base columns
            for col in base_feature_cols:
                if col in data.columns:
                    for lag in range(1, window_size + 1):
                        processed_data[f'{col}_lag_{lag}'] = data[col].shift(lag)
                    
                    # Create rolling statistics
                    processed_data[f'{col}_rolling_mean_3'] = data[col].rolling(window=3).mean()
                    processed_data[f'{col}_rolling_mean_5'] = data[col].rolling(window=5).mean()
                    processed_data[f'{col}_rolling_std_5'] = data[col].rolling(window=5).std()
                    processed_data[f'{col}_rolling_min_5'] = data[col].rolling(window=5).min()
                    processed_data[f'{col}_rolling_max_5'] = data[col].rolling(window=5).max()
            
            # Drop rows with NaN values
            processed_data.dropna(inplace=True)
            
            # Get the last data point for prediction
            if len(processed_data) == 0:
                return None  # Not enough data after feature engineering
                
            last_data_point = processed_data.iloc[-1:]
            
            # Extract features and scale
            feature_values = last_data_point[metadata['feature_columns']].values
            scaled_features = feature_scaler.transform(feature_values)
            
            # Get the last date
            last_date = data.index[-1]
            
            # Get next trading days
            forecast_dates = get_next_trading_days(last_date, forecast_days)
            
            # Prepare for autoregressive predictions
            future_features = scaled_features.copy()
            future_predictions = []
            
            # Generate predictions for each future day
            for _ in range(forecast_days):
                # Predict next values
                next_pred = model.predict(future_features)
                future_predictions.append(next_pred[0])
                
                # Update features for next prediction
                # This is a simplified approach and may need refinement for real use
                future_features = np.roll(future_features, -1, axis=0)
                future_features[0, :len(next_pred[0])] = next_pred[0]
            
            # Convert predictions to numpy array
            future_predictions = np.array(future_predictions)
            
            # Inverse transform predictions
            future_predictions = target_scaler.inverse_transform(future_predictions)
            
        else:
            # For traditional models (ARIMA, Prophet)
            # Use a different approach or delegate to another function
            return None
        
        # Create DataFrame with predictions
        predictions_df = pd.DataFrame(
            future_predictions,
            columns=[f'predicted_{col}' for col in metadata['target_columns']],
            index=forecast_dates
        )
        
        # Add upper and lower bounds (simple approach: ±10%)
        for col in metadata['target_columns']:
            predictions_df[f'upper_bound_{col}'] = predictions_df[f'predicted_{col}'] * 1.1
            predictions_df[f'lower_bound_{col}'] = predictions_df[f'predicted_{col}'] * 0.9
        
        # For backward compatibility with the existing code
        predictions_df['upper_bound'] = predictions_df['predicted_close'] * 1.1
        predictions_df['lower_bound'] = predictions_df['predicted_close'] * 0.9
        
        return predictions_df
    except Exception as e:
        print(f"Error making predictions: {str(e)}")
        return None

def evaluate_predictions(predictions_df, actual_data, target_column='close'):
    """
    Evaluate prediction accuracy against actual data
    
    Args:
        predictions_df (pd.DataFrame): DataFrame with predictions
        actual_data (pd.DataFrame): DataFrame with actual stock data
        target_column (str): Column to evaluate (default: 'close')
        
    Returns:
        tuple: (accuracy metrics, accuracy percentage)
    """
    # Extract dates that exist in both predictions and actual data
    common_dates = predictions_df.index.intersection(actual_data.index)
    
    if len(common_dates) == 0:
        return None, 0
    
    # Extract predicted and actual values
    predicted = predictions_df.loc[common_dates, f'predicted_{target_column}'].values
    actual = actual_data.loc[common_dates, target_column].values
    
    # Calculate metrics
    metrics = {
        'mse': mean_squared_error(actual, predicted),
        'rmse': np.sqrt(mean_squared_error(actual, predicted)),
        'mae': mean_absolute_error(actual, predicted),
        'mape': np.mean(np.abs((actual - predicted) / actual)) * 100
    }
    
    # Calculate simple accuracy percentage (based on direction)
    direction_actual = np.diff(actual)
    direction_predicted = np.diff(predicted)
    correct_direction = np.sum((direction_actual > 0) == (direction_predicted > 0))
    accuracy_pct = correct_direction / len(direction_actual) * 100 if len(direction_actual) > 0 else 0
    
    return metrics, accuracy_pct