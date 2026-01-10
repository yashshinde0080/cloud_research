"""
LSTM-based Prediction for Bursty Workloads
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import logging
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logger = logging.getLogger(__name__)


class LSTMPredictor:
    """LSTM prediction for bursty and complex workloads"""
    
    def __init__(self, 
                 lookback: int = 12,
                 units: int = 32,
                 epochs: int = 50,
                 batch_size: int = 32,
                 verbose: int = 0):
        self.lookback = lookback
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None
        self.scaler = None
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create input sequences for LSTM"""
        X, y = [], []
        for i in range(len(data) - self.lookback):
            X.append(data[i:i+self.lookback])
            y.append(data[i+self.lookback])
        return np.array(X), np.array(y)
    
    def _build_model(self, input_shape: Tuple) -> None:
        """Build LSTM architecture"""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.optimizers import Adam
            
            self.model = Sequential([
                LSTM(self.units, input_shape=input_shape, return_sequences=False),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1)
            ])
            
            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            logger.info(f"LSTM model built: {self.units} units")
            
        except ImportError:
            logger.error("TensorFlow not installed. Install with: pip install tensorflow")
            raise
    
    def fit(self, time_series: np.ndarray) -> 'LSTMPredictor':
        """Fit LSTM model"""
        from sklearn.preprocessing import MinMaxScaler
        
        # Scale data
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(time_series.reshape(-1, 1)).flatten()
        
        if len(scaled_data) < self.lookback + 10:
            logger.warning("Insufficient data for LSTM training")
            self._fallback_value = time_series[-1] if len(time_series) > 0 else 0
            return self
        
        # Create sequences
        X, y = self._create_sequences(scaled_data)
        X = X.reshape(-1, self.lookback, 1)
        
        # Build and train model
        self._build_model((self.lookback, 1))
        
        try:
            from tensorflow.keras.callbacks import EarlyStopping
            
            early_stop = EarlyStopping(
                monitor='loss',
                patience=10,
                restore_best_weights=True
            )
            
            self.model.fit(
                X, y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=self.verbose,
                callbacks=[early_stop]
            )
            
            logger.info("LSTM training complete")
            
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            self._fallback_value = time_series[-1]
        
        return self
    
    def predict(self, last_sequence: np.ndarray, steps: int = 6) -> np.ndarray:
        """Predict future values"""
        if self.model is None:
            return np.full(steps, self._fallback_value)
        
        try:
            # Scale input
            scaled_seq = self.scaler.transform(last_sequence.reshape(-1, 1)).flatten()
            current_seq = scaled_seq[-self.lookback:]
            
            predictions = []
            for _ in range(steps):
                # Reshape for prediction
                X = current_seq.reshape(1, self.lookback, 1)
                pred = self.model.predict(X, verbose=0)[0, 0]
                predictions.append(pred)
                
                # Update sequence
                current_seq = np.append(current_seq[1:], pred)
            
            # Inverse transform
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = self.scaler.inverse_transform(predictions).flatten()
            
            # Clip to valid range
            predictions = np.clip(predictions, 0, 1)
            
            return predictions
            
        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            return np.full(steps, self._fallback_value)
    
    def fit_predict(self, time_series: np.ndarray, steps: int = 6) -> np.ndarray:
        """Convenience method: fit and predict"""
        self.fit(time_series)
        return self.predict(time_series, steps)


class SimpleLSTMPredictor:
    """
    Lightweight LSTM alternative using numpy
    For environments without TensorFlow
    """
    
    def __init__(self, lookback: int = 12):
        self.lookback = lookback
        self._weights = None
    
    def fit(self, time_series: np.ndarray) -> 'SimpleLSTMPredictor':
        """Fit using exponential smoothing as approximation"""
        self._last_values = time_series[-self.lookback:]
        self._mean = np.mean(time_series)
        self._std = np.std(time_series)
        
        # Simple exponential weights
        alpha = 0.3
        self._weights = np.array([alpha * (1-alpha)**i for i in range(self.lookback)])
        self._weights = self._weights / self._weights.sum()
        
        return self
    
    def predict(self, steps: int = 6) -> np.ndarray:
        """Predict using weighted moving average"""
        predictions = []
        current = self._last_values.copy()
        
        for _ in range(steps):
            pred = np.sum(current * self._weights)
            # Add small noise for realism
            pred += np.random.normal(0, self._std * 0.1)
            pred = np.clip(pred, 0, 1)
            predictions.append(pred)
            
            # Shift window
            current = np.append(current[1:], pred)
        
        return np.array(predictions)