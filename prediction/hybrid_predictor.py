"""
Hybrid Prediction System
Selects prediction strategy based on workload cluster
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple, Union
import logging
import joblib
from pathlib import Path

from .arima_predictor import ARIMAPredictor
from .lstm_predictor import LSTMPredictor, SimpleLSTMPredictor

logger = logging.getLogger(__name__)


class HybridPredictor:
    """
    Cluster-aware hybrid prediction system
    Core methodology of the paper
    """
    
    STRATEGY_MAP = {
        'stable_low_load': 'arima',
        'periodic': 'seasonal_arima',
        'bursty': 'lstm',
        'volatile': 'conservative'
    }
    
    def __init__(self, 
                 arima_order: Tuple = (5, 1, 0),
                 seasonal_order: Tuple = (1, 1, 1, 12),
                 lstm_lookback: int = 12,
                 lstm_epochs: int = 50,
                 use_tensorflow: bool = True):
        
        self.arima_order = arima_order
        self.seasonal_order = seasonal_order
        self.lstm_lookback = lstm_lookback
        self.lstm_epochs = lstm_epochs
        self.use_tensorflow = use_tensorflow
        
        # Model cache per machine
        self.models = {}
        self.prediction_errors = {}
    
    def get_strategy(self, workload_type: str) -> str:
        """Map workload type to prediction strategy"""
        return self.STRATEGY_MAP.get(workload_type, 'arima')
    
    def fit_machine(self, 
                    machine_id: str,
                    time_series: np.ndarray,
                    workload_type: str) -> None:
        """Fit appropriate model for a machine based on its workload type"""
        strategy = self.get_strategy(workload_type)
        
        logger.debug(f"Fitting {strategy} for machine {machine_id}")
        
        if strategy == 'arima':
            model = ARIMAPredictor(order=self.arima_order)
            model.fit(time_series)
            
        elif strategy == 'seasonal_arima':
            model = ARIMAPredictor(
                order=self.arima_order,
                seasonal_order=self.seasonal_order
            )
            model.fit(time_series)
            
        elif strategy == 'lstm':
            if self.use_tensorflow:
                try:
                    model = LSTMPredictor(
                        lookback=self.lstm_lookback,
                        epochs=self.lstm_epochs
                    )
                    model.fit(time_series)
                except ImportError:
                    logger.warning("TensorFlow unavailable, using simple LSTM")
                    model = SimpleLSTMPredictor(lookback=self.lstm_lookback)
                    model.fit(time_series)
            else:
                model = SimpleLSTMPredictor(lookback=self.lstm_lookback)
                model.fit(time_series)
                
        else:  # conservative
            model = ConservativePredictor()
            model.fit(time_series)
        
        self.models[machine_id] = {
            'model': model,
            'strategy': strategy,
            'workload_type': workload_type
        }
    
    def predict_machine(self, 
                        machine_id: str,
                        last_sequence: Optional[np.ndarray] = None,
                        steps: int = 6) -> np.ndarray:
        """Generate predictions for a machine"""
        if machine_id not in self.models:
            logger.warning(f"Machine {machine_id} not fitted, using default prediction")
            return np.full(steps, 0.5)
        
        model_info = self.models[machine_id]
        model = model_info['model']
        strategy = model_info['strategy']
        
        if strategy in ['arima', 'seasonal_arima']:
            predictions = model.predict(steps)
        elif strategy == 'lstm':
            if last_sequence is not None:
                predictions = model.predict(last_sequence, steps)
            else:
                predictions = model.predict(steps)
        else:
            predictions = model.predict(steps)
        
        return predictions
    
    def fit_all_machines(self,
                          machine_data: Dict[str, np.ndarray],
                          machine_workload_types: Dict[str, str]) -> None:
        """Fit models for all machines"""
        logger.info(f"Fitting models for {len(machine_data)} machines")
        
        for machine_id, time_series in machine_data.items():
            workload_type = machine_workload_types.get(machine_id, 'periodic')
            self.fit_machine(machine_id, time_series, workload_type)
        
        logger.info("All machine models fitted")
    
    def predict_all_machines(self,
                              machine_sequences: Dict[str, np.ndarray],
                              steps: int = 6) -> Dict[str, np.ndarray]:
        """Generate predictions for all machines"""
        predictions = {}
        
        for machine_id, sequence in machine_sequences.items():
            predictions[machine_id] = self.predict_machine(
                machine_id,
                sequence,
                steps
            )
        
        return predictions
    
    def evaluate_predictions(self,
                              predictions: Dict[str, np.ndarray],
                              actuals: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate prediction errors"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        all_preds = []
        all_actuals = []
        
        for machine_id in predictions:
            if machine_id in actuals:
                pred = predictions[machine_id]
                actual = actuals[machine_id]
                
                # Align lengths
                min_len = min(len(pred), len(actual))
                all_preds.extend(pred[:min_len])
                all_actuals.extend(actual[:min_len])
        
        if not all_preds:
            return {'rmse': float('inf'), 'mae': float('inf')}
        
        all_preds = np.array(all_preds)
        all_actuals = np.array(all_actuals)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(all_actuals, all_preds)),
            'mae': mean_absolute_error(all_actuals, all_preds),
            'mape': np.mean(np.abs((all_actuals - all_preds) / (all_actuals + 1e-10))) * 100
        }
        
        return metrics
    
    def save_models(self, path: Union[str, Path]) -> None:
        """Save trained models to disk"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model metadata and objects
        save_data = {
            'models': self.models,
            'prediction_errors': self.prediction_errors,
            'config': {
                'arima_order': self.arima_order,
                'seasonal_order': self.seasonal_order,
                'lstm_lookback': self.lstm_lookback
            }
        }
        
        try:
            joblib.dump(save_data, path / 'hybrid_predictor_models.joblib')
            logger.info(f"Models saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save models: {e}")

    def load_models(self, path: Union[str, Path]) -> None:
        """Load trained models from disk"""
        path = Path(path)
        model_file = path / 'hybrid_predictor_models.joblib'
        
        if not model_file.exists():
            logger.warning(f"No model file found at {model_file}")
            return
            
        try:
            data = joblib.load(model_file)
            self.models = data['models']
            self.prediction_errors = data.get('prediction_errors', {})
            logger.info(f"Models loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")


class ConservativePredictor:
    """
    Conservative prediction for volatile workloads
    Uses upper percentile to avoid underprovisioning
    """
    
    def __init__(self, percentile: float = 75):
        self.percentile = percentile
        self._prediction_value = 0.5
    
    def fit(self, time_series: np.ndarray) -> 'ConservativePredictor':
        """Compute conservative prediction value"""
        self._prediction_value = np.percentile(time_series, self.percentile)
        self._mean = np.mean(time_series)
        return self
    
    def predict(self, steps: int = 6) -> np.ndarray:
        """Return conservative flat prediction"""
        return np.full(steps, self._prediction_value)