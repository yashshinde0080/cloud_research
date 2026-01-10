"""
ARIMA-based Prediction for Stable Workloads
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
import warnings
import logging

# Suppress convergence warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ARIMAPredictor:
    """ARIMA prediction for stable and periodic workloads"""
    
    def __init__(self, order: Tuple[int, int, int] = (5, 1, 0),
                 seasonal_order: Optional[Tuple[int, int, int, int]] = None):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
    
    def fit(self, time_series: np.ndarray) -> 'ARIMAPredictor':
        """Fit ARIMA model to time series"""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            
            # Clean data
            ts = pd.Series(time_series).dropna()
            
            if len(ts) < 20:
                logger.warning("Insufficient data for ARIMA, using naive prediction")
                self.fitted_model = None
                self._last_value = ts.iloc[-1] if len(ts) > 0 else 0
                return self
            
            if self.seasonal_order:
                self.model = SARIMAX(
                    ts,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            else:
                self.model = ARIMA(ts, order=self.order)
            
            self.fitted_model = self.model.fit()
            logger.debug(f"ARIMA fitted successfully. AIC: {self.fitted_model.aic:.2f}")
            
        except Exception as e:
            logger.warning(f"ARIMA fitting failed: {e}. Using fallback.")
            self.fitted_model = None
            self._last_value = time_series[-1] if len(time_series) > 0 else 0
        
        return self
    
    def predict(self, steps: int = 6) -> np.ndarray:
        """Predict future values"""
        if self.fitted_model is None:
            # Naive prediction: repeat last value
            return np.full(steps, self._last_value)
        
        try:
            forecast = self.fitted_model.forecast(steps=steps)
            predictions = np.array(forecast)
            
            # Clip to valid range
            predictions = np.clip(predictions, 0, 1)
            
            return predictions
            
        except Exception as e:
            logger.warning(f"ARIMA prediction failed: {e}")
            return np.full(steps, self._last_value)
    
    def fit_predict(self, time_series: np.ndarray, steps: int = 6) -> np.ndarray:
        """Convenience method: fit and predict"""
        self.fit(time_series)
        return self.predict(steps)
    
    @staticmethod
    def find_best_order(time_series: np.ndarray, 
                        max_p: int = 5, max_d: int = 2, max_q: int = 5) -> Tuple[int, int, int]:
        """
        Find optimal ARIMA order using AIC
        Use sparingly - computationally expensive
        """
        from statsmodels.tsa.arima.model import ARIMA
        from itertools import product
        
        ts = pd.Series(time_series).dropna()
        best_aic = float('inf')
        best_order = (1, 1, 0)
        
        for p, d, q in product(range(max_p+1), range(max_d+1), range(max_q+1)):
            if p == 0 and q == 0:
                continue
            try:
                model = ARIMA(ts, order=(p, d, q))
                fitted = model.fit()
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = (p, d, q)
            except:
                continue
        
        logger.info(f"Best ARIMA order: {best_order} with AIC: {best_aic:.2f}")
        return best_order