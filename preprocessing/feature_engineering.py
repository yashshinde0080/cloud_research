# TODO: implement

"""
Feature Engineering Module
Extracts features for clustering and prediction
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Extract features for ML pipeline"""
    
    def __init__(self):
        self.feature_names = []
        self.scaler = None
    
    def extract_temporal_features(self, df: pd.DataFrame, 
                                   time_col: str = 'time') -> pd.DataFrame:
        """Extract time-based features"""
        df = df.copy()
        
        if time_col not in df.columns:
            logger.warning(f"Time column {time_col} not found")
            return df
        
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col])
        
        # Hour of day (cyclical encoding)
        hour = df[time_col].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        # Day of week (cyclical encoding)
        dow = df[time_col].dt.dayofweek
        df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        df['dow_cos'] = np.cos(2 * np.pi * dow / 7)
        
        # Is weekend
        df['is_weekend'] = (dow >= 5).astype(int)
        
        # Is business hours (9-17)
        df['is_business_hours'] = ((hour >= 9) & (hour <= 17)).astype(int)
        
        logger.info("Extracted temporal features")
        return df
    
    def extract_workload_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract workload behavior features per machine"""
        df = df.copy()
        
        # Burstiness metric
        if 'average_usage_mean' in df.columns and 'maximum_usage_max' in df.columns:
            df['burstiness'] = (df['maximum_usage_max'] - df['average_usage_mean']) / (df['average_usage_mean'] + 1e-10)
        
        # Stability metric (inverse of CV)
        if 'average_usage_std' in df.columns and 'average_usage_mean' in df.columns:
            cv = df['average_usage_std'] / (df['average_usage_mean'] + 1e-10)
            df['stability'] = 1 / (1 + cv)
        
        return df
    
    def create_clustering_features(self, profiles: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Create feature matrix for clustering
        Returns normalized features and feature names
        """
        feature_columns = [
            'cpu_mean', 'cpu_std', 'cpu_cv', 'peak_usage',
            'over_provision_ratio', 'memory_mean'
        ]
        
        # Use only columns that exist
        available_features = [c for c in feature_columns if c in profiles.columns]
        
        if not available_features:
            raise ValueError("No valid features found for clustering")
        
        logger.info(f"Using features for clustering: {available_features}")
        
        X = profiles[available_features].values
        
        # Handle any remaining NaN
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Standardize features
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.feature_names = available_features
        
        return X_scaled, available_features
    
    def create_prediction_sequences(self, df: pd.DataFrame,
                                      target_col: str = 'average_usage_mean',
                                      lookback: int = 12) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM prediction
        lookback: number of time steps to use as input
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found")
        
        # Sort by time
        df = df.sort_values('time')
        values = df[target_col].values
        
        X, y = [], []
        for i in range(len(values) - lookback):
            X.append(values[i:i+lookback])
            y.append(values[i+lookback])
        
        return np.array(X), np.array(y)
    
    def calculate_over_provisioning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate over-provisioning metrics
        Core contribution of the paper
        """
        df = df.copy()
        
        if 'resource_request_mean' in df.columns and 'average_usage_mean' in df.columns:
            df['over_provision_ratio'] = (
                df['resource_request_mean'] / (df['average_usage_mean'] + 1e-10)
            )
            
            # Wasted resources
            df['wasted_resources'] = np.maximum(
                0, df['resource_request_mean'] - df['average_usage_mean']
            )
            
            # Utilization efficiency
            df['utilization_efficiency'] = df['average_usage_mean'] / (df['resource_request_mean'] + 1e-10)
            df['utilization_efficiency'] = df['utilization_efficiency'].clip(0, 1)
        
        return df