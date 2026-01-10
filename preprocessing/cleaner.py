# TODO: implement

"""
Data Cleaning Module
Handles missing values, outliers, and data quality issues
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataCleaner:
    """Clean and preprocess raw cluster trace data"""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.cleaning_stats = {}
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full cleaning pipeline"""
        logger.info(f"Starting cleaning. Initial rows: {len(df)}")
        
        df = self._handle_missing_values(df)
        df = self._remove_invalid_rows(df)
        df = self._handle_outliers(df)
        df = self._normalize_timestamps(df)
        df = self._sort_data(df)
        
        logger.info(f"Cleaning complete. Final rows: {len(df)}")
        logger.info(f"Cleaning stats: {self.cleaning_stats}")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values appropriately per column type"""
        initial_nulls = df.isnull().sum().sum()
        
        # Critical columns - drop rows if missing
        critical_cols = ['machine_id', 'time', 'average_usage']
        for col in critical_cols:
            if col in df.columns:
                df = df.dropna(subset=[col])
        
        # Numeric columns - forward fill then backward fill
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df.groupby('machine_id')[col].transform(
                lambda x: x.fillna(method='ffill').fillna(method='bfill')
            )
        
        # Remaining nulls - fill with 0
        df = df.fillna(0)
        
        final_nulls = df.isnull().sum().sum()
        self.cleaning_stats['missing_values_handled'] = initial_nulls - final_nulls
        
        return df
    
    def _remove_invalid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with invalid values"""
        initial_rows = len(df)
        
        # Remove negative usage values
        if 'average_usage' in df.columns:
            df = df[df['average_usage'] >= 0]
        
        if 'maximum_usage' in df.columns:
            df = df[df['maximum_usage'] >= 0]
        
        # Remove impossible values (usage > 100%)
        if 'average_usage' in df.columns:
            df = df[df['average_usage'] <= 1.5]  # Allow some headroom
        
        # Remove zero-duration entries
        if 'start_time' in df.columns and 'end_time' in df.columns:
            df = df[df['end_time'] > df['start_time']]
        
        removed = initial_rows - len(df)
        self.cleaning_stats['invalid_rows_removed'] = removed
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame, 
                         method: str = 'iqr', 
                         columns: Optional[list] = None) -> pd.DataFrame:
        """Handle outliers using IQR or Z-score method"""
        if columns is None:
            columns = ['average_usage', 'maximum_usage', 'resource_request']
        
        columns = [c for c in columns if c in df.columns]
        outliers_capped = 0
        
        for col in columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
            else:  # z-score
                mean = df[col].mean()
                std = df[col].std()
                lower = mean - 3 * std
                upper = mean + 3 * std
            
            # Cap outliers instead of removing
            outliers_mask = (df[col] < lower) | (df[col] > upper)
            outliers_capped += outliers_mask.sum()
            df[col] = df[col].clip(lower=max(0, lower), upper=upper)
        
        self.cleaning_stats['outliers_capped'] = outliers_capped
        
        return df
    
    def _normalize_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure timestamps are in correct format"""
        time_cols = ['time', 'start_time', 'end_time']
        
        for col in time_cols:
            if col in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
    
    def _sort_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort by machine and time for time series operations"""
        sort_cols = []
        if 'machine_id' in df.columns:
            sort_cols.append('machine_id')
        if 'time' in df.columns:
            sort_cols.append('time')
        
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)
        
        return df
    
    def get_cleaning_report(self) -> dict:
        """Return cleaning statistics for paper documentation"""
        return self.cleaning_stats.copy()