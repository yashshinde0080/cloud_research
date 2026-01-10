# TODO: implement

"""
Time-Window Aggregation Module
Aggregates raw metrics into analysis windows
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class TimeAggregator:
    """Aggregate metrics over time windows for each machine"""
    
    def __init__(self, window_minutes: int = 5):
        self.window_minutes = window_minutes
        self.window_str = f'{window_minutes}min'
    
    def aggregate(self, df: pd.DataFrame, 
                  group_col: str = 'machine_id',
                  time_col: str = 'time') -> pd.DataFrame:
        """
        Aggregate metrics over time windows
        Returns one row per (machine, time_window)
        """
        logger.info(f"Aggregating with {self.window_minutes}-minute windows")
        
        # Ensure time is datetime
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col])
        
        # Create time bucket
        df['time_bucket'] = df[time_col].dt.floor(self.window_str)
        
        # Define aggregation functions
        agg_funcs = {
            'average_usage': ['mean', 'std', 'min', 'max'],
            'maximum_usage': ['mean', 'max'],
            'resource_request': ['mean', 'max'],
            'assigned_memory': ['mean'],
            'page_cache_memory': ['mean'],
            'cycles_per_instruction': ['mean'],
            'memory_accesses_per_instruction': ['mean']
        }
        
        # Filter to columns that exist
        agg_funcs = {k: v for k, v in agg_funcs.items() if k in df.columns}
        
        # Perform aggregation
        aggregated = df.groupby([group_col, 'time_bucket']).agg(agg_funcs)
        
        # Flatten column names
        aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns]
        aggregated = aggregated.reset_index()
        
        # Rename time column
        aggregated = aggregated.rename(columns={'time_bucket': 'time'})
        
        logger.info(f"Aggregation complete: {len(aggregated)} rows")
        
        return aggregated
    
    def create_machine_profiles(self, df: pd.DataFrame,
                                 group_col: str = 'machine_id') -> pd.DataFrame:
        """
        Create per-machine summary profiles for clustering
        One row per machine with aggregate statistics
        """
        logger.info("Creating machine profiles for clustering")
        
        profile_aggs = {}
        
        # CPU metrics
        if 'average_usage_mean' in df.columns:
            profile_aggs['cpu_mean'] = ('average_usage_mean', 'mean')
            profile_aggs['cpu_std'] = ('average_usage_mean', 'std')
            profile_aggs['cpu_max'] = ('average_usage_mean', 'max')
            profile_aggs['cpu_min'] = ('average_usage_mean', 'min')
        
        if 'maximum_usage_max' in df.columns:
            profile_aggs['peak_usage'] = ('maximum_usage_max', 'max')
        
        # Resource request
        if 'resource_request_mean' in df.columns:
            profile_aggs['request_mean'] = ('resource_request_mean', 'mean')
        
        # Memory
        if 'assigned_memory_mean' in df.columns:
            profile_aggs['memory_mean'] = ('assigned_memory_mean', 'mean')
        
        # CPI
        if 'cycles_per_instruction_mean' in df.columns:
            profile_aggs['cpi_mean'] = ('cycles_per_instruction_mean', 'mean')
        
        # Perform aggregation
        profiles = df.groupby(group_col).agg(**profile_aggs)
        profiles = profiles.reset_index()
        
        # Calculate derived features
        if 'cpu_mean' in profiles.columns and 'cpu_std' in profiles.columns:
            profiles['cpu_cv'] = profiles['cpu_std'] / (profiles['cpu_mean'] + 1e-10)
        
        if 'request_mean' in profiles.columns and 'cpu_mean' in profiles.columns:
            profiles['over_provision_ratio'] = profiles['request_mean'] / (profiles['cpu_mean'] + 1e-10)
        
        # Count observations per machine
        obs_counts = df.groupby(group_col).size().reset_index(name='n_observations')
        profiles = profiles.merge(obs_counts, on=group_col)
        
        logger.info(f"Created profiles for {len(profiles)} machines")
        
        return profiles