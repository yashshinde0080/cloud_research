# TODO: implement

"""
Data Loader for Google Cluster Workload Traces 2019
Handles BigQuery export CSVs and validates schema
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Union
import logging
import glob
import ast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and validate Google Cluster Trace data"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.required_columns = [
            'machine_id', 'time', 'average_usage', 'maximum_usage',
            'resource_request', 'assigned_memory'
        ]
    
    def load_single_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load a single CSV file with proper dtypes"""
        # Removed float parsing for columns that might be JSON strings
        dtype_map = {
            'machine_id': str,
            'instance_index': 'Int64',
            'cluster': str,
            # 'resource_request': float,  # Parsed later
            # 'average_usage': float,     # Parsed later
            # 'maximum_usage': float,     # Parsed later
            # 'random_sample_usage': float,
            # 'assigned_memory': float,
            # 'page_cache_memory': float,
            # 'cycles_per_instruction': float,
            # 'memory_accesses_per_instruction': float,
            'sample_rate': float
        }
        
        try:
            df = pd.read_csv(
                file_path,
                dtype=dtype_map,
                # parse_dates=['time'], # 'time' seems to be int index in this dataset
                low_memory=False
            )
            
            # Parse JSON-like columns
            metric_cols = [
                'resource_request', 'average_usage', 'maximum_usage',
                'random_sample_usage', 'assigned_memory', 'page_cache_memory',
                'cycles_per_instruction', 'memory_accesses_per_instruction'
            ]
            
            for col in metric_cols:
                if col in df.columns:
                    # Check first non-null value to see if parsing is needed
                    first_valid = df[col].dropna().iloc[0] if not df[col].dropna().empty else 0
                    if isinstance(first_valid, str):
                        df[col] = df[col].apply(self._parse_cpu_metric)
                    else:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert time to datetime if it's an integer timestamp (assuming microseconds/nanoseconds or seconds?)
            # Google traces usually use microseconds.
            if 'time' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['time']):
                # Heuristic: if time is huge, it's micros.
                if df['time'].mean() > 1e12:
                    df['time'] = pd.to_datetime(df['time'], unit='us', errors='coerce')
                else:
                    df['time'] = pd.to_datetime(df['time'], unit='s', errors='coerce')
            
            # Drop rows with invalid time
            if df['time'].isnull().any():
                logger.warning(f"Dropping {df['time'].isnull().sum()} rows with invalid time")
                df = df.dropna(subset=['time'])
            
            logger.info(f"Loaded {len(df)} rows from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise

    def _parse_cpu_metric(self, val):
        """Extract CPU value from dictionary string"""
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            try:
                # Handle dictionary string {'cpus': 0.1, ...}
                d = ast.literal_eval(val)
                if isinstance(d, dict):
                    return float(d.get('cpus', 0.0))
                # Handle list string [0.1, 0.2]
                if isinstance(d, list):
                    return np.mean(d) if d else 0.0
            except:
                pass
        return 0.0
    
    def load_all_data(self, pattern: str = "*.csv") -> pd.DataFrame:
        """Load all CSV files matching pattern"""
        files = list(self.data_path.glob(pattern))
        
        if not files:
            raise FileNotFoundError(f"No files matching {pattern} in {self.data_path}")
        
        logger.info(f"Found {len(files)} files to load")
        
        dfs = []
        for f in files:
            try:
                df = self.load_single_file(f)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Skipping {f}: {e}")
                continue
        
        if not dfs:
            raise ValueError("No data loaded successfully")
        
        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Total rows loaded: {len(combined)}")
        
        return combined
    
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Check that required columns exist"""
        missing = set(self.required_columns) - set(df.columns)
        
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return False
        
        logger.info("Schema validation passed")
        return True
    
    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """Generate summary statistics for paper's dataset section"""
        summary = {
            'total_rows': len(df),
            'unique_machines': df['machine_id'].nunique(),
            'time_range': {
                'start': df['time'].min(),
                'end': df['time'].max()
            },
            'columns': list(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1e6,
            'missing_values': df.isnull().sum().to_dict()
        }
        
        return summary
    
    def sample_machines(self, df: pd.DataFrame, n_machines: int = 1000, 
                        random_state: int = 42) -> pd.DataFrame:
        """Sample subset of machines for tractable analysis"""
        unique_machines = df['machine_id'].unique()
        
        if len(unique_machines) <= n_machines:
            logger.info(f"Using all {len(unique_machines)} machines")
            return df
        
        np.random.seed(random_state)
        sampled_machines = np.random.choice(unique_machines, n_machines, replace=False)
        
        sampled_df = df[df['machine_id'].isin(sampled_machines)].copy()
        logger.info(f"Sampled {n_machines} machines, {len(sampled_df)} rows")
        
        return sampled_df


def create_synthetic_data(n_machines: int = 100, n_days: int = 7) -> pd.DataFrame:
    """
    Create synthetic data matching Google Cluster Trace schema
    Use this ONLY for testing code, not for paper results
    """
    np.random.seed(42)
    
    timestamps = pd.date_range(
        start='2019-05-01',
        periods=n_days * 24 * 12,  # 5-minute intervals
        freq='5min'
    )
    
    records = []
    
    for machine_id in range(n_machines):
        # Assign workload type
        workload_type = np.random.choice(['stable', 'periodic', 'bursty', 'volatile'], 
                                         p=[0.4, 0.3, 0.2, 0.1])
        
        for ts in timestamps:
            hour = ts.hour
            
            # Base usage patterns by type
            if workload_type == 'stable':
                base_usage = 0.3 + np.random.normal(0, 0.05)
            elif workload_type == 'periodic':
                base_usage = 0.3 + 0.3 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 0.05)
            elif workload_type == 'bursty':
                base_usage = 0.2 + (0.6 if np.random.random() < 0.1 else 0) + np.random.normal(0, 0.05)
            else:  # volatile
                base_usage = np.random.uniform(0.1, 0.9)
            
            base_usage = np.clip(base_usage, 0, 1)
            max_usage = min(base_usage + np.random.uniform(0.1, 0.3), 1.0)
            
            # Over-provisioning simulation
            resource_request = base_usage * np.random.uniform(1.5, 3.0)
            
            records.append({
                'machine_id': f'machine_{machine_id:04d}',
                'instance_index': 0,
                'cluster': f'cluster_{machine_id % 8}',
                'time': ts,
                'start_time': ts,
                'end_time': ts + pd.Timedelta(minutes=5),
                'resource_request': resource_request,
                'average_usage': base_usage,
                'maximum_usage': max_usage,
                'random_sample_usage': base_usage + np.random.normal(0, 0.02),
                'assigned_memory': np.random.uniform(0.5, 2.0),
                'page_cache_memory': np.random.uniform(0.1, 0.5),
                'cycles_per_instruction': np.random.uniform(0.5, 2.0),
                'memory_accesses_per_instruction': np.random.uniform(0.001, 0.01),
                'sample_rate': 1.0
            })
    
    df = pd.DataFrame(records)
    logger.info(f"Created synthetic data: {len(df)} rows, {n_machines} machines")
    
    return df