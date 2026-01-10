# TODO: implement

"""
Configuration file for Cloud Cost Optimization Research
All hyperparameters centralized here for reproducibility
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict
from pathlib import Path


@dataclass
class DataConfig:
    """Data-related configuration"""
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"
    
    # Google Cluster Trace specific columns
    identity_columns: List[str] = field(default_factory=lambda: [
        'machine_id', 'instance_index', 'cluster', 'time', 'start_time', 'end_time'
    ])
    
    resource_columns: List[str] = field(default_factory=lambda: [
        'resource_request', 'average_usage', 'maximum_usage', 
        'random_sample_usage', 'assigned_memory', 'page_cache_memory'
    ])
    
    cpu_columns: List[str] = field(default_factory=lambda: [
        'cpu_usage_distribution', 'tail_cpu_usage_distribution',
        'cycles_per_instruction', 'memory_accesses_per_instruction', 'sample_rate'
    ])
    
    # Columns to ignore (document in paper)
    ignored_columns: List[str] = field(default_factory=lambda: [
        'user', 'scheduler', 'collection_name', 'priority', 'constraint', 'failed'
    ])
    
    # Sampling configuration
    time_window_minutes: int = 5
    analysis_days: int = 7
    max_machines: int = 1000  # For computational tractability


@dataclass
class ClusteringConfig:
    """Clustering hyperparameters"""
    # KMeans
    n_clusters: int = 4
    kmeans_random_state: int = 42
    kmeans_n_init: int = 10
    kmeans_max_iter: int = 300
    
    # DBSCAN
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 10
    
    # Cluster semantics (expected)
    cluster_labels: Dict[int, str] = field(default_factory=lambda: {
        0: 'stable_low_load',
        1: 'periodic_workload',
        2: 'bursty_workload',
        3: 'highly_volatile'
    })


@dataclass
class PredictionConfig:
    """Prediction model hyperparameters"""
    # ARIMA
    arima_order: tuple = (5, 1, 0)
    seasonal_order: tuple = (1, 1, 1, 12)  # For periodic workloads
    
    # LSTM
    lstm_units: int = 32
    lstm_layers: int = 1
    lstm_dropout: float = 0.2
    lstm_epochs: int = 50
    lstm_batch_size: int = 32
    lstm_lookback: int = 12  # 12 * 5min = 1 hour
    
    # Prediction window
    prediction_horizon_minutes: int = 30
    prediction_steps: int = 6  # 30min / 5min


@dataclass
class ScalingConfig:
    """Scaling policy thresholds"""
    scale_up_threshold: float = 0.75
    scale_down_threshold: float = 0.40
    
    # Cost model (normalized pricing)
    price_per_cpu_hour: float = 0.05
    price_per_gb_hour: float = 0.01
    
    # SLA constraints
    max_allowed_utilization: float = 0.95
    sla_violation_threshold: float = 0.90


@dataclass
class ExperimentConfig:
    """Experimental setup"""
    n_runs: int = 5
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42


@dataclass
class Config:
    """Master configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    # Output paths
    results_path: str = "results"
    figures_path: str = "figures"
    models_path: str = "models"
    
    def __post_init__(self):
        """Create output directories"""
        for path in [self.results_path, self.figures_path, self.models_path]:
            Path(path).mkdir(parents=True, exist_ok=True)


# Global config instance
config = Config()