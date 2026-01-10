"""
Cluster Analysis and Interpretation Module
Maps clusters to workload semantics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class ClusterAnalyzer:
    """Analyze and interpret cluster characteristics"""
    
    # Expected cluster semantics
    WORKLOAD_TYPES = {
        'stable_low_load': {
            'cpu_mean': (0, 0.4),
            'cpu_cv': (0, 0.3),
            'description': 'Low resource usage with minimal variance'
        },
        'periodic': {
            'cpu_mean': (0.3, 0.7),
            'cpu_cv': (0.2, 0.5),
            'description': 'Moderate usage with regular patterns'
        },
        'bursty': {
            'cpu_mean': (0.2, 0.6),
            'cpu_cv': (0.5, 1.0),
            'description': 'Variable usage with occasional spikes'
        },
        'volatile': {
            'cpu_mean': (0.4, 1.0),
            'cpu_cv': (0.6, 2.0),
            'description': 'High and unpredictable resource usage'
        }
    }
    
    def __init__(self):
        self.cluster_labels = {}
        self.cluster_stats = {}
    
    def analyze_clusters(self, X: np.ndarray, 
                          labels: np.ndarray,
                          feature_names: List[str]) -> Dict:
        """
        Comprehensive cluster analysis
        Returns statistics and semantic labels for each cluster
        """
        df = pd.DataFrame(X, columns=feature_names)
        df['cluster'] = labels
        
        analysis = {}
        
        for cluster_id in sorted(df['cluster'].unique()):
            if cluster_id == -1:  # Noise in DBSCAN
                continue
                
            cluster_data = df[df['cluster'] == cluster_id]
            
            stats = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df) * 100
            }
            
            # Calculate statistics for each feature
            for feature in feature_names:
                stats[f'{feature}_mean'] = cluster_data[feature].mean()
                stats[f'{feature}_std'] = cluster_data[feature].std()
            
            # Assign semantic label
            stats['workload_type'] = self._assign_workload_type(stats)
            
            analysis[cluster_id] = stats
            self.cluster_stats[cluster_id] = stats
        
        return analysis
    
    def _assign_workload_type(self, stats: Dict) -> str:
        """Map cluster statistics to workload type"""
        cpu_mean = stats.get('cpu_mean_mean', 0.5)
        cpu_cv = stats.get('cpu_cv_mean', 0.5)
        
        # Simple rule-based assignment
        if cpu_mean < 0.3 and cpu_cv < 0.3:
            return 'stable_low_load'
        elif cpu_cv > 0.6:
            if cpu_mean > 0.5:
                return 'volatile'
            else:
                return 'bursty'
        else:
            return 'periodic'
    
    def get_prediction_strategy(self, workload_type: str) -> str:
        """Return recommended prediction model for workload type"""
        strategies = {
            'stable_low_load': 'ARIMA',
            'periodic': 'Seasonal_ARIMA',
            'bursty': 'LSTM',
            'volatile': 'Conservative_Baseline'
        }
        return strategies.get(workload_type, 'ARIMA')
    
    def create_analysis_report(self, analysis: Dict) -> pd.DataFrame:
        """Create formatted analysis report for paper"""
        rows = []
        
        for cluster_id, stats in analysis.items():
            row = {
                'Cluster': cluster_id,
                'Size': stats['size'],
                'Percentage': f"{stats['percentage']:.1f}%",
                'Workload Type': stats['workload_type'],
                'Prediction Strategy': self.get_prediction_strategy(stats['workload_type'])
            }
            
            # Add key metrics
            if 'cpu_mean_mean' in stats:
                row['Avg CPU'] = f"{stats['cpu_mean_mean']:.3f}"
            if 'cpu_cv_mean' in stats:
                row['CPU Variability'] = f"{stats['cpu_cv_mean']:.3f}"
            if 'over_provision_ratio_mean' in stats:
                row['Over-Provision Ratio'] = f"{stats['over_provision_ratio_mean']:.2f}"
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def assign_machines_to_strategies(self, 
                                       machine_ids: np.ndarray,
                                       cluster_labels: np.ndarray,
                                       cluster_analysis: Dict) -> pd.DataFrame:
        """Assign each machine to its prediction strategy"""
        assignments = []
        
        for machine_id, cluster in zip(machine_ids, cluster_labels):
            if cluster == -1:  # Noise
                workload_type = 'volatile'
            else:
                workload_type = cluster_analysis.get(cluster, {}).get('workload_type', 'periodic')
            
            assignments.append({
                'machine_id': machine_id,
                'cluster': cluster,
                'workload_type': workload_type,
                'prediction_strategy': self.get_prediction_strategy(workload_type)
            })
        
        return pd.DataFrame(assignments)