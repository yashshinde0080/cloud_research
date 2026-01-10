"""
Evaluation Metrics Module
All metrics used in the paper
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for the paper
    
    Categories:
    1. Prediction Metrics (RMSE, MAE, MAPE)
    2. Cost Metrics (Total, Wasted, Reduction)
    3. Utilization Metrics
    4. SLA Metrics
    """
    
    # ==================== Prediction Metrics ====================
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error"""
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error"""
        # Avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            return float('inf')
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Symmetric Mean Absolute Percentage Error"""
        denominator = np.abs(y_true) + np.abs(y_pred)
        mask = denominator != 0
        if not np.any(mask):
            return 0.0
        return np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
    
    # ==================== Cost Metrics ====================
    
    @staticmethod
    def total_cost(capacity: np.ndarray, 
                   price_per_unit: float = 0.05,
                   time_hours: float = 1/12) -> float:
        """Calculate total resource cost"""
        return np.sum(capacity) * price_per_unit * time_hours
    
    @staticmethod
    def wasted_cost(capacity: np.ndarray, 
                    usage: np.ndarray,
                    price_per_unit: float = 0.05,
                    time_hours: float = 1/12) -> float:
        """Calculate cost of wasted (unused) resources"""
        waste = np.maximum(0, capacity - usage)
        return np.sum(waste) * price_per_unit * time_hours
    
    @staticmethod
    def cost_reduction(baseline_cost: float, proposed_cost: float) -> float:
        """Calculate percentage cost reduction"""
        if baseline_cost == 0:
            return 0.0
        return (baseline_cost - proposed_cost) / baseline_cost * 100
    
    @staticmethod
    def cost_efficiency(usage: np.ndarray, capacity: np.ndarray) -> float:
        """Cost efficiency ratio (usage per unit cost)"""
        return np.sum(usage) / (np.sum(capacity) + 1e-10)
    
    # ==================== Utilization Metrics ====================
    
    @staticmethod
    def average_utilization(usage: np.ndarray, capacity: np.ndarray) -> float:
        """Average resource utilization"""
        return np.mean(usage / (capacity + 1e-10))
    
    @staticmethod
    def utilization_variance(usage: np.ndarray, capacity: np.ndarray) -> float:
        """Variance in utilization (stability metric)"""
        utilization = usage / (capacity + 1e-10)
        return np.var(utilization)
    
    @staticmethod
    def over_provisioning_ratio(capacity: np.ndarray, usage: np.ndarray) -> float:
        """Average over-provisioning ratio"""
        return np.mean(capacity / (usage + 1e-10))
    
    # ==================== SLA Metrics ====================
    
    @staticmethod
    def sla_violation_count(usage: np.ndarray, 
                             capacity: np.ndarray,
                             threshold: float = 0.95) -> int:
        """Count number of SLA violations"""
        return int(np.sum(usage > threshold * capacity))
    
    @staticmethod
    def sla_violation_rate(usage: np.ndarray,
                            capacity: np.ndarray,
                            threshold: float = 0.95) -> float:
        """Percentage of time with SLA violations"""
        violations = np.sum(usage > threshold * capacity)
        return violations / len(usage) * 100
    
    @staticmethod
    def sla_compliance_rate(usage: np.ndarray,
                             capacity: np.ndarray,
                             threshold: float = 0.95) -> float:
        """Percentage of time meeting SLA"""
        return 100 - EvaluationMetrics.sla_violation_rate(usage, capacity, threshold)
    
    # ==================== Clustering Metrics ====================
    
    @staticmethod
    def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
        """Silhouette score for clustering quality"""
        from sklearn.metrics import silhouette_score
        if len(np.unique(labels)) < 2:
            return 0.0
        return silhouette_score(X, labels)
    
    @staticmethod
    def davies_bouldin_score(X: np.ndarray, labels: np.ndarray) -> float:
        """Davies-Bouldin score (lower is better)"""
        from sklearn.metrics import davies_bouldin_score
        if len(np.unique(labels)) < 2:
            return float('inf')
        return davies_bouldin_score(X, labels)
    
    # ==================== Comprehensive Evaluation ====================
    
    @classmethod
    def evaluate_all(cls,
                      y_true: np.ndarray,
                      y_pred: np.ndarray,
                      capacity: np.ndarray,
                      usage: np.ndarray,
                      baseline_cost: Optional[float] = None) -> Dict[str, float]:
        """Compute all metrics"""
        
        metrics = {
            # Prediction
            'rmse': cls.rmse(y_true, y_pred),
            'mae': cls.mae(y_true, y_pred),
            'mape': cls.mape(y_true, y_pred),
            
            # Cost
            'total_cost': cls.total_cost(capacity),
            'wasted_cost': cls.wasted_cost(capacity, usage),
            
            # Utilization
            'avg_utilization': cls.average_utilization(usage, capacity),
            'over_provision_ratio': cls.over_provisioning_ratio(capacity, usage),
            
            # SLA
            'sla_violation_count': cls.sla_violation_count(usage, capacity),
            'sla_violation_rate': cls.sla_violation_rate(usage, capacity),
        }
        
        if baseline_cost is not None:
            metrics['cost_reduction_pct'] = cls.cost_reduction(baseline_cost, metrics['total_cost'])
        
        return metrics
    
    @classmethod
    def format_results_table(cls, results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Format multiple method results as comparison table"""
        rows = []
        for method, metrics in results.items():
            row = {'Method': method}
            row.update(metrics)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df