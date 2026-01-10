"""
Baseline Methods for Comparison
Static provisioning and threshold-based auto-scaling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    """Result from baseline method"""
    name: str
    capacity_sequence: np.ndarray
    total_cost: float
    utilization: float
    sla_violations: int


class BaselineModels:
    """
    Baseline scaling methods for comparison
    
    1. Static Provisioning: Fixed capacity at peak
    2. Threshold-based Auto-scaling: Reactive scaling
    3. Moving Average Prediction: Simple predictive baseline
    """
    
    def __init__(self, 
                 time_interval_minutes: int = 5,
                 price_per_unit_hour: float = 0.05):
        self.time_interval_hours = time_interval_minutes / 60
        self.price_per_unit_hour = price_per_unit_hour
    
    def static_provisioning(self, 
                             usage_sequence: np.ndarray,
                             buffer_factor: float = 1.2) -> BaselineResult:
        """
        Static provisioning at peak usage + buffer
        Most conservative and expensive approach
        """
        peak = np.max(usage_sequence)
        capacity = peak * buffer_factor
        capacity_sequence = np.full(len(usage_sequence), capacity)
        
        # Calculate metrics
        total_cost = self._calculate_cost(capacity_sequence)
        utilization = np.mean(usage_sequence) / capacity
        sla_violations = self._count_violations(capacity_sequence, usage_sequence)
        
        return BaselineResult(
            name='static',
            capacity_sequence=capacity_sequence,
            total_cost=total_cost,
            utilization=utilization,
            sla_violations=sla_violations
        )
    
    def threshold_autoscaling(self,
                               usage_sequence: np.ndarray,
                               upper_threshold: float = 0.8,
                               lower_threshold: float = 0.3,
                               cooldown_periods: int = 3,
                               scale_factor: float = 1.25) -> BaselineResult:
        """
        Reactive threshold-based auto-scaling
        Standard cloud provider approach (AWS, GCP, Azure)
        """
        n = len(usage_sequence)
        capacity_sequence = np.zeros(n)
        
        # Initialize at median usage with buffer
        current_capacity = np.median(usage_sequence) * 1.5
        cooldown_counter = 0
        
        for i in range(n):
            usage = usage_sequence[i]
            utilization = usage / (current_capacity + 1e-10)
            
            if cooldown_counter > 0:
                cooldown_counter -= 1
            else:
                if utilization > upper_threshold:
                    # Scale up
                    current_capacity = min(current_capacity * scale_factor, 1.0)
                    cooldown_counter = cooldown_periods
                elif utilization < lower_threshold:
                    # Scale down
                    new_capacity = current_capacity / scale_factor
                    # Don't scale below current usage
                    current_capacity = max(new_capacity, usage * 1.2, 0.1)
                    cooldown_counter = cooldown_periods
            
            capacity_sequence[i] = current_capacity
        
        total_cost = self._calculate_cost(capacity_sequence)
        utilization = np.mean(usage_sequence / (capacity_sequence + 1e-10))
        sla_violations = self._count_violations(capacity_sequence, usage_sequence)
        
        return BaselineResult(
            name='threshold',
            capacity_sequence=capacity_sequence,
            total_cost=total_cost,
            utilization=utilization,
            sla_violations=sla_violations
        )
    
    def moving_average_prediction(self,
                                    usage_sequence: np.ndarray,
                                    window_size: int = 12,
                                    buffer_factor: float = 1.2) -> BaselineResult:
        """
        Simple moving average prediction baseline
        Uses past window to predict and provision
        """
        n = len(usage_sequence)
        capacity_sequence = np.zeros(n)
        
        for i in range(n):
            if i < window_size:
                # Not enough history, use current value
                predicted = usage_sequence[:i+1].mean() if i > 0 else usage_sequence[0]
            else:
                predicted = usage_sequence[i-window_size:i].mean()
            
            capacity_sequence[i] = predicted * buffer_factor
        
        # Ensure minimum capacity
        capacity_sequence = np.maximum(capacity_sequence, 0.1)
        
        total_cost = self._calculate_cost(capacity_sequence)
        utilization = np.mean(usage_sequence / (capacity_sequence + 1e-10))
        sla_violations = self._count_violations(capacity_sequence, usage_sequence)
        
        return BaselineResult(
            name='moving_average',
            capacity_sequence=capacity_sequence,
            total_cost=total_cost,
            utilization=utilization,
            sla_violations=sla_violations
        )
    
    def run_all_baselines(self, usage_sequence: np.ndarray) -> Dict[str, BaselineResult]:
        """Run all baseline methods"""
        results = {}
        
        results['static'] = self.static_provisioning(usage_sequence)
        results['threshold'] = self.threshold_autoscaling(usage_sequence)
        results['moving_average'] = self.moving_average_prediction(usage_sequence)
        
        return results
    
    def _calculate_cost(self, capacity_sequence: np.ndarray) -> float:
        """Calculate total cost"""
        return np.sum(capacity_sequence) * self.time_interval_hours * self.price_per_unit_hour
    
    def _count_violations(self, 
                           capacity_sequence: np.ndarray,
                           usage_sequence: np.ndarray,
                           threshold: float = 0.95) -> int:
        """Count SLA violations"""
        return int(np.sum(usage_sequence > threshold * capacity_sequence))
    
    def compare_to_proposed(self,
                             usage_sequence: np.ndarray,
                             proposed_capacity: np.ndarray) -> pd.DataFrame:
        """Compare all baselines to proposed method"""
        
        # Run baselines
        baselines = self.run_all_baselines(usage_sequence)
        
        # Calculate proposed metrics
        proposed_cost = self._calculate_cost(proposed_capacity)
        proposed_util = np.mean(usage_sequence / (proposed_capacity + 1e-10))
        proposed_violations = self._count_violations(proposed_capacity, usage_sequence)
        
        # Build comparison table
        rows = []
        
        for name, result in baselines.items():
            rows.append({
                'method': name,
                'total_cost': result.total_cost,
                'utilization': result.utilization * 100,
                'sla_violations': result.sla_violations,
                'sla_violation_rate': result.sla_violations / len(usage_sequence) * 100
            })
        
        rows.append({
            'method': 'proposed',
            'total_cost': proposed_cost,
            'utilization': proposed_util * 100,
            'sla_violations': proposed_violations,
            'sla_violation_rate': proposed_violations / len(usage_sequence) * 100
        })
        
        df = pd.DataFrame(rows)
        
        # Calculate cost reduction vs static
        static_cost = df[df['method'] == 'static']['total_cost'].values[0]
        df['cost_reduction_pct'] = (static_cost - df['total_cost']) / static_cost * 100
        
        return df