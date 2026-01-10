"""
Cloud Cost Model
Calculates and compares costs across strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CostBreakdown:
    """Detailed cost breakdown"""
    total_cost: float
    compute_cost: float
    wasted_cost: float
    utilization: float
    sla_violations: int
    
    def to_dict(self) -> Dict:
        return {
            'total_cost': self.total_cost,
            'compute_cost': self.compute_cost,
            'wasted_cost': self.wasted_cost,
            'utilization': self.utilization,
            'sla_violations': self.sla_violations
        }


class CostModel:
    """
    Cloud cost calculation and comparison
    Formula: Cost = Σ (VM_active_hours × price_per_hour)
    """
    
    def __init__(self,
                 price_per_cpu_hour: float = 0.05,
                 price_per_memory_gb_hour: float = 0.01,
                 time_interval_minutes: int = 5):
        
        self.price_per_cpu_hour = price_per_cpu_hour
        self.price_per_memory_gb_hour = price_per_memory_gb_hour
        self.time_interval_hours = time_interval_minutes / 60
    
    def calculate_cost(self,
                        capacity_sequence: np.ndarray,
                        memory_sequence: Optional[np.ndarray] = None) -> float:
        """Calculate total cost for a capacity sequence"""
        cpu_cost = np.sum(capacity_sequence) * self.time_interval_hours * self.price_per_cpu_hour
        
        if memory_sequence is not None:
            memory_cost = np.sum(memory_sequence) * self.time_interval_hours * self.price_per_memory_gb_hour
        else:
            memory_cost = 0
        
        return cpu_cost + memory_cost
    
    def calculate_wasted_cost(self,
                               capacity_sequence: np.ndarray,
                               usage_sequence: np.ndarray) -> float:
        """Calculate cost of unused (wasted) resources"""
        waste = np.maximum(0, capacity_sequence - usage_sequence)
        return np.sum(waste) * self.time_interval_hours * self.price_per_cpu_hour
    
    def calculate_utilization(self,
                               capacity_sequence: np.ndarray,
                               usage_sequence: np.ndarray) -> float:
        """Calculate average resource utilization"""
        return np.mean(usage_sequence / (capacity_sequence + 1e-10))
    
    def count_sla_violations(self,
                              capacity_sequence: np.ndarray,
                              usage_sequence: np.ndarray,
                              threshold: float = 0.95) -> int:
        """Count periods where usage exceeds capacity threshold"""
        violations = usage_sequence > (threshold * capacity_sequence)
        return int(np.sum(violations))
    
    def evaluate_strategy(self,
                           capacity_sequence: np.ndarray,
                           usage_sequence: np.ndarray,
                           memory_sequence: Optional[np.ndarray] = None) -> CostBreakdown:
        """Complete cost evaluation for a scaling strategy"""
        
        total_cost = self.calculate_cost(capacity_sequence, memory_sequence)
        compute_cost = self.calculate_cost(capacity_sequence)
        wasted_cost = self.calculate_wasted_cost(capacity_sequence, usage_sequence)
        utilization = self.calculate_utilization(capacity_sequence, usage_sequence)
        sla_violations = self.count_sla_violations(capacity_sequence, usage_sequence)
        
        return CostBreakdown(
            total_cost=total_cost,
            compute_cost=compute_cost,
            wasted_cost=wasted_cost,
            utilization=utilization,
            sla_violations=sla_violations
        )
    
    def compare_strategies(self,
                            usage_sequence: np.ndarray,
                            strategies: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Compare multiple scaling strategies
        
        Parameters:
            usage_sequence: Actual usage over time
            strategies: Dict mapping strategy name to capacity sequence
        """
        results = []
        
        for name, capacity_sequence in strategies.items():
            breakdown = self.evaluate_strategy(capacity_sequence, usage_sequence)
            
            result = breakdown.to_dict()
            result['strategy'] = name
            results.append(result)
        
        df = pd.DataFrame(results)
        
        # Calculate relative metrics
        baseline_cost = df[df['strategy'] == 'static']['total_cost'].values[0]
        df['cost_reduction_pct'] = (baseline_cost - df['total_cost']) / baseline_cost * 100
        
        return df
    
        def create_strategy_capacities(self,
                                    usage_sequence: np.ndarray,
                                    predictions: np.ndarray,
                                    scaling_policy) -> Dict[str, np.ndarray]:
         """Create capacity sequences for different strategies"""
        n = len(usage_sequence)
        
        strategies = {}
        
        # 1. Static provisioning (baseline)
        # Provision at peak + 20% buffer
        peak_usage = np.max(usage_sequence)
        strategies['static'] = np.full(n, peak_usage * 1.2)
        
        # 2. Threshold-based auto-scaling
        threshold_capacity = self._threshold_scaling(usage_sequence)
        strategies['threshold'] = threshold_capacity
        
        # 3. Proposed method (prediction-based)
        proposed_capacity = self._proposed_scaling(predictions, scaling_policy)
        strategies['proposed'] = proposed_capacity
        
        # 4. Oracle (perfect knowledge - for reference only)
        oracle_capacity = usage_sequence * 1.1  # 10% buffer over actual
        oracle_capacity = np.maximum(oracle_capacity, 0.1)
        strategies['oracle'] = oracle_capacity
        
        return strategies
    
    def _threshold_scaling(self, 
                            usage_sequence: np.ndarray,
                            upper_threshold: float = 0.8,
                            lower_threshold: float = 0.3,
                            scale_factor: float = 1.25) -> np.ndarray:
        """
        Simulate reactive threshold-based auto-scaling
        This is the baseline method in most cloud providers
        """
        n = len(usage_sequence)
        capacity = np.zeros(n)
        
        # Start with initial capacity
        current_capacity = usage_sequence[0] * 1.5
        
        for i in range(n):
            utilization = usage_sequence[i] / (current_capacity + 1e-10)
            
            if utilization > upper_threshold:
                # Scale up
                current_capacity = min(current_capacity * scale_factor, 1.0)
            elif utilization < lower_threshold:
                # Scale down
                current_capacity = max(current_capacity / scale_factor, 0.1)
            
            capacity[i] = current_capacity
        
        return capacity
    
    def _proposed_scaling(self,
                           predictions: np.ndarray,
                           scaling_policy) -> np.ndarray:
        """
        Apply proposed prediction-based scaling
        """
        n = len(predictions)
        capacity = np.zeros(n)
        
        current_capacity = predictions[0] * 1.3
        
        for i in range(n):
            decision = scaling_policy.recommend(
                machine_id='temp',
                current_capacity=current_capacity,
                predicted_usage=predictions[i]
            )
            current_capacity = decision.recommended_capacity
            capacity[i] = current_capacity
        
        return capacity
    
    def generate_cost_report(self, comparison_df: pd.DataFrame) -> str:
        """Generate formatted cost comparison report for paper"""
        report = []
        report.append("=" * 60)
        report.append("COST COMPARISON REPORT")
        report.append("=" * 60)
        
        for _, row in comparison_df.iterrows():
            report.append(f"\nStrategy: {row['strategy'].upper()}")
            report.append("-" * 40)
            report.append(f"  Total Cost:        ${row['total_cost']:.4f}")
            report.append(f"  Wasted Cost:       ${row['wasted_cost']:.4f}")
            report.append(f"  Utilization:       {row['utilization']*100:.1f}%")
            report.append(f"  SLA Violations:    {row['sla_violations']}")
            report.append(f"  Cost Reduction:    {row['cost_reduction_pct']:.1f}%")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)