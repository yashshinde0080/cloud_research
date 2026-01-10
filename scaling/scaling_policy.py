"""
Scaling Recommendation Engine
Generates scaling decisions based on predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class ScalingAction(Enum):
    """Possible scaling actions"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    KEEP = "keep"


@dataclass
class ScalingDecision:
    """Represents a scaling decision"""
    machine_id: str
    timestamp: pd.Timestamp
    current_capacity: float
    predicted_usage: float
    action: ScalingAction
    recommended_capacity: float
    confidence: float


class ScalingPolicy:
    """
    Scaling recommendation engine
    Core decision logic for the paper
    """
    
    def __init__(self,
                 scale_up_threshold: float = 0.75,
                 scale_down_threshold: float = 0.40,
                 min_capacity: float = 0.1,
                 max_capacity: float = 1.0,
                 scaling_factor: float = 1.25):
        
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.scaling_factor = scaling_factor
        
        self.decisions_log = []
    
    def recommend(self,
                   machine_id: str,
                   current_capacity: float,
                   predicted_usage: float,
                   timestamp: Optional[pd.Timestamp] = None) -> ScalingDecision:
        """
        Generate scaling recommendation for a single machine
        
        Logic (as specified in methodology):
        - If predicted_usage > 0.75 × capacity: scale_up
        - If predicted_usage < 0.40 × capacity: scale_down
        - Otherwise: keep
        """
        utilization_ratio = predicted_usage / (current_capacity + 1e-10)
        
        if utilization_ratio > self.scale_up_threshold:
            action = ScalingAction.SCALE_UP
            recommended = min(current_capacity * self.scaling_factor, self.max_capacity)
            
        elif utilization_ratio < self.scale_down_threshold:
            action = ScalingAction.SCALE_DOWN
            recommended = max(current_capacity / self.scaling_factor, self.min_capacity)
            # Don't scale below predicted usage
            recommended = max(recommended, predicted_usage * 1.2)
            
        else:
            action = ScalingAction.KEEP
            recommended = current_capacity
        
        # Confidence based on how clear the decision is
        if action == ScalingAction.KEEP:
            confidence = 1 - abs(utilization_ratio - 0.5) / 0.5
        else:
            confidence = abs(utilization_ratio - 0.5) / 0.5
        
        decision = ScalingDecision(
            machine_id=machine_id,
            timestamp=timestamp or pd.Timestamp.now(),
            current_capacity=current_capacity,
            predicted_usage=predicted_usage,
            action=action,
            recommended_capacity=recommended,
            confidence=min(confidence, 1.0)
        )
        
        self.decisions_log.append(decision)
        
        return decision
    
    def recommend_batch(self,
                         machine_capacities: Dict[str, float],
                         predicted_usages: Dict[str, float],
                         timestamp: Optional[pd.Timestamp] = None) -> List[ScalingDecision]:
        """Generate recommendations for multiple machines"""
        decisions = []
        
        for machine_id in machine_capacities:
            if machine_id in predicted_usages:
                decision = self.recommend(
                    machine_id=machine_id,
                    current_capacity=machine_capacities[machine_id],
                    predicted_usage=predicted_usages[machine_id],
                    timestamp=timestamp
                )
                decisions.append(decision)
        
        return decisions
    
    def simulate_scaling_window(self,
                                  machine_id: str,
                                  initial_capacity: float,
                                  predicted_sequence: np.ndarray,
                                  actual_sequence: np.ndarray) -> Dict:
        """
        Simulate scaling decisions over a time window
        Returns metrics for evaluation
        """
        capacity = initial_capacity
        decisions = []
        costs = []
        sla_violations = 0
        
        for t, (predicted, actual) in enumerate(zip(predicted_sequence, actual_sequence)):
            # Make decision based on prediction
            decision = self.recommend(
                machine_id=machine_id,
                current_capacity=capacity,
                predicted_usage=predicted
            )
            
            # Apply scaling
            new_capacity = decision.recommended_capacity
            
            # Check for SLA violation (actual usage > 90% of capacity)
            if actual > 0.9 * new_capacity:
                sla_violations += 1
            
            # Calculate cost for this period
            period_cost = new_capacity  # Proportional to capacity
            costs.append(period_cost)
            
            decisions.append({
                'time_step': t,
                'capacity': new_capacity,
                'predicted': predicted,
                'actual': actual,
                'action': decision.action.value
            })
            
            capacity = new_capacity
        
        return {
            'decisions': decisions,
            'total_cost': sum(costs),
            'sla_violations': sla_violations,
            'sla_violation_rate': sla_violations / len(predicted_sequence) * 100,
            'avg_capacity': np.mean([d['capacity'] for d in decisions]),
            'avg_utilization': np.mean(actual_sequence) / np.mean([d['capacity'] for d in decisions])
        }
    
    def get_decision_summary(self) -> pd.DataFrame:
        """Summarize all decisions made"""
        if not self.decisions_log:
            return pd.DataFrame()
        
        records = []
        for d in self.decisions_log:
            records.append({
                'machine_id': d.machine_id,
                'timestamp': d.timestamp,
                'current_capacity': d.current_capacity,
                'predicted_usage': d.predicted_usage,
                'action': d.action.value,
                'recommended_capacity': d.recommended_capacity,
                'confidence': d.confidence
            })
        
        return pd.DataFrame(records)
    
    def get_action_distribution(self) -> Dict[str, int]:
        """Get distribution of scaling actions"""
        actions = [d.action.value for d in self.decisions_log]
        unique, counts = np.unique(actions, return_counts=True)
        return dict(zip(unique, counts.tolist()))