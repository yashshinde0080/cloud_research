"""
Experiment Runner
Complete experimental pipeline for paper results
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import logging
from datetime import datetime

# Import all modules
import sys
sys.path.append('..')

from preprocessing import DataLoader, DataCleaner, TimeAggregator, FeatureEngineer
from clustering import KMeansClustering, DBSCANClustering, ClusterAnalyzer
from prediction import HybridPredictor
from scaling import ScalingPolicy, CostModel
from .baselines import BaselineModels
from .metrics import EvaluationMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Complete experimental pipeline
    Reproduces all results for the paper
    """
    
    def __init__(self, config, results_path: str = 'results'):
        self.config = config
        self.results_path = Path(results_path)
        self.results_path.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_loader = DataLoader(config.data.raw_data_path)
        self.cleaner = DataCleaner()
        self.aggregator = TimeAggregator(config.data.time_window_minutes)
        self.feature_engineer = FeatureEngineer()
        
        self.kmeans = KMeansClustering(n_clusters=config.clustering.n_clusters)
        self.dbscan = DBSCANClustering(
            eps=config.clustering.dbscan_eps,
            min_samples=config.clustering.dbscan_min_samples
        )
        self.cluster_analyzer = ClusterAnalyzer()
        
        self.predictor = HybridPredictor(
            arima_order=config.prediction.arima_order,
            lstm_lookback=config.prediction.lstm_lookback
        )
        
        self.scaling_policy = ScalingPolicy(
            scale_up_threshold=config.scaling.scale_up_threshold,
            scale_down_threshold=config.scaling.scale_down_threshold
        )
        
        self.cost_model = CostModel(
            price_per_cpu_hour=config.scaling.price_per_cpu_hour,
            time_interval_minutes=config.data.time_window_minutes
        )
        
        self.baselines = BaselineModels()
        self.metrics = EvaluationMetrics()
        
        # Results storage
        self.results = {}
    
    def run_full_experiment(self, 
                             data: Optional[pd.DataFrame] = None,
                             n_runs: int = 5) -> Dict:
        """
        Run complete experiment pipeline
        
        Steps:
        1. Load and preprocess data
        2. Extract features
        3. Cluster workloads
        4. Train predictors per cluster
        5. Generate scaling recommendations
        6. Evaluate and compare
        """
        logger.info("=" * 60)
        logger.info("STARTING FULL EXPERIMENT")
        logger.info("=" * 60)
        
        # Step 1: Data Loading
        logger.info("\n[Step 1] Loading and preprocessing data...")
        if data is None:
            try:
                data = self.data_loader.load_all_data()
            except FileNotFoundError:
                logger.warning("No real data found, using synthetic data")
                from preprocessing.data_loader import create_synthetic_data
                data = create_synthetic_data(
                    n_machines=self.config.data.max_machines,
                    n_days=self.config.data.analysis_days
                )
        
        data = self.cleaner.clean(data)
        logger.info(f"Data loaded: {len(data)} rows")
        
        # Step 2: Aggregation and Feature Engineering
        logger.info("\n[Step 2] Aggregating and engineering features...")
        aggregated = self.aggregator.aggregate(data)
        logger.info(f"Aggregated columns: {aggregated.columns.tolist()}")
        aggregated = self.feature_engineer.extract_temporal_features(aggregated)
        aggregated = self.feature_engineer.calculate_over_provisioning(aggregated)
        
        # Create machine profiles for clustering
        profiles = self.aggregator.create_machine_profiles(aggregated)
        logger.info(f"Created profiles for {len(profiles)} machines")
        
        # Step 3: Clustering
        logger.info("\n[Step 3] Clustering workloads...")
        X_cluster, feature_names = self.feature_engineer.create_clustering_features(profiles)
        
        # Run KMeans
        self.kmeans.fit(X_cluster)
        profiles['cluster'] = self.kmeans.labels_
        
        # Analyze clusters
        cluster_analysis = self.cluster_analyzer.analyze_clusters(
            X_cluster, self.kmeans.labels_, feature_names
        )
        
        cluster_report = self.cluster_analyzer.create_analysis_report(cluster_analysis)
        logger.info("\nCluster Analysis:")
        logger.info(cluster_report.to_string())
        
        # Step 4: Train Predictors
        logger.info("\n[Step 4] Training predictors...")
        
        # Prepare time series per machine
        machine_series = {}
        # Sort by observation count to pick machines with most data
        top_machines = profiles.sort_values('n_observations', ascending=False)['machine_id'].unique()[:100]
        
        for machine_id in top_machines:
            machine_data = aggregated[aggregated['machine_id'] == machine_id]
            if 'average_usage_mean' in machine_data.columns and len(machine_data) > 5:
                machine_series[machine_id] = machine_data['average_usage_mean'].values
        
        logger.info(f"Collected time series for {len(machine_series)} machines")
        if not machine_series:
            logger.warning("No machines had sufficient data (>5 points). Summary of data:")
            logger.warning(f"Total rows in aggregated: {len(aggregated)}")
            if not aggregated.empty:
                 logger.warning(f"Sample aggregated: {aggregated.iloc[0].to_dict()}")
        
        # Get workload types
        machine_workload_types = {}
        for _, row in profiles.iterrows():
            if row['machine_id'] in machine_series:
                cluster = row['cluster']
                workload_type = cluster_analysis.get(cluster, {}).get('workload_type', 'periodic')
                machine_workload_types[row['machine_id']] = workload_type
        
        # Fit predictors
        # Fit predictors
        self.predictor.fit_all_machines(machine_series, machine_workload_types)
        
        # Save trained models
        self.predictor.save_models(self.config.models_path)
        
        # Step 5: Generate predictions and scaling decisions
        logger.info("\n[Step 5] Generating predictions and scaling decisions...")
        
        all_predictions = {}
        all_actuals = {}
        all_capacities = {}
        
        prediction_steps = 2 # Force small steps for sparse data
        logger.info(f"Prediction Steps: {prediction_steps}")
        
        skipped_count = 0
        accepted_count = 0
        
        for machine_id, series in machine_series.items():
            if len(series) < prediction_steps + 1: # Very relaxed check
                if skipped_count < 5:
                     logger.info(f"Skipping machine {machine_id}: length {len(series)} < {prediction_steps + 5}")
                skipped_count += 1
                continue
            
            accepted_count += 1
            
            # Split into train/test
            train_size = int(len(series) * 0.8)
            train = series[:train_size]
            test = series[train_size:train_size + prediction_steps]
            
            # Predict
            pred = self.predictor.predict_machine(machine_id, train, prediction_steps)
            
            all_predictions[machine_id] = pred
            all_actuals[machine_id] = test
            
            # Generate capacity sequence using proposed method
            capacity = []
            current_cap = train[-1] * 1.3
            for p in pred:
                decision = self.scaling_policy.recommend(machine_id, current_cap, p)
                current_cap = decision.recommended_capacity
                capacity.append(current_cap)
            all_capacities[machine_id] = np.array(capacity)
        
        # Step 6: Evaluation
        logger.info("\n[Step 6] Evaluating results...")
        
        # Aggregate all results
        # Calculate metrics
        if len(all_predictions) > 0:
            all_preds_flat = np.concatenate(list(all_predictions.values()))
            all_actuals_flat = np.concatenate(list(all_actuals.values()))
            all_capacity_flat = np.concatenate(list(all_capacities.values()))
            
            baseline_results = self.baselines.run_all_baselines(all_actuals_flat)
            
            prediction_metrics = {
                'rmse': self.metrics.rmse(all_actuals_flat, all_preds_flat),
                'mae': self.metrics.mae(all_actuals_flat, all_preds_flat),
                'mape': self.metrics.mape(all_actuals_flat, all_preds_flat)
            }
            proposed_metrics = {
                'total_cost': self.metrics.total_cost(all_capacity_flat),
                'utilization': self.metrics.average_utilization(all_actuals_flat, all_capacity_flat),
                'sla_violations': self.metrics.sla_violation_count(all_actuals_flat, all_capacity_flat),
                'sla_violation_rate': self.metrics.sla_violation_rate(all_actuals_flat, all_capacity_flat)
            }
        else:
            logger.warning("No predictions generated. Skipping evaluation metrics.")
            baseline_results = {
                'static': type('obj', (object,), {'total_cost': 0, 'utilization': 0, 'sla_violations': 0})(),
                'threshold': type('obj', (object,), {'total_cost': 0, 'utilization': 0, 'sla_violations': 0, 'cost_reduction_pct': 0})(),
                'moving_average': type('obj', (object,), {'total_cost': 0, 'utilization': 0, 'sla_violations': 0})()
            }
            prediction_metrics = {'rmse': 0, 'mae': 0, 'mape': 0}
            proposed_metrics = {'total_cost': 0, 'utilization': 0, 'sla_violations': 0, 'sla_violation_rate': 0}
            all_preds_flat = np.array([])
            all_actuals_flat = np.array([])

        results = {
            'prediction_metrics': prediction_metrics,
            'proposed': proposed_metrics,
            'baselines': {}
        }
        
        # Add baseline results
        for name, baseline in baseline_results.items():
            results['baselines'][name] = {
                'total_cost': baseline.total_cost,
                'utilization': baseline.utilization,
                'sla_violations': baseline.sla_violations
            }
        
        # Calculate cost reduction
        static_cost = results['baselines']['static']['total_cost']
        if static_cost > 0:
            results['proposed']['cost_reduction_pct'] = (
                (static_cost - results['proposed']['total_cost']) / static_cost * 100
            )
            
            for name in results['baselines']:
                baseline_cost = results['baselines'][name]['total_cost']
                results['baselines'][name]['cost_reduction_pct'] = (
                    (static_cost - baseline_cost) / static_cost * 100
                )
        else:
             results['proposed']['cost_reduction_pct'] = 0.0
             for name in results['baselines']:
                results['baselines'][name]['cost_reduction_pct'] = 0.0
        
        # Store results
        self.results = results
        self.results['cluster_analysis'] = cluster_analysis
        self.results['n_machines'] = len(machine_series)
        self.results['n_observations'] = len(all_actuals_flat)
        
        # Add visualization data
        self.results['vis_data'] = {
            'X_cluster': X_cluster,
            'labels': self.kmeans.labels_,
            'all_predictions': all_predictions,
            'all_actuals': all_actuals,
            'all_capacities': all_capacities,
            'machine_series': machine_series
        }
        
        # Print summary
        self._print_summary(results)
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _print_summary(self, results: Dict):
        """Print formatted results summary"""
        print("\n" + "=" * 60)
        print("EXPERIMENT RESULTS SUMMARY")
        print("=" * 60)
        
        print("\n--- Prediction Performance ---")
        pm = results['prediction_metrics']
        print(f"RMSE:  {pm['rmse']:.4f}")
        print(f"MAE:   {pm['mae']:.4f}")
        print(f"MAPE:  {pm['mape']:.2f}%")
        
        print("\n--- Cost Comparison ---")
        print(f"{'Method':<20} {'Cost':>10} {'Reduction':>12} {'Utilization':>12} {'SLA Viol.':>10}")
        print("-" * 64)
        
        # Static baseline
        static = results['baselines']['static']
        print(f"{'Static':<20} {static['total_cost']:>10.4f} {'0.0%':>12} {static['utilization']*100:>11.1f}% {static['sla_violations']:>10}")
        
        # Threshold baseline
        threshold = results['baselines']['threshold']
        print(f"{'Threshold':<20} {threshold['total_cost']:>10.4f} {threshold['cost_reduction_pct']:>11.1f}% {threshold['utilization']*100:>11.1f}% {threshold['sla_violations']:>10}")
        
        # Proposed
        proposed = results['proposed']
        print(f"{'Proposed':<20} {proposed['total_cost']:>10.4f} {proposed['cost_reduction_pct']:>11.1f}% {proposed['utilization']*100:>11.1f}% {proposed['sla_violations']:>10}")
        
        print("\n" + "=" * 60)
    
    def _save_results(self, results: Dict):
        """Save results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, dict):
                return {str(k): convert(v) for k, v in obj.items()}
            return obj
        
        # Exclude visualization data from JSON
        results_copy = results.copy()
        if 'vis_data' in results_copy:
            del results_copy['vis_data']
            
        results_serializable = convert(results_copy)
        
        output_file = self.results_path / f"experiment_results_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
    
    def run_sensitivity_analysis(self, 
                                   data: pd.DataFrame,
                                   param_name: str,
                                   param_values: List) -> pd.DataFrame:
        """Run sensitivity analysis on a parameter"""
        results = []
        
        for value in param_values:
            logger.info(f"Testing {param_name} = {value}")
            
            # Update config
            if param_name == 'n_clusters':
                self.kmeans = KMeansClustering(n_clusters=value)
            elif param_name == 'scale_up_threshold':
                self.scaling_policy.scale_up_threshold = value
            elif param_name == 'scale_down_threshold':
                self.scaling_policy.scale_down_threshold = value
            
            # Run experiment
            exp_results = self.run_full_experiment(data, n_runs=1)
            
            results.append({
                param_name: value,
                'cost_reduction': exp_results['proposed']['cost_reduction_pct'],
                'utilization': exp_results['proposed']['utilization'],
                'sla_violations': exp_results['proposed']['sla_violation_rate']
            })
        
        return pd.DataFrame(results)