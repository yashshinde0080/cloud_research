"""
Main Pipeline
Complete execution of the research project
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.config import Config, config
from preprocessing import DataLoader, DataCleaner, TimeAggregator, FeatureEngineer
from preprocessing.data_loader import create_synthetic_data
from clustering import KMeansClustering, ClusterAnalyzer
from prediction import HybridPredictor
from scaling import ScalingPolicy, CostModel
from evaluation import ExperimentRunner
from visualization import Visualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Cloud Cost Optimization Using Workload Pattern Clustering'
    )
    
    parser.add_argument('--data-path', type=str, default='data/raw',
                       help='Path to raw data directory')
    parser.add_argument('--use-synthetic', action='store_true',
                       help='Use synthetic data for testing')
    parser.add_argument('--n-machines', type=int, default=100,
                       help='Number of machines to analyze')
    parser.add_argument('--n-days', type=int, default=7,
                       help='Number of days to analyze')
    parser.add_argument('--n-clusters', type=int, default=4,
                       help='Number of workload clusters')
    parser.add_argument('--output-path', type=str, default='results',
                       help='Path for output results')
    parser.add_argument('--generate-figures', action='store_true',
                       help='Generate all figures')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test with minimal data')
    
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_args()
    
    logger.info("=" * 70)
    logger.info("CLOUD COST OPTIMIZATION USING WORKLOAD PATTERN CLUSTERING")
    logger.info("=" * 70)
    
    # Update config with command line args
    config.data.raw_data_path = args.data_path
    config.data.max_machines = args.n_machines
    config.data.analysis_days = args.n_days
    config.clustering.n_clusters = args.n_clusters
    config.results_path = args.output_path
    
    # Quick test mode
    if args.quick_test:
        logger.info("Running in quick test mode...")
        config.data.max_machines = 20
        config.data.analysis_days = 2
        config.prediction.lstm_epochs = 10
    
    # Load or create data
    if args.use_synthetic:
        logger.info("Creating synthetic data...")
        data = create_synthetic_data(
            n_machines=config.data.max_machines,
            n_days=config.data.analysis_days
        )
    else:
        logger.info(f"Loading data from {config.data.raw_data_path}...")
        try:
            loader = DataLoader(config.data.raw_data_path)
            data = loader.load_all_data()
            data = loader.sample_machines(data, config.data.max_machines)
        except FileNotFoundError:
            logger.warning("No data found, falling back to synthetic data")
            data = create_synthetic_data(
                n_machines=config.data.max_machines,
                n_days=config.data.analysis_days
            )
    
    logger.info(f"Data shape: {data.shape}")
    
    # Run experiment
    runner = ExperimentRunner(config, results_path=config.results_path)
    results = runner.run_full_experiment(data)
    
    # Generate figures
    # Generate figures
    if args.generate_figures:
        logger.info("\nGenerating figures...")
        
        visualizer = Visualizer(output_path=config.figures_path)
        
        if 'vis_data' in results:
            vis_data = results['vis_data']
            
            # Pick a representative machine (first one with data)
            machine_ids = list(vis_data['all_actuals'].keys())
            if machine_ids:
                mid = machine_ids[0]
                logger.info(f"Using machine {mid} as representative for time-series plots")
                
                usage = vis_data['all_actuals'][mid]
                proposed_cap = vis_data['all_capacities'][mid]
                pred = vis_data['all_predictions'][mid]
                
                # Re-compute baselines for this machine
                # Access baselines from runner
                static_res = runner.baselines.static_provisioning(usage)
                threshold_res = runner.baselines.threshold_autoscaling(usage)
                
                capacities = {
                    'proposed': proposed_cap,
                    'static': static_res.capacity_sequence,
                    'threshold': threshold_res.capacity_sequence
                }
                
                visualizer.generate_all_figures(
                    experiment_results=results,
                    X_cluster=vis_data['X_cluster'],
                    labels=vis_data['labels'],
                    usage=usage,
                    capacities=capacities,
                    predictions=pred,
                    actuals=usage
                )
            else:
                logger.warning("No machine data available for visualization")
        else:
            # Fallback if no vis_data (should not happen with updated experiments.py)
            visualizer.plot_system_architecture()
            logger.warning("Visualization data missing, only system architecture saved")
            
        logger.info(f"Figures saved to {config.figures_path}")
    
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 70)
    
    return results


def demo():
    """Quick demonstration with minimal setup"""
    logger.info("Running demonstration...")
    
    # Create synthetic data
    data = create_synthetic_data(n_machines=50, n_days=3)
    
    # Clean and preprocess
    cleaner = DataCleaner()
    data = cleaner.clean(data)
    
    # Aggregate
    aggregator = TimeAggregator(window_minutes=5)
    aggregated = aggregator.aggregate(data)
    
    # Create profiles
    profiles = aggregator.create_machine_profiles(aggregated)
    
    # Feature engineering
    fe = FeatureEngineer()
    X, features = fe.create_clustering_features(profiles)
    
    # Clustering
    kmeans = KMeansClustering(n_clusters=4)
    kmeans.fit(X)
    
    # Analyze clusters
    analyzer = ClusterAnalyzer()
    analysis = analyzer.analyze_clusters(X, kmeans.labels_, features)
    
    report = analyzer.create_analysis_report(analysis)
    print("\n" + "=" * 50)
    print("CLUSTER ANALYSIS")
    print("=" * 50)
    print(report.to_string())
    
    # Visualize
    visualizer = Visualizer()
    visualizer.plot_cluster_pca(X, kmeans.labels_, 
                                 {i: analysis[i]['workload_type'] for i in analysis})
    
    logger.info("Demonstration complete!")
    

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        demo()
    else:
        main()