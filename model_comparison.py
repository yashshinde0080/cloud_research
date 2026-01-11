
"""
Model Comparison Script
Trains ARIMA, LSTM, and Hybrid models on data and compares them using RMSE and MAPE.
Saves results and figures to 'results/model_comparison'.
"""

import sys
import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.config import config
from preprocessing import DataLoader, DataCleaner, TimeAggregator, FeatureEngineer
from clustering import KMeansClustering, ClusterAnalyzer
from prediction.arima_predictor import ARIMAPredictor
from prediction.lstm_predictor import LSTMPredictor, SimpleLSTMPredictor
from prediction.hybrid_predictor import HybridPredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_metrics(actual, predicted):
    """Calculate RMSE and MAPE"""
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    # Avoid division by zero
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-10))) * 100
    return rmse, mape

def run_comparison():
    # Setup output directories
    output_dir = Path("results/model_comparison")
    figures_dir = output_dir / "figures"
    values_dir = output_dir / "values"
    figures_dir.mkdir(parents=True, exist_ok=True)
    values_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading and processing data...")
    
    # Load Data
    try:
        loader = DataLoader(config.data.raw_data_path)
        data = loader.load_all_data()
        # Sample if too large, but for comparison we want quality
        data = loader.sample_machines(data, n_machines=20) # Process 20 machines
    except FileNotFoundError:
        logger.warning("Data not found, generating synthetic data")
        from preprocessing.data_loader import create_synthetic_data
        data = create_synthetic_data(n_machines=20, n_days=7)

    cleaner = DataCleaner()
    data = cleaner.clean(data)
    
    # Aggregation
    aggregator = TimeAggregator(window_minutes=5)
    aggregated = aggregator.aggregate(data)
    
    # Clustering (needed for Hybrid)
    aggregator = TimeAggregator(window_minutes=60) # Larger window for profiling
    profiles = aggregator.create_machine_profiles(aggregated)
    fe = FeatureEngineer()
    X_cluster, feature_names = fe.create_clustering_features(profiles)
    
    kmeans = KMeansClustering(n_clusters=4)
    kmeans.fit(X_cluster)
    analyzer = ClusterAnalyzer()
    cluster_analysis = analyzer.analyze_clusters(X_cluster, kmeans.labels_, feature_names)
    
    # Map machine to workload type
    machine_workload = {}
    for idx, row in profiles.iterrows():
        cluster = kmeans.labels_[idx]
        workload = cluster_analysis[cluster]['workload_type']
        machine_workload[row['machine_id']] = workload

    # Comparison Loop
    results = []
    
    # Process each machine
    unique_machines = aggregated['machine_id'].unique()
    logger.info(f"Comparing models on {len(unique_machines)} machines...")

    for machine_id in unique_machines:
        machine_data = aggregated[aggregated['machine_id'] == machine_id].sort_values('time')
        if len(machine_data) < 50:
            continue
            
        series = machine_data['average_usage_mean'].values
        
        # Train/Test Split (80/20)
        split_idx = int(len(series) * 0.8)
        train = series[:split_idx]
        test = series[split_idx:]
        
        if len(test) < 1:
            continue

        workload_type = machine_workload.get(machine_id, 'periodic')
        logger.info(f"Processing Machine {machine_id} (Type: {workload_type})")

        # 1. ARIMA
        arima = ARIMAPredictor()
        arima.fit(train)
        pred_arima = arima.predict(steps=len(test))
        rmse_arima, mape_arima = calculate_metrics(test, pred_arima)

        # 2. LSTM
        try:
            lstm = LSTMPredictor(epochs=10, verbose=0)
            lstm.fit(train)
            pred_lstm = lstm.predict(train, steps=len(test))
        except:
            lstm = SimpleLSTMPredictor()
            lstm.fit(train)
            pred_lstm = lstm.predict(steps=len(test))
            
        rmse_lstm, mape_lstm = calculate_metrics(test, pred_lstm)

        # 3. Hybrid (Simulate logic: It picks the best strategy based on workload)
        # We can implement it by just reinstantiating HybridPredictor and letting it transform
        hybrid = HybridPredictor()
        # HybridPredictor usually fits the specific model. We mimic its selection logic here:
        strategy = hybrid.get_strategy(workload_type)
        
        # For the sake of this comparison, "Hybrid" outcome is basically one of the above 
        # (or Seasonal ARIMA / Conservative).
        # We will re-run the exact HybridPredictor logic to be sure.
        hybrid.fit_machine(machine_id, train, workload_type)
        pred_hybrid = hybrid.predict_machine(machine_id, train, steps=len(test))
        rmse_hybrid, mape_hybrid = calculate_metrics(test, pred_hybrid)

        # Store Metrics
        results.append({
            'machine_id': machine_id,
            'workload_type': workload_type,
            'ARIMA_RMSE': rmse_arima, 'ARIMA_MAPE': mape_arima,
            'LSTM_RMSE': rmse_lstm, 'LSTM_MAPE': mape_lstm,
            'Hybrid_RMSE': rmse_hybrid, 'Hybrid_MAPE': mape_hybrid,
            'Hybrid_Strategy': strategy
        })

        # Save Values
        df_values = pd.DataFrame({
            'Actual': test,
            'ARIMA': pred_arima,
            'LSTM': pred_lstm,
            'Hybrid': pred_hybrid
        })
        df_values.to_csv(values_dir / f"{machine_id}_values.csv", index=False)

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(test, label='Actual', color='black', linewidth=1.5)
        plt.plot(pred_arima, label=f'ARIMA (RMSE={rmse_arima:.3f})',  linestyle='--')
        plt.plot(pred_lstm, label=f'LSTM (RMSE={rmse_lstm:.3f})', linestyle='-.')
        plt.plot(pred_hybrid, label=f'Hybrid (RMSE={rmse_hybrid:.3f})', linewidth=2, alpha=0.7)
        
        plt.title(f"Model Comparison - Machine {machine_id} ({workload_type})\nHybrid chose: {strategy}")
        plt.legend()
        plt.xlabel("Time Steps")
        plt.ylabel("Usage")
        plt.grid(True, alpha=0.3)
        plt.savefig(figures_dir / f"{machine_id}_comparison.png")
        plt.close()

    # Save Summary Metrics
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results.to_csv(output_dir / "model_comparison_metrics.csv", index=False)
        
        # Calculate Averages
        avg_results = df_results.mean(numeric_only=True)
        print("\nAverage Performance Metrics:")
        print(avg_results)
        
        # Save Average
        avg_results.to_csv(output_dir / "average_metrics.csv")
    else:
        logger.warning("No results generated.")

if __name__ == "__main__":
    run_comparison()
