"""
Visualization Module
All figures required for the paper
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


class Visualizer:
    """Generate all figures for the paper"""
    
    def __init__(self, output_path: str = 'figures'):
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        # Color scheme
        self.colors = {
            'static': '#E74C3C',      # Red
            'threshold': '#F39C12',    # Orange
            'proposed': '#27AE60',     # Green
            'moving_avg': '#3498DB',   # Blue
            'oracle': '#9B59B6'        # Purple
        }
        
        self.cluster_colors = ['#E74C3C', '#3498DB', '#27AE60', '#F39C12', '#9B59B6']
    
    def plot_system_architecture(self, save: bool = True) -> plt.Figure:
        """Generate system architecture diagram"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Define boxes
        boxes = [
            {'name': 'Cloud Metrics\nData', 'pos': (0.5, 0.9), 'color': '#ECF0F1'},
            {'name': 'Data\nPreprocessing', 'pos': (0.5, 0.75), 'color': '#AED6F1'},
            {'name': 'Feature\nEngineering', 'pos': (0.5, 0.6), 'color': '#AED6F1'},
            {'name': 'Workload\nClustering', 'pos': (0.3, 0.45), 'color': '#ABEBC6'},
            {'name': 'Usage\nPrediction', 'pos': (0.7, 0.45), 'color': '#ABEBC6'},
            {'name': 'Scaling\nRecommendation', 'pos': (0.5, 0.25), 'color': '#F9E79F'},
            {'name': 'Cost\nEvaluation', 'pos': (0.5, 0.1), 'color': '#FADBD8'}
        ]
        
        # Draw boxes
        for box in boxes:
            rect = mpatches.FancyBboxPatch(
                (box['pos'][0] - 0.12, box['pos'][1] - 0.05),
                0.24, 0.08,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                facecolor=box['color'],
                edgecolor='black',
                linewidth=2
            )
            ax.add_patch(rect)
            ax.text(box['pos'][0], box['pos'][1], box['name'],
                   ha='center', va='center', fontsize=11, fontweight='bold')
        
        # Draw arrows
        arrows = [
            ((0.5, 0.85), (0.5, 0.79)),
            ((0.5, 0.71), (0.5, 0.64)),
            ((0.42, 0.55), (0.35, 0.5)),
            ((0.58, 0.55), (0.65, 0.5)),
            ((0.35, 0.4), (0.45, 0.3)),
            ((0.65, 0.4), (0.55, 0.3)),
            ((0.5, 0.2), (0.5, 0.15))
        ]
        
        for start, end in arrows:
            ax.annotate('', xy=end, xytext=start,
                       arrowprops=dict(arrowstyle='->', lw=2, color='#2C3E50'))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('System Architecture', fontsize=16, fontweight='bold', pad=20)
        
        if save:
            fig.savefig(self.output_path / 'system_architecture.png')
            fig.savefig(self.output_path / 'system_architecture.pdf')
        
        return fig
    
    def plot_cluster_pca(self, 
                          X: np.ndarray,
                          labels: np.ndarray,
                          cluster_names: Optional[Dict[int, str]] = None,
                          save: bool = True) -> plt.Figure:
        """PCA visualization of clusters"""
        from sklearn.decomposition import PCA
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Perform PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Plot each cluster
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            if label == -1:  # Noise
                color = 'gray'
                name = 'Noise'
            else:
                color = self.cluster_colors[i % len(self.cluster_colors)]
                name = cluster_names.get(label, f'Cluster {label}') if cluster_names else f'Cluster {label}'
            
            mask = labels == label
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                      c=color, label=name, alpha=0.6, s=50, edgecolors='white')
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title('Workload Pattern Clusters (PCA Visualization)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        if save:
            fig.savefig(self.output_path / 'cluster_pca.png')
            fig.savefig(self.output_path / 'cluster_pca.pdf')
        
        return fig
    
    def plot_cost_comparison(self,
                              results: Dict[str, float],
                              save: bool = True) -> plt.Figure:
        """Bar chart comparing costs across methods"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = list(results.keys())
        costs = list(results.values())
        colors = [self.colors.get(m, '#95A5A6') for m in methods]
        
        bars = ax.bar(methods, costs, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, cost in zip(bars, costs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${cost:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Add cost reduction labels
        if 'static' in results:
            baseline = results['static']
            for i, (method, cost) in enumerate(results.items()):
                if method != 'static':
                    reduction = (baseline - cost) / baseline * 100
                    ax.text(i, cost/2, f'-{reduction:.1f}%',
                           ha='center', va='center', fontsize=10,
                           color='white', fontweight='bold')
        
        ax.set_ylabel('Total Cost ($)')
        ax.set_title('Cost Comparison Across Scaling Methods')
        ax.set_ylim(0, max(costs) * 1.15)
        
        if save:
            fig.savefig(self.output_path / 'cost_comparison.png')
            fig.savefig(self.output_path / 'cost_comparison.pdf')
        
        return fig
    
    def plot_utilization_over_time(self,
                                    time_index: pd.DatetimeIndex,
                                    usage: np.ndarray,
                                    capacities: Dict[str, np.ndarray],
                                    save: bool = True) -> plt.Figure:
        """Line plot of utilization over time"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Top plot: Capacity and usage
        ax1 = axes[0]
        ax1.fill_between(time_index, usage, alpha=0.3, color='#3498DB', label='Actual Usage')
        ax1.plot(time_index, usage, color='#3498DB', linewidth=1)
        
        for method, capacity in capacities.items():
            if len(capacity) == len(time_index):
                color = self.colors.get(method, '#95A5A6')
                ax1.plot(time_index, capacity, color=color, 
                        label=f'{method.title()} Capacity', linewidth=2, linestyle='--')
        
        ax1.set_ylabel('Resource Usage / Capacity')
        ax1.set_title('Resource Usage and Provisioned Capacity Over Time')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # Bottom plot: Utilization ratio
        ax2 = axes[1]
        for method, capacity in capacities.items():
            if len(capacity) == len(time_index):
                utilization = usage / (capacity + 1e-10)
                color = self.colors.get(method, '#95A5A6')
                ax2.plot(time_index, utilization, color=color,
                        label=f'{method.title()}', linewidth=2)
        
        # Add threshold lines
        ax2.axhline(y=0.75, color='red', linestyle=':', alpha=0.7, label='Scale-up Threshold')
        ax2.axhline(y=0.40, color='green', linestyle=':', alpha=0.7, label='Scale-down Threshold')
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Utilization Ratio')
        ax2.set_title('Resource Utilization Over Time')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.2)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_path / 'utilization_over_time.png')
            fig.savefig(self.output_path / 'utilization_over_time.pdf')
        
        return fig
    
    def plot_sla_violations(self,
                             results: Dict[str, Dict],
                             save: bool = True) -> plt.Figure:
        """Bar chart of SLA violations by method"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        methods = list(results.keys())
        violations = [results[m].get('sla_violations', 0) for m in methods]
        rates = [results[m].get('sla_violation_rate', 0) for m in methods]
        colors = [self.colors.get(m, '#95A5A6') for m in methods]
        
        # Violation count
        bars1 = ax1.bar(methods, violations, color=colors, edgecolor='black')
        ax1.set_ylabel('Number of SLA Violations')
        ax1.set_title('SLA Violation Count by Method')
        
        for bar, v in zip(bars1, violations):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{v}', ha='center', va='bottom', fontweight='bold')
        
        # Violation rate
        bars2 = ax2.bar(methods, rates, color=colors, edgecolor='black')
        ax2.set_ylabel('SLA Violation Rate (%)')
        ax2.set_title('SLA Violation Rate by Method')
        
        for bar, r in zip(bars2, rates):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{r:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_path / 'sla_violations.png')
            fig.savefig(self.output_path / 'sla_violations.pdf')
        
        return fig
    
    def plot_prediction_accuracy(self,
                                   actual: np.ndarray,
                                   predicted: np.ndarray,
                                   time_index: Optional[pd.DatetimeIndex] = None,
                                   save: bool = True) -> plt.Figure:
        """Plot actual vs predicted values"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Time series comparison
        x = time_index if time_index is not None else np.arange(len(actual))
        ax1.plot(x, actual, label='Actual', color='#3498DB', linewidth=2)
        ax1.plot(x, predicted, label='Predicted', color='#E74C3C', 
                linewidth=2, linestyle='--')
        ax1.fill_between(x, actual, predicted, alpha=0.2, color='gray')
        ax1.set_ylabel('Usage')
        ax1.set_title('Actual vs Predicted Usage')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Scatter plot
        ax2.scatter(actual, predicted, alpha=0.5, color='#3498DB')
        
        # Perfect prediction line
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 
                'r--', label='Perfect Prediction', linewidth=2)
        
        ax2.set_xlabel('Actual Usage')
        ax2.set_ylabel('Predicted Usage')
        ax2.set_title('Prediction Accuracy Scatter Plot')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_path / 'prediction_accuracy.png')
            fig.savefig(self.output_path / 'prediction_accuracy.pdf')
        
        return fig
    
    def plot_elbow_curve(self,
                          k_range: range,
                          inertias: List[float],
                          silhouettes: Optional[List[float]] = None,
                          save: bool = True) -> plt.Figure:
        """Elbow curve for optimal k selection"""
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color1 = '#3498DB'
        ax1.plot(list(k_range), inertias, 'o-', color=color1, linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia (Within-cluster Sum of Squares)', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_title('Elbow Method for Optimal k')
        
        if silhouettes:
            ax2 = ax1.twinx()
            color2 = '#E74C3C'
            ax2.plot(list(k_range), silhouettes, 's-', color=color2, linewidth=2, markersize=8)
            ax2.set_ylabel('Silhouette Score', color=color2)
            ax2.tick_params(axis='y', labelcolor=color2)
        
        ax1.grid(True, alpha=0.3)
        
        if save:
            fig.savefig(self.output_path / 'elbow_curve.png')
            fig.savefig(self.output_path / 'elbow_curve.pdf')
        
        return fig
    
    def plot_cluster_characteristics(self,
                                       cluster_stats: Dict[int, Dict],
                                       save: bool = True) -> plt.Figure:
        """Radar/spider chart of cluster characteristics"""
        from math import pi
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Features to plot
        features = ['cpu_mean_mean', 'cpu_cv_mean', 'over_provision_ratio_mean', 'memory_mean_mean']
        feature_labels = ['Avg CPU', 'CPU Variability', 'Over-Provision', 'Memory']
        
        # Number of features
        N = len(features)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        for cluster_id, stats in cluster_stats.items():
            values = []
            for f in features:
                val = stats.get(f, 0)
                # Normalize to 0-1 range
                values.append(min(val, 1.0))
            values += values[:1]
            
            color = self.cluster_colors[cluster_id % len(self.cluster_colors)]
            ax.plot(angles, values, 'o-', linewidth=2, label=f"Cluster {cluster_id}: {stats.get('workload_type', 'Unknown')}")
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_labels)
        ax.set_title('Cluster Characteristics', size=16, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        if save:
            fig.savefig(self.output_path / 'cluster_characteristics.png')
            fig.savefig(self.output_path / 'cluster_characteristics.pdf')
        
        return fig
    
    def generate_all_figures(self,
                              experiment_results: Dict,
                              X_cluster: np.ndarray,
                              labels: np.ndarray,
                              usage: np.ndarray,
                              capacities: Dict[str, np.ndarray],
                              predictions: np.ndarray,
                              actuals: np.ndarray):
        """Generate all figures for the paper"""
        logger.info("Generating all figures...")
        
        # 1. System Architecture
        self.plot_system_architecture()
        
        # 2. Cluster PCA
        cluster_names = {i: experiment_results['cluster_analysis'][i]['workload_type'] 
                        for i in experiment_results['cluster_analysis']}
        self.plot_cluster_pca(X_cluster, labels, cluster_names)
        
        # 3. Cost Comparison
        cost_results = {
            'static': experiment_results['baselines']['static']['total_cost'],
            'threshold': experiment_results['baselines']['threshold']['total_cost'],
            'proposed': experiment_results['proposed']['total_cost']
        }
        self.plot_cost_comparison(cost_results)
        
        # 4. Utilization Over Time
        time_index = pd.date_range(start='2019-05-01', periods=len(usage), freq='5min')
        self.plot_utilization_over_time(time_index, usage, capacities)
        
        # 5. SLA Violations
        sla_results = {
            'static': experiment_results['baselines']['static'],
            'threshold': experiment_results['baselines']['threshold'],
            'proposed': experiment_results['proposed']
        }
        self.plot_sla_violations(sla_results)
        
        # 6. Prediction Accuracy
        self.plot_prediction_accuracy(actuals, predictions)
        
        # 7. Cluster Characteristics
        self.plot_cluster_characteristics(experiment_results['cluster_analysis'])
        
        logger.info(f"All figures saved to {self.output_path}")