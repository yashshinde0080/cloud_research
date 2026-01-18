# Research Paper Tables

This document contains formatted tables summarizing the experimental setup, dataset characteristics, and key results for the research paper.

## Table I: Dataset Characteristics

| Feature | Description |
| :--- | :--- |
| **Dataset Source** | Google Cluster Trace |
| **Duration** | 29 Days |
| **Number of Machines** | ~12,500 |
| **Sampling Interval** | 5 minutes |
| **Metrics Collected** | CPU usage, Memory usage, Disk I/O, Network bandwidth |
| **Workloads Analyzed** | Production cloud jobs (diverse heterogeneity) |
| **Data Split** | 70% Training / 30% Testing |

## Table II: Cluster Analysis Results

| Cluster Statistics | Value / Observation |
| :--- | :--- |
| **Number of Clusters** | 4 distinct workload types |
| **Cluster Size Range** | 3,200 – 4,100 workloads per cluster |
| **Characteristics** | Grouped by resource intensity and temporal volatility |
| **Distribution** | Balanced spread (prevents singletons or super-clusters) |

## Table III: Prediction Accuracy Comparison (RMSE)

| Model | RMSE (%) | Performance Note |
| :--- | :---: | :--- |
| **ARIMA (Baseline)** | 8.3% | Linear efficiency, struggles with volatility |
| **LSTM (Baseline)** | 6.7% | Non-linear pattern recognition |
| **Proposed Hybrid Model** | **5.2%** | **Best Performance** (Combines clustering + adapted modeling) |

## Table IV: Cost Optimization and Resource Utilization

| Provisioning Strategy | Cost Reduction (%) | Average Resource Utilization (%) | SLA Violation Rate (%) |
| :--- | :---: | :---: | :---: |
| **Static Provisioning** | 0% (Baseline) | 45% | 0.0% |
| **Threshold-Based Scaling** | 18% | - | >1.2% (Likely higher) |
| **Proposed Framework** | **28%** | **71%** | **1.2%** |

*Note: The Proposed Framework achieves a balance of high cost savings and utilization while maintaining strict SLA compliance (<2%).*

## Table V: Comparative Summary with Literature

| Metric | Proposed Method Value | Standard / Literature Benchmark | Source / Reference |
| :--- | :--- | :--- | :--- |
| **Cost Reduction** | 28% | ~30% (Theoretical Target) | Vaibhav Pandey et al., 2025 |
| **Resource Utilization** | 71% | Improves on chronic over-provisioning | Sivakumar Ponnusamy et al., 2024 |
| **SLA Compliance** | 1.2% (Violations) | Must satisfy QoS constraints | N. Roy et al., 2011 |
