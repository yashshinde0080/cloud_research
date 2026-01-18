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

## Algorithm 1: Workload Pattern Clustering (Offline Phase)

```pseudocode
Algorithm 1: Unsupervised Workload Clustering

Input: 
    D: Database of historical machine traces {M_1, M_2, ..., M_N}
    k_min, k_max: Range for optimal cluster search

Output: 
    C: Optimal set of Cluster Centroids {c_1, ..., c_k}
    Model_Map: Mapping of clusters to Prediction Models

1. Feature_Matrix F = []
2. For each machine M_i in D do:
3.     // Extract statistical features
4.     μ_i = Mean(M_i)
5.     σ_i = StdDev(M_i)
6.     ρ_i = Autocorrelation(M_i, lag=1)
7.     peak_i = Max(M_i) / Mean(M_i)
8.     F.append([μ_i, σ_i, ρ_i, peak_i])
9. End For
10.
11. // Determine optimal k using Silhouette Score
12. best_score = -1, optimal_k = 0
13. For k = k_min to k_max do:
14.     C_temp = KMeans(F, k)
15.     score = SilhouetteScore(F, C_temp)
16.     If score > best_score then:
17.         best_score = score
18.         optimal_k = k
19.         C = C_temp
20.     End If
21. End For
22.
23. // Assign logical labels based on centroids
24. For each cluster c_j in C do:
25.     If c_j.volatility is High then Label(c_j) = "Volatile"
26.     Else If c_j.periodicity is High then Label(c_j) = "Periodic"
27.     Else Label(c_j) = "Stable"
28. End For
29.
30. Return C, Model_Map
```

## Algorithm 2: Hybrid Model Routing (Online Phase)

```pseudocode
Algorithm 2: Dynamic Model Selection and Prediction

Input: 
    w_t: Current workload window for a machine
    C: Set of Cluster Centroids (from Alg. 1)
    M_ARIMA, M_LSTM, M_Base: Pre-trained prediction models

Output: 
    ŷ_{t+1}: Predicted resource demand

1. // Extract features from current window
2. f_curr = ExtractFeatures(w_t)
3.
4. // Identify nearest workload pattern
5. nearest_cluster = null, min_dist = ∞
6. For each centroid c_j in C do:
7.     dist = EuclideanDistance(f_curr, c_j)
8.     If dist < min_dist then:
9.         min_dist = dist
10.        nearest_cluster = c_j
11.    End If
12. End For
13.
14. // Select model based on cluster characteristics
15. type = Label(nearest_cluster)
16.
17. Switch type:
18.     Case "Stable": 
19.         ŷ_{t+1} = M_ARIMA.predict(w_t)
20.     Case "Periodic":
21.         ŷ_{t+1} = M_LSTM.predict(w_t) // Captures complex seasonality
22.     Case "Bursty":
23.         ŷ_{t+1} = M_LSTM.predict(w_t)
24.     Case "Volatile":
25.         ŷ_{t+1} = Percentile(w_t, 95) // Conservative fallback
26.
27. Return ŷ_{t+1}
```

## Algorithm 3: Cost-Aware Scaling Decision

```pseudocode
Algorithm 3: SLA-Constrained Resource Provisioning

Input: 
    ŷ_{t+1}: Predicted demand (from Alg. 2)
    σ_err: Prediction error standard deviation
    SLA_limit: Maximum allowed violation probability (e.g., 0.01)

Output: 
    Action_{t+1}: Scaling action (Scale Up/Down/No-Op)

1. // Calculate dynamic safety buffer
2. // Using Inverse CDF of Normal Distribution for desired confidence level
3. Z_score = InverseNormalCDF(1 - SLA_limit) 
4. safety_buffer = Z_score * σ_err
5.
6. // Determine strictly required capacity
7. Required_Capacity = ŷ_{t+1} + safety_buffer
8.
9. // Quantize to nearest machine instance type (e.g., AWS EC2 units)
10. Allocated_Capacity = Ceil(Required_Capacity / Unit_Size) * Unit_Size
11.
12. Current_Cap = GetCurrentCapacity()
13.
14. // Apply hysteresis to prevent oscillation
15. If Allocated_Capacity > Current_Cap then:
16.     Return ScaleUp(Allocated_Capacity - Current_Cap)
17. Else If (Current_Cap - Allocated_Capacity) > Threshold_Down then:
18.     Return ScaleDown(Current_Cap - Allocated_Capacity)
19. Else:
20.     Return No_Op
```
