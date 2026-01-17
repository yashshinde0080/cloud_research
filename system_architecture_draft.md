# System Architecture

The proposed system architecture is designed as a closed-loop control framework for automated cloud resource provisioning. It operates in four distinct stages:

1.  **Data Ingestion and Preprocessing**: Raw telemetry data (CPU, memory usage) is ingested and aggregated into fixed time intervals. Noise reduction and normalization techniques are applied to ensure data quality.
2.  **Workload Pattern Recognition**: An offline clustering module utilizes the K-Means algorithm to analyze historical usage patterns, categorizing machines into distinct behavioral groups (e.g., Stable, Periodic, Bursty) based on statistical features.
3.  **Hybrid Prediction Engine**: An intelligent router directs real-time workload data to the most suitable forecasting model—using ARIMA for linear, stable trends and LSTM networks for complex, non-linear patterns—thereby maximizing prediction accuracy for each specific workload type.
4.  **Scaling and Optimization**: The predicted demand is fed into a cost-aware scaling policy that dynamically adjusts resource capacity defined by a utility function. This function balances the trade-off between minimizing operational costs and preventing SLA violations, resulting in an optimal provisioning decision for the next time interval.
