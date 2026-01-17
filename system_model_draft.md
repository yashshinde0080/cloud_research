# VI. PROPOSED SYSTEM MODEL

This section describes the architecture of the proposed **Clustering-Based Cloud Cost Optimization Framework**. The system aims to dynamically provision resources by categorizing workload patterns and applying specialized prediction strategies, balancing cost efficiency with Service Level Agreement (SLA) compliance.

A high-level overview of the system architecture is presented in **Figure 1** (conceptual), consisting of four primary modules: **Data Preprocessing**, **Workload Clustering**, **Hybrid Prediction**, and **Scaling Decision**.

## A. Workload Characterization and Clustering Logic

Cloud workloads exhibit diverse behaviors, ranging from stable, predictable usage to highly volatile, bursty patterns. A "one-size-fits-all" prediction model often fails to capture these nuances effectively. Our system employs an unsupervised learning approach to categorize machines based on their resource usage signatures.

1.  **Feature Extraction**: For each machine $m_i$, we extract a feature vector $X_i$ from the historical time-series data $T_i$. Key statistical features include:
    *   **Mean Utilization ($\mu$)**: Represents the average load.
    *   **Variance ($\sigma^2$)**: Captures the stability of the workload.
    *   **Peak-to-Mean Ratio**: Indicates the "burstiness" of the task.
    *   **Autocorrelation**: Measures the degree of seasonality and periodicity.

2.  **Pattern Identification**: We utilize the **K-Means clustering algorithm** to partition the feature space into $k$ distinct clusters ($C_1, C_2, ..., C_k$). The optimal number of clusters is determined using the **Silhouette Coefficient**. This process groups machines with similar behavioral characteristics, such as "Stable/Low Load," "Periodic," "Bursty," and "Volatile."

## B. Cluster-Aware Hybrid Prediction Engine

The core innovation of the proposed system is the **Hybrid Predictor**, which dynamically selects the most appropriate forecasting model based on the identified workload cluster. This allows the system to leverage the strengths of different algorithms for different usage patterns.

Let $W_{type}$ be the workload type identified for a machine. The prediction strategy $S$ is defined as:

$$
S(W_{type}) = 
\begin{cases} 
\text{ARIMA}, & \text{if } W_{type} \in \{\text{Stable, Low Load}\} \\
\text{Seasonal ARIMA}, & \text{if } W_{type} \in \{\text{Periodic}\} \\
\text{LSTM}, & \text{if } W_{type} \in \{\text{Bursty}\} \\
\text{Conservative Baseline}, & \text{if } W_{type} \in \{\text{Volatile}\}
\end{cases}
$$

*   **ARIMA (AutoRegressive Integrated Moving Average)**: Used for linear, stable datasets where statistical properties are constant over time. It is computationally lightweight and effective for short-term trends.
*   **LSTM (Long Short-Term Memory)**: A recurrent neural network employed for complex, non-linear, and bursty patterns. Its ability to retain long-term dependencies makes it superior for unpredictable workloads, albeit at a higher computational cost.
*   **Conservative Baseline**: For highly volatile workloads where prediction accuracy is low, the system defaults to a robust percentile-based provisioning (e.g., $P_{95}$) to prevent under-provisioning.

## C. Cost-Aware Scaling Policy

The final module translates the predicted resource demand $\hat{D}_{t+1}$ into a capacity allocation decision $C_{t+1}$. The objective is to minimize the total operational cost while maintaining strict SLA adherence.

The **Total Cost ($Cost_{total}$)** is modeled as the sum of the compute cost and the penalty for wasted resources:

$$
Cost_{total} = \sum_{t=1}^{T} (C_t \cdot P_{compute}) + \sum_{t=1}^{T} \max(0, C_t - D_t) \cdot P_{waste}
$$

Where:
*   $C_t$ is the allocated capacity at time $t$.
*   $D_t$ is the actual demand at time $t$.
*   $P_{compute}$ is the unit price of the resource.
*   $P_{waste}$ represents the implicit cost of over-provisioning (efficiency loss).

**SLA Constraint**: To ensure reliability, the system incorporates a safety margin $\delta$ (determined by the cluster volatility) such that:

$$
C_{t+1} = \hat{D}_{t+1} \cdot (1 + \delta)
$$

This ensures that even if the prediction slightly underestimates the load, the allocated capacity remains sufficient to prevent SLA violations (where $D_t > C_t$).

## D. Optimization Objective

The primary objective of the proposed framework is to minimize the total cost of resource provisioning while satisfying strict Service Level Agreements (SLAs). We formulate this as a constrained optimization problem where the goal is to determine the optimal capacity allocation $C_t$ for each time interval $t$ that minimizes the aggregate compute cost, subject to the constraint that the probability of resource contention (demand $D_t$ exceeding capacity $C_t$) remains below a predefined tolerance threshold $\epsilon$. This ensures that the system maximizes economic efficiency by reducing resource wastage without compromising application performance or reliability.

Mathematically, the objective function is defined as:

$$
\min_{C_t} \sum_{t=1}^{T} Cost(C_t) \quad \text{s.t.} \quad P(D_t > C_t) \le \epsilon
$$

where $Cost(C_t)$ represents the monetary cost of the allocated resources and $\epsilon$ denotes the maximum allowable SLA violation rate (e.g., 0.01%).
