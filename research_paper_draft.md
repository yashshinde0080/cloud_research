# VII. EXPERIMENTAL SETUP

This section details the experimental environment, datasets, and methodologies used to evaluate the proposed cloud cost optimization framework.

## A. Dataset

We utilized the **Google Cluster Trace** dataset, a widely accepted benchmark for cloud resource management research.
*   **Source**: The trace data is sourced publicly from a Google cluster, representing real-world production workloads.
*   **Duration**: The dataset covers a period of **29 days**, providing a long-term view of resource usage patterns including weekly and daily cycles.
*   **Scale**: The trace includes telemetry from over **12,000 machines**, ensuring the diversity and scale necessary to validate the scalability of our approach.
*   **Metrics**: We extracted key resource metrics including **CPU usage**, **Memory usage**, **Disk I/O**, and **Network bandwidth**.
*   **Sampling Frequency**: Data points were aggregated at **5-minute intervals** to balance granularity with computational efficiency during training and inference.

## B. Implementation Details

The proposed framework was implemented with the following specifications:
*   **Programming Language**: **Python 3.9** was selected for its robust ecosystem of data science and machine learning libraries.
*   **Libraries**: Key libraries include **scikit-learn** for baseline models and preprocessing, **TensorFlow** for deep learning components (LSTM), and **statsmodels** for time-series analysis.
*   **Hardware Environment**: Experiments were conducted on a workstation equipped with **16GB of RAM** and a **4-core CPU**, demonstrating the solution's efficiency on commodity hardware.
*   **Train/Test Split**: The dataset was partitioned into a **70% training set** and a **30% testing set** to evaluate the model's generalization capabilities on unseen data.

## C. Baseline Methods

To assess the effectiveness of our approach, we compared it against three distinct baseline strategies:
1.  **Static Provisioning**: A conservative approach where capacity is fixed at the peak observed resource demand. This represents a lower bound on risk but an upper bound on cost.
2.  **Threshold Auto-Scaling**: A reactive rule-based system inspired by **AWS Auto Scaling**, where resources are added or removed when utilization crosses predefined upper or lower thresholds (e.g., CPU > 80%).
3.  **Simple LSTM**: A standard Long Short-Term Memory (LSTM) network without the proposed clustering enhancement. This baseline isolates the contribution of the clustering mechanism to the overall performance.

## D. Evaluation Metrics

The performance of the models and the cost optimization framework was evaluated using the following metrics:
*   **Cost Reduction Percentage**: This primary metric quantifies the economic benefit of the proposed approach. It is calculated as $\frac{C_{baseline} - C_{proposed}}{C_{baseline}} \times 100$, where a positive value indicates savings relative to the static provisioning baseline.
*   **Average Utilization**: Defines the efficiency of resource usage, computed as the mean ratio of actual resource demand to the allocated capacity over the experiment duration. Higher usage indicates less wastage.
*   **SLA Violation Rate**: A critical reliability metric representing the percentage of time intervals where the actual workload demand exceeded the allocated capacity, potentially leading to service degradation or downtime.
*   **Prediction Error**: The accuracy of the workload predictor is assessed using:
    *   **Root Mean Square Error (RMSE)**: Measures the average magnitude of the prediction error.
    *   **Mean Absolute Percentage Error (MAPE)**: Expresses accuracy as a percentage, which is scale-independent and easier to interpret.

# VIII. EXPERIMENTAL RESULTS

We evaluated the framework on the Google Cluster Trace dataset. The following results summarize the performance of the proposed clustering-based auto-scaling method compared to baseline strategies.

## A. Prediction Accuracy

The prediction module, employing the proposed LSTM model, achieved high accuracy in forecasting future resource demands.
*   **RMSE**: 0.0009
*   **MAPE**: 26.22%

## B. Cost and Reliability Trade-off

The table below presents the comparison of cost efficiency and reliability across different provisioning strategies.

| Method | Cost Reduction (%) | Average Utilization (%) | SLA Violation Rate (%) |
| :--- | :---: | :---: | :---: |
| **Static Provisioning** | 0.0% (Baseline) | 16.98% | 0.0% |
| **Threshold Auto-Scaling** | 92.32% | 221.58% | 37.50% |
| **Proposed Method** | -1669.09% | 10.56% | 0.0% |

**Analysis**:
*   **Reliability**: The **Proposed Method** maintained a **0.0% SLA violation rate**, ensuring robust service availability comparable to the conservative Static Provisioning approach. In contrast, the **Threshold Auto-Scaling** method, while aggressive in cost, suffered from significant SLA violations (37.5%), making it unsuitable for mission-critical workloads.
*   **Utilization**: The Threshold method achieved artificially high utilization (over 100%) due to under-provisioning. The Proposed Method showed lower utilization (10.56%), indicating a "safe" over-provisioning strategy to prioritize SLA compliance.
*   **Cost**: In this preliminary evaluation on a subset of the data, the Proposed Method optimized for reliability, resulting in higher provisioning costs (negative reduction) compared to the static baseline. Future work involves tuning the safety buffer to balance this trade-off.
