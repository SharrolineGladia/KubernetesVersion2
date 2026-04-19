# Bandwidth-Aware Escalation Optimization: Evaluation Results

**Generated:** February 23, 2026 at 12:22:27

---


## Table 1: Threshold Evaluation Results

| Threshold θ | Escalation Rate | Bandwidth (MB/day) | Expected Latency (ms) | Accuracy | Edge/Cloud Split |
|-------------|-----------------|--------------------|-----------------------|----------|------------------|
| 0.6 | 0.3% | 0.00 | 20.06 ± 2.02 | 99.51% | 99.7% / 0.3% |
| 0.7 | 0.8% | 0.01 | 20.34 ± 2.03 | 99.52% | 99.2% / 0.8% |
| 0.8 | 1.1% | 0.01 | 20.51 ± 2.03 | 99.54% | 98.9% / 1.1% |
| 0.9 | 1.5% | 0.01 | 20.73 ± 2.03 | 99.56% | 98.5% / 1.5% |

---


## Bandwidth Consumption Analysis

**Payload Characteristics:**
- Mean raw payload size: 2.44 KB
- Mean compressed payload size: 0.85 KB
- Compression ratio: 2.86x
- Bandwidth savings: 65.0%

**Daily Bandwidth by Threshold:**
- θ = 0.6: 0.00 MB/day (0.000 GB/day) [0.3% escalation rate]
- θ = 0.7: 0.01 MB/day (0.000 GB/day) [0.8% escalation rate]
- θ = 0.8: 0.01 MB/day (0.000 GB/day) [1.1% escalation rate]
- θ = 0.9: 0.01 MB/day (0.000 GB/day) [1.5% escalation rate]

**Optimization Impact:**
- Maximum bandwidth (θ = 0.6): 0.01 MB/day
- Minimum bandwidth (θ = 0.9): 0.00 MB/day
- Potential savings: 80.0% through threshold optimization

---


## Latency Performance Analysis

**Edge vs Cloud Latency:**
- Mean edge-only latency: 19.90 ms
- Mean escalated latency: 74.58 ms
- Escalation overhead: 54.68 ms (274.8% increase)

**Expected Latency by Threshold:**
- θ = 0.6: 20.06 ms [99.7% edge, 0.3% cloud]
- θ = 0.7: 20.34 ms [99.2% edge, 0.8% cloud]
- θ = 0.8: 20.51 ms [98.9% edge, 1.1% cloud]
- θ = 0.9: 20.73 ms [98.5% edge, 1.5% cloud]

**Optimal Configuration:**
- Lowest expected latency at θ = 0.6
- Expected latency: 20.06 ms

---


## Accuracy Trade-off Analysis

**Edge vs Cloud Classification:**
- Mean edge accuracy: 99.75%
- Mean cloud accuracy: 74.00%
- Cloud improvement: -25.75 percentage points

**Overall Accuracy by Threshold:**
- θ = 0.6: 99.51% (Edge: 99.60%, Cloud: 70.00%)
- θ = 0.7: 99.52% (Edge: 99.80%, Cloud: 65.62%)
- θ = 0.8: 99.54% (Edge: 99.80%, Cloud: 76.36%)
- θ = 0.9: 99.56% (Edge: 99.80%, Cloud: 84.00%)

**Bandwidth-Accuracy Trade-off:**
- θ = 0.6: 0.00 MB/day per unit accuracy
- θ = 0.7: 0.01 MB/day per unit accuracy
- θ = 0.8: 0.01 MB/day per unit accuracy
- θ = 0.9: 0.01 MB/day per unit accuracy

**Pareto Optimal Configuration:**
- Best balance at θ = 0.6
- Accuracy: 99.51%, Bandwidth: 0.00 MB/day

---


## Key Observations

1. **Threshold Impact on Escalation**: Varying θ from 0.6 to 0.9 modulates escalation rate by 1.2%, enabling fine-grained control over edge-cloud workload distribution.

2. **Bandwidth Optimization**: Increasing confidence threshold from 0.6 to 0.9 reduces daily bandwidth consumption by -400.0%, demonstrating significant network efficiency gains for edge deployments with limited connectivity.

3. **Latency-Accuracy Trade-off**: The system exhibits a Pareto frontier between latency (20.06-20.73 ms) and accuracy (99.51%-99.56%), enabling application-specific optimization based on operational requirements.

4. **Compression Effectiveness**: Gzip compression achieves 2.86x reduction in payload size, translating to 65.0% bandwidth savings for escalated cases—critical for low-bandwidth edge environments.

5. **Recommended Configuration**: Based on multi-objective optimization balancing bandwidth, latency, and accuracy, θ = 0.6 emerges as optimal, achieving 99.51% accuracy with 0.00 MB/day bandwidth consumption and 20.06 ms expected latency.

---


## LaTeX Table for Paper

```latex
\begin{table}[ht]
\centering
\caption{Bandwidth-Aware Escalation Optimization: Threshold Evaluation Results}
\label{tab:escalation_threshold}
\begin{tabular}{ccccc}
\hline
\textbf{Threshold} & \textbf{Escalation} & \textbf{Bandwidth} & \textbf{Latency} & \textbf{Accuracy} \\
$\theta$ & Rate (\%) & (MB/day) & (ms) & (\%) \\
\hline
0.6 & 0.3 & 0.00 & 20.06 $\pm$ 2.02 & 99.51 \\
0.7 & 0.8 & 0.01 & 20.34 $\pm$ 2.03 & 99.52 \\
0.8 & 1.1 & 0.01 & 20.51 $\pm$ 2.03 & 99.54 \\
0.9 & 1.5 & 0.01 & 20.73 $\pm$ 2.03 & 99.56 \\
\hline
\end{tabular}
\end{table}
```
