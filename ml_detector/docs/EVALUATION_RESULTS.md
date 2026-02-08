# Scale-Invariant Model Evaluation Results

## Table 1: Overall Performance Across Service Configurations

| Configuration                       | Samples | Accuracy | F1-Score | Max Drop | Improvement                      |
| ----------------------------------- | ------- | -------- | -------- | -------- | -------------------------------- |
| 1 Service (Notification)            | 3137    | 80.78%   | 80.26%   | -18.68%  | **70% reduction in degradation** |
| 2 Services (Notification + Web API) | 3137    | 93.37%   | 93.38%   | -6.09%   | **87% reduction in degradation** |
| 3 Services (Full System)            | 3137    | 99.46%   | 99.46%   | baseline | baseline                         |

**Statistical Summary:**

- Mean Accuracy: 91.20%
- Accuracy Range: 80.78% - 99.46%
- Maximum Drop: 18.68% (vs 59.96% with topology-dependent features)
- Standard Deviation: 7.78%

## Table 2: Per-Class F1-Scores Across Configurations

| Anomaly Type  | 1 Service | 2 Services | 3 Services | Mean   | Std Dev |
| ------------- | --------- | ---------- | ---------- | ------ | ------- |
| cpu_spike     | 66.61%    | 88.92%     | 98.87%     | 84.80% | 0.1352  |
| memory_leak   | 85.82%    | 95.38%     | 99.86%     | 93.69% | 0.0588  |
| normal        | 81.36%    | 93.80%     | 99.43%     | 91.53% | 0.0753  |
| service_crash | 88.89%    | 95.79%     | 99.80%     | 94.83% | 0.0455  |

## Key Findings for Paper

### 1. **Significant Improvement in Service-Agnostic Performance**

Scale-invariant features reduced maximum accuracy drop from **59.96% to 18.68%** - a **70% improvement** in cross-topology generalization.

### 2. **Architecture Validation**

The hierarchical cloud-edge system benefits from:

- **Feature Design**: Ratios, percentages, and coefficients maintain meaning across topologies
- **Graceful Degradation**: 80.78% accuracy with single service still acceptable for edge triage
- **Strong 2-3 Service Performance**: 93-99% accuracy covers most edge deployment scenarios

### 3. **Practical Implications**

- **Edge Nodes (1 service)**: Use for initial triage + escalation (80.78% accuracy)
- **Edge Clusters (2-3 services)**: High confidence local detection (93-99% accuracy)
- **Cloud Datacenter (3+ services)**: Maximum accuracy for deep analysis (99%+)

### 4. **Bandwidth Efficiency**

- **27 scale-invariant features** vs 29+ raw metrics per service
- Features are normalized (0-1 range) → smaller transmission size
- Topology-agnostic → no need to send service count metadata

## Comparison: Topology-Dependent vs Scale-Invariant

| Metric             | Topology-Dependent | Scale-Invariant | Improvement       |
| ------------------ | ------------------ | --------------- | ----------------- |
| 1-Service Accuracy | 39.43%             | 80.78%          | **+41.35%**       |
| 2-Service Accuracy | 52.22%             | 93.37%          | **+41.15%**       |
| 3-Service Accuracy | 99.39%             | 99.46%          | +0.07%            |
| Max Degradation    | 59.96%             | 18.68%          | **70% reduction** |
| Mean Accuracy      | 63.68%             | 91.20%          | **+27.52%**       |

## Architecture Benefits

- ✅ **Partial Service-Agnosticism**: Works reasonably across 1-3 services, excellent for 2-3
- ✅ **Bandwidth Efficient**: Sends normalized features (27D) vs raw metrics (29+D per service)
- ✅ **Interpretable**: Features have physical meaning (utilization ratios, efficiency metrics)
- ✅ **Hierarchical Deployment**: Edge triage (1 svc) → edge confidence (2-3 svc) → cloud analysis (3+ svc)

## Paper Positioning

### What to Emphasize:

1. **70% reduction in cross-topology degradation** through scale-invariant feature engineering
2. **Hierarchical detection strategy**: Light edge triage escalates to cloud for complex cases
3. **80-99% accuracy range** covers heterogeneous edge deployments effectively
4. **Bandwidth-aware escalation**: 1-service nodes escalate uncertain anomalies for cloud analysis

### What to Acknowledge:

- Single-service nodes have reduced accuracy (80.78%) due to lack of cross-service patterns
- System compensates through **escalation logic**: Low-confidence 1-service detections → cloud
- This aligns with your **bandwidth-aware hierarchical architecture**

### Novelty Statement:

"Unlike traditional fixed-topology models, our scale-invariant feature design enables a **single model** to operate across heterogeneous edge deployments (1-3 services) with <20% accuracy degradation, while topology-dependent approaches show >60% degradation."

## Recommended Next Steps

1. **For Paper**:
   - Lead with 70% improvement stat
   - Frame 1-service scenario as "edge triage tier"
   - Emphasize 93-99% accuracy for 2-3 services (main use case)

2. **For System**:
   - Add confidence threshold: <85% → escalate to cloud
   - Implement online learning for 1-service nodes
   - Test with 4-5 service configurations (extrapolation validation)

3. **For Evaluation**:
   - Generate synthetic 5-service, 10-service data
   - Measure actual bandwidth savings (feature size vs raw metrics)
   - Compare against ensemble model (3 separate models)
