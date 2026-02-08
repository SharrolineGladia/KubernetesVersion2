# Dual-Feature Detection Architecture

## Overview

This directory contains the **dual-feature architecture** that solves a critical challenge in cloud-edge anomaly detection:

**Challenge**: How do you detect anomalies across variable service topologies (1-10 services) while still providing actionable, service-specific explanations?

**Solution**: Two-phase architecture

1. **Phase 1 - Detection**: Use 27 scale-invariant features for topology-agnostic anomaly classification
2. **Phase 2 - Explainability**: Use service-specific metrics for root cause analysis and recommendations

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ RAW SERVICE METRICS (per-service granularity)              │
│ notification: {cpu:95, memory:78, requests:150, ...}       │
│ web_api: {cpu:45, memory:56, requests:120, ...}            │
│ processor: {cpu:32, memory:48, requests:30, ...}           │
└──────────────────────┬──────────────────────────────────────┘
                       │
         ┌─────────────┴────────────┐
         │                          │
    ┌────▼────────────┐     ┌──────▼─────────┐
    │ TRANSFORM TO    │     │ PRESERVE RAW   │
    │ SCALE-INVARIANT │     │ SERVICE DATA   │
    │ (27 features)   │     │ (8×N features) │
    └────┬────────────┘     └──────┬─────────┘
         │                          │
    ┌────▼──────────────────────────▼─────────┐
    │ UNIFIED SNAPSHOT                        │
    │ • features_scaleinvariant (detection)   │
    │ • service_metrics (explainability)      │
    │ • metadata (topology, timestamp)        │
    └────┬────────────────────────────────────┘
         │
         ├───────────────┬──────────────────┐
         │               │                  │
    ┌────▼────┐    ┌────▼────┐      ┌─────▼──────┐
    │ XGBoost │    │ Stored  │      │ Metadata   │
    │ Predict │    │ for RCA │      │ (topology) │
    └────┬────┘    └────┬────┘      └─────┬──────┘
         │               │                  │
   [Anomaly Type]       │                  │
    cpu_spike            │                  │
    memory_leak          │                  │
    service_crash        │                  │
         │               │                  │
         └───────┬───────┴──────────────────┘
                 │
            [If Anomaly]
                 │
         ┌───────▼────────┐
         │ EXPLAINABILITY │
         │ • SHAP values  │
         │ • RCA analysis │
         │ • Service ID   │
         │ • Remediation  │
         └────────────────┘
```

---

## File Structure

```
scripts/
├── explainability_layer.py          # RCA engine (service-level analysis)
├── dual_feature_detector.py         # Unified detector (detection + RCA)
├── online_detector_integration.py   # Integration with EWMA detector
├── demo_dual_feature.py             # Complete workflow demo
├── transform_dataset_scaleinvariant.py  # Dataset transformation
├── train_scaleinvariant_model.py    # Model training
├── create_eval_scaleinvariant.py    # Evaluation dataset generation
├── evaluate_scaleinvariant_model.py # Cross-topology evaluation
├── test_extrapolation.py            # 4-5 service extrapolation test
└── test_10_services.py              # 10-service stress test
```

---

## Module Descriptions

### 1. `explainability_layer.py`

**Purpose**: Root Cause Analysis (RCA) using service-specific metrics

**Key Classes**:

- `ServiceMetrics`: Container for per-service metrics
- `RCAResult`: Structured RCA output with root cause, severity, recommendations
- `AnomalyExplainer`: Main RCA engine

**Features**:

- **Service Attribution**: Identifies which service caused the anomaly
- **Factor Analysis**: Determines contributing metrics (CPU, memory, errors, etc.)
- **Severity Calculation**: Classifies as low/medium/high/critical
- **Actionable Recommendations**: Generates recovery steps (scale, restart, profile, etc.)
- **SHAP Integration**: (Optional) Explains model decisions using SHAP values

**Example**:

```python
from explainability_layer import AnomalyExplainer, ServiceMetrics

explainer = AnomalyExplainer(model_path='../models/anomaly_detector_scaleinvariant.pkl')

service_metrics = {
    'notification': ServiceMetrics(
        service_name='notification',
        cpu=95, memory=78, network_in=50, network_out=45,
        disk_io=30, requests=120, errors=8, latency=250
    )
}

rca_result = explainer.explain_anomaly(
    anomaly_type='cpu_spike',
    service_metrics=service_metrics,
    scale_invariant_features={...}
)

print(f"Root cause: {rca_result.root_cause_service}")
print(f"Severity: {rca_result.severity}")
print(f"Recommendations: {rca_result.recommendations}")
```

---

### 2. `dual_feature_detector.py`

**Purpose**: Unified detector combining scale-invariant detection with service-specific RCA

**Key Classes**:

- `DetectionSnapshot`: Unified data structure with both feature sets
- `DualFeatureDetector`: Main detector orchestrator

**Workflow**:

1. Accept raw service metrics (any number of services)
2. Transform to 27 scale-invariant features
3. Classify anomaly type using XGBoost
4. If anomaly detected → perform RCA using service-specific metrics
5. Return unified snapshot with both detection + explanation

**Example**:

```python
from dual_feature_detector import DualFeatureDetector

detector = DualFeatureDetector(
    model_path='../models/anomaly_detector_scaleinvariant.pkl',
    confidence_threshold=0.85
)

# Raw metrics from 2 services
raw_data = {
    'notification': {
        'cpu': 95, 'memory': 78, 'network_in': 50, 'network_out': 45,
        'disk_io': 30, 'requests': 120, 'errors': 8, 'latency': 250
    },
    'web_api': {
        'cpu': 68, 'memory': 65, 'network_in': 70, 'network_out': 62,
        'disk_io': 22, 'requests': 180, 'errors': 10, 'latency': 180
    }
}

# One-shot detection + RCA
snapshot = detector.detect_from_raw(raw_data, perform_rca=True)

print(f"Anomaly: {snapshot.anomaly_type}")
print(f"Confidence: {snapshot.detection_confidence:.1%}")
if snapshot.rca_result:
    print(f"Root cause service: {snapshot.rca_result.root_cause_service}")
```

---

### 3. `online_detector_integration.py`

**Purpose**: Integration layer between EWMA stress detector and XGBoost classifier

**Key Classes**:

- `EnrichedDetector`: Combines EWMA stress detection with XGBoost classification

**Integration Points**:

1. Monitors EWMA stress score
2. Triggers XGBoost classification when stress > threshold
3. Captures service metrics snapshot
4. Performs classification + RCA
5. Logs results for escalation/recovery

**Usage in `online_detector/main.py`**:

```python
from ml_detector.scripts.online_detector_integration import EnrichedDetector

# Initialize
enriched_detector = EnrichedDetector(
    model_path="../ml_detector/models/anomaly_detector_scaleinvariant.pkl",
    stress_threshold=0.6,
    confidence_threshold=0.80
)

# In monitoring loop
while True:
    # Calculate EWMA stress (existing code)
    ewma_stress_score = calculate_stress(...)

    # Trigger classification if stress high
    snapshot = enriched_detector.classify_and_explain(
        stress_score=ewma_stress_score,
        current_time=datetime.utcnow()
    )

    # Handle results
    if snapshot and snapshot.anomaly_type != 'normal':
        if snapshot.detection_confidence >= 0.85:
            trigger_recovery(snapshot.rca_result)
        else:
            escalate_to_cloud(snapshot)
```

---

### 4. `demo_dual_feature.py`

**Purpose**: Comprehensive demonstration of dual-feature architecture

**Scenarios**:

1. **1-service edge node**: Memory leak detection (80.78% accuracy)
2. **2-service edge cluster**: CPU spike detection (93.37% accuracy)
3. **3-service cloud**: Service crash detection (99.46% accuracy)
4. **Feature comparison**: Scale-invariant vs raw metrics
5. **Bandwidth analysis**: Efficiency comparison

**Run**:

```bash
cd ml_detector
python scripts/demo_dual_feature.py
```

---

## Key Concepts

### Why Dual Features?

**Problem**:

- Scale-invariant features enable topology-agnostic detection (works with 1-10 services)
- But they lose service-level granularity needed for actionable RCA

**Solution**:

- **Detection phase**: Use scale-invariant features (27 dimensions, works anywhere)
- **Explainability phase**: Use service-specific metrics (8×N dimensions, detailed RCA)

**Benefits**:

1. ✅ Single model works across all topologies
2. ✅ Actionable insights ("restart notification-service")
3. ✅ Bandwidth efficient (only send raw metrics on escalation)
4. ✅ Interpretable (both high-level + detailed analysis)

---

### Scale-Invariant Features (27 features)

**Categories**:

- **CPU** (4): mean, max, variance_coef, imbalance
- **Memory** (4): mean, max, variance_coef, imbalance
- **Network** (5): in_rate, out_rate, in_variance, out_variance, asymmetry
- **Disk** (2): io_rate, variance_coef
- **Requests** (4): rate, variance, error_rate, error_variance
- **Latency** (3): mean, p95, variance_coef
- **System** (5): stress, efficiency, density, cpu_memory_corr, degradation

**Key Properties**:

- **Normalized**: Values in [0, 1] range (consistent meaning)
- **Ratios**: Scale-free (σ/μ, errors/requests)
- **Percentages**: Inherently normalized
- **Topology-agnostic**: Semantics preserved across 1-10 services

**Example**:

```python
# Raw metrics change meaning with topology
max_cpu_3svc = 95  # "One service spiking"
max_cpu_10svc = 95  # "Maybe normal for heavily loaded cluster"

# Scale-invariant features preserve meaning
cpu_variance_coef = std(cpu)/mean(cpu)  # 0.28 = "28% relative spread"
# Same interpretation regardless of service count!
```

---

### Service-Specific Metrics (8 per service)

**Per-Service Metrics**:

- `cpu`: CPU utilization (0-100%)
- `memory`: Memory usage (0-100%)
- `network_in`: Incoming traffic rate
- `network_out`: Outgoing traffic rate
- `disk_io`: Disk I/O rate
- `requests`: Request throughput
- `errors`: Error count
- `latency`: Response time (ms)

**Usage**:

- **NOT used for detection** (topology-dependent)
- **ONLY used for RCA** (after anomaly detected)
- Enables service-level attribution and recommendations

---

## Bandwidth Efficiency

### Traditional Approach (All Raw Metrics)

- 1 service: 8 metrics = **8 values**
- 3 services: 8×3 = **24 values**
- 10 services: 8×10 = **80 values**

### Dual-Feature Approach

- **Detection**: Always send 27 scale-invariant features
- **Explainability**: Only send raw metrics on escalation (low confidence)

**Effective Bandwidth**:

- 1-3 services: ~Equivalent (small overhead, but enables single model)
- 4-5 services: 30-40% savings (27 vs 32-40)
- 10 services: **66% savings** (27 vs 80)

**Additional Savings**:

- High confidence (85%+) → No escalation → No raw metrics sent
- Edge handles 93-99% locally → Rare cloud communication

---

## Integration Guide

### Step 1: Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost shap
```

### Step 2: Train Model (if not already done)

```bash
cd ml_detector/scripts
python transform_dataset_scaleinvariant.py
python train_scaleinvariant_model.py
```

### Step 3: Test Dual-Feature Detection

```bash
python demo_dual_feature.py
```

### Step 4: Integrate with Online Detector

**Option A: Modify existing `online_detector/main.py`**

Add to imports:

```python
from ml_detector.scripts.online_detector_integration import EnrichedDetector
```

Initialize in `main()`:

```python
enriched_detector = EnrichedDetector(
    model_path="../ml_detector/models/anomaly_detector_scaleinvariant.pkl",
    stress_threshold=0.6,
    confidence_threshold=0.85
)
```

In monitoring loop:

```python
# After calculating EWMA stress
snapshot = enriched_detector.classify_and_explain(
    stress_score=channel_risk_score,
    current_time=timestamp
)

if snapshot and snapshot.anomaly_type != 'normal':
    handle_anomaly(snapshot)
```

**Option B: Standalone enriched detector**

Use `online_detector_integration.py` as standalone:

```bash
python ml_detector/scripts/online_detector_integration.py
```

---

## Testing & Validation

### Unit Tests

Test explainability layer:

```bash
python ml_detector/scripts/explainability_layer.py
```

Test dual-feature detector:

```bash
python ml_detector/scripts/dual_feature_detector.py
```

### Integration Tests

Full workflow demo:

```bash
python ml_detector/scripts/demo_dual_feature.py
```

### Performance Validation

Check model accuracy across topologies:

```bash
cd ml_detector/scripts
python evaluate_scaleinvariant_model.py
```

Results:

- 1 service: 80.78%
- 2 services: 93.37%
- 3 services: 99.46%
- 4 services: 94.87% (extrapolation)
- 5 services: 93.66% (extrapolation)

---

## Troubleshooting

### Error: Model file not found

```bash
# Ensure model exists
ls ml_detector/models/anomaly_detector_scaleinvariant.pkl

# If missing, train model
cd ml_detector/scripts
python transform_dataset_scaleinvariant.py
python train_scaleinvariant_model.py
```

### Error: SHAP library not installed

```bash
pip install shap
```

Note: SHAP is optional for RCA. Basic RCA works without it.

### Error: Import errors

```bash
# Run from project root
cd c:\other drive\sem 7\project\implementation3\demo
python ml_detector/scripts/demo_dual_feature.py
```

---

## Future Enhancements

1. **Online Learning**: Incremental model updates as new topologies observed
2. **Temporal Features**: Add time-series patterns (5-min rolling mean, 1-hr trend)
3. **Multi-Modal SHAP**: Explain both scale-invariant features AND service attribution
4. **Federated RCA**: Distributed explainability across edge-cloud hierarchy
5. **Automated Remediation**: Execute recommendations automatically (scale, restart, etc.)

---

## References

- **Research Documentation**: `../docs/RESEARCH_DOCUMENTATION.md`
- **Evaluation Results**: `../docs/EVALUATION_RESULTS.md`
- **Implementation Summary**: `../docs/IMPLEMENTATION_SUMMARY.md`
- **Model File**: `../models/anomaly_detector_scaleinvariant.pkl`
- **Training Data**: `../datasets/metrics_dataset_scaleinvariant.csv`

---

**Author**: Scale-Invariant Anomaly Detection Research Team  
**Last Updated**: February 8, 2026  
**Version**: 1.0
