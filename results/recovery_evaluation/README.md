# Recovery Evaluation Module

This module provides comprehensive evaluation of the Kubernetes anomaly detection and recovery framework.

## Components

### 1. `enhanced_orchestrator.py`

Enhanced recovery orchestrator with detailed performance tracking:

- Timestamps for each stage of recovery
- Success/failure tracking
- Kubernetes action latency measurement
- Pod restart duration monitoring

### 2. `comprehensive_evaluator.py`

Full evaluation framework for live Kubernetes environments:

- Real anomaly injection via notification service
- Prometheus metrics monitoring
- End-to-end recovery tracking
- **Requires**: Running Kubernetes cluster, Prometheus, notification service

### 3. `simulated_evaluator.py` ⭐ **Recommended for initial testing**

Simulated evaluation that doesn't require Kubernetes:

- Realistic timing models
- Reproducible results
- Fast execution
- Statistical analysis
- **Use this first** to validate evaluation logic

## Quick Start

### Run Simulated Evaluation (No Kubernetes Required)

```bash
cd results/recovery_evaluation
python simulated_evaluator.py --injections 20 --seed 42
```

This will generate:

- `simulated_evaluation_TIMESTAMP.csv` - Raw data
- `simulated_metrics_TIMESTAMP.json` - Aggregate metrics
- `evaluation_plots_TIMESTAMP.png` - Comprehensive visualizations

### Run Live Evaluation (Requires Kubernetes + Services)

```bash
# Ensure Kubernetes cluster, Prometheus, and notification service are running
python comprehensive_evaluator.py --injections 20 --model ../../ml_detector/models/anomaly_detector_scaleinvariant.pkl
```

## Metrics Collected

### 1️⃣ Recovery Execution Metrics (Orchestrator Level)

- **Recovery Success Rate (%)**: % of recovery actions that resolved anomaly
- **Mean Time To Recovery (MTTR)**: From detection → metrics normalized
- **Recovery Execution Latency**: Time from trigger → Kubernetes action executed
- **Pod Restart Duration**: Time from restart command → pod Ready state
- **Recurrence Rate (%)**: % of anomalies reappearing within 5 minutes
- **Detection Latency**: Time from injection → detection

### 2️⃣ Counterfactual-Guided Recovery Impact (Novel Contribution)

- **Anomaly Probability Reduction**: Model confidence before vs after recovery
- **Risk Score Reduction**: Risk assessment change
- **Action Ranking Change Rate**: Impact of counterfactual reasoning

### 3️⃣ End-to-End System Performance

- **End-to-End Latency**: Detection → Recovery completion
- **Autonomous Resolution Rate**: % anomalies resolved without human intervention
- **Framework Resource Overhead**: CPU, memory usage

## Output Files

### Raw Data (CSV)

Contains per-injection records with all timestamps and metrics:

- injection_id, anomaly_type
- All timestamps (injection, detection, recovery, etc.)
- Pre/post recovery metrics
- Calculated latencies
- Success/failure flags

### Aggregate Metrics (JSON)

Summary statistics:

```json
{
  "total_injections": 20,
  "detection_success_rate": 95.0,
  "recovery_success_rate": 98.0,
  "mean_mttr": 25.5,
  "mean_anomaly_prob_reduction": 0.75,
  ...
}
```

### Visualizations (PNG)

Comprehensive dashboard with:

1. MTTR distribution histogram
2. Anomaly probability reduction histogram
3. Success rates by anomaly type
4. End-to-end latency over time
5. Pod restart duration distribution
6. Detection latency distribution
7. CPU before/after recovery
8. Memory before/after recovery
9. Summary statistics table

## Evaluation Pipeline

```
1. Inject Anomaly → 2. Wait for Detection → 3. Generate Explanation
                                                      ↓
5. Check Recurrence ← 4. Trigger Recovery ← Recovery Decision
         ↓
6. Record Metrics & Generate Reports
```

## Configuration

Edit constants in the evaluator files:

```python
# Number of injections
NUM_INJECTIONS = 20

# Anomaly types to test
ANOMALY_TYPES = ["cpu_spike", "memory_leak", "service_crash"]

# Normalization thresholds
NORMALIZATION_THRESHOLD = {
    'cpu_percent': 80.0,
    'memory_percent': 75.0,
    'error_rate': 0.1,
    'response_time_p95': 200.0
}

# Recurrence monitoring window
RECURRENCE_WINDOW = 300  # 5 minutes
```

## Example Usage

### Python API

```python
from simulated_evaluator import SimulatedRecoveryEvaluator

# Initialize
evaluator = SimulatedRecoveryEvaluator(seed=42)

# Run evaluation
df = evaluator.run_evaluation(num_injections=20)

# Generate visualizations
evaluator.generate_visualizations()

# Print report
evaluator.print_summary_report()

# Save results
evaluator.save_results()
```

### Command Line

```bash
# Run with custom parameters
python simulated_evaluator.py --injections 50 --seed 123

# Run live evaluation
python comprehensive_evaluator.py \
    --injections 20 \
    --model ../../ml_detector/models/anomaly_detector_scaleinvariant.pkl \
    --service-url http://localhost:8003 \
    --prometheus-url http://localhost:9090
```

## Interpreting Results

### Key Metrics for Publication

1. **MTTR < 30s**: Shows fast recovery
2. **Recovery Success Rate > 95%**: High reliability
3. **Anomaly Prob Reduction > 0.70**: Effective recovery
4. **Recurrence Rate < 10%**: Sustainable fixes
5. **Autonomous Resolution > 90%**: Minimal human intervention

### Expected Ranges (Based on Simulation)

- Detection Latency: 5-15s
- MTTR: 20-50s
- End-to-End Latency: 30-80s
- Pod Restart: 10-30s
- Anomaly Prob Reduction: 0.60-0.90

## Troubleshooting

### Simulated Evaluator

- ✅ No external dependencies
- ✅ Always works
- ⚠️ Results are synthetic (but realistic)

### Live Evaluator

- ❌ Requires running services
- ❌ May fail if services unavailable
- ✅ Real-world results
- Check: Kubernetes cluster, Prometheus, notification service

## Integration with Existing Code

This module integrates with:

- `recovery-orchestrator/orchestrator.py` - Recovery actions
- `ml_detector/scripts/dual_feature_detector.py` - Anomaly classification
- `ml_detector/scripts/explainability_layer.py` - RCA generation
- `anomaly-trigger/trigger_anomaly.py` - Anomaly injection
- `online_detector/` - Detection pipeline

## Future Enhancements

- [ ] Resource overhead monitoring (framework CPU/memory)
- [ ] Multi-anomaly stress testing
- [ ] Bandwidth-constrained scenarios
- [ ] Edge vs cloud handling split
- [ ] Cascading failure detection
- [ ] Action ranking change rate tracking
- [ ] Integration with online detector logs
- [ ] Real-time dashboard

## Citation

If you use this evaluation framework in your research, please cite:

```
[Your Paper Title]
[Authors]
[Conference/Journal]
```
