# Integration Guide: Connecting Evaluation to Live System

This guide shows how to integrate the recovery evaluation framework with your running Kubernetes system.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATION FRAMEWORK                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Anomaly Injection          2. Detection Monitoring          │
│     ├─ trigger_anomaly.py          ├─ EWMA detector logs       │
│     └─ simulate-critical           └─ Prometheus metrics       │
│                                                                 │
│  3. Classification & RCA       4. Recovery Execution            │
│     ├─ XGBoost detector            ├─ Enhanced orchestrator     │
│     └─ SHAP explainer              └─ Kubernetes API           │
│                                                                 │
│  5. Metrics Collection         6. Result Storage               │
│     ├─ Timestamps                  ├─ CSV (raw data)           │
│     ├─ Pre/post metrics            ├─ JSON (aggregates)        │
│     └─ Normalization check         └─ PNG (visualizations)     │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

### For Simulated Evaluation ✅

- Python 3.8+
- numpy, pandas, matplotlib
- No external services required

### For Live Evaluation ⚠️

#### 1. Kubernetes Cluster

```bash
# Check cluster is running
kubectl cluster-info
kubectl get nodes

# Verify namespace exists
kubectl get namespace journal-implementation
```

#### 2. Notification Service

```bash
# Check service is running
kubectl get pods -n journal-implementation -l app=notification-service

# Verify service endpoint
curl http://localhost:8003/health

# Port-forward if needed
kubectl port-forward -n journal-implementation svc/notification-service 8003:8003
```

#### 3. Prometheus

```bash
# Check Prometheus is running
curl http://localhost:9090/-/healthy

# Verify metrics are being collected
curl http://localhost:9090/api/v1/query?query=up

# Port-forward if needed
kubectl port-forward -n monitoring svc/prometheus 9090:9090
```

#### 4. Trained XGBoost Model

```bash
# Verify model exists
ls ml_detector/models/anomaly_detector_scaleinvariant.pkl

# If not, train it first
cd ml_detector/scripts
python train_scaleinvariant_model.py
```

## Integration Steps

### Step 1: Install Dependencies

```bash
cd results/recovery_evaluation
pip install -r requirements.txt
```

### Step 2: Configure Endpoints

Edit the configuration in `comprehensive_evaluator.py` if your endpoints differ:

```python
SERVICE_URL = "http://localhost:8003"  # Notification service
PROMETHEUS_URL = "http://localhost:9090"  # Prometheus
```

### Step 3: Run Evaluation

#### Quick Test (Simulated)

```bash
python simulated_evaluator.py --injections 5 --seed 42
```

#### Full Evaluation (Live System)

```bash
python comprehensive_evaluator.py \
    --injections 20 \
    --model ../../ml_detector/models/anomaly_detector_scaleinvariant.pkl \
    --service-url http://localhost:8003 \
    --prometheus-url http://localhost:9090
```

## Integration with Online Detector

### Option 1: API-Based Integration (Recommended)

Add REST API to online detector for evaluation hooks:

```python
# Add to online_detector/main.py

from flask import Flask, jsonify
import threading

app = Flask(__name__)

@app.route('/detector/status', methods=['GET'])
def get_status():
    """Return current detector state."""
    return jsonify({
        'status': 'running',
        'last_detection': last_detection_time,
        'anomaly_type': current_anomaly_type,
        'confidence': current_confidence
    })

@app.route('/detector/history', methods=['GET'])
def get_history():
    """Return detection history."""
    return jsonify({
        'detections': detection_history
    })

# Run Flask in background thread
def start_api():
    app.run(host='0.0.0.0', port=5001)

api_thread = threading.Thread(target=start_api, daemon=True)
api_thread.start()
```

Then modify evaluator to query this API:

```python
# In comprehensive_evaluator.py

def wait_for_detection(self, timeout=60):
    """Poll detector API for detection."""
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get('http://localhost:5001/detector/status')
            data = response.json()

            if data['last_detection'] > self.injection_time:
                return True, data['anomaly_type'], data['last_detection']
        except:
            pass

        time.sleep(2)

    return False, None, None
```

### Option 2: Log-Based Integration

Monitor detector logs for detection events:

```python
# In comprehensive_evaluator.py

def wait_for_detection(self, timeout=60):
    """Monitor logs for detection."""
    import subprocess

    start_time = time.time()

    while time.time() - start_time < timeout:
        # Read last N lines of detector logs
        try:
            result = subprocess.run(
                ['kubectl', 'logs', '-n', 'journal-implementation',
                 '-l', 'app=online-detector', '--tail=10'],
                capture_output=True,
                text=True
            )

            logs = result.stdout

            # Parse logs for detection
            if 'ANOMALY DETECTED' in logs:
                # Extract timestamp and type
                detection_time = time.time()
                anomaly_type = self._extract_anomaly_type(logs)
                return True, anomaly_type, detection_time
        except:
            pass

        time.sleep(2)

    return False, None, None
```

### Option 3: Event-Driven Integration

Use Kubernetes events or custom CRDs:

```python
# Create custom event when anomaly detected

from kubernetes import client

def emit_detection_event(anomaly_type, confidence):
    """Emit Kubernetes event for detection."""
    v1 = client.CoreV1Api()

    event = client.V1Event(
        metadata=client.V1ObjectMeta(
            name=f'anomaly-detected-{int(time.time())}',
            namespace='journal-implementation'
        ),
        involved_object=client.V1ObjectReference(
            kind='Pod',
            name='online-detector',
            namespace='journal-implementation'
        ),
        reason='AnomalyDetected',
        message=f'Anomaly detected: {anomaly_type} (confidence: {confidence:.2f})',
        type='Warning'
    )

    v1.create_namespaced_event(namespace='journal-implementation', body=event)
```

Then watch events in evaluator:

```python
def wait_for_detection(self, timeout=60):
    """Watch Kubernetes events for detection."""
    from kubernetes import client, watch

    v1 = client.CoreV1Api()
    w = watch.Watch()

    start_time = time.time()

    for event in w.stream(
        v1.list_namespaced_event,
        namespace='journal-implementation',
        timeout_seconds=timeout
    ):
        if event['object'].reason == 'AnomalyDetected':
            detection_time = time.time()
            anomaly_type = self._parse_anomaly_type(event['object'].message)
            w.stop()
            return True, anomaly_type, detection_time

        if time.time() - start_time > timeout:
            w.stop()
            break

    return False, None, None
```

## Enhanced Recovery Orchestrator Integration

The enhanced orchestrator is already compatible with live systems:

```python
from enhanced_orchestrator import EnhancedRecoveryOrchestrator

# Initialize
orchestrator = EnhancedRecoveryOrchestrator(
    namespace='journal-implementation',
    app_label='notification-service'
)

# Execute recovery with full tracking
metrics = orchestrator.execute_recovery(
    recovery_action='pod_restart',
    anomaly_type='cpu_spike',
    root_cause_service='notification-service',
    detection_timestamp=detection_time,
    explanation_timestamp=explanation_time
)

# Metrics object contains all timestamps
print(f"MTTR: {metrics.mttr}s")
print(f"Pod restart duration: {metrics.pod_restart_duration}s")
print(f"Success: {metrics.recovery_success}")
```

## Monitoring Live Evaluation

### Terminal 1: Online Detector

```bash
cd online_detector
python -m online_detector.main
```

### Terminal 2: Recovery Orchestrator (if running as service)

```bash
cd recovery-orchestrator
python enhanced_orchestrator.py
```

### Terminal 3: Evaluation Runner

```bash
cd results/recovery_evaluation
python comprehensive_evaluator.py --injections 10
```

### Terminal 4: Monitoring (optional)

```bash
# Watch pods
watch kubectl get pods -n journal-implementation

# Watch events
kubectl get events -n journal-implementation --watch

# Watch logs
kubectl logs -f -n journal-implementation -l app=online-detector
```

## Troubleshooting

### Issue: Detection not triggering

**Cause:** Anomaly injection may not be strong enough

**Solution:**

```bash
# Manually verify injection works
curl -X POST http://localhost:8003/simulate-critical

# Check metrics increase
curl "http://localhost:9090/api/v1/query?query=container_cpu_usage_seconds_total"
```

### Issue: Recovery not executing

**Cause:** Kubernetes permissions or orchestrator not running

**Solution:**

```bash
# Check RBAC permissions
kubectl auth can-i delete pods -n journal-implementation

# Verify orchestrator can connect
kubectl get pods -n journal-implementation
```

### Issue: Metrics not normalizing

**Cause:** Recovery action may not be appropriate for anomaly type

**Solution:**

- Check anomaly persists after pod restart
- May need different recovery action (scaling, resource adjustment)
- Verify anomaly was actually injected correctly

## Performance Tuning

### Reduce Evaluation Time

```python
# Reduce wait times
DETECTION_TIMEOUT = 30  # Instead of 60
NORMALIZATION_TIMEOUT = 60  # Instead of 120
RECURRENCE_WINDOW = 180  # Instead of 300

# Reduce interval between injections
INTER_INJECTION_DELAY = 10  # Instead of 30
```

### Increase Accuracy

```python
# Longer timeouts for detection
DETECTION_TIMEOUT = 120

# More aggressive anomaly injection
INJECTION_INTENSITY = 'high'  # More CPU/memory
INJECTION_DURATION = 60  # Longer anomaly period
```

## Results Storage

All evaluation results are automatically saved to:

```
results/recovery_evaluation/
├── simulated_evaluation_TIMESTAMP.csv      # Raw data
├── simulated_metrics_TIMESTAMP.json        # Aggregate metrics
├── evaluation_plots_TIMESTAMP.png          # Visualizations
├── RESULTS_SUMMARY.md                      # Human-readable summary
└── evaluation_log_TIMESTAMP.txt            # Detailed logs (optional)
```

## Next Steps

1. **Baseline Comparison:** Implement rule-based recovery and compare
2. **Stress Testing:** Multiple concurrent anomalies
3. **Edge Deployment:** Test on resource-constrained nodes
4. **Multi-Service:** Test cascading failures
5. **Publication:** Use results for paper/presentation

## Getting Help

If you encounter issues:

1. Check Prerequisites section
2. Review Troubleshooting section
3. Run simulated evaluation first to verify framework works
4. Check logs for specific error messages
5. Verify all services are accessible

## Example Full Workflow

```bash
# 1. Start all services
kubectl apply -f k8s/
kubectl port-forward -n monitoring svc/prometheus 9090:9090 &
kubectl port-forward -n journal-implementation svc/notification-service 8003:8003 &

# 2. Verify services
curl http://localhost:9090/-/healthy
curl http://localhost:8003/health

# 3. Run quick test (simulated)
cd results/recovery_evaluation
python simulated_evaluator.py --injections 5

# 4. Run live evaluation
python comprehensive_evaluator.py --injections 10

# 5. View results
python run_evaluation.py  # Select option 3

# 6. Generate report
cat RESULTS_SUMMARY.md
```

## References

- Enhanced Orchestrator: `recovery-orchestrator/enhanced_orchestrator.py`
- Comprehensive Evaluator: `results/recovery_evaluation/comprehensive_evaluator.py`
- Simulated Evaluator: `results/recovery_evaluation/simulated_evaluator.py`
- Online Detector: `online_detector/main.py`
- XGBoost Detector: `ml_detector/scripts/dual_feature_detector.py`
- SHAP Explainer: `ml_detector/scripts/explainability_layer.py`
