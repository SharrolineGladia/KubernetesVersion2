# Online Detector Configuration

## Quick Start

The detector works with **sensible defaults** out of the box. For custom deployments, override thresholds using environment variables.

---

## Configuration Values

All thresholds can be customized via environment variables:

### Resource Saturation (CPU/Memory/Threads)
```bash
CPU_SOFT_LIMIT=80.0              # % CPU baseline (default: 80)
CPU_HARD_LIMIT=95.0              # % CPU critical (default: 95)
MEMORY_SOFT_LIMIT=80.0           # % Memory baseline (default: 80)
MEMORY_HARD_LIMIT=95.0           # % Memory critical (default: 95)
THREAD_SOFT_LIMIT=200            # Thread count baseline (default: 200)
THREAD_HARD_LIMIT=500            # Thread count critical (default: 500)
```

### Performance Degradation (Response Time)
```bash
P95_RESPONSE_TIME_BASELINE=150   # ms - Normal operation (default: 150)
P95_RESPONSE_TIME_STRESSED=300   # ms - Degraded state (default: 300)
P95_RESPONSE_TIME_CRITICAL=500   # ms - Critical state (default: 500)
```

### Backpressure/Overload (Queue Depth)
```bash
QUEUE_DEPTH_BASELINE=10          # Normal queue size (default: 10)
QUEUE_DEPTH_STRESSED=50          # Degraded queue size (default: 50)
QUEUE_DEPTH_CRITICAL=100         # Critical queue size (default: 100)
```

### EWMA Parameters
```bash
EWMA_ALPHA=0.05                  # Smoothing factor (default: 0.05)
Z_MAX=6.0                        # Max Z-score for normalization (default: 6.0)
```

---

## How to Use

### Local Development
Just run with defaults:
```bash
python -m online_detector.main
```

### Kubernetes Deployment

**1. Create ConfigMap for your service type:**

```yaml
# High-throughput service (relaxed thresholds)
apiVersion: v1
kind: ConfigMap
metadata:
  name: detector-config-high-throughput
data:
  P95_RESPONSE_TIME_BASELINE: "300"
  P95_RESPONSE_TIME_CRITICAL: "1000"
  QUEUE_DEPTH_BASELINE: "50"
  QUEUE_DEPTH_CRITICAL: "200"
```

```yaml
# Latency-sensitive service (strict thresholds)
apiVersion: v1
kind: ConfigMap
metadata:
  name: detector-config-latency-sensitive
data:
  P95_RESPONSE_TIME_BASELINE: "50"
  P95_RESPONSE_TIME_CRITICAL: "200"
  QUEUE_DEPTH_BASELINE: "5"
  QUEUE_DEPTH_CRITICAL: "20"
```

**2. Reference in Deployment:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: online-detector
spec:
  template:
    spec:
      containers:
      - name: detector
        image: your-registry/online-detector:latest
        envFrom:
        - configMapRef:
            name: detector-config-high-throughput  # Choose appropriate ConfigMap
```

**3. Apply:**
```bash
kubectl apply -f configmap.yaml
kubectl apply -f deployment.yaml
```

---

## Determining Thresholds

### Step 1: Measure Baseline
Query Prometheus for historical p95 latency over 1-2 weeks:
```promql
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
```

### Step 2: Set Thresholds
- **Baseline**: p95 of normal operation
- **Stressed**: Baseline + 50-100% headroom
- **Critical**: Baseline + 200-300% headroom

### Step 3: Validate
Monitor for false positives/negatives and adjust accordingly.

---

## Service Type Examples

| Service Type | P95 Baseline | P95 Critical | Queue Baseline | Queue Critical |
|--------------|--------------|--------------|----------------|----------------|
| **Real-time API** | 50ms | 200ms | 5 | 20 |
| **Standard API** | 150ms | 500ms | 10 | 100 |
| **Batch Processor** | 300ms | 1000ms | 50 | 200 |
| **High-Throughput** | 500ms | 2000ms | 100 | 500 |

---

## Files
- Configuration: [config.py](online_detector/config.py)
- Kubernetes examples: [k8s/online-detector-configmap.yaml](k8s/online-detector-configmap.yaml)
