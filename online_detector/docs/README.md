# Online Anomaly Detector

A **multi-channel EWMA-based anomaly detection system** with snapshot-freeze capability for Kubernetes environments.

---

## What It Does

Monitors three channels independently and freezes a 10-minute observation window when critical state is detected:

1. **Resource Saturation** (5s interval): CPU, Memory, Threads
2. **Performance Degradation** (45s interval): p95 Response Time
3. **Backpressure Overload** (30s interval): Queue Depth

Each channel uses **hybrid detection** (EWMA + absolute thresholds) to catch both transient spikes and sustained degradation.

---

## Architecture

### Core Components

**1. EWMA Metric** (`EWMAMetric` class)
- Exponential Weighted Moving Average with Z-score normalization
- Returns stress level (0-1) indicating deviation from mean
- Parameters: `alpha=0.05`, `z_max=6.0`

**2. Persistence State Machine** (`PersistenceStateMachine`)
- States: `normal` → `stressed` → `critical`
- Requires sustained stress over multiple windows before transitioning
- Prevents false positives from transient spikes

**3. Channel Detectors**
- `ResourceSaturationChannel`: Monitors CPU/Memory/Threads
- `PerformanceDegradationChannel`: Monitors p95 response time
- `BackpressureOverloadChannel`: Monitors queue depth

**4. Snapshot Freeze**
- **Rolling buffer**: Last 10 minutes of observations (using `deque`)
- **Freeze trigger**: On transition to `critical` state
- **Single freeze guarantee**: Won't freeze again until state returns to `normal`
- **Snapshot structure**:
  ```python
  {
      "channel": "ResourceSaturationChannel",
      "trigger_time": "2026-02-03T14:30:00Z",
      "snapshot_window_seconds": 600,
      "data": [
          {"timestamp": 123456789, "cpu": 85.2, "memory": 78.5, ...},
          # ... 120 observations
      ]
  }
  ```

---

## How It Works

### Hybrid Detection Algorithm

**Problem**: Pure EWMA adapts to sustained high values, treating them as "new normal"

**Solution**: `combined_stress = max(ewma_stress, absolute_stress)`

- **EWMA stress**: Detects transient spikes (relative changes)
- **Absolute stress**: Detects sustained degradation (static thresholds)
- **Combined**: Catches both scenarios

Example:
```
Time    Metric    EWMA Stress    Absolute Stress    Combined
0s      100ms     0.0            0.0                0.0
30s     600ms     0.8            1.0                1.0    ← Spike detected
60s     600ms     0.4            1.0                1.0    ← Still critical (EWMA adapting)
90s     600ms     0.2            1.0                1.0    ← Still critical (absolute prevents)
```

### State Transitions

```
         sustained stress (3 windows)
normal ─────────────────────────────→ stressed
                                           │
                                           │ sustained stress (3 windows)
                                           ↓
                                        critical  ← SNAPSHOT FROZEN HERE
                                           │
                                           │ recovery (5 windows)
                                           ↓
                                        stressed
                                           │
                                           │ recovery (3 windows)
                                           ↓
                                        normal    ← Snapshot reset
```

---

## Running the Detector

### Prerequisites
- Python 3.x
- Prometheus accessible at `http://localhost:9090`
- Target services instrumented with metrics

### Installation
```bash
cd online_detector
pip install -r requirements.txt
```

### Run
```bash
python -m online_detector.main
```

### Expected Output
```
=== ResourceSaturationChannel ===
State: normal | Stress: 0.12 | CPU: 45.2% | Mem: 38.5% | Threads: 120

=== PerformanceDegradationChannel ===
State: stressed | Stress: 0.68 | p95: 285.3ms

=== BackpressureOverloadChannel ===
State: critical | Stress: 0.95 | Queue: 95 items
🔒 SNAPSHOT FROZEN for BackpressureOverloadChannel at 2026-02-03T14:30:15Z
```

---

## Configuration

See [CONFIGURATION.md](CONFIGURATION.md) for details on customizing thresholds via environment variables or Kubernetes ConfigMaps.

**Quick example:**
```bash
# Override thresholds
export P95_RESPONSE_TIME_BASELINE=300
export P95_RESPONSE_TIME_CRITICAL=1000
python -m online_detector.main
```

---

## Testing

Run unit tests:
```bash
python test_channels.py
```

Test hybrid detection:
```bash
python test_hybrid_detection.py
```

---

## Files

| File | Purpose |
|------|---------|
| `detector.py` | Core EWMA, FSM, and channel implementations |
| `config.py` | Configuration with environment variable support |
| `main.py` | Multi-channel orchestration with independent polling |
| `metrics_reader.py` | Prometheus query client |
| `test_channels.py` | Unit tests for all channels |

---

## Key Features

✅ **Multi-channel independent polling** (5s, 45s, 30s)  
✅ **Hybrid detection** (EWMA + absolute thresholds)  
✅ **10-minute snapshot freeze** on critical transitions  
✅ **Single freeze guarantee** per critical episode  
✅ **Kubernetes-native configuration** (ConfigMaps)  
✅ **False positive prevention** (persistence-based FSM)  

---

## Integration

The frozen snapshots are designed to be consumed by the **recovery orchestrator** for root cause analysis and automated remediation.

**Next steps:**
1. Deploy to Kubernetes cluster
2. Measure baseline performance over 1-2 weeks
3. Set thresholds with 25-50% headroom
4. Connect snapshots to recovery system
