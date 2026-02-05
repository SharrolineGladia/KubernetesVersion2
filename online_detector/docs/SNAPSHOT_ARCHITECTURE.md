# Snapshot Architecture: Time-Series vs System-Wide

## ❌ The Problem

Your current snapshot extracts **30 time-series statistical features** from a single channel's 10-minute history, but your XGBoost model expects **29 concurrent system metrics** across 3 services.

### Mismatch Summary

| What You Have | What Your Model Needs |
|--------------|----------------------|
| Time-series analytics of ONE metric | Cross-sectional state of 29 metrics |
| Vertical slice (one signal over time) | Horizontal slice (all signals at one moment) |
| `stress_mean=0.568, velocity=0.001237, ...` | `notification_cpu=45.2, web_api_cpu=67.8, ...` |
| Answers "How did this metric evolve?" | Answers "What's the system state NOW?" |

## ✅ The Solution: Dual-Snapshot Architecture

You need **BOTH** snapshot types working together:

### 1️⃣ Time-Series Snapshot (feature_extraction.py)
**Purpose:** Detect **WHEN** anomaly occurs  
**Scope:** Single channel over 10 minutes  
**Features:** 30 temporal statistics  
**Output:** Stress patterns, trends, velocities  
**Model:** EWMA + Finite State Machine  
**Trigger:** Threshold-based detection  

### 2️⃣ System-Wide Snapshot (system_snapshot.py) 
**Purpose:** Classify **WHAT TYPE** of anomaly  
**Scope:** All 3 services at trigger moment  
**Features:** 29 concurrent metrics  
**Output:** System state for classification  
**Model:** XGBoost (your trained model)  
**Trigger:** When time-series detector fires  

## 📊 Your XGBoost Model's Features (29)

### Notification Service (9 metrics)
1. `notification_cpu`
2. `notification_memory`
3. `notification_error_rate`
4. `notification_api_health`
5. `notification_delivery_success`
6. `notification_message_rate`
7. `notification_queue`
8. `notification_queue_depth`
9. `notification_thread_count`

### Web API Service (10 metrics)
10. `web_api_cpu`
11. `web_api_memory`
12. `web_api_response_time_p95`
13. `web_api_db_connections`
14. `web_api_errors`
15. `web_api_queue_depth`
16. `web_api_redis_health`
17. `web_api_requests`
18. `web_api_requests_per_second`
19. `web_api_thread_count`

### Processor Service (10 metrics)
20. `processor_cpu`
21. `processor_memory`
22. `processor_response_time_p95`
23. `processor_db_connections`
24. `processor_memory_growth`
25. `processor_processing_rate`
26. `processor_queue`
27. `processor_queue_depth`
28. `processor_redis_health`
29. `processor_thread_count`

## 🔄 Complete Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  1. EWMA Channels Monitor Individual Metrics                │
│     • ResourceSaturation: CPU/Memory/Threads                │
│     • PerformanceDegradation: P95 Latency                   │
│     • BackpressureOverload: Queue Depth                     │
└────────────────────┬────────────────────────────────────────┘
                     │ Threshold exceeded
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Time-Series Snapshot Frozen                             │
│     • 10-minute rolling buffer                              │
│     • 30 temporal features extracted                        │
│     • Provides CONTEXT: "How did we get here?"             │
└────────────────────┬────────────────────────────────────────┘
                     │ Trigger system-wide capture
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  3. System-Wide Snapshot Captured                           │
│     • Query ALL 29 metrics from Prometheus                  │
│     • Capture at exact trigger timestamp                    │
│     • Provides STATE: "What's happening now?"              │
└────────────────────┬────────────────────────────────────────┘
                     │ Feed to XGBoost
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  4. XGBoost Classification                                  │
│     • Input: 29-dimensional feature vector                  │
│     • Output: {cpu_spike, memory_leak, service_crash}      │
│     • Confidence: 87%                                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  5. Recovery Action                                         │
│     • Type-specific remediation                             │
│     • Targeted to affected services                         │
│     • Informed by both snapshots                            │
└─────────────────────────────────────────────────────────────┘
```

## 💻 Implementation

### Step 1: Integrate SystemSnapshotCollector

```python
from online_detector.system_snapshot import SystemSnapshotCollector

# Initialize collector
collector = SystemSnapshotCollector(
    prometheus_url="http://localhost:9090",
    namespace="default"
)

# In your main detection loop
if detector.get_frozen_snapshot():
    # Time-series snapshot already frozen
    ts_snapshot = detector.get_frozen_snapshot()
    
    # Capture system-wide state
    sys_snapshot = collector.collect_snapshot(
        trigger_channel=ts_snapshot["channel"],
        timestamp=datetime.fromisoformat(ts_snapshot["trigger_time"])
    )
    
    # Feed to XGBoost
    model_input = sys_snapshot.to_model_input()
    prediction = xgb_model.predict([model_input])
    
    print(f"Anomaly Type: {prediction}")
```

### Step 2: Update Prometheus Queries

Your `system_snapshot.py` assumes metric names. Update `query_metric()` to match your actual Prometheus metrics:

```python
def query_metric(metric_name: str, service: str) -> float:
    # Adjust metric names to match your Prometheus schema
    metric_map = {
        "notification_cpu": "notification_service_cpu_percent",
        "web_api_cpu": "web_api_service_cpu_percent",
        # ... etc
    }
    actual_metric = metric_map.get(metric_name, metric_name)
    # Query Prometheus...
```

## 📈 Benefits of Dual-Snapshot

| Benefit | Time-Series | System-Wide | Combined |
|---------|------------|-------------|----------|
| **Detection** | ✅ When anomaly starts | ❌ | ✅ Accurate timing |
| **Classification** | ❌ | ✅ What type | ✅ Accurate typing |
| **Context** | ✅ How it evolved | ❌ | ✅ Full story |
| **Recovery** | ⚠️ Generic | ✅ Type-specific | ✅ Optimal action |
| **Root Cause** | ⚠️ Single metric | ✅ System-wide | ✅ Complete picture |

## 🎯 Answer to Your Question

> **"Is my snapshot enough for this model to detect anomaly?"**

**No, the current time-series snapshot (30 features) is NOT compatible with your XGBoost model (29 features).**

**What you need:**
1. **Keep** your time-series snapshot for detection
2. **Add** system-wide snapshot for classification
3. **Use** `system_snapshot.py` to query all 29 metrics when anomaly is detected
4. **Feed** the 29-metric vector to your XGBoost model

## 📁 Files Created

1. **`online_detector/system_snapshot.py`** - System-wide snapshot module
2. **`demo_dual_snapshot.py`** - Complete workflow demonstration
3. **`SNAPSHOT_ARCHITECTURE.md`** - This documentation

## 🚀 Next Steps

1. **Test system_snapshot.py** with your Prometheus instance
2. **Verify metric names** match your Prometheus schema
3. **Integrate** with your XGBoost model
4. **Add** classification to your main detection loop
5. **Validate** predictions against your 3,137-sample dataset

## 📊 Validation Strategy

```python
# Collect snapshots for your dataset
for incident in historical_incidents:
    sys_snapshot = collector.collect_snapshot(
        trigger_channel=incident["channel"],
        timestamp=incident["timestamp"]
    )
    
    # Compare with dataset labels
    model_prediction = model.predict([sys_snapshot.to_model_input()])
    actual_type = incident["anomaly_type"]
    
    print(f"Predicted: {model_prediction}, Actual: {actual_type}")
```

## ✅ Summary

- **Time-series snapshot**: ✅ Working, use for detection
- **System-wide snapshot**: ✅ Now implemented, use for classification
- **XGBoost compatibility**: ✅ Now compatible (29 features)
- **Architecture**: ✅ Dual-snapshot approach recommended
- **Next action**: Integrate `SystemSnapshotCollector` into your main loop
