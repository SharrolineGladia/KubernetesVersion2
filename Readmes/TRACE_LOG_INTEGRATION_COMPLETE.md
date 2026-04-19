# 🎯 Trace & Log Integration - Complete!

## What Was Done

I've integrated **distributed traces** (Jaeger) and **structured logs** into your RCA pipeline so it now uses ALL THREE observability signals:

✅ **Metrics** (Prometheus) - What happened  
✅ **Traces** (Jaeger) - How it propagated  
✅ **Logs** (JSON) - Why it failed

## New Files Created

### 1. Core Analyzers

- **`ml_detector/scripts/trace_analyzer.py`** (290 lines)
  - Fetches traces from Jaeger API
  - Identifies slow operations (>1000ms)
  - Reconstructs error propagation chains
  - Builds service dependency graphs

- **`ml_detector/scripts/log_analyzer.py`** (310 lines)
  - Parses structured JSON logs
  - Identifies recurring error patterns
  - Correlates logs with traces via trace_id
  - Extracts critical error details

### 2. Enhanced RCA

- **`ml_detector/scripts/explainability_layer.py`** (MODIFIED)
  - Now initializes TraceAnalyzer and LogAnalyzer
  - Enhanced `explain_anomaly()` to gather trace + log context
  - Enhanced `_analyze_contributing_factors()` to include trace/log factors
  - Enhanced `_calculate_severity()` to escalate based on traces/logs
  - Enhanced `_generate_recommendations()` with specific trace/log insights
  - RCAResult now includes `trace_context` and `log_context` fields

### 3. Demo & Testing

- **`scripts/demos/demo_integrated_rca.py`** - Shows RCA with all three data sources
- **`scripts/tests/test_integration_verification.py`** - Quick test to verify setup
- **`scripts/tools/capture_service_logs.py`** - Utility to capture logs from containers/pods
- **`INTEGRATION_GUIDE.md`** - Complete setup and usage guide

## How It Works

### Before (Metrics Only)

```
Anomaly Detected → Analyze Metrics → Generate Recommendations
                    (CPU high, memory high)
```

### After (Multi-Modal)

```
Anomaly Detected → Analyze Metrics → Fetch Traces → Parse Logs → Enhanced RCA
                    (CPU 92%)        (3 slow ops)   (12 errors)
                                                         ↓
                         "CPU high + send_notification slow (850ms) +
                          repeated 'Connection timeout' errors"
```

## Example Output

```
🎯 Root Cause: notification-service
🔍 Anomaly Type: cpu_spike
📊 Confidence: 89%
⚠️  Severity: CRITICAL (escalated by trace data)

Contributing Factors (Multi-Modal):
1. [cpu_utilization] 🔴 CRITICAL
   Value: 92.3%
   CPU usage extremely high

2. [trace_slow_operation] 🟠 HIGH
   Value: 850ms
   Slow: send_notification in notification-service

3. [log_error_pattern] 🟠 HIGH
   Value: 12 occurrences
   Repeated: 'Connection timeout to external API'

Trace Analysis:
✓ Trace data available
  Error rate (traces): 35%
  Services involved: notification-service, api-gateway
  Slow operations:
    - send_notification: 850ms
    - validate_request: 420ms
  Error propagation chain:
    - notification-service.send_notification: Connection timeout
    - api-gateway.forward_request: Upstream error

Log Analysis:
✓ Log data available
  Error rate (logs): 28%
  Error patterns:
    - 'Connection timeout to external API' (12x)
    - 'Failed to send notification' (8x)
  Critical errors:
    - [notification-service] Connection refused to smtp.example.com:587

Recommendations:
1. 🔧 Scale notification-service horizontally (add replicas)
2. 🔍 Optimize send_notification - takes 850ms
3. 🚨 IMMEDIATE: Address error 'Connection timeout to external API'
4. 🔗 Check cascade: 2 services affected by error propagation

Data sources used:
  ✓ Traces
  ✓ Logs
  ✓ Metrics
```

## How to Use

### Step 1: Start Infrastructure

```powershell
# Start Jaeger
docker-compose -f docker-compose-jaeger.yml up -d

# Verify Jaeger UI
start http://localhost:16686
```

### Step 2: Start Your Microservice

```powershell
# Option A: Docker
cd services\notification-service
docker build -t notification-service .
docker run -p 8000:8000 --name notification-service notification-service

# Option B: Direct Python
cd services\notification-service
python notification_service.py
```

### Step 3: Generate Traffic & Capture Logs

```powershell
# Terminal 1: Generate load
python scripts/tools/generate_load.py

# Terminal 2: Capture logs (in another terminal)
python scripts/tools/capture_service_logs.py notification-service 5
```

### Step 4: Verify Integration

```powershell
python scripts/tests/test_integration_verification.py
```

Should show:

```
✅ PASS - Trace Analyzer
✅ PASS - Log Analyzer
✅ PASS - RCA Integration
```

### Step 5: Run Full RCA Demo

```powershell
python scripts/demos/demo_integrated_rca.py
```

Choose option 1 for full integrated demo.

## Key Features

### 1. Real Data Collection

- ✅ Captures actual traces from your running services
- ✅ Parses real structured logs (JSON format)
- ✅ No dummy data - everything is from live system

### 2. Proper Integration

- ✅ Trace analyzer called by RCA layer
- ✅ Log analyzer called by RCA layer
- ✅ Contributing factors enriched with trace/log insights
- ✅ Severity calculation considers all three signals
- ✅ Recommendations tailored to specific errors found

### 3. Trace Capabilities

- Service dependency graph (who calls who)
- Error propagation chains (failure cascade)
- Slow operation detection (latency spikes)
- Cross-service impact analysis

### 4. Log Capabilities

- Error pattern recognition (recurring issues)
- Critical error extraction (most recent/severe)
- Trace correlation (link logs to distributed traces)
- Error message parsing (exception types, status codes)

## Technical Details

### Trace Data Flow

```
Microservice → OpenTelemetry SDK → Jaeger Agent → Jaeger Collector
                                                         ↓
                                                    Jaeger Query API
                                                         ↓
                                                  trace_analyzer.py
                                                         ↓
                                                  RCA (explainability_layer.py)
```

### Log Data Flow

```
Microservice → stdout/stderr (JSON logs) → scripts/tools/capture_service_logs.py → service_logs.json
                                                                            ↓
                                                                      log_analyzer.py
                                                                            ↓
                                                                      RCA (explainability_layer.py)
```

### RCA Enhancement

```python
# Before
rca_result = explainer.explain_anomaly(
    anomaly_type='cpu_spike',
    service_metrics=metrics,
    scale_invariant_features=features
)
# Only metrics analyzed

# After
rca_result = explainer.explain_anomaly(
    anomaly_type='cpu_spike',
    service_metrics=metrics,
    scale_invariant_features=features,
    timestamp=datetime.utcnow(),
    service_name='notification-service'  # NEW
)
# Metrics + traces + logs analyzed
# rca_result.trace_context - trace insights
# rca_result.log_context - log insights
```

## Verification Checklist

Before considering integration complete:

- [ ] Jaeger running and accessible at localhost:16686
- [ ] Microservice emitting traces (check Jaeger UI)
- [ ] service_logs.json exists and has recent entries
- [ ] scripts/tests/test_integration_verification.py passes all 3 tests
- [ ] scripts/demos/demo_integrated_rca.py shows trace and log data
- [ ] Contributing factors include trace/log entries
- [ ] Recommendations reference specific errors from logs
- [ ] Severity properly escalated when traces show high error rate

## What's Next

After verifying integration works:

1. **Create data aligner** - Align metrics, traces, logs in same time window
2. **Test full pipeline** - Run end-to-end with anomaly trigger
3. **Add to online detector** - Integrate RCA into main detection loop

## Troubleshooting

**"No trace data available"**

- Verify Jaeger: `docker ps | findstr jaeger`
- Check service sends traces: Jaeger UI → Search for your service
- Ensure tracing_utils.py imported in service code

**"No log data available"**

- Check file exists: `type service_logs.json`
- Verify timestamps are recent (within last 5-10 minutes)
- Capture fresh logs: `python scripts/tools/capture_service_logs.py notification-service 1`

**Import errors**

- Install requests: `pip install requests`
- Check Python path includes ml_detector/scripts

## Success Criteria

✅ Integration is successful when:

1. scripts/tests/test_integration_verification.py passes all tests
2. scripts/demos/demo_integrated_rca.py shows:
   - ✓ Trace data available
   - ✓ Log data available
   - Contributing factors include entries with type='trace' and type='log'
3. Recommendations reference specific errors from traces/logs
4. Severity is escalated when traces show high error rates

---

**You now have a multi-modal RCA system!** 🎉

Metrics tell you **what** happened, traces show you **how** it spread, and logs explain **why** it failed.
