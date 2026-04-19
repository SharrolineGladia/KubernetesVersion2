# Real Data Integration Guide

## Overview

This guide shows how to run your microservices and verify traces/logs integration.

## Architecture

```
[Microservices] → Emit metrics, logs, traces
                    ↓
[Prometheus]     ← Scrapes metrics
[Jaeger]         ← Receives traces
[Log Files]      ← Writes structured JSON logs
                    ↓
[RCA Pipeline]   ← Analyzes ALL THREE data sources
```

## Quick Start

### 1. Start Infrastructure

**Start Jaeger (for traces):**

```bash
docker-compose -f docker-compose-jaeger.yml up -d
```

Verify: Open http://localhost:16686 in browser (Jaeger UI)

**Start Prometheus (for metrics):**

```bash
# Check if Prometheus config exists
dir services\notification-service\prometheus.yml

# Start Prometheus (if config exists)
docker run -d -p 9090:9090 \
  -v ${PWD}/services/notification-service/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus
```

Verify: Open http://localhost:9090 (Prometheus UI)

### 2. Start Notification Service

**Option A: Docker**

```bash
cd services\notification-service
docker build -t notification-service .
docker run -p 8000:8000 --name notification-service notification-service
```

**Option B: Direct Python**

```bash
cd services\notification-service
pip install -r requirements.txt
python notification_service.py
```

### 3. Generate Traffic & Logs

**Terminal 1: Generate load**

```bash
python scripts/tools/generate_load.py
```

**Terminal 2: Capture logs** (while service is running)

```bash
python scripts/tools/capture_service_logs.py notification-service 5
```

This will:

- Stream logs from the container/pod
- Write to `service_logs.json` in JSON lines format
- Run for 5 minutes (or Ctrl+C to stop)

### 4. Verify Data Collection

**Check Traces (Jaeger):**

```bash
# Via browser: http://localhost:16686
# Search for service: notification-service
# Should see traces with spans
```

**Check Metrics (Prometheus):**

```bash
# Via browser: http://localhost:9090
# Query: notification_requests_total
# Should see metrics
```

**Check Logs:**

```bash
# View captured logs
type service_logs.json | Select-Object -First 10
```

Each log should have:

- timestamp
- service
- level
- message
- trace_id (correlation with Jaeger)

### 5. Run Integrated RCA Demo

```bash
python scripts/demos/demo_integrated_rca.py
```

Select option 1 (Full integrated RCA)

Expected output:

```
✓ Trace analyzer: Enabled
✓ Log analyzer: Enabled

Contributing Factors (Multi-Modal):
1. [cpu_utilization] 🔴 CRITICAL
2. [trace_slow_operation] 🟠 HIGH
3. [log_error_pattern] 🟠 HIGH

Trace Analysis:
✓ Trace data available
  Error rate (traces): 35%
  Slow operations detected:
    - send_notification: 850ms

Log Analysis:
✓ Log data available
  Error rate (logs): 28%
  Error patterns:
    - 'Connection timeout to external API' (12x)

Data sources used:
  ✓ Traces
  ✓ Logs
  ✓ Metrics
```

## Troubleshooting

### "No trace data available"

- Check Jaeger is running: `docker ps | findstr jaeger`
- Verify service is sending traces: Check Jaeger UI at localhost:16686
- Ensure tracing_utils.py is imported in notification_service.py

### "No log data available"

- Check service_logs.json exists and has recent entries
- Verify logs have timestamps within last 5 minutes
- Make sure logs are in JSON format (one JSON object per line)

### "Failed to fetch traces from Jaeger"

- Verify Jaeger API is accessible: `curl http://localhost:16686/api/traces`
- Check firewall/network settings
- Try increasing timeout in trace_analyzer.py

### Service not emitting traces

- Check notification_service.py imports DistributedTracer
- Verify tracer.start_trace() is called on requests
- Check JAEGER_AGENT_HOST environment variable

## File Locations

- **Traces**: Jaeger (query via API at localhost:16686)
- **Logs**: `service_logs.json` (JSON lines format)
- **Metrics**: Prometheus (query at localhost:9090)
- **RCA Output**: Console output from scripts/demos/demo_integrated_rca.py

## Next Steps

Once you verify data is flowing:

1. **Run full detection pipeline:**

   ```bash
   python scripts/demos/demo_full_pipeline.py
   ```

2. **Test with anomaly trigger:**

   ```bash
   cd anomaly-trigger
   python trigger_cpu_spike.py
   ```

3. **Run integrated RCA** to see traces/logs in action:
   ```bash
   python scripts/demos/demo_integrated_rca.py
   ```

## Architecture Files

- `trace_analyzer.py` - Fetches and parses Jaeger traces
- `log_analyzer.py` - Parses structured JSON logs
- `explainability_layer.py` - RCA with trace/log integration
- `scripts/tools/capture_service_logs.py` - Helper to capture logs from containers

## What Gets Integrated

### From Traces (Jaeger):

- Error rate across requests
- Slow operations (>1000ms threshold)
- Error propagation chain (Service A → B → C)
- Service dependency graph

### From Logs (JSON):

- Error rate from log levels
- Recurring error patterns
- Critical error messages
- Trace ID correlation

### From Metrics (Prometheus):

- CPU, memory, error rate
- Request rate, latency
- Thread count, queue depth

**All three are combined** in the RCA to provide:

- **Metrics**: What happened (resource exhaustion)
- **Traces**: How it spread (service call chain)
- **Logs**: Why it failed (error messages)

This gives you complete observability and explainability!
