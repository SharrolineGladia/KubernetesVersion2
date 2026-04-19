# System Verification Guide

Complete guide to verify the anomaly detection system with real Kubernetes deployments.

---

## 📋 Prerequisites

### 1. **Kubernetes Cluster Running**

```bash
kubectl cluster-info
kubectl get nodes
```

### 2. **Prometheus Installed**

```bash
kubectl get pods -n monitoring
# Should see prometheus pods running
```

### 3. **Namespace Created**

```bash
kubectl create namespace journal-implementation
```

---

## 🚀 Step-by-Step Verification

### **Step 1: Build and Deploy Notification Service**

#### Build Docker Image

```bash
cd services/notification-service
docker build -t notification-service:latest .
```

#### Deploy to Kubernetes

```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

#### Verify Deployment

```bash
kubectl get pods -n journal-implementation
# Wait for notification-service pod to be Running

kubectl logs -n journal-implementation -l app=notification-service
# Should see: "Notification service running on port 8003"
```

---

### **Step 2: Set Up Port Forwarding**

#### Forward notification-service (for load generation)

```bash
kubectl port-forward -n journal-implementation svc/notification-service 8003:8003
```

#### Forward Prometheus (for detector to query)

```bash
# In a separate terminal
kubectl port-forward -n monitoring svc/prometheus 9090:9090
```

#### Verify Connectivity

```bash
# Test notification-service
curl http://localhost:8003/health
# Expected: {"status": "healthy", ...}

# Test Prometheus
curl http://localhost:9090/api/v1/query?query=up
# Expected: JSON response with metrics
```

---

### **Step 3: Verify Metrics in Prometheus**

Open browser: http://localhost:9090/graph

Test queries:

```promql
# CPU usage
process_cpu_seconds_total{job="notification-service"}

# Memory usage
process_resident_memory_bytes{job="notification-service"}

# Thread count
python_info{job="notification-service"}

# Request rate
rate(http_requests_total{job="notification-service"}[1m])
```

**⚠️ Important:** If metrics don't appear, check ServiceMonitor:

```bash
kubectl get servicemonitor -n journal-implementation
kubectl apply -f k8s/notification-service/notification-service-servicemonitor.yaml
```

---

### **Step 4: Run Online Detector (EWMA + XGBoost)**

#### Option A: Run Locally (Python) - **RECOMMENDED**

```bash
cd online_detector
python -m online_detector.main
```

**Expected Startup Output:**

```
✅ XGBoost classifier loaded (topology-agnostic detection + RCA)
🚀 Multi-Channel EWMA Online Detector started
📡 Resource Saturation: every 5s
📡 Performance Degradation: every 10s
📡 Backpressure Overload: every 15s
🤖 XGBoost Classification: enabled (cooldown: 60s)
```

#### Option B: Run in Kubernetes

```bash
kubectl apply -f k8s/online-detector-configmap.yaml
kubectl apply -f k8s/online-detector-deployment.yaml

# Watch logs
kubectl logs -n journal-implementation -l app=online-detector -f
```

**Expected Output (Normal State):**

```json
{
  "timestamp": "2026-02-08T12:00:00",
  "channel": "resource_saturation",
  "stress_score": 0.123,
  "ewma_stress_score": 0.089,
  "resource_pressure": 0.045,
  "state": "NORMAL",
  "state_duration": 0,
  "transition_reason": "none",
  "raw": {
    "cpu_percent": 5.2,
    "memory_mb": 128.4,
    "threads": 12
  }
}
```

---

### **Step 5: Trigger Anomalies**

#### **Method 1: Specific Anomaly Types**

**CPU/Memory Spike:**

```bash
cd anomaly-trigger
python trigger_anomaly.py --resource
```

**Performance Degradation:**

```bash
python trigger_anomaly.py --performance
```

**All Anomalies:**

```bash
python trigger_anomaly.py --all
```

**Stop Anomalies:**

```bash
python trigger_anomaly.py --stop
```

#### **Method 2: Heavy Load (Realistic Scenario)**

```bash
python scripts/tools/generate_load.py
# Runs 100 threads × 200 req/sec for 120 seconds
```

---

### **Step 6: Observe COMPLETE Detection Pipeline**

Watch the online detector output for the **FULL WORKFLOW**:

#### **1. Normal → Stressed**

```json
{
  "timestamp": "2026-02-08T12:01:30",
  "channel": "resource_saturation",
  "stress_score": 0.423, // exceeded 0.35 threshold
  "state": "STRESSED",
  "state_duration": 0,
  "transition_reason": "stress_score 0.423 > normal_threshold 0.35",
  "raw": {
    "cpu_percent": 45.8,
    "memory_mb": 256.2,
    "threads": 35
  }
}
```

#### **2. Stressed → Critical**

```json
{
  "timestamp": "2026-02-08T12:02:00",
  "channel": "resource_saturation",
  "stress_score": 0.687, // exceeded 0.60 threshold
  "state": "CRITICAL",
  "state_duration": 0,
  "transition_reason": "stress_score 0.687 > anomaly_threshold 0.60",
  "raw": {
    "cpu_percent": 85.3,
    "memory_mb": 450.7,
    "threads": 78
  }
}
```

#### **3. Snapshot Frozen → XGBoost Classification**

```
================================================================================
📸 Snapshot frozen for resource_saturation: 2026-02-08T12:02:00
🤖 Running XGBoost classification...

────────────────────────────────────────────────────────────────────────────────
🎯 ANOMALY CLASSIFICATION RESULTS
────────────────────────────────────────────────────────────────────────────────
   Anomaly Type: CPU_SPIKE
   Confidence: 92.3%
   Active Services: notification
   Timestamp: 2026-02-08T12:02:00.123456
```

#### **4. Root Cause Analysis (Explainability)**

```
────────────────────────────────────────────────────────────────────────────────
🔍 ROOT CAUSE ANALYSIS
────────────────────────────────────────────────────────────────────────────────
   Root Cause Service: notification
   RCA Confidence: 89.5%
   Severity: HIGH

   Contributing Factors:
      • cpu_percent: critical
      • thread_count: high
      • memory_percent: medium

   💡 Recommendations:
      1. Scale notification to 3 replicas to distribute CPU load
      2. Check notification logs for CPU-intensive operations
      3. Profile notification application for performance bottlenecks
      4. Review notification resource limits and requests
================================================================================
```

#### **Complete Workflow Visualization:**

```
EWMA Monitoring (5s intervals)
    ↓
Stress > 0.35 → STRESSED state
    ↓
Stress > 0.60 → CRITICAL state
    ↓
📸 Snapshot Frozen (contains 10min history)
    ↓
🤖 XGBoost Classifier (27 scale-invariant features)
    ↓
🎯 Anomaly Type: CPU_SPIKE / MEMORY_LEAK / SERVICE_CRASH / NORMAL
    ↓
🔍 Root Cause Analysis (8 metrics per service)
    ↓
💡 Actionable Recommendations
```

---

### **Step 7: XGBoost Classification Integration** (Optional - Future Step)

Once you modify [online_detector/main.py](online_detector/main.py) per [README.md](README.md), you'll see:

```json
{
  "timestamp": "2026-02-08T12:02:05",
  "channel": "resource_saturation",
  "stress_score": 0.712,
  "state": "CRITICAL",

  // NEW: XGBoost classification outputs
  "anomaly_type": "cpu_spike",
  "confidence": 0.923,
  "root_cause": "notification",
  "severity": "high",
  "recommendations": [
    "Scale notification to 3 replicas",
    "Check notification logs for CPU intensive operations",
    "Profile notification for performance bottlenecks"
  ]
}
```

---

## ✅ Verification Checklist

Use this to confirm system is working:

### **Metrics Collection**

- [ ] Prometheus scraping notification-service metrics
- [ ] `process_cpu_seconds_total` available in Prometheus
- [ ] `process_resident_memory_bytes` available
- [ ] Port forwards active (8003, 9090)

### **EWMA Detector**

- [ ] Online detector running (no errors)
- [ ] Detector querying Prometheus every 5 seconds
- [ ] Detector showing NORMAL state during idle
- [ ] Detector showing raw metrics (CPU, memory, threads)

### **Anomaly Detection**

- [ ] Stress score increases when load applied
- [ ] State transitions: NORMAL → STRESSED → CRITICAL
- [ ] Frozen snapshots captured at critical state
- [ ] Multiple channels monitored independently

### **Load Generation**

- [ ] `scripts/tools/generate_load.py` successfully connects to service
- [ ] CPU/memory increases visible in raw metrics
- [ ] Anomaly triggered within 30-60 seconds of load start

### **Recovery**

- [ ] State returns to NORMAL after stopping load
- [ ] Stress score decreases back below thresholds
- [ ] System stabilizes within 2-3 minutes

---

## 🐛 Troubleshooting

### **"Failed to query Prometheus"**

```bash
# Check Prometheus is accessible
curl http://localhost:9090/api/v1/query?query=up

# Verify port forward is active
netstat -an | findstr 9090  # Windows
netstat -an | grep 9090     # Linux/Mac

# Restart port forward
kubectl port-forward -n monitoring svc/prometheus 9090:9090
```

### **"No metrics returned"**

```bash
# Check if notification-service is being scraped
kubectl get servicemonitor -n journal-implementation

# Check Prometheus targets
# Open: http://localhost:9090/targets
# Look for notification-service target, should be "UP"

# If missing, apply ServiceMonitor:
kubectl apply -f k8s/notification-service/notification-service-servicemonitor.yaml
```

### **"Service not responding on port 8003"**

```bash
# Check pod status
kubectl get pods -n journal-implementation

# Check pod logs
kubectl logs -n journal-implementation -l app=notification-service

# Check service endpoints
kubectl get endpoints -n journal-implementation notification-service

# Restart port forward
kubectl port-forward -n journal-implementation svc/notification-service 8003:8003
```

### **"Detector shows very low stress even under load"**

This might mean:

1. EWMA alpha too high (adapts too fast) - Check `config.py` EWMA_ALPHA (should be ~0.05)
2. Thresholds too high - Check NORMAL_THRESHOLD (0.35) and ANOMALY_THRESHOLD (0.60)
3. Load not actually reaching service - Check `kubectl top pods` to see real CPU/memory usage

### **"State stuck in CRITICAL"**

The detector uses persistence windows to prevent flapping:

- CRITICAL → STRESSED: Requires 3 consecutive "below anomaly threshold" readings
- STRESSED → NORMAL: Requires 5 consecutive "below normal threshold" readings

This is intentional to avoid false recoveries. Wait ~30 seconds after stopping load.

---

## 📊 Expected Timeline (Realistic Load Test)

```
00:00 - Start scripts/tools/generate_load.py
00:05 - Detector shows stress_score increasing (0.2 → 0.4)
00:15 - State transition: NORMAL → STRESSED (stress_score > 0.35)
00:30 - State transition: STRESSED → CRITICAL (stress_score > 0.60)
00:32 - 📸 Frozen snapshot captured
00:33 - 🤖 XGBoost classification triggered
00:34 - 🎯 Anomaly type: CPU_SPIKE (confidence: 92%)
00:35 - 🔍 Root cause: notification service
00:36 - 💡 Recommendations displayed (scale, profile, check logs)
02:00 - scripts/tools/generate_load.py completes
02:15 - State transition: CRITICAL → STRESSED (3 consecutive below 0.60)
02:45 - State transition: STRESSED → NORMAL (5 consecutive below 0.35)
03:00 - System stabilized
```

---

## 🔍 What to Look For

### **Successful Detection Indicators:**

1. **Gradual Stress Increase**
   - stress_score: 0.1 → 0.3 → 0.5 → 0.7
   - ewma_stress_score follows similar pattern
   - resource_pressure shows raw metric changes

2. **Clear State Transitions**
   - Transition reasons logged
   - State duration resets on transition
   - Frozen snapshot timestamp matches critical transition

3. **XGBoost Classification Workflow** (NEW)
   - "🤖 Running XGBoost classification..." message appears
   - Anomaly type correctly classified (cpu_spike for CPU load, memory_leak for memory growth)
   - Confidence score is reasonable (>75% for valid detection)
   - Classification happens within 1-3 seconds of snapshot freeze

4. **Root Cause Analysis** (NEW)
   - Root cause service identified (should be "notification" in single-service test)
   - Contributing factors listed with severity levels (critical/high/medium/low)
   - Recommendations are actionable and relevant to anomaly type
   - RCA confidence typically 70-95%

5. **Multi-Channel Behavior**
   - resource_saturation: Fast polling (5s)
   - performance_degradation: Medium polling (10s)
   - complete pipeline is verified working:

6. **Test Different Anomaly Types**
   - CPU spike: Use `scripts/tools/generate_load.py` or `trigger_anomaly.py --resource`
   - Memory leak: Modify trigger to gradually increase memory
   - Service crash: Stop notification service pod (`kubectl delete pod -n journal-implementation -l app=notification-service`)

7. **Tune Classification Thresholds**
   - XGBoost confidence threshold: Currently logs all detections
   - Add action logic: if confidence > 85%, auto-remediate; if 60-85%, alert; if <60%, escalate
   - Adjust classification cooldown (default 60s) based on alert volume

8. **Test Multi-Service Scenarios**
   - Deploy web_api and processor services
   - Generate load across multiple services
   - Verify scale-invariant features work with 2-3 services
   - Verify RCA correctly attributes root cause to specific service

9. **Production Integration**
   - Add real Prometheus queries for error_rate, request_rate, response_time
   - Implement automated recovery actions based on recommendations
   - Set up alerting/notification for critical anomalies
   - Configure hierarchical escalation (edge → cluster → cloud)

10. **Monitoring & Logging**
    - Enable detection_log.jsonl for audit trail
    - Set up Grafana dashboard for visualization
    - Monitor classification accuracy over time
    - Track false positive/negative ratesation)

11. **Test Multi-Service Scenarios**
    - Deploy web_api and processor services
    - Generate load across multiple services
    - Verify scale-invariant features work with 2-3 services
    - Verify RCA correctly attributes root cause

12. **Production Tuning**
    - Adjust EWMA_ALPHA for your workload characteristics
    - Tune thresholds based on baseline measurements
    - Configure classification cooldown for alert volume
    - Set confidence thresholds for escalation logic

--- &
kubectl port-forward -n monitoring svc/prometheus 9090:9090 &

# Terminal 2: Run detector (with XGBoost integrated)

cd online_detector
python -m online_detector.main

# Watch for:

# ✅ XGBoost classifier loaded

# NORMAL state outputs

# Then COMPLETE workflow after load applied

# Terminal 3: Generate load

python scripts/tools/generate_load.py

# Watch Terminal 2 for:

# State transitions → Snapshot → Classification → RCA

# Terminal 4: Monitor Kubernetes

kubectl top pods -n journal-implementation

# Watch CPU/memory increase in real-time

```

**Success Criteria:**
- ✅ Detector transitions to CRITICAL within 30-60s
- ✅ Frozen snapshot captured
- ✅ XGBoost classification runs automatically
- ✅ Anomaly type identified (cpu_spike, memory_leak, or service_crash)
- ✅ Root cause analysis shows notification as culprit
- ✅ Recommendations displayed (scale, restart, profile, investigate)implementation
# Watch CPU/memory increase in real-time
```

**Success Criteria:**

- ✅ Detector transitions to CRITICAL within 30-60s
- ✅ Frozen snapshot captured
- ✅ System recovers to NORMAL after load stops
- ✅ No Prometheus query errors
- ✅ All three channels operating independently

---

**Status:** System ready for COMPLETE pipeline verification (EWMA → XGBoost → RCA)  
**Estimated Test Duration:** 5-10 minutes (including setup)  
**Required Components:** Kubernetes, Prometheus, notification-service, online_detector with XGBoost  
**Complete Workflow:** Detection → Classification → Explainability → Recommendations

For questions or issues, refer to [README.md](README.md) or check troubleshooting section above.
