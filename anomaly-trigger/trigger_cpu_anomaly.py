import time
import requests
from datetime import datetime

# -----------------------------
# CONFIGURATION
# -----------------------------

PROMETHEUS_URL = "http://localhost:9090"
CPU_QUERY = 'avg(rate(process_cpu_seconds_total[30s])) * 100'

CPU_THRESHOLD = 80.0          # % CPU
SUSTAINED_SECONDS = 20        # anomaly must persist
POLL_INTERVAL = 5             # seconds

ORCHESTRATOR_URL = "http://localhost:5000/recover"
TARGET_NAMESPACE = "journal-implementation"
TARGET_APP_LABEL = "notification-service"

# -----------------------------
# STATE
# -----------------------------

anomaly_start_time = None

# -----------------------------
# FUNCTIONS
# -----------------------------

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def query_cpu_usage():
    try:
        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": CPU_QUERY},
            timeout=5
        )
        result = response.json()
        value = float(result["data"]["result"][0]["value"][1])
        return value
    except Exception as e:
        log(f"ERROR querying Prometheus: {e}")
        return None

def trigger_recovery(cpu_value):
    payload = {
        "namespace": TARGET_NAMESPACE,
        "app_label": TARGET_APP_LABEL,
        "anomaly_type": "CPU_SATURATION",
        "cpu_value": cpu_value,
        "explanation": "Sustained high CPU detected; proactive pod restart recommended"
    }

    try:
        r = requests.post(ORCHESTRATOR_URL, json=payload, timeout=5)
        log(f"Recovery triggered → status={r.status_code}")
    except Exception as e:
        log(f"ERROR calling orchestrator: {e}")

# -----------------------------
# MAIN LOOP
# -----------------------------

log("Starting CPU anomaly trigger")

while True:
    cpu = query_cpu_usage()

    if cpu is None:
        time.sleep(POLL_INTERVAL)
        continue

    log(f"CPU usage = {cpu:.2f}%")

    if cpu > CPU_THRESHOLD:
        if anomaly_start_time is None:
            anomaly_start_time = time.time()
            log("CPU anomaly detected (starting timer)")
        elif time.time() - anomaly_start_time >= SUSTAINED_SECONDS:
            log("CPU anomaly sustained → triggering recovery")
            trigger_recovery(cpu)
            anomaly_start_time = None
            time.sleep(30)  # cooldown to avoid restart loop
    else:
        anomaly_start_time = None

    time.sleep(POLL_INTERVAL)
