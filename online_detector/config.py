# config.py

PROMETHEUS_URL = "http://localhost:9090"

# Sampling
SCRAPE_INTERVAL_SECONDS = 5

# EWMA parameters
# Lower alpha = slower adaptation, better anomaly detection (0.05-0.1 for production)
# Set to 0.05 to prevent stress from becoming "new normal" within minutes
EWMA_ALPHA = 0.05
EPSILON = 1e-6

# Z-score normalization
# Higher ceiling allows stronger anomalies to be properly detected
Z_MAX = 6.0

# Channel weights (resource saturation)
CPU_WEIGHT = 0.5
MEMORY_WEIGHT = 0.3
THREAD_WEIGHT = 0.2

# Thresholds (adjusted for baseline variation tolerance)
# NORMAL_THRESHOLD: when combined stress score exceeds this, start counting toward stressed
# ANOMALY_THRESHOLD: when combined stress score exceeds this, start counting toward critical
NORMAL_THRESHOLD = 0.35
ANOMALY_THRESHOLD = 0.60

# Persistence FSM windows (consecutive samples).
# With SCRAPE_INTERVAL_SECONDS=5, each window is ~5s.
# Increased windows for more stable detection and baseline establishment
NORMAL_TO_STRESSED_WINDOWS = 3
STRESSED_TO_CRITICAL_WINDOWS = 3
STRESSED_TO_NORMAL_WINDOWS = 5
CRITICAL_TO_STRESSED_WINDOWS = 3

# Target service
SERVICE_NAME = "notification-service"
NAMESPACE = "journal-implementation"

# Soft limits for static "pressure" scoring (0..1). Tune these per environment.
# CPU is expressed as percent (psutil style; can exceed 100 on multi-core).
# Set these ABOVE baseline to give headroom:
# - Baseline memory ~140MB, set limit at 250MB (gives pressure ~0.56 at baseline)
# - Baseline threads ~10, set limit at 60
CPU_SOFT_LIMIT = 100.0
MEMORY_SOFT_LIMIT_MB = 250.0
THREAD_SOFT_LIMIT = 60.0
