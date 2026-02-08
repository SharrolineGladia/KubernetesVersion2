# Online Detector - Project Structure

## 📁 Reorganized Directory Layout

```
online_detector/
├── 🎯 Core Modules (Root Level)
│   ├── config.py              # Configuration with environment variables
│   ├── detector.py            # EWMA channels & FSM logic
│   ├── main.py                # Multi-channel orchestration
│   ├── metrics_reader.py      # Prometheus client
│   └── requirements.txt       # Dependencies
│
├── 📸 snapshots/              # Incident snapshot modules
│   ├── __init__.py
│   ├── feature_extraction.py  # Time-series analysis (30 features)
│   ├── system_snapshot.py     # System-wide state (29 features)
│   └── role_based_snapshot.py # Multi-service XGBoost compat (NEW)
│
├── 🧪 tests/                  # Test suite
│   ├── __init__.py
│   ├── test_channels.py       # Channel tests (5/5 pass ✅)
│   └── test_feature_extraction.py  # Feature tests (15/15 pass ✅)
│
└── 📚 docs/                   # Documentation
    ├── README.md              # Main documentation
    ├── CONFIGURATION.md       # Kubernetes ConfigMap guide
    ├── FEATURE_EXTRACTION.md  # Time-series features guide
    └── SNAPSHOT_ARCHITECTURE.md  # Dual-snapshot architecture
```

## 🚀 Usage

### Import Core Detector

```python
from online_detector.detector import ResourceSaturationDetector
from online_detector.config import NORMAL_THRESHOLD
from online_detector.metrics_reader import PrometheusClient
```

### Import Snapshot Modules

```python
# Time-series snapshot (for detection)
from online_detector.snapshots import IncidentSnapshot, SnapshotFeatureExtractor

# System-wide snapshot (for XGBoost classification)
from online_detector.snapshots import SystemSnapshot, SystemSnapshotCollector

# Role-based snapshot (for multi-service XGBoost with synthetic metrics) [NEW]
from online_detector.snapshots import (
    RoleBasedSnapshot,
    create_role_based_snapshot_from_frozen,
    aggregate_resource_saturation_metrics
)
```

### Run Detector

```python
python -m online_detector.main
```

### Run Tests

```python
# Run all channel tests
python -m online_detector.tests.test_channels

# Run feature extraction tests
python -m online_detector.tests.test_feature_extraction
```

## 📖 Documentation

All documentation moved to [`docs/`](docs/) subdirectory:

- **[docs/README.md](docs/README.md)** - Architecture overview
- **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)** - Kubernetes setup
- **[docs/FEATURE_EXTRACTION.md](docs/FEATURE_EXTRACTION.md)** - Time-series features
- **[docs/SNAPSHOT_ARCHITECTURE.md](docs/SNAPSHOT_ARCHITECTURE.md)** - Dual snapshots
- **[docs/ROLE_BASED_SNAPSHOT.md](docs/ROLE_BASED_SNAPSHOT.md)** - Multi-service XGBoost compat 🆕

## 🎯 Benefits of New Structure

✅ **Cleaner root** - Only core modules at top level  
✅ **Logical grouping** - Related files in subdirectories  
✅ **Easy navigation** - Clear separation of concerns  
✅ **Import clarity** - `from online_detector.snapshots import ...`  
✅ **Professional layout** - Standard Python package structure
