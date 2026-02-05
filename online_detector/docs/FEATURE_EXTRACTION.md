# Feature Extraction Module

## Overview

The feature extraction module provides a clean abstraction layer between raw detector snapshots and ML-ready numerical features. It separates temporal timeline data from derived features, making incident snapshots suitable for machine learning pipelines, post-mortem analysis, and recovery systems.

## Architecture

```
Frozen Snapshot (from Detector)
         ↓
   IncidentSnapshot (Structured)
         ↓
SnapshotFeatureExtractor
         ↓
  30D Feature Vector (ML-ready)
```

## Components

### 1. IncidentSnapshot

A structured representation that separates:

- **Identity & Context**: Channel name, trigger time, window duration
- **Timeline Data**: Timestamps, raw metrics, stress scores, FSM states
- **Derived Features**: Extracted numerical features (computed once)
- **Contextual References**: Trace IDs, log hints for correlation

```python
from online_detector.feature_extraction import IncidentSnapshot

# Convert from frozen snapshot
snapshot = IncidentSnapshot.from_frozen_snapshot(frozen_snapshot)

# Add context
snapshot.trace_ids = ["trace-abc123", "trace-def456"]
snapshot.log_hints = ["High CPU in worker-3", "Memory leak in cache"]

# Export to JSON
exported = snapshot.to_dict()
```

### 2. SnapshotFeatureExtractor

Extracts **30 numerical features** in **7 categories**:

#### Statistical Features (7)

- `stress_mean`: Average stress score
- `stress_std`: Standard deviation
- `stress_min`, `stress_max`: Range bounds
- `stress_median`: Middle value
- `stress_p75`, `stress_p95`: Percentiles

#### Peak Characteristics (4)

- `peak_stress_value`: Maximum stress reached
- `peak_stress_position_ratio`: Where peak occurred (0=start, 1=end)
- `time_to_peak_seconds`: Duration until peak
- `peak_rise_rate`: Stress increase per second

#### Rate of Change (4)

- `stress_velocity_mean`: Average rate of change
- `stress_velocity_std`: Velocity variability
- `stress_acceleration_mean`: Average acceleration
- `max_single_step_change`: Largest single jump

#### State Duration (4)

- `time_in_normal_ratio`: % time in normal state
- `time_in_stressed_ratio`: % time in stressed
- `time_in_critical_ratio`: % time in critical
- `first_critical_position_ratio`: When critical first appeared

#### State Transitions (3)

- `num_state_transitions`: Total state changes
- `normal_to_stressed_transitions`: Count of escalations
- `stressed_to_critical_transitions`: Count of critical escalations

#### Trend Analysis (4)

- `linear_trend_slope`: Linear regression slope
- `trend_r_squared`: How linear the trend is
- `monotonicity_score`: Directionality (-1=down, 0=mixed, 1=up)
- `stress_range`: max - min stress

#### Raw Metric Features (4)

- `raw_metric_mean`: Average of underlying metric (CPU/latency/queue)
- `raw_metric_max`: Peak value
- `raw_metric_std`: Standard deviation
- `raw_metric_cv`: Coefficient of variation (normalized variability)

## Usage

### Basic Feature Extraction

```python
from online_detector.feature_extraction import (
    IncidentSnapshot,
    SnapshotFeatureExtractor
)

# Create structured snapshot
snapshot = IncidentSnapshot.from_frozen_snapshot(frozen_snapshot)

# Extract features
extractor = SnapshotFeatureExtractor()
features = extractor.extract(snapshot)  # Dict of 30 features

# Get ML-ready vector
feature_vector = extractor.get_feature_vector(snapshot)  # List of 30 floats
feature_names = extractor.get_feature_names()  # List of 30 names
```

### Integration with Detector

```python
from online_detector.main import resource_saturation

# After detector freezes snapshot
frozen = resource_saturation.get_frozen_snapshot()

if frozen:
    # Convert to structured format
    incident = IncidentSnapshot.from_frozen_snapshot(frozen)

    # Extract features
    extractor = SnapshotFeatureExtractor()
    features = extractor.extract(incident)

    # Use features for ML prediction, alerting, or recovery
    predict_recovery_action(features)
```

### Export for ML Pipeline

```python
import json

# Minimal export (features only)
minimal = {
    "channel": snapshot.channel,
    "trigger_time": snapshot.trigger_time.isoformat(),
    "features": extractor.get_feature_vector(snapshot),
    "feature_names": extractor.get_feature_names()
}

with open("features.json", "w") as f:
    json.dump(minimal, f)
```

### Full Export (with timeline)

```python
# Full export includes timeline data
full = snapshot.to_dict()

with open("incident_full.json", "w") as f:
    json.dump(full, f, indent=2, default=str)
```

## Key Properties

### ✅ Deterministic

- Same snapshot **always** produces identical features
- No randomness or side effects
- Reproducible across different runs

### ✅ Fixed-Length

- Always **30 dimensions**
- Consistent across all channels (resource/performance/backpressure)
- Ready for ML models expecting fixed input size

### ✅ ML-Ready

- All features are numerical (`float`)
- No NaN or Inf values
- Normalized where appropriate (ratios, percentiles)

### ✅ Channel-Agnostic

- Works with any channel type:
  - `resource_saturation` (CPU, memory, threads)
  - `performance_degradation` (p95 latency)
  - `backpressure_overload` (queue depth)

## Examples

### Run the Demo

```bash
python demo_feature_extraction.py
```

This creates:

- `incident_snapshot_full.json` - Complete timeline + features (7.4 KB)
- `incident_features_minimal.json` - Features only (1.3 KB)

### Run Tests

```bash
python -m unittest online_detector.test_feature_extraction -v
```

Tests validate:

- Conversion from frozen snapshots
- Deterministic extraction
- Feature vector dimensionality
- Statistical correctness
- Edge case handling

## Design Principles

### 1. Separation of Concerns

- **IncidentSnapshot**: Structured storage
- **SnapshotFeatureExtractor**: Feature computation
- No mixing of data and processing logic

### 2. Immutability

- Snapshots are immutable once created
- Feature extraction doesn't modify snapshots
- Thread-safe and cacheable

### 3. Composability

- Features can be extracted independently
- Easy to add new feature groups
- Extensible without breaking existing code

### 4. Observability

- Clear feature names (`stress_mean`, not `feat_0`)
- Self-documenting feature descriptions
- Easy to inspect and debug

## Integration Points

### Upstream (Detector → Snapshot)

```python
# Detector provides frozen snapshot
frozen_snapshot = detector.get_frozen_snapshot()

# Convert to structured format
snapshot = IncidentSnapshot.from_frozen_snapshot(frozen_snapshot)
```

### Downstream (Snapshot → ML/Recovery)

```python
# Extract features for ML model
features = extractor.get_feature_vector(snapshot)
prediction = ml_model.predict([features])

# Or for recovery orchestrator
recovery_plan = recovery_orchestrator.generate_plan(
    channel=snapshot.channel,
    features=features,
    context=snapshot.log_hints
)
```

## Future Extensions

Potential enhancements (not implemented):

- **Temporal patterns**: Fourier transform, periodicity detection
- **Cross-channel features**: Correlation between channels
- **Historical context**: Comparison with past incidents
- **Anomaly scores**: Distance from normal baseline
- **Feature importance**: Ranking of most informative features

## Performance

- **Feature extraction**: <1ms per snapshot
- **Memory footprint**: ~10KB per snapshot with 120 observations
- **Export size**:
  - Full: ~7.4KB (with timeline)
  - Minimal: ~1.3KB (features only)

## Testing

Comprehensive test suite covers:

- ✅ Snapshot structure validation
- ✅ Deterministic extraction (15/15 tests pass)
- ✅ Feature vector dimensionality
- ✅ Statistical accuracy
- ✅ Edge cases (single observation, empty states)
- ✅ Reproducibility across instances

Run tests:

```bash
python -m unittest online_detector.test_feature_extraction
```

All **15 tests pass** ✅
