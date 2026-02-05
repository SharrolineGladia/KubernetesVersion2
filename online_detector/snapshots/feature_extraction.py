"""
Feature Extraction Module for Incident Snapshots

This module provides a clean abstraction layer between raw detector snapshots
and ML-ready numerical features. It separates temporal timeline data from
derived numerical features suitable for machine learning models.

Design Principles:
- Deterministic: Same snapshot always produces same features
- Reproducible: No randomness or side effects
- Fixed-length: Output vector has consistent dimensionality
- ML-ready: All features are numerical and normalized where appropriate
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import statistics
import math


@dataclass
class IncidentSnapshot:
    """
    Structured representation of a frozen anomaly snapshot.
    
    Separates raw timeline data from derived features and contextual metadata,
    making it suitable for ML pipelines, post-mortem analysis, and recovery systems.
    """
    
    # === Identity & Context ===
    channel: str
    trigger_time: datetime
    snapshot_window_seconds: int
    
    # === Timeline Data ===
    timestamps: List[datetime]
    raw_metrics: List[float]
    stress_scores: List[float]
    states: List[str]
    
    # === Derived Features (computed once) ===
    features: Optional[Dict[str, float]] = None
    
    # === Contextual References ===
    trace_ids: List[str] = field(default_factory=list)
    log_hints: List[str] = field(default_factory=list)
    
    @classmethod
    def from_frozen_snapshot(cls, frozen_snapshot: Dict) -> 'IncidentSnapshot':
        """
        Create an IncidentSnapshot from the detector's frozen snapshot format.
        
        Args:
            frozen_snapshot: Dict with keys {channel, trigger_time, snapshot_window_seconds, data}
        
        Returns:
            IncidentSnapshot instance with structured data
        """
        data = frozen_snapshot["data"]
        
        return cls(
            channel=frozen_snapshot["channel"],
            trigger_time=datetime.fromisoformat(frozen_snapshot["trigger_time"]),
            snapshot_window_seconds=frozen_snapshot["snapshot_window_seconds"],
            timestamps=[obs["timestamp"] for obs in data],
            raw_metrics=[obs["raw_metric"] for obs in data],
            stress_scores=[obs["stress_score"] for obs in data],
            states=[obs["state"] for obs in data],
            trace_ids=[],  # Can be populated from external sources
            log_hints=[]   # Can be populated from external sources
        )
    
    def to_dict(self) -> Dict:
        """Export snapshot as dictionary for serialization."""
        return {
            "channel": self.channel,
            "trigger_time": self.trigger_time.isoformat(),
            "snapshot_window_seconds": self.snapshot_window_seconds,
            "timeline": {
                "timestamps": [ts.isoformat() for ts in self.timestamps],
                "raw_metrics": self.raw_metrics,
                "stress_scores": self.stress_scores,
                "states": self.states
            },
            "features": self.features,
            "context": {
                "trace_ids": self.trace_ids,
                "log_hints": self.log_hints
            }
        }


class SnapshotFeatureExtractor:
    """
    Extracts fixed-length numerical feature vectors from incident snapshots.
    
    Produces ML-ready features capturing:
    - Statistical properties (mean, std, percentiles)
    - Temporal dynamics (rate of change, velocity, acceleration)
    - Peak characteristics (max values, timing, rise rate)
    - State transitions (duration in states, transition counts)
    - Trend indicators (slope, monotonicity, stability)
    
    All features are deterministic and reproducible.
    """
    
    def __init__(self):
        self.feature_names = self._define_feature_names()
    
    def _define_feature_names(self) -> List[str]:
        """Define the ordered list of feature names for the output vector."""
        return [
            # === Statistical Features (7) ===
            "stress_mean",
            "stress_std",
            "stress_min",
            "stress_max",
            "stress_median",
            "stress_p75",
            "stress_p95",
            
            # === Peak Characteristics (4) ===
            "peak_stress_value",
            "peak_stress_position_ratio",
            "time_to_peak_seconds",
            "peak_rise_rate",
            
            # === Rate of Change (4) ===
            "stress_velocity_mean",
            "stress_velocity_std",
            "stress_acceleration_mean",
            "max_single_step_change",
            
            # === State Duration Features (4) ===
            "time_in_normal_ratio",
            "time_in_stressed_ratio",
            "time_in_critical_ratio",
            "first_critical_position_ratio",
            
            # === State Transition Features (3) ===
            "num_state_transitions",
            "normal_to_stressed_transitions",
            "stressed_to_critical_transitions",
            
            # === Trend Features (4) ===
            "linear_trend_slope",
            "trend_r_squared",
            "monotonicity_score",
            "stress_range",
            
            # === Raw Metric Features (4) ===
            "raw_metric_mean",
            "raw_metric_max",
            "raw_metric_std",
            "raw_metric_cv",
        ]
    
    def extract(self, snapshot: IncidentSnapshot) -> Dict[str, float]:
        """
        Extract fixed-length feature vector from incident snapshot.
        
        Args:
            snapshot: IncidentSnapshot instance
        
        Returns:
            Dictionary mapping feature names to numerical values
        """
        features = {}
        
        # Extract all feature groups
        features.update(self._extract_statistical_features(snapshot))
        features.update(self._extract_peak_features(snapshot))
        features.update(self._extract_rate_of_change_features(snapshot))
        features.update(self._extract_state_duration_features(snapshot))
        features.update(self._extract_state_transition_features(snapshot))
        features.update(self._extract_trend_features(snapshot))
        features.update(self._extract_raw_metric_features(snapshot))
        
        # Attach features to snapshot
        snapshot.features = features
        
        return features
    
    def _extract_statistical_features(self, snapshot: IncidentSnapshot) -> Dict[str, float]:
        """Extract statistical properties of stress scores."""
        stress = snapshot.stress_scores
        
        return {
            "stress_mean": statistics.mean(stress),
            "stress_std": statistics.stdev(stress) if len(stress) > 1 else 0.0,
            "stress_min": min(stress),
            "stress_max": max(stress),
            "stress_median": statistics.median(stress),
            "stress_p75": self._percentile(stress, 0.75),
            "stress_p95": self._percentile(stress, 0.95),
        }
    
    def _extract_peak_features(self, snapshot: IncidentSnapshot) -> Dict[str, float]:
        """Extract peak stress characteristics."""
        stress = snapshot.stress_scores
        timestamps = snapshot.timestamps
        
        peak_value = max(stress)
        peak_idx = stress.index(peak_value)
        
        # Position ratio: 0.0 = start, 1.0 = end
        peak_position_ratio = peak_idx / (len(stress) - 1) if len(stress) > 1 else 0.5
        
        # Time to peak in seconds
        time_to_peak = (timestamps[peak_idx] - timestamps[0]).total_seconds()
        
        # Rise rate: stress increase per second from start to peak
        stress_increase = peak_value - stress[0]
        peak_rise_rate = stress_increase / time_to_peak if time_to_peak > 0 else 0.0
        
        return {
            "peak_stress_value": peak_value,
            "peak_stress_position_ratio": peak_position_ratio,
            "time_to_peak_seconds": time_to_peak,
            "peak_rise_rate": peak_rise_rate,
        }
    
    def _extract_rate_of_change_features(self, snapshot: IncidentSnapshot) -> Dict[str, float]:
        """Extract velocity and acceleration features."""
        stress = snapshot.stress_scores
        timestamps = snapshot.timestamps
        
        if len(stress) < 2:
            return {
                "stress_velocity_mean": 0.0,
                "stress_velocity_std": 0.0,
                "stress_acceleration_mean": 0.0,
                "max_single_step_change": 0.0,
            }
        
        # Compute velocities (stress change per second)
        velocities = []
        for i in range(1, len(stress)):
            dt = (timestamps[i] - timestamps[i-1]).total_seconds()
            if dt > 0:
                velocity = (stress[i] - stress[i-1]) / dt
                velocities.append(velocity)
        
        # Compute accelerations (velocity change per second)
        accelerations = []
        for i in range(1, len(velocities)):
            dt = (timestamps[i+1] - timestamps[i]).total_seconds()
            if dt > 0:
                acceleration = (velocities[i] - velocities[i-1]) / dt
                accelerations.append(acceleration)
        
        # Max single-step change (absolute)
        single_step_changes = [abs(stress[i] - stress[i-1]) for i in range(1, len(stress))]
        
        return {
            "stress_velocity_mean": statistics.mean(velocities) if velocities else 0.0,
            "stress_velocity_std": statistics.stdev(velocities) if len(velocities) > 1 else 0.0,
            "stress_acceleration_mean": statistics.mean(accelerations) if accelerations else 0.0,
            "max_single_step_change": max(single_step_changes) if single_step_changes else 0.0,
        }
    
    def _extract_state_duration_features(self, snapshot: IncidentSnapshot) -> Dict[str, float]:
        """Extract time spent in each state."""
        states = snapshot.states
        total_observations = len(states)
        
        if total_observations == 0:
            return {
                "time_in_normal_ratio": 0.0,
                "time_in_stressed_ratio": 0.0,
                "time_in_critical_ratio": 0.0,
                "first_critical_position_ratio": 1.0,
            }
        
        normal_count = states.count("normal")
        stressed_count = states.count("stressed")
        critical_count = states.count("critical")
        
        # Find first critical position
        try:
            first_critical_idx = states.index("critical")
            first_critical_ratio = first_critical_idx / (total_observations - 1) if total_observations > 1 else 0.0
        except ValueError:
            first_critical_ratio = 1.0  # No critical state found
        
        return {
            "time_in_normal_ratio": normal_count / total_observations,
            "time_in_stressed_ratio": stressed_count / total_observations,
            "time_in_critical_ratio": critical_count / total_observations,
            "first_critical_position_ratio": first_critical_ratio,
        }
    
    def _extract_state_transition_features(self, snapshot: IncidentSnapshot) -> Dict[str, float]:
        """Extract state transition counts."""
        states = snapshot.states
        
        if len(states) < 2:
            return {
                "num_state_transitions": 0.0,
                "normal_to_stressed_transitions": 0.0,
                "stressed_to_critical_transitions": 0.0,
            }
        
        total_transitions = 0
        normal_to_stressed = 0
        stressed_to_critical = 0
        
        for i in range(1, len(states)):
            if states[i] != states[i-1]:
                total_transitions += 1
                
                if states[i-1] == "normal" and states[i] == "stressed":
                    normal_to_stressed += 1
                elif states[i-1] == "stressed" and states[i] == "critical":
                    stressed_to_critical += 1
        
        return {
            "num_state_transitions": float(total_transitions),
            "normal_to_stressed_transitions": float(normal_to_stressed),
            "stressed_to_critical_transitions": float(stressed_to_critical),
        }
    
    def _extract_trend_features(self, snapshot: IncidentSnapshot) -> Dict[str, float]:
        """Extract trend and directionality features."""
        stress = snapshot.stress_scores
        
        if len(stress) < 2:
            return {
                "linear_trend_slope": 0.0,
                "trend_r_squared": 0.0,
                "monotonicity_score": 0.0,
                "stress_range": 0.0,
            }
        
        # Linear regression: y = mx + b
        x = list(range(len(stress)))
        slope, r_squared = self._linear_regression(x, stress)
        
        # Monotonicity: ratio of increasing steps to total steps
        increases = sum(1 for i in range(1, len(stress)) if stress[i] > stress[i-1])
        decreases = sum(1 for i in range(1, len(stress)) if stress[i] < stress[i-1])
        total_changes = increases + decreases
        
        if total_changes > 0:
            monotonicity = (increases - decreases) / total_changes
        else:
            monotonicity = 0.0
        
        return {
            "linear_trend_slope": slope,
            "trend_r_squared": r_squared,
            "monotonicity_score": monotonicity,
            "stress_range": max(stress) - min(stress),
        }
    
    def _extract_raw_metric_features(self, snapshot: IncidentSnapshot) -> Dict[str, float]:
        """Extract features from raw metric values (CPU, memory, latency, queue, etc.)."""
        raw = snapshot.raw_metrics
        
        mean_val = statistics.mean(raw)
        std_val = statistics.stdev(raw) if len(raw) > 1 else 0.0
        
        # Coefficient of variation (normalized variability)
        cv = (std_val / mean_val) if mean_val > 0 else 0.0
        
        return {
            "raw_metric_mean": mean_val,
            "raw_metric_max": max(raw),
            "raw_metric_std": std_val,
            "raw_metric_cv": cv,
        }
    
    def _extract_temporal_context_features(self, snapshot: IncidentSnapshot) -> Dict[str, float]:
        """Extract temporal metadata features."""
        timestamps = snapshot.timestamps
        
        window_duration = (timestamps[-1] - timestamps[0]).total_seconds()
        sampling_rate = len(timestamps) / window_duration if window_duration > 0 else 0.0
        
        return {
            "window_duration_seconds": float(snapshot.snapshot_window_seconds),
            "sampling_rate_hz": sampling_rate,
        }
    
    # === Helper Methods ===
    
    def _percentile(self, data: List[float], p: float) -> float:
        """Calculate percentile using linear interpolation."""
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        
        if f == c:
            return sorted_data[int(k)]
        
        d0 = sorted_data[int(f)] * (c - k)
        d1 = sorted_data[int(c)] * (k - f)
        return d0 + d1
    
    def _linear_regression(self, x: List[float], y: List[float]) -> Tuple[float, float]:
        """
        Compute linear regression slope and R² coefficient.
        
        Returns:
            (slope, r_squared)
        """
        n = len(x)
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(y)
        
        # Slope: Σ((x - x̄)(y - ȳ)) / Σ((x - x̄)²)
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        slope = numerator / denominator if denominator > 0 else 0.0
        
        # R²: 1 - (SS_res / SS_tot)
        y_pred = [slope * (i - x_mean) + y_mean for i in x]
        ss_res = sum((y[i] - y_pred[i]) ** 2 for i in range(n))
        ss_tot = sum((y[i] - y_mean) ** 2 for i in range(n))
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return slope, r_squared
    
    def get_feature_vector(self, snapshot: IncidentSnapshot) -> List[float]:
        """
        Extract features and return as ordered list (ML-ready vector).
        
        Args:
            snapshot: IncidentSnapshot instance
        
        Returns:
            List of 30 numerical features in consistent order
        """
        features = self.extract(snapshot)
        return [features[name] for name in self.feature_names]
    
    def get_feature_names(self) -> List[str]:
        """Return ordered list of feature names."""
        return self.feature_names.copy()


# === Example Usage ===

if __name__ == "__main__":
    # Example: Load a frozen snapshot from detector
    example_frozen_snapshot = {
        "channel": "resource_saturation",
        "trigger_time": "2026-02-03T16:23:00.000000",
        "snapshot_window_seconds": 600,
        "data": [
            {
                "timestamp": datetime(2026, 2, 3, 16, 13, 0),
                "raw_metric": 25.0,
                "stress_score": 0.15,
                "state": "normal"
            },
            {
                "timestamp": datetime(2026, 2, 3, 16, 14, 0),
                "raw_metric": 30.0,
                "stress_score": 0.20,
                "state": "normal"
            },
            {
                "timestamp": datetime(2026, 2, 3, 16, 15, 0),
                "raw_metric": 45.0,
                "stress_score": 0.40,
                "state": "stressed"
            },
            {
                "timestamp": datetime(2026, 2, 3, 16, 16, 0),
                "raw_metric": 65.0,
                "stress_score": 0.65,
                "state": "critical"
            },
            {
                "timestamp": datetime(2026, 2, 3, 16, 17, 0),
                "raw_metric": 75.0,
                "stress_score": 0.75,
                "state": "critical"
            },
        ]
    }
    
    # Create structured snapshot
    snapshot = IncidentSnapshot.from_frozen_snapshot(example_frozen_snapshot)
    
    # Extract features
    extractor = SnapshotFeatureExtractor()
    features = extractor.extract(snapshot)
    
    print("=== Feature Extraction Example ===\n")
    print(f"Channel: {snapshot.channel}")
    print(f"Trigger Time: {snapshot.trigger_time}")
    print(f"Window: {snapshot.snapshot_window_seconds}s\n")
    
    print("=== Extracted Features ===")
    for name, value in features.items():
        print(f"{name:40s} = {value:10.4f}")
    
    print(f"\n=== Feature Vector (ML-ready) ===")
    feature_vector = extractor.get_feature_vector(snapshot)
    print(f"Dimension: {len(feature_vector)}")
    print(f"Vector: {feature_vector}")
    
    print(f"\n=== Export as Dictionary ===")
    import json
    print(json.dumps(snapshot.to_dict(), indent=2, default=str))
