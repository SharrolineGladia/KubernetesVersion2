"""
Snapshot modules for incident capture and feature extraction.

This package provides three complementary snapshot types:
- feature_extraction: Time-series analysis for single-channel anomaly detection
- system_snapshot: Cross-sectional system state for XGBoost classification
- role_based_snapshot: Multi-service structure with synthetic derivation for XGBoost compatibility
"""

from .feature_extraction import IncidentSnapshot, SnapshotFeatureExtractor
from .system_snapshot import SystemSnapshot, SystemSnapshotCollector
from .role_based_snapshot import (
    RoleBasedSnapshot,
    SyntheticMetricDerivation,
    create_role_based_snapshot_from_frozen,
    aggregate_resource_saturation_metrics,
    aggregate_performance_degradation_metrics,
    aggregate_backpressure_metrics
)

__all__ = [
    'IncidentSnapshot',
    'SnapshotFeatureExtractor',
    'SystemSnapshot',
    'SystemSnapshotCollector',
    'RoleBasedSnapshot',
    'SyntheticMetricDerivation',
    'create_role_based_snapshot_from_frozen',
    'aggregate_resource_saturation_metrics',
    'aggregate_performance_degradation_metrics',
    'aggregate_backpressure_metrics',
]
