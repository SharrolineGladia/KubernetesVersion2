"""
Snapshot modules for incident capture and feature extraction.

This package provides two complementary snapshot types:
- feature_extraction: Time-series analysis for single-channel anomaly detection
- system_snapshot: Cross-sectional system state for XGBoost classification
"""

from .feature_extraction import IncidentSnapshot, SnapshotFeatureExtractor
from .system_snapshot import SystemSnapshot, SystemSnapshotCollector

__all__ = [
    'IncidentSnapshot',
    'SnapshotFeatureExtractor',
    'SystemSnapshot',
    'SystemSnapshotCollector',
]
