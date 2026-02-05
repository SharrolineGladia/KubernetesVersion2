"""
Test suite for Feature Extraction Module

Validates that:
1. IncidentSnapshot correctly structures frozen snapshot data
2. SnapshotFeatureExtractor produces fixed-length feature vectors
3. Feature extraction is deterministic and reproducible
4. Edge cases are handled gracefully
"""

import unittest
from datetime import datetime, timedelta
from ..snapshots import IncidentSnapshot, SnapshotFeatureExtractor


class TestIncidentSnapshot(unittest.TestCase):
    """Test IncidentSnapshot structure and conversion."""
    
    def setUp(self):
        """Create sample frozen snapshot."""
        self.frozen_snapshot = {
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
    
    def test_from_frozen_snapshot_conversion(self):
        """Test conversion from frozen snapshot to IncidentSnapshot."""
        snapshot = IncidentSnapshot.from_frozen_snapshot(self.frozen_snapshot)
        
        self.assertEqual(snapshot.channel, "resource_saturation")
        self.assertEqual(snapshot.snapshot_window_seconds, 600)
        self.assertEqual(len(snapshot.timestamps), 5)
        self.assertEqual(len(snapshot.raw_metrics), 5)
        self.assertEqual(len(snapshot.stress_scores), 5)
        self.assertEqual(len(snapshot.states), 5)
    
    def test_to_dict_export(self):
        """Test export to dictionary for serialization."""
        snapshot = IncidentSnapshot.from_frozen_snapshot(self.frozen_snapshot)
        exported = snapshot.to_dict()
        
        self.assertIn("channel", exported)
        self.assertIn("timeline", exported)
        self.assertIn("features", exported)
        self.assertIn("context", exported)
        
        self.assertEqual(exported["channel"], "resource_saturation")
        self.assertEqual(len(exported["timeline"]["timestamps"]), 5)


class TestSnapshotFeatureExtractor(unittest.TestCase):
    """Test feature extraction logic."""
    
    def setUp(self):
        """Create sample snapshot and extractor."""
        self.extractor = SnapshotFeatureExtractor()
        
        # Create a realistic snapshot with gradual degradation
        base_time = datetime(2026, 2, 3, 16, 0, 0)
        data = []
        
        # Phase 1: Normal (0-3 min) - low stress
        for i in range(36):
            data.append({
                "timestamp": base_time + timedelta(seconds=i*5),
                "raw_metric": 20.0 + i * 0.5,
                "stress_score": 0.10 + i * 0.003,
                "state": "normal"
            })
        
        # Phase 2: Stressed (3-6 min) - medium stress
        for i in range(36):
            data.append({
                "timestamp": base_time + timedelta(seconds=(36+i)*5),
                "raw_metric": 40.0 + i * 1.0,
                "stress_score": 0.35 + i * 0.005,
                "state": "stressed"
            })
        
        # Phase 3: Critical (6-10 min) - high stress
        for i in range(48):
            data.append({
                "timestamp": base_time + timedelta(seconds=(72+i)*5),
                "raw_metric": 70.0 + i * 0.8,
                "stress_score": 0.60 + i * 0.003,
                "state": "critical"
            })
        
        self.frozen_snapshot = {
            "channel": "resource_saturation",
            "trigger_time": base_time.isoformat(),
            "snapshot_window_seconds": 600,
            "data": data
        }
        
        self.snapshot = IncidentSnapshot.from_frozen_snapshot(self.frozen_snapshot)
    
    def test_feature_vector_dimension(self):
        """Test that feature vector has consistent dimension."""
        feature_vector = self.extractor.get_feature_vector(self.snapshot)
        feature_names = self.extractor.get_feature_names()
        
        self.assertEqual(len(feature_vector), 30, "Feature vector should have 30 dimensions")
        self.assertEqual(len(feature_names), 30, "Feature names should have 30 elements")
    
    def test_deterministic_extraction(self):
        """Test that extraction is deterministic (same input = same output)."""
        features1 = self.extractor.extract(self.snapshot)
        features2 = self.extractor.extract(self.snapshot)
        
        self.assertEqual(features1, features2, "Feature extraction should be deterministic")
    
    def test_statistical_features(self):
        """Test statistical feature extraction."""
        features = self.extractor.extract(self.snapshot)
        
        # Check that statistical features are computed
        self.assertIn("stress_mean", features)
        self.assertIn("stress_std", features)
        self.assertIn("stress_min", features)
        self.assertIn("stress_max", features)
        self.assertIn("stress_median", features)
        
        # Validate ranges
        self.assertGreaterEqual(features["stress_min"], 0.0)
        self.assertLessEqual(features["stress_max"], 1.0)
        self.assertLessEqual(features["stress_min"], features["stress_mean"])
        self.assertLessEqual(features["stress_mean"], features["stress_max"])
    
    def test_peak_features(self):
        """Test peak characteristic extraction."""
        features = self.extractor.extract(self.snapshot)
        
        # Check peak features exist
        self.assertIn("peak_stress_value", features)
        self.assertIn("peak_stress_position_ratio", features)
        self.assertIn("time_to_peak_seconds", features)
        self.assertIn("peak_rise_rate", features)
        
        # Peak should be at or near end (critical phase)
        self.assertGreater(features["peak_stress_position_ratio"], 0.5, 
                          "Peak should be in second half (stressed/critical phase)")
        self.assertGreater(features["peak_rise_rate"], 0.0,
                          "Peak rise rate should be positive")
    
    def test_state_duration_features(self):
        """Test state duration extraction."""
        features = self.extractor.extract(self.snapshot)
        
        # Check state duration features
        self.assertIn("time_in_normal_ratio", features)
        self.assertIn("time_in_stressed_ratio", features)
        self.assertIn("time_in_critical_ratio", features)
        
        # Ratios should sum to 1.0
        total_ratio = (features["time_in_normal_ratio"] + 
                      features["time_in_stressed_ratio"] + 
                      features["time_in_critical_ratio"])
        self.assertAlmostEqual(total_ratio, 1.0, places=6,
                              msg="State duration ratios should sum to 1.0")
        
        # Critical phase should be longest (48 samples out of 120)
        self.assertGreater(features["time_in_critical_ratio"], 
                          features["time_in_normal_ratio"],
                          "Critical phase should be longest")
    
    def test_state_transition_features(self):
        """Test state transition counting."""
        features = self.extractor.extract(self.snapshot)
        
        # Check transition features
        self.assertIn("num_state_transitions", features)
        self.assertIn("normal_to_stressed_transitions", features)
        self.assertIn("stressed_to_critical_transitions", features)
        
        # Should have 2 major transitions: normal→stressed, stressed→critical
        self.assertEqual(features["normal_to_stressed_transitions"], 1.0,
                        "Should have 1 normal→stressed transition")
        self.assertEqual(features["stressed_to_critical_transitions"], 1.0,
                        "Should have 1 stressed→critical transition")
    
    def test_trend_features(self):
        """Test trend analysis features."""
        features = self.extractor.extract(self.snapshot)
        
        # Check trend features
        self.assertIn("linear_trend_slope", features)
        self.assertIn("trend_r_squared", features)
        self.assertIn("monotonicity_score", features)
        self.assertIn("stress_range", features)
        
        # Slope should be positive (stress increasing)
        self.assertGreater(features["linear_trend_slope"], 0.0,
                          "Trend slope should be positive (stress increasing)")
        
        # Monotonicity should be high (mostly increasing)
        self.assertGreater(features["monotonicity_score"], 0.5,
                          "Monotonicity should indicate increasing trend")
    
    def test_temporal_context_features(self):
        """Test that temporal information is captured in the snapshot structure."""
        # Temporal context is stored in snapshot itself, not as extracted features
        self.assertEqual(self.snapshot.snapshot_window_seconds, 600,
                        "Window should be 600 seconds (10 minutes)")
        
        # Calculate actual sampling rate from timestamps
        duration = (self.snapshot.timestamps[-1] - self.snapshot.timestamps[0]).total_seconds()
        sampling_rate = len(self.snapshot.timestamps) / duration
        self.assertAlmostEqual(sampling_rate, 0.2, places=1,
                              msg="Sampling rate should be ~0.2 Hz (5s intervals)")
    
    def test_edge_case_single_observation(self):
        """Test handling of snapshot with single observation."""
        single_obs = {
            "channel": "test",
            "trigger_time": datetime(2026, 2, 3, 16, 0, 0).isoformat(),
            "snapshot_window_seconds": 5,
            "data": [{
                "timestamp": datetime(2026, 2, 3, 16, 0, 0),
                "raw_metric": 50.0,
                "stress_score": 0.5,
                "state": "critical"
            }]
        }
        
        snapshot = IncidentSnapshot.from_frozen_snapshot(single_obs)
        features = self.extractor.extract(snapshot)
        
        # Should not crash, should produce valid features
        self.assertEqual(len(features), 30)
        self.assertEqual(features["stress_std"], 0.0, "Single observation has 0 std")
        self.assertEqual(features["num_state_transitions"], 0.0, "No transitions possible")
    
    def test_feature_reproducibility_across_instances(self):
        """Test that different extractor instances produce same results."""
        extractor1 = SnapshotFeatureExtractor()
        extractor2 = SnapshotFeatureExtractor()
        
        features1 = extractor1.extract(self.snapshot)
        features2 = extractor2.extract(self.snapshot)
        
        self.assertEqual(features1, features2,
                        "Different extractor instances should produce same results")


class TestFeatureVectorML(unittest.TestCase):
    """Test ML-ready feature vector properties."""
    
    def setUp(self):
        """Create sample snapshot."""
        base_time = datetime(2026, 2, 3, 16, 0, 0)
        data = []
        
        for i in range(60):
            data.append({
                "timestamp": base_time + timedelta(seconds=i*10),
                "raw_metric": 30.0 + i * 0.5,
                "stress_score": 0.20 + i * 0.01,
                "state": "normal" if i < 20 else ("stressed" if i < 40 else "critical")
            })
        
        frozen = {
            "channel": "test",
            "trigger_time": base_time.isoformat(),
            "snapshot_window_seconds": 600,
            "data": data
        }
        
        self.snapshot = IncidentSnapshot.from_frozen_snapshot(frozen)
        self.extractor = SnapshotFeatureExtractor()
    
    def test_feature_vector_is_numeric(self):
        """Test that all features are numeric (float)."""
        feature_vector = self.extractor.get_feature_vector(self.snapshot)
        
        for value in feature_vector:
            self.assertIsInstance(value, float, "All features should be float type")
    
    def test_no_nan_or_inf_values(self):
        """Test that features contain no NaN or Inf values."""
        import math
        
        feature_vector = self.extractor.get_feature_vector(self.snapshot)
        
        for value in feature_vector:
            self.assertFalse(math.isnan(value), "Features should not contain NaN")
            self.assertFalse(math.isinf(value), "Features should not contain Inf")
    
    def test_feature_names_match_vector_length(self):
        """Test that feature names match vector length."""
        feature_vector = self.extractor.get_feature_vector(self.snapshot)
        feature_names = self.extractor.get_feature_names()
        
        self.assertEqual(len(feature_vector), len(feature_names),
                        "Feature vector length should match feature names length")


if __name__ == "__main__":
    unittest.main()
