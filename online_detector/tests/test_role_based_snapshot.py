# test_role_based_snapshot.py
"""
Tests for role-based snapshot with synthetic metric derivation.

Verifies:
1. Snapshot structure follows role-based format
2. Synthetic derivation is deterministic
3. Feature extraction produces correct vector format
4. Integration with existing detector snapshots
"""

import unittest
from datetime import datetime, timedelta
from online_detector.snapshots.role_based_snapshot import (
    RoleBasedSnapshot,
    SyntheticMetricDerivation,
    create_role_based_snapshot_from_frozen,
    aggregate_resource_saturation_metrics,
    aggregate_performance_degradation_metrics,
    aggregate_backpressure_metrics
)


class TestRoleBasedSnapshot(unittest.TestCase):
    """Test role-based snapshot data structure."""
    
    def test_snapshot_structure(self):
        """Verify snapshot has correct role-based structure."""
        primary = {"cpu_mean": 45.0, "memory_mean": 512.0}
        upstream = {"cpu_mean": 30.0, "memory_mean": 400.0}
        downstream = {"cpu_mean": 60.0, "memory_mean": 700.0}
        
        snapshot = RoleBasedSnapshot(
            timestamp="2026-02-05T10:00:00",
            channel="resource_saturation",
            window_seconds=600,
            primary_metrics=primary,
            upstream_metrics=upstream,
            downstream_metrics=downstream
        )
        
        # Check structure
        self.assertEqual(snapshot.channel, "resource_saturation")
        self.assertEqual(snapshot.window_seconds, 600)
        self.assertIn("primary", snapshot.services)
        self.assertIn("upstream", snapshot.services)
        self.assertIn("downstream", snapshot.services)
        
        # Check values
        self.assertEqual(snapshot.services["primary"]["cpu_mean"], 45.0)
        self.assertEqual(snapshot.services["upstream"]["cpu_mean"], 30.0)
        self.assertEqual(snapshot.services["downstream"]["cpu_mean"], 60.0)
    
    def test_to_dict_export(self):
        """Verify snapshot can be exported to dict."""
        primary = {"cpu_mean": 45.0}
        snapshot = RoleBasedSnapshot(
            timestamp="2026-02-05T10:00:00",
            channel="resource_saturation",
            window_seconds=600,
            primary_metrics=primary
        )
        
        data = snapshot.to_dict()
        
        self.assertIsInstance(data, dict)
        self.assertEqual(data["timestamp"], "2026-02-05T10:00:00")
        self.assertEqual(data["channel"], "resource_saturation")
        self.assertIn("services", data)
    
    def test_feature_extraction_simple(self):
        """Test feature extraction with simple schema."""
        primary = {"cpu_mean": 45.0, "memory_mean": 512.0}
        upstream = {"cpu_mean": 30.0, "memory_mean": 400.0}
        downstream = {"cpu_mean": 60.0, "memory_mean": 700.0}
        
        snapshot = RoleBasedSnapshot(
            timestamp="2026-02-05T10:00:00",
            channel="resource_saturation",
            window_seconds=600,
            primary_metrics=primary,
            upstream_metrics=upstream,
            downstream_metrics=downstream
        )
        
        # Define feature schema (role_metric format)
        schema = [
            "primary_cpu_mean",
            "upstream_cpu_mean",
            "downstream_cpu_mean",
            "primary_memory_mean",
            "upstream_memory_mean",
            "downstream_memory_mean"
        ]
        
        vector = snapshot.to_model_input(schema)
        
        # Check vector properties
        self.assertEqual(len(vector), 6)
        self.assertIsInstance(vector, list)
        self.assertTrue(all(isinstance(x, float) for x in vector))
        
        # Check values
        self.assertEqual(vector[0], 45.0)  # primary_cpu_mean
        self.assertEqual(vector[1], 30.0)  # upstream_cpu_mean
        self.assertEqual(vector[2], 60.0)  # downstream_cpu_mean
        self.assertEqual(vector[3], 512.0)  # primary_memory_mean
        self.assertEqual(vector[4], 400.0)  # upstream_memory_mean
        self.assertEqual(vector[5], 700.0)  # downstream_memory_mean
    
    def test_missing_feature_handling(self):
        """Test that missing features return 0.0 as safe default."""
        primary = {"cpu_mean": 45.0}
        snapshot = RoleBasedSnapshot(
            timestamp="2026-02-05T10:00:00",
            channel="resource_saturation",
            window_seconds=600,
            primary_metrics=primary
        )
        
        schema = [
            "primary_cpu_mean",
            "primary_latency_p95",  # Not in metrics
            "upstream_cpu_mean",    # Upstream not provided
            "nonexistent_metric"    # Completely invalid
        ]
        
        vector = snapshot.to_model_input(schema)
        
        self.assertEqual(len(vector), 4)
        self.assertEqual(vector[0], 45.0)  # Exists
        self.assertEqual(vector[1], 0.0)   # Missing in primary
        self.assertEqual(vector[2], 0.0)   # Empty upstream
        self.assertEqual(vector[3], 0.0)   # Invalid format


class TestSyntheticDerivation(unittest.TestCase):
    """Test synthetic metric derivation logic."""
    
    def test_deterministic_factor(self):
        """Verify deterministic factor generation is reproducible."""
        timestamp = "2026-02-05T10:00:00"
        
        # Same inputs should produce same output
        factor1 = SyntheticMetricDerivation._get_deterministic_factor(
            timestamp, "upstream", "cpu"
        )
        factor2 = SyntheticMetricDerivation._get_deterministic_factor(
            timestamp, "upstream", "cpu"
        )
        
        self.assertEqual(factor1, factor2)
        
        # Different inputs should produce different outputs
        factor3 = SyntheticMetricDerivation._get_deterministic_factor(
            timestamp, "downstream", "cpu"
        )
        
        self.assertNotEqual(factor1, factor3)
        
        # Factor should be in [0, 1] range
        self.assertGreaterEqual(factor1, 0.0)
        self.assertLessEqual(factor1, 1.0)
    
    def test_upstream_cpu_derivation(self):
        """Test upstream CPU is 60-80% of primary."""
        primary = {"cpu_mean": 100.0}
        timestamp = "2026-02-05T10:00:00"
        
        upstream = SyntheticMetricDerivation.derive_upstream_metrics(
            primary, timestamp, []
        )
        
        self.assertIn("cpu_mean", upstream)
        # Should be 60-80% of primary (60-80)
        self.assertGreaterEqual(upstream["cpu_mean"], 60.0)
        self.assertLessEqual(upstream["cpu_mean"], 80.0)
    
    def test_downstream_cpu_amplification(self):
        """Test downstream CPU is 120-180% of primary (amplification)."""
        primary = {"cpu_mean": 50.0}
        timestamp = "2026-02-05T10:00:00"
        
        downstream = SyntheticMetricDerivation.derive_downstream_metrics(
            primary, timestamp, []
        )
        
        self.assertIn("cpu_mean", downstream)
        # Should be 120-180% of primary (60-90)
        self.assertGreaterEqual(downstream["cpu_mean"], 60.0)
        self.assertLessEqual(downstream["cpu_mean"], 90.0)
    
    def test_latency_relationships(self):
        """Test latency propagation follows causal logic."""
        primary = {"latency_p95": 200.0, "latency_mean": 150.0}
        timestamp = "2026-02-05T10:00:00"
        
        upstream = SyntheticMetricDerivation.derive_upstream_metrics(
            primary, timestamp, []
        )
        downstream = SyntheticMetricDerivation.derive_downstream_metrics(
            primary, timestamp, []
        )
        
        # Upstream latency: 70-90% of primary (doesn't include downstream)
        self.assertGreaterEqual(upstream["latency_p95"], 140.0)
        self.assertLessEqual(upstream["latency_p95"], 180.0)
        
        # Downstream latency: 40-70% of primary (component of total)
        self.assertGreaterEqual(downstream["latency_p95"], 80.0)
        self.assertLessEqual(downstream["latency_p95"], 140.0)
    
    def test_queue_backpressure(self):
        """Test queue depth backpressure propagation."""
        primary = {"queue_depth_mean": 50.0, "queue_depth_p95": 75.0}
        timestamp = "2026-02-05T10:00:00"
        
        upstream = SyntheticMetricDerivation.derive_upstream_metrics(
            primary, timestamp, []
        )
        downstream = SyntheticMetricDerivation.derive_downstream_metrics(
            primary, timestamp, []
        )
        
        # Upstream queue: 80-120% of primary (backpressure)
        self.assertGreaterEqual(upstream["queue_depth_mean"], 40.0)
        self.assertLessEqual(upstream["queue_depth_mean"], 60.0)
        
        # Downstream queue: 130-200% of primary (bottleneck)
        self.assertGreaterEqual(downstream["queue_depth_mean"], 65.0)
        self.assertLessEqual(downstream["queue_depth_mean"], 100.0)
    
    def test_throughput_patterns(self):
        """Test throughput correlation across services."""
        primary = {"throughput_mean": 1000.0}
        timestamp = "2026-02-05T10:00:00"
        
        upstream = SyntheticMetricDerivation.derive_upstream_metrics(
            primary, timestamp, []
        )
        downstream = SyntheticMetricDerivation.derive_downstream_metrics(
            primary, timestamp, []
        )
        
        # Upstream throughput: 100-150% of primary (may handle more traffic)
        self.assertGreaterEqual(upstream["throughput_mean"], 1000.0)
        self.assertLessEqual(upstream["throughput_mean"], 1500.0)
        
        # Downstream throughput: 150-300% of primary (query amplification)
        self.assertGreaterEqual(downstream["throughput_mean"], 1500.0)
        self.assertLessEqual(downstream["throughput_mean"], 3000.0)
    
    def test_deterministic_across_metrics(self):
        """Verify all derived metrics are deterministic."""
        primary = {
            "cpu_mean": 50.0,
            "memory_mean": 500.0,
            "latency_p95": 200.0,
            "queue_depth_mean": 30.0
        }
        timestamp = "2026-02-05T10:00:00"
        
        # Derive twice
        upstream1 = SyntheticMetricDerivation.derive_upstream_metrics(
            primary, timestamp, []
        )
        upstream2 = SyntheticMetricDerivation.derive_upstream_metrics(
            primary, timestamp, []
        )
        
        downstream1 = SyntheticMetricDerivation.derive_downstream_metrics(
            primary, timestamp, []
        )
        downstream2 = SyntheticMetricDerivation.derive_downstream_metrics(
            primary, timestamp, []
        )
        
        # Should be identical
        self.assertEqual(upstream1, upstream2)
        self.assertEqual(downstream1, downstream2)


class TestAggregationFunctions(unittest.TestCase):
    """Test observation aggregation functions."""
    
    def test_resource_saturation_aggregation(self):
        """Test aggregation of resource saturation observations."""
        observations = [
            {
                "timestamp": "2026-02-05T10:00:00",
                "raw_metric": {"cpu": 40.0, "memory": 500.0, "threads": 10},
                "stress_score": 0.3,
                "state": "normal"
            },
            {
                "timestamp": "2026-02-05T10:00:05",
                "raw_metric": {"cpu": 60.0, "memory": 600.0, "threads": 15},
                "stress_score": 0.5,
                "state": "stressed"
            },
            {
                "timestamp": "2026-02-05T10:00:10",
                "raw_metric": {"cpu": 80.0, "memory": 700.0, "threads": 20},
                "stress_score": 0.7,
                "state": "critical"
            }
        ]
        
        metrics = aggregate_resource_saturation_metrics(observations)
        
        # Check aggregated metrics exist
        self.assertIn("cpu_mean", metrics)
        self.assertIn("cpu_p95", metrics)
        self.assertIn("memory_mean", metrics)
        self.assertIn("thread_count_mean", metrics)
        self.assertIn("stress_score_mean", metrics)
        self.assertIn("stress_score_max", metrics)
        
        # Check values
        self.assertAlmostEqual(metrics["cpu_mean"], 60.0, places=1)
        self.assertAlmostEqual(metrics["memory_mean"], 600.0, places=1)
        self.assertAlmostEqual(metrics["stress_score_max"], 0.7, places=1)
    
    def test_performance_degradation_aggregation(self):
        """Test aggregation of performance degradation observations."""
        observations = [
            {
                "timestamp": "2026-02-05T10:00:00",
                "raw_metric": 100.0,  # p95 latency in ms
                "ewma_signal": 0.2,
                "stress_score": 0.2,
                "state": "normal"
            },
            {
                "timestamp": "2026-02-05T10:00:45",
                "raw_metric": 250.0,
                "ewma_signal": 0.4,
                "stress_score": 0.5,
                "state": "stressed"
            },
            {
                "timestamp": "2026-02-05T10:01:30",
                "raw_metric": 400.0,
                "ewma_signal": 0.7,
                "stress_score": 0.8,
                "state": "critical"
            }
        ]
        
        metrics = aggregate_performance_degradation_metrics(observations)
        
        self.assertIn("latency_p95", metrics)
        self.assertIn("latency_mean", metrics)
        self.assertIn("latency_max", metrics)
        self.assertIn("stress_score_mean", metrics)
        
        # Check values
        self.assertEqual(metrics["latency_max"], 400.0)
        self.assertAlmostEqual(metrics["latency_mean"], 250.0, places=1)
    
    def test_backpressure_aggregation(self):
        """Test aggregation of backpressure observations."""
        observations = [
            {
                "timestamp": "2026-02-05T10:00:00",
                "raw_metric": 5.0,  # queue depth
                "ewma_signal": 0.1,
                "stress_score": 0.1,
                "state": "normal"
            },
            {
                "timestamp": "2026-02-05T10:01:00",
                "raw_metric": 45.0,
                "ewma_signal": 0.5,
                "stress_score": 0.6,
                "state": "stressed"
            },
            {
                "timestamp": "2026-02-05T10:02:00",
                "raw_metric": 95.0,
                "ewma_signal": 0.9,
                "stress_score": 0.95,
                "state": "critical"
            }
        ]
        
        metrics = aggregate_backpressure_metrics(observations)
        
        self.assertIn("queue_depth_mean", metrics)
        self.assertIn("queue_depth_p95", metrics)
        self.assertIn("queue_depth_max", metrics)
        
        self.assertEqual(metrics["queue_depth_max"], 95.0)
        self.assertAlmostEqual(metrics["queue_depth_mean"], 48.3, places=1)


class TestIntegrationWithDetector(unittest.TestCase):
    """Test integration with existing detector snapshots."""
    
    def test_frozen_snapshot_conversion(self):
        """Test converting detector frozen snapshot to role-based format."""
        # Simulate frozen snapshot from detector
        frozen = {
            "channel": "resource_saturation",
            "trigger_time": "2026-02-05T10:05:00",
            "snapshot_window_seconds": 600,
            "data": [
                {
                    "timestamp": "2026-02-05T10:00:00",
                    "raw_metric": {"cpu": 45.0, "memory": 550.0, "threads": 12},
                    "stress_score": 0.4,
                    "state": "stressed"
                },
                {
                    "timestamp": "2026-02-05T10:00:05",
                    "raw_metric": {"cpu": 65.0, "memory": 650.0, "threads": 18},
                    "stress_score": 0.7,
                    "state": "critical"
                }
            ]
        }
        
        # Convert to role-based snapshot
        snapshot = create_role_based_snapshot_from_frozen(
            frozen,
            aggregate_resource_saturation_metrics
        )
        
        # Verify structure
        self.assertIsInstance(snapshot, RoleBasedSnapshot)
        self.assertEqual(snapshot.channel, "resource_saturation")
        self.assertEqual(snapshot.timestamp, "2026-02-05T10:05:00")
        
        # Verify all roles present
        self.assertIn("primary", snapshot.services)
        self.assertIn("upstream", snapshot.services)
        self.assertIn("downstream", snapshot.services)
        
        # Verify primary metrics are aggregated
        primary = snapshot.services["primary"]
        self.assertIn("cpu_mean", primary)
        self.assertAlmostEqual(primary["cpu_mean"], 55.0, places=1)
        
        # Verify synthetic metrics exist
        upstream = snapshot.services["upstream"]
        downstream = snapshot.services["downstream"]
        self.assertIn("cpu_mean", upstream)
        self.assertIn("cpu_mean", downstream)
        
        # Verify relationships (upstream < primary < downstream for CPU)
        self.assertLess(upstream["cpu_mean"], primary["cpu_mean"])
        self.assertGreater(downstream["cpu_mean"], primary["cpu_mean"])
    
    def test_model_input_generation(self):
        """Test generating XGBoost-compatible feature vector."""
        # Create role-based snapshot
        frozen = {
            "channel": "resource_saturation",
            "trigger_time": "2026-02-05T10:00:00",
            "snapshot_window_seconds": 600,
            "data": [
                {
                    "timestamp": "2026-02-05T10:00:00",
                    "raw_metric": {"cpu": 50.0, "memory": 600.0, "threads": 15},
                    "stress_score": 0.5,
                    "state": "critical"
                }
            ]
        }
        
        snapshot = create_role_based_snapshot_from_frozen(
            frozen,
            aggregate_resource_saturation_metrics
        )
        
        # Define XGBoost feature schema (example)
        xgboost_schema = [
            "primary_cpu_mean",
            "primary_memory_mean",
            "primary_thread_count_mean",
            "upstream_cpu_mean",
            "upstream_memory_mean",
            "downstream_cpu_mean",
            "downstream_memory_mean",
            "downstream_queue_depth_mean",  # Will be 0.0 (not in primary)
        ]
        
        # Generate feature vector
        vector = snapshot.to_model_input(xgboost_schema)
        
        # Verify vector properties
        self.assertEqual(len(vector), len(xgboost_schema))
        self.assertTrue(all(isinstance(x, float) for x in vector))
        self.assertTrue(all(x >= 0.0 for x in vector))  # No negative values
        
        # Verify no NaN or Inf
        import math
        self.assertTrue(all(math.isfinite(x) for x in vector))
        
        # Verify primary metrics are present
        self.assertGreater(vector[0], 0.0)  # primary_cpu_mean
        self.assertGreater(vector[1], 0.0)  # primary_memory_mean
        
        # Verify synthetic metrics are present
        self.assertGreater(vector[3], 0.0)  # upstream_cpu_mean
        self.assertGreater(vector[5], 0.0)  # downstream_cpu_mean


if __name__ == "__main__":
    unittest.main()
