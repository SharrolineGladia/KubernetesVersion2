"""
Bandwidth and Latency Calculator for Edge-Cloud Escalation

Calculates:
- Payload sizes (raw and compressed) for escalated cases
- Transmission times over constrained links
- Latency profiles for edge vs cloud processing
"""

import json
import zlib
import gzip
import pickle
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class PayloadMetrics:
    """Metrics for a single payload transmission."""
    raw_size_bytes: int
    compressed_size_bytes: int
    compression_ratio: float
    transmission_time_ms: float  # At 10 Mbps
    compression_method: str
    
    def to_dict(self) -> Dict:
        return {
            'raw_size_mb': self.raw_size_bytes / (1024 * 1024),
            'compressed_size_mb': self.compressed_size_bytes / (1024 * 1024),
            'compression_ratio': self.compression_ratio,
            'transmission_time_ms': self.transmission_time_ms,
            'compression_method': self.compression_method
        }


@dataclass
class LatencyProfile:
    """Latency profile for edge vs cloud processing."""
    edge_detection_ms: float  # EWMA detection latency
    edge_classification_ms: float  # XGBoost inference at edge
    edge_total_ms: float  # Total edge-only latency
    
    cloud_transmission_ms: float  # Network transmission to cloud
    cloud_classification_ms: float  # XGBoost inference at cloud
    cloud_total_ms: float  # Total escalated latency
    
    escalation_overhead_ms: float  # Extra latency from escalation
    
    def to_dict(self) -> Dict:
        return {
            'edge_detection_ms': self.edge_detection_ms,
            'edge_classification_ms': self.edge_classification_ms,
            'edge_total_ms': self.edge_total_ms,
            'cloud_transmission_ms': self.cloud_transmission_ms,
            'cloud_classification_ms': self.cloud_classification_ms,
            'cloud_total_ms': self.cloud_total_ms,
            'escalation_overhead_ms': self.escalation_overhead_ms
        }


class BandwidthCalculator:
    """Calculate bandwidth consumption for escalation scenarios."""
    
    def __init__(self, link_bandwidth_mbps: float = 10.0):
        """
        Initialize calculator.
        
        Args:
            link_bandwidth_mbps: Edge-to-cloud link bandwidth in Mbps
        """
        self.link_bandwidth_mbps = link_bandwidth_mbps
        self.link_bandwidth_bytes_per_sec = (link_bandwidth_mbps * 1024 * 1024) / 8
    
    def calculate_payload_size(
        self,
        features: np.ndarray,
        service_metrics: Dict,
        use_compression: bool = True
    ) -> PayloadMetrics:
        """
        Calculate payload size for escalating a snapshot to cloud.
        
        Args:
            features: 27-dim scale-invariant feature vector
            service_metrics: Dict of service metrics
            use_compression: Whether to use gzip compression
        
        Returns:
            PayloadMetrics with size and transmission time
        """
        # Create payload as JSON (realistic format)
        payload = {
            'timestamp': '2026-02-23T10:00:00.000000',
            'trigger': 'resource_saturation',
            'features_scaleinvariant': features.tolist(),
            'service_metrics': {
                name: {
                    'cpu_percent': float(metrics.cpu_percent),
                    'memory_percent': float(metrics.memory_percent),
                    'error_rate': float(metrics.error_rate),
                    'request_rate': float(metrics.request_rate),
                    'response_time_p95': float(metrics.response_time_p95),
                    'thread_count': int(metrics.thread_count),
                    'queue_depth': float(metrics.queue_depth),
                    'requests_per_second': float(metrics.requests_per_second)
                }
                for name, metrics in service_metrics.items()
            },
            'metadata': {
                'service_count': len(service_metrics),
                'edge_node': 'edge-01',
                'escalation_reason': 'low_confidence'
            }
        }
        
        # Serialize to JSON
        json_str = json.dumps(payload, indent=2)
        raw_bytes = json_str.encode('utf-8')
        raw_size = len(raw_bytes)
        
        # Apply compression if enabled
        if use_compression:
            compressed_bytes = gzip.compress(raw_bytes, compresslevel=6)
            compressed_size = len(compressed_bytes)
            compression_method = 'gzip'
        else:
            compressed_bytes = raw_bytes
            compressed_size = raw_size
            compression_method = 'none'
        
        compression_ratio = raw_size / compressed_size if compressed_size > 0 else 1.0
        
        # Calculate transmission time (ms)
        transmission_time_s = compressed_size / self.link_bandwidth_bytes_per_sec
        transmission_time_ms = transmission_time_s * 1000
        
        return PayloadMetrics(
            raw_size_bytes=raw_size,
            compressed_size_bytes=compressed_size,
            compression_ratio=compression_ratio,
            transmission_time_ms=transmission_time_ms,
            compression_method=compression_method
        )
    
    def calculate_daily_bandwidth(
        self,
        escalation_rate: float,
        daily_anomalies: int,
        mean_payload_size_bytes: int
    ) -> Tuple[float, float]:
        """
        Calculate daily bandwidth consumption.
        
        Args:
            escalation_rate: Proportion of anomalies escalated to cloud
            daily_anomalies: Total anomalies detected per day
            mean_payload_size_bytes: Average payload size per escalation
        
        Returns:
            (daily_bandwidth_mb, daily_bandwidth_gb)
        """
        escalated_count = escalation_rate * daily_anomalies
        daily_bytes = escalated_count * mean_payload_size_bytes
        daily_mb = daily_bytes / (1024 * 1024)
        daily_gb = daily_mb / 1024
        
        return daily_mb, daily_gb


class LatencyCalculator:
    """Calculate latency profiles for edge vs cloud processing."""
    
    def __init__(self):
        """Initialize with realistic latency values based on system architecture."""
        # Edge latencies (measured from existing system)
        self.ewma_detection_latency_ms = 5.0  # EWMA computation
        self.xgboost_edge_latency_ms = 15.0   # XGBoost inference at edge (GPU accelerated)
        
        # Cloud latencies
        self.xgboost_cloud_latency_ms = 8.0   # XGBoost inference at cloud (powerful GPU)
        
        # Network characteristics
        self.rtt_ms = 50.0  # Round-trip time edge-to-cloud (typical for regional cloud)
        self.transmission_overhead_ms = 10.0  # Protocol overhead (TCP/TLS handshake)
    
    def calculate_latency_profile(
        self,
        payload_metrics: PayloadMetrics,
        is_escalated: bool
    ) -> LatencyProfile:
        """
        Calculate complete latency profile.
        
        Args:
            payload_metrics: Payload size metrics for transmission calculation
            is_escalated: Whether this case was escalated to cloud
        
        Returns:
            LatencyProfile with edge and cloud latencies
        """
        # Edge-only latency
        edge_total = self.ewma_detection_latency_ms + self.xgboost_edge_latency_ms
        
        if is_escalated:
            # Escalated to cloud: EWMA + transmission + cloud XGBoost
            transmission_time = (
                payload_metrics.transmission_time_ms +
                self.rtt_ms +
                self.transmission_overhead_ms
            )
            cloud_total = (
                self.ewma_detection_latency_ms +
                transmission_time +
                self.xgboost_cloud_latency_ms
            )
            escalation_overhead = cloud_total - edge_total
        else:
            # Handled at edge
            transmission_time = 0.0
            cloud_total = 0.0
            escalation_overhead = 0.0
        
        return LatencyProfile(
            edge_detection_ms=self.ewma_detection_latency_ms,
            edge_classification_ms=self.xgboost_edge_latency_ms,
            edge_total_ms=edge_total,
            cloud_transmission_ms=transmission_time,
            cloud_classification_ms=self.xgboost_cloud_latency_ms if is_escalated else 0.0,
            cloud_total_ms=cloud_total,
            escalation_overhead_ms=escalation_overhead
        )
    
    def calculate_expected_latency(
        self,
        escalation_rate: float,
        edge_latency_ms: float,
        cloud_latency_ms: float
    ) -> float:
        """
        Calculate expected latency given escalation rate.
        
        Args:
            escalation_rate: Proportion of cases escalated
            edge_latency_ms: Mean edge-only latency
            cloud_latency_ms: Mean escalated latency
        
        Returns:
            Expected latency in ms
        """
        return (1 - escalation_rate) * edge_latency_ms + escalation_rate * cloud_latency_ms


def main():
    """Test bandwidth and latency calculations."""
    
    print("=" * 80)
    print("  BANDWIDTH & LATENCY CALCULATOR TEST")
    print("=" * 80)
    
    # Test payload size calculation
    bw_calc = BandwidthCalculator(link_bandwidth_mbps=10.0)
    
    # Create sample data
    features = np.random.rand(27)
    
    # Create dummy service metrics class
    from dataclasses import dataclass as dc
    
    @dc
    class ServiceMetrics:
        cpu_percent: float
        memory_percent: float
        error_rate: float
        request_rate: float
        response_time_p95: float
        thread_count: int
        queue_depth: float
        requests_per_second: float
    
    service_metrics = {
        'service-1': ServiceMetrics(45.2, 51.2, 0.02, 150.0, 125.3, 20, 5.5, 150.0),
        'service-2': ServiceMetrics(52.1, 48.0, 0.01, 180.0, 110.5, 25, 3.2, 180.0),
        'service-3': ServiceMetrics(38.7, 39.0, 0.03, 120.0, 145.8, 18, 7.1, 120.0)
    }
    
    # Calculate payload metrics
    print("\n📦 Payload Analysis:")
    
    payload_raw = bw_calc.calculate_payload_size(features, service_metrics, use_compression=False)
    print(f"   Raw JSON:")
    print(f"      Size: {payload_raw.raw_size_bytes:,} bytes ({payload_raw.raw_size_bytes / 1024:.2f} KB)")
    print(f"      Transmission: {payload_raw.transmission_time_ms:.2f} ms")
    
    payload_compressed = bw_calc.calculate_payload_size(features, service_metrics, use_compression=True)
    print(f"\n   Compressed (gzip):")
    print(f"      Size: {payload_compressed.compressed_size_bytes:,} bytes ({payload_compressed.compressed_size_bytes / 1024:.2f} KB)")
    print(f"      Compression Ratio: {payload_compressed.compression_ratio:.2f}x")
    print(f"      Transmission: {payload_compressed.transmission_time_ms:.2f} ms")
    print(f"      Savings: {(1 - 1/payload_compressed.compression_ratio) * 100:.1f}%")
    
    # Calculate daily bandwidth
    print("\n📊 Daily Bandwidth Consumption:")
    escalation_rates = [0.2, 0.5, 0.8]
    daily_anomalies = 1000
    
    for rate in escalation_rates:
        daily_mb, daily_gb = bw_calc.calculate_daily_bandwidth(
            escalation_rate=rate,
            daily_anomalies=daily_anomalies,
            mean_payload_size_bytes=payload_compressed.compressed_size_bytes
        )
        print(f"   {rate:.0%} escalation: {daily_mb:.2f} MB/day ({daily_gb:.3f} GB/day)")
    
    # Test latency calculation
    print("\n⏱️  Latency Analysis:")
    latency_calc = LatencyCalculator()
    
    # Edge-only
    profile_edge = latency_calc.calculate_latency_profile(payload_compressed, is_escalated=False)
    print(f"   Edge-only: {profile_edge.edge_total_ms:.2f} ms")
    
    # Escalated
    profile_cloud = latency_calc.calculate_latency_profile(payload_compressed, is_escalated=True)
    print(f"   Escalated: {profile_cloud.cloud_total_ms:.2f} ms")
    print(f"   Overhead:  {profile_cloud.escalation_overhead_ms:.2f} ms ({profile_cloud.escalation_overhead_ms / profile_edge.edge_total_ms * 100:.1f}%)")
    
    # Expected latency for different escalation rates
    print("\n📈 Expected Latency by Escalation Rate:")
    for rate in [0.0, 0.2, 0.5, 0.8, 1.0]:
        expected = latency_calc.calculate_expected_latency(
            rate, profile_edge.edge_total_ms, profile_cloud.cloud_total_ms
        )
        print(f"   {rate:.0%} escalation: {expected:.2f} ms")
    
    print("\n" + "=" * 80)
    print("✅ TEST COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
