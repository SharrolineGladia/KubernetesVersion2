# role_based_snapshot.py
"""
Role-Based Snapshot for Multi-Service Anomaly Classification

This module extends the single-service snapshot to support XGBoost models trained
on multi-service metrics. It uses deterministic synthetic derivation for upstream
and downstream services when only the primary service is available.

DESIGN RATIONALE:
- Primary: Real metrics from the monitored service (e.g., notification-service)
- Upstream/Downstream: Synthetically derived using deterministic transformations
- Enables compatibility with XGBoost models expecting multi-service features
- Prepares architecture for future deployment of actual upstream/downstream services

FUTURE EXTENSIBILITY:
When additional services are deployed, simply replace synthetic derivation with
real Prometheus queries while maintaining the same data structure.
"""

import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional


class RoleBasedSnapshot:
    """
    Multi-service snapshot with role-based structure (primary, upstream, downstream).
    
    The snapshot contains:
    - Primary: Real metrics from the monitored service
    - Upstream: Synthetically derived metrics (deterministic transformations)
    - Downstream: Synthetically derived metrics (deterministic transformations)
    
    Attributes:
        timestamp: ISO format timestamp of snapshot freeze
        channel: Channel name that triggered the snapshot
        window_seconds: Duration of observation window
        services: Dict with keys 'primary', 'upstream', 'downstream'
    """
    
    def __init__(
        self,
        timestamp: str,
        channel: str,
        window_seconds: int,
        primary_metrics: Dict[str, Any],
        upstream_metrics: Optional[Dict[str, Any]] = None,
        downstream_metrics: Optional[Dict[str, Any]] = None
    ):
        self.timestamp = timestamp
        self.channel = channel
        self.window_seconds = window_seconds
        
        self.services = {
            "primary": primary_metrics,
            "upstream": upstream_metrics or {},
            "downstream": downstream_metrics or {}
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Export snapshot as dictionary."""
        return {
            "timestamp": self.timestamp,
            "channel": self.channel,
            "window_seconds": self.window_seconds,
            "services": self.services
        }
    
    def to_model_input(self, feature_schema: List[str]) -> List[float]:
        """
        Convert snapshot to flat feature vector for XGBoost model.
        
        Args:
            feature_schema: Ordered list of feature names expected by model
                           e.g., ['primary_cpu', 'upstream_latency_p95', ...]
        
        Returns:
            List of floats matching feature_schema order
            
        Example:
            >>> schema = ['primary_cpu', 'downstream_queue_depth', 'upstream_memory']
            >>> snapshot.to_model_input(schema)
            [45.2, 78.5, 512.0]
        """
        feature_vector = []
        
        for feature_name in feature_schema:
            value = self._extract_feature(feature_name)
            feature_vector.append(value)
        
        return feature_vector
    
    def _extract_feature(self, feature_name: str) -> float:
        """
        Extract a single feature value from the snapshot.
        
        Feature name format: {role}_{metric_name}
        Examples: primary_cpu, upstream_latency_p95, downstream_queue_depth
        
        Returns 0.0 if feature not found (safe default for missing features).
        """
        # Parse role and metric from feature name
        parts = feature_name.split('_', 1)
        if len(parts) < 2:
            return 0.0
        
        role, metric_name = parts[0], parts[1]
        
        # Get service metrics
        if role not in self.services:
            return 0.0
        
        service_metrics = self.services[role]
        
        # Handle nested metric paths (e.g., 'latency_p95' or 'cpu_percent')
        return self._get_nested_value(service_metrics, metric_name, 0.0)
    
    def _get_nested_value(self, data: Dict, key_path: str, default: float) -> float:
        """
        Get value from potentially nested dict structure.
        
        Args:
            data: Dictionary to search
            key_path: Metric name (may contain nested keys)
            default: Value to return if not found
        """
        # Try direct key first
        if key_path in data:
            value = data[key_path]
            return float(value) if isinstance(value, (int, float)) else default
        
        # Try common patterns
        # e.g., 'cpu_percent' might be stored as {'cpu': {'percent': X}}
        parts = key_path.split('_')
        current = data
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        
        return float(current) if isinstance(current, (int, float)) else default


class SyntheticMetricDerivation:
    """
    Deterministic synthetic metric generation for upstream/downstream services.
    
    DESIGN PRINCIPLES:
    1. Deterministic: Same primary metrics always produce same synthetic metrics
    2. Statistically plausible: Mimics real service correlation patterns
    3. Causal relationships: Upstream issues propagate downstream
    4. NO RANDOMNESS: Uses deterministic transformations only
    
    TRANSFORMATION STRATEGIES:
    - CPU/Memory: Apply role-specific load factors with correlation
    - Latency: Propagate with cumulative delays (upstream → primary → downstream)
    - Queue depth: Backpressure propagates upstream
    - Error rates: Correlated with delay patterns
    - Throughput: Inversely correlated with latency
    """
    
    # Deterministic seed generation based on timestamp
    # Ensures reproducibility: same timestamp → same transformations
    @staticmethod
    def _get_deterministic_factor(timestamp: str, role: str, metric: str) -> float:
        """
        Generate a deterministic factor in range [0, 1] based on inputs.
        
        Uses cryptographic hashing to create stable, evenly distributed values.
        Same inputs always produce same output.
        """
        seed_string = f"{timestamp}:{role}:{metric}"
        hash_digest = hashlib.sha256(seed_string.encode()).hexdigest()
        # Use first 8 hex chars as integer, normalize to [0, 1]
        hash_value = int(hash_digest[:8], 16)
        return (hash_value % 1000) / 1000.0  # Range: 0.000 to 0.999
    
    @staticmethod
    def derive_upstream_metrics(
        primary_metrics: Dict[str, Any],
        timestamp: str,
        observation_window: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Derive upstream service metrics from primary service.
        
        UPSTREAM LOGIC:
        - Upstream services are request sources (API gateways, load balancers, clients)
        - Primary performance degradation often caused by upstream load spikes
        - Upstream metrics should show early warning signs
        
        Args:
            primary_metrics: Aggregated metrics from primary service
            timestamp: Snapshot timestamp for deterministic derivation
            observation_window: Time-series observations for pattern detection
        
        Returns:
            Dict of synthetic upstream metrics
        """
        upstream = {}
        
        # ============================================
        # CPU & MEMORY: Upstream typically has LOWER load
        # (primary does the processing work)
        # ============================================
        if "cpu_mean" in primary_metrics:
            # Upstream CPU: 60-80% of primary
            factor = 0.6 + 0.2 * SyntheticMetricDerivation._get_deterministic_factor(
                timestamp, "upstream", "cpu"
            )
            upstream["cpu_mean"] = primary_metrics["cpu_mean"] * factor
            
            if "cpu_p95" in primary_metrics:
                upstream["cpu_p95"] = primary_metrics["cpu_p95"] * factor
        
        if "memory_mean" in primary_metrics:
            # Upstream memory: 50-70% of primary
            factor = 0.5 + 0.2 * SyntheticMetricDerivation._get_deterministic_factor(
                timestamp, "upstream", "memory"
            )
            upstream["memory_mean"] = primary_metrics["memory_mean"] * factor
        
        # ============================================
        # LATENCY: Upstream adds network + processing delay
        # ============================================
        if "latency_p95" in primary_metrics:
            # Upstream latency: includes its own processing + call to primary
            # Typically 70-90% of primary's latency (doesn't include downstream)
            base_latency = primary_metrics["latency_p95"]
            factor = 0.7 + 0.2 * SyntheticMetricDerivation._get_deterministic_factor(
                timestamp, "upstream", "latency"
            )
            upstream["latency_p95"] = base_latency * factor
            
            if "latency_mean" in primary_metrics:
                upstream["latency_mean"] = primary_metrics["latency_mean"] * factor
        
        # ============================================
        # QUEUE DEPTH: Backpressure propagates UPSTREAM
        # ============================================
        if "queue_depth_mean" in primary_metrics:
            # When primary queue fills, upstream also experiences queueing
            primary_queue = primary_metrics["queue_depth_mean"]
            
            # Upstream queue: 80-120% of primary (backpressure effect)
            factor = 0.8 + 0.4 * SyntheticMetricDerivation._get_deterministic_factor(
                timestamp, "upstream", "queue"
            )
            upstream["queue_depth_mean"] = primary_queue * factor
            
            if "queue_depth_p95" in primary_metrics:
                upstream["queue_depth_p95"] = primary_metrics["queue_depth_p95"] * factor
        
        # ============================================
        # ERROR RATE: Correlated but upstream may have different patterns
        # ============================================
        if "error_rate_mean" in primary_metrics:
            # Upstream errors: 50-90% of primary (may have different failure modes)
            factor = 0.5 + 0.4 * SyntheticMetricDerivation._get_deterministic_factor(
                timestamp, "upstream", "error"
            )
            upstream["error_rate_mean"] = primary_metrics["error_rate_mean"] * factor
        
        # ============================================
        # THROUGHPUT: Typically HIGHER than primary
        # (upstream handles more requests, primary may be bottleneck)
        # ============================================
        if "throughput_mean" in primary_metrics:
            # Upstream throughput: 100-150% of primary
            factor = 1.0 + 0.5 * SyntheticMetricDerivation._get_deterministic_factor(
                timestamp, "upstream", "throughput"
            )
            upstream["throughput_mean"] = primary_metrics["throughput_mean"] * factor
        
        # ============================================
        # THREAD COUNT: Similar to CPU relationship
        # ============================================
        if "thread_count_mean" in primary_metrics:
            factor = 0.6 + 0.3 * SyntheticMetricDerivation._get_deterministic_factor(
                timestamp, "upstream", "threads"
            )
            upstream["thread_count_mean"] = primary_metrics["thread_count_mean"] * factor
        
        return upstream
    
    @staticmethod
    def derive_downstream_metrics(
        primary_metrics: Dict[str, Any],
        timestamp: str,
        observation_window: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Derive downstream service metrics from primary service.
        
        DOWNSTREAM LOGIC:
        - Downstream services are called by primary (databases, caches, APIs)
        - Primary performance issues CAUSE downstream degradation
        - Downstream metrics amplify primary stress patterns
        
        Args:
            primary_metrics: Aggregated metrics from primary service
            timestamp: Snapshot timestamp for deterministic derivation
            observation_window: Time-series observations for pattern detection
        
        Returns:
            Dict of synthetic downstream metrics
        """
        downstream = {}
        
        # ============================================
        # CPU & MEMORY: Downstream typically has HIGHER load
        # (databases/caches under heavy query load)
        # ============================================
        if "cpu_mean" in primary_metrics:
            # Downstream CPU: 120-180% of primary (amplification effect)
            factor = 1.2 + 0.6 * SyntheticMetricDerivation._get_deterministic_factor(
                timestamp, "downstream", "cpu"
            )
            downstream["cpu_mean"] = primary_metrics["cpu_mean"] * factor
            
            if "cpu_p95" in primary_metrics:
                downstream["cpu_p95"] = primary_metrics["cpu_p95"] * factor
        
        if "memory_mean" in primary_metrics:
            # Downstream memory: 100-150% of primary (cache pressure)
            factor = 1.0 + 0.5 * SyntheticMetricDerivation._get_deterministic_factor(
                timestamp, "downstream", "memory"
            )
            downstream["memory_mean"] = primary_metrics["memory_mean"] * factor
        
        # ============================================
        # LATENCY: Downstream is COMPONENT of primary latency
        # ============================================
        if "latency_p95" in primary_metrics:
            # Downstream latency: 40-70% of primary's total latency
            # (primary latency = downstream latency + own processing)
            base_latency = primary_metrics["latency_p95"]
            factor = 0.4 + 0.3 * SyntheticMetricDerivation._get_deterministic_factor(
                timestamp, "downstream", "latency"
            )
            downstream["latency_p95"] = base_latency * factor
            
            if "latency_mean" in primary_metrics:
                downstream["latency_mean"] = primary_metrics["latency_mean"] * factor
        
        # ============================================
        # QUEUE DEPTH: Downstream queues fill FIRST
        # ============================================
        if "queue_depth_mean" in primary_metrics:
            # Downstream queue: 130-200% of primary (downstream bottleneck)
            primary_queue = primary_metrics["queue_depth_mean"]
            factor = 1.3 + 0.7 * SyntheticMetricDerivation._get_deterministic_factor(
                timestamp, "downstream", "queue"
            )
            downstream["queue_depth_mean"] = primary_queue * factor
            
            if "queue_depth_p95" in primary_metrics:
                downstream["queue_depth_p95"] = primary_metrics["queue_depth_p95"] * factor
        
        # ============================================
        # ERROR RATE: Downstream errors CAUSE primary errors
        # ============================================
        if "error_rate_mean" in primary_metrics:
            # Downstream errors: 120-180% of primary (source of failures)
            factor = 1.2 + 0.6 * SyntheticMetricDerivation._get_deterministic_factor(
                timestamp, "downstream", "error"
            )
            downstream["error_rate_mean"] = primary_metrics["error_rate_mean"] * factor
        
        # ============================================
        # THROUGHPUT: HIGHER than primary
        # (primary makes multiple downstream calls per request)
        # ============================================
        if "throughput_mean" in primary_metrics:
            # Downstream throughput: 150-300% of primary (query amplification)
            factor = 1.5 + 1.5 * SyntheticMetricDerivation._get_deterministic_factor(
                timestamp, "downstream", "throughput"
            )
            downstream["throughput_mean"] = primary_metrics["throughput_mean"] * factor
        
        # ============================================
        # CONNECTION POOL: Downstream-specific metric
        # ============================================
        if "connection_pool_usage" in primary_metrics:
            # Downstream connections: directly related to primary usage
            factor = 1.1 + 0.3 * SyntheticMetricDerivation._get_deterministic_factor(
                timestamp, "downstream", "connections"
            )
            downstream["connection_pool_usage"] = primary_metrics["connection_pool_usage"] * factor
        
        # ============================================
        # THREAD COUNT: Higher for concurrent queries
        # ============================================
        if "thread_count_mean" in primary_metrics:
            factor = 1.3 + 0.5 * SyntheticMetricDerivation._get_deterministic_factor(
                timestamp, "downstream", "threads"
            )
            downstream["thread_count_mean"] = primary_metrics["thread_count_mean"] * factor
        
        return downstream


def create_role_based_snapshot_from_frozen(
    frozen_snapshot: Dict[str, Any],
    extract_primary_metrics_fn: callable
) -> RoleBasedSnapshot:
    """
    Convert a traditional frozen snapshot into a role-based snapshot.
    
    This function bridges the existing EWMA detector with the new role-based format.
    It extracts primary metrics from the frozen snapshot and derives synthetic
    upstream/downstream metrics.
    
    Args:
        frozen_snapshot: Traditional snapshot from detector._freeze_snapshot()
                        Format: {
                            "channel": str,
                            "trigger_time": str,
                            "snapshot_window_seconds": int,
                            "data": List[Dict]  # observation buffer
                        }
        extract_primary_metrics_fn: Function that aggregates observation buffer
                                    into primary service metrics
    
    Returns:
        RoleBasedSnapshot with primary (real) + upstream/downstream (synthetic)
    
    Example:
        >>> def extract_fn(observations):
        ...     return {
        ...         "cpu_mean": statistics.mean(o["stress_score"] for o in observations),
        ...         "latency_p95": ...,
        ...     }
        >>> snapshot = create_role_based_snapshot_from_frozen(
        ...     detector.frozen_snapshot,
        ...     extract_fn
        ... )
    """
    # Extract basic snapshot info
    channel = frozen_snapshot["channel"]
    trigger_time = frozen_snapshot["trigger_time"]
    window_seconds = frozen_snapshot["snapshot_window_seconds"]
    observations = frozen_snapshot["data"]
    
    # Extract primary metrics using provided aggregation function
    primary_metrics = extract_primary_metrics_fn(observations)
    
    # Derive synthetic metrics for upstream/downstream
    upstream_metrics = SyntheticMetricDerivation.derive_upstream_metrics(
        primary_metrics, trigger_time, observations
    )
    downstream_metrics = SyntheticMetricDerivation.derive_downstream_metrics(
        primary_metrics, trigger_time, observations
    )
    
    # Create role-based snapshot
    return RoleBasedSnapshot(
        timestamp=trigger_time,
        channel=channel,
        window_seconds=window_seconds,
        primary_metrics=primary_metrics,
        upstream_metrics=upstream_metrics,
        downstream_metrics=downstream_metrics
    )


# ============================================================================
# EXAMPLE AGGREGATION FUNCTIONS FOR DIFFERENT CHANNELS
# ============================================================================

def aggregate_resource_saturation_metrics(observations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate resource saturation observations into primary metrics.
    
    Input observations have structure:
    {
        "timestamp": "...",
        "raw_metric": {"cpu": X, "memory": Y, "threads": Z},
        "stress_score": float,
        "state": str
    }
    """
    import statistics
    
    if not observations:
        return {}
    
    # Extract time series
    cpu_values = [o.get("raw_metric", {}).get("cpu", 0) for o in observations if o.get("raw_metric")]
    memory_values = [o.get("raw_metric", {}).get("memory", 0) for o in observations if o.get("raw_metric")]
    thread_values = [o.get("raw_metric", {}).get("threads", 0) for o in observations if o.get("raw_metric")]
    stress_scores = [o["stress_score"] for o in observations if "stress_score" in o]
    
    metrics = {}
    
    if cpu_values:
        metrics["cpu_mean"] = statistics.mean(cpu_values)
        metrics["cpu_p95"] = sorted(cpu_values)[int(len(cpu_values) * 0.95)] if len(cpu_values) > 1 else cpu_values[0]
        metrics["cpu_std"] = statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0.0
    
    if memory_values:
        metrics["memory_mean"] = statistics.mean(memory_values)
        metrics["memory_p95"] = sorted(memory_values)[int(len(memory_values) * 0.95)] if len(memory_values) > 1 else memory_values[0]
    
    if thread_values:
        metrics["thread_count_mean"] = statistics.mean(thread_values)
        metrics["thread_count_p95"] = sorted(thread_values)[int(len(thread_values) * 0.95)] if len(thread_values) > 1 else thread_values[0]
    
    if stress_scores:
        metrics["stress_score_mean"] = statistics.mean(stress_scores)
        metrics["stress_score_max"] = max(stress_scores)
    
    return metrics


def aggregate_performance_degradation_metrics(observations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate performance degradation observations (p95 latency).
    
    Input observations have structure:
    {
        "timestamp": "...",
        "raw_metric": float,  # p95_response_time_ms
        "ewma_signal": float,
        "absolute_stress": float,
        "stress_score": float,
        "state": str
    }
    """
    import statistics
    
    if not observations:
        return {}
    
    # Extract latency values
    latency_values = [o["raw_metric"] for o in observations if "raw_metric" in o]
    stress_scores = [o["stress_score"] for o in observations if "stress_score" in o]
    
    metrics = {}
    
    if latency_values:
        metrics["latency_p95"] = sorted(latency_values)[int(len(latency_values) * 0.95)] if len(latency_values) > 1 else latency_values[0]
        metrics["latency_mean"] = statistics.mean(latency_values)
        metrics["latency_max"] = max(latency_values)
        metrics["latency_std"] = statistics.stdev(latency_values) if len(latency_values) > 1 else 0.0
    
    if stress_scores:
        metrics["stress_score_mean"] = statistics.mean(stress_scores)
        metrics["stress_score_max"] = max(stress_scores)
    
    return metrics


def aggregate_backpressure_metrics(observations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate backpressure overload observations (queue depth).
    
    Input observations have structure:
    {
        "timestamp": "...",
        "raw_metric": float,  # queue_depth
        "ewma_signal": float,
        "absolute_stress": float,
        "stress_score": float,
        "state": str
    }
    """
    import statistics
    
    if not observations:
        return {}
    
    # Extract queue depth values
    queue_values = [o["raw_metric"] for o in observations if "raw_metric" in o]
    stress_scores = [o["stress_score"] for o in observations if "stress_score" in o]
    
    metrics = {}
    
    if queue_values:
        metrics["queue_depth_mean"] = statistics.mean(queue_values)
        metrics["queue_depth_p95"] = sorted(queue_values)[int(len(queue_values) * 0.95)] if len(queue_values) > 1 else queue_values[0]
        metrics["queue_depth_max"] = max(queue_values)
        metrics["queue_depth_std"] = statistics.stdev(queue_values) if len(queue_values) > 1 else 0.0
    
    if stress_scores:
        metrics["stress_score_mean"] = statistics.mean(stress_scores)
        metrics["stress_score_max"] = max(stress_scores)
    
    return metrics
