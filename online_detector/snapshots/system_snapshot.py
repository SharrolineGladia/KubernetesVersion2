"""
System-Wide Snapshot for XGBoost Anomaly Classification

This module captures cross-sectional system state (29 metrics across 3 services)
compatible with the XGBoost model trained on multi-service anomaly dataset.

Unlike feature_extraction.py (time-series statistics for single channel),
this captures CONCURRENT readings from all services at the moment of anomaly.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import json


@dataclass
class SystemSnapshot:
    """
    Cross-sectional snapshot of entire system state.
    
    Captures 29 metrics across 3 services (notification, web_api, processor)
    at a single point in time, matching XGBoost model's training data format.
    """
    
    # === Identity ===
    timestamp: datetime
    trigger_channel: str  # Which channel triggered the snapshot
    
    # === Notification Service Metrics (9) ===
    notification_cpu: float
    notification_memory: float
    notification_error_rate: float
    notification_api_health: float
    notification_delivery_success: float
    notification_message_rate: float
    notification_queue: float
    notification_queue_depth: float
    notification_thread_count: float
    
    # === Web API Service Metrics (10) ===
    web_api_cpu: float
    web_api_memory: float
    web_api_response_time_p95: float
    web_api_db_connections: float
    web_api_errors: float
    web_api_queue_depth: float
    web_api_redis_health: float
    web_api_requests: float
    web_api_requests_per_second: float
    web_api_thread_count: float
    
    # === Processor Service Metrics (10) ===
    processor_cpu: float
    processor_memory: float
    processor_response_time_p95: float
    processor_db_connections: float
    processor_memory_growth: float
    processor_processing_rate: float
    processor_queue: float
    processor_queue_depth: float
    processor_redis_health: float
    processor_thread_count: float
    
    # === Optional Context ===
    anomaly_label: Optional[int] = None  # For validation: 0=normal, 1=anomaly
    anomaly_type: Optional[str] = None   # For validation: cpu_spike, memory_leak, etc.
    
    def to_model_input(self) -> List[float]:
        """
        Convert to feature vector for XGBoost model inference.
        
        Returns ordered list of 29 metrics matching model training format.
        """
        return [
            # Notification service (9)
            self.notification_cpu,
            self.notification_memory,
            self.notification_error_rate,
            self.notification_api_health,
            self.notification_delivery_success,
            self.notification_message_rate,
            self.notification_queue,
            self.notification_queue_depth,
            self.notification_thread_count,
            
            # Web API service (10)
            self.web_api_cpu,
            self.web_api_memory,
            self.web_api_response_time_p95,
            self.web_api_db_connections,
            self.web_api_errors,
            self.web_api_queue_depth,
            self.web_api_redis_health,
            self.web_api_requests,
            self.web_api_requests_per_second,
            self.web_api_thread_count,
            
            # Processor service (10)
            self.processor_cpu,
            self.processor_memory,
            self.processor_response_time_p95,
            self.processor_db_connections,
            self.processor_memory_growth,
            self.processor_processing_rate,
            self.processor_queue,
            self.processor_queue_depth,
            self.processor_redis_health,
            self.processor_thread_count,
        ]
    
    @staticmethod
    def get_feature_names() -> List[str]:
        """Return ordered list of feature names matching model training data."""
        return [
            "notification_cpu", "notification_memory", "notification_error_rate",
            "notification_api_health", "notification_delivery_success",
            "notification_message_rate", "notification_queue",
            "notification_queue_depth", "notification_thread_count",
            "web_api_cpu", "web_api_memory", "web_api_response_time_p95",
            "web_api_db_connections", "web_api_errors", "web_api_queue_depth",
            "web_api_redis_health", "web_api_requests", "web_api_requests_per_second",
            "web_api_thread_count", "processor_cpu", "processor_memory",
            "processor_response_time_p95", "processor_db_connections",
            "processor_memory_growth", "processor_processing_rate",
            "processor_queue", "processor_queue_depth", "processor_redis_health",
            "processor_thread_count"
        ]
    
    def to_dict(self) -> Dict:
        """Export snapshot as dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "trigger_channel": self.trigger_channel,
            "metrics": {
                "notification": {
                    "cpu": self.notification_cpu,
                    "memory": self.notification_memory,
                    "error_rate": self.notification_error_rate,
                    "api_health": self.notification_api_health,
                    "delivery_success": self.notification_delivery_success,
                    "message_rate": self.notification_message_rate,
                    "queue": self.notification_queue,
                    "queue_depth": self.notification_queue_depth,
                    "thread_count": self.notification_thread_count,
                },
                "web_api": {
                    "cpu": self.web_api_cpu,
                    "memory": self.web_api_memory,
                    "response_time_p95": self.web_api_response_time_p95,
                    "db_connections": self.web_api_db_connections,
                    "errors": self.web_api_errors,
                    "queue_depth": self.web_api_queue_depth,
                    "redis_health": self.web_api_redis_health,
                    "requests": self.web_api_requests,
                    "requests_per_second": self.web_api_requests_per_second,
                    "thread_count": self.web_api_thread_count,
                },
                "processor": {
                    "cpu": self.processor_cpu,
                    "memory": self.processor_memory,
                    "response_time_p95": self.processor_response_time_p95,
                    "db_connections": self.processor_db_connections,
                    "memory_growth": self.processor_memory_growth,
                    "processing_rate": self.processor_processing_rate,
                    "queue": self.processor_queue,
                    "queue_depth": self.processor_queue_depth,
                    "redis_health": self.processor_redis_health,
                    "thread_count": self.processor_thread_count,
                }
            },
            "validation": {
                "anomaly_label": self.anomaly_label,
                "anomaly_type": self.anomaly_type
            }
        }
    
    def to_csv_row(self) -> str:
        """Export as CSV row matching training data format."""
        features = self.to_model_input()
        row = ",".join(f"{v:.2f}" for v in features)
        row += f",{self.timestamp.isoformat()}"
        row += f",{self.anomaly_label if self.anomaly_label is not None else ''}"
        row += f",{self.anomaly_type if self.anomaly_type else ''}"
        return row


class SystemSnapshotCollector:
    """
    Collects system-wide metrics from Prometheus when anomaly is detected.
    
    Queries all 29 metrics across 3 services at the trigger moment.
    """
    
    def __init__(self, prometheus_url: str, namespace: str = "default"):
        self.prometheus_url = prometheus_url
        self.namespace = namespace
    
    def collect_snapshot(
        self,
        trigger_channel: str,
        timestamp: Optional[datetime] = None
    ) -> SystemSnapshot:
        """
        Collect system-wide metrics at given timestamp.
        
        Args:
            trigger_channel: Which channel triggered the snapshot
            timestamp: When to collect metrics (default: now)
        
        Returns:
            SystemSnapshot with all 29 metrics
        """
        import requests
        
        timestamp = timestamp or datetime.utcnow()
        
        # Query Prometheus for all metrics at this timestamp
        def query_metric(metric_name: str, service: str) -> float:
            query = f'{metric_name}{{namespace="{self.namespace}",service="{service}"}}'
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": query, "time": timestamp.isoformat()}
            )
            result = response.json()
            
            if result["status"] == "success" and result["data"]["result"]:
                return float(result["data"]["result"][0]["value"][1])
            return 0.0  # Default if metric not found
        
        # Collect all metrics
        return SystemSnapshot(
            timestamp=timestamp,
            trigger_channel=trigger_channel,
            
            # Notification service
            notification_cpu=query_metric("notification_service_cpu_percent", "notification"),
            notification_memory=query_metric("notification_service_memory_mb", "notification"),
            notification_error_rate=query_metric("notification_service_error_rate", "notification"),
            notification_api_health=query_metric("notification_service_api_health", "notification"),
            notification_delivery_success=query_metric("notification_service_delivery_success", "notification"),
            notification_message_rate=query_metric("notification_service_message_rate", "notification"),
            notification_queue=query_metric("notification_service_queue", "notification"),
            notification_queue_depth=query_metric("notification_service_internal_queue_depth", "notification"),
            notification_thread_count=query_metric("notification_service_thread_count", "notification"),
            
            # Web API service
            web_api_cpu=query_metric("web_api_cpu_percent", "web-api"),
            web_api_memory=query_metric("web_api_memory_mb", "web-api"),
            web_api_response_time_p95=query_metric("web_api_response_time_p95_ms", "web-api"),
            web_api_db_connections=query_metric("web_api_db_connections", "web-api"),
            web_api_errors=query_metric("web_api_errors", "web-api"),
            web_api_queue_depth=query_metric("web_api_queue_depth", "web-api"),
            web_api_redis_health=query_metric("web_api_redis_health", "web-api"),
            web_api_requests=query_metric("web_api_requests", "web-api"),
            web_api_requests_per_second=query_metric("web_api_requests_per_second", "web-api"),
            web_api_thread_count=query_metric("web_api_thread_count", "web-api"),
            
            # Processor service
            processor_cpu=query_metric("processor_cpu_percent", "processor"),
            processor_memory=query_metric("processor_memory_mb", "processor"),
            processor_response_time_p95=query_metric("processor_response_time_p95_ms", "processor"),
            processor_db_connections=query_metric("processor_db_connections", "processor"),
            processor_memory_growth=query_metric("processor_memory_growth", "processor"),
            processor_processing_rate=query_metric("processor_processing_rate", "processor"),
            processor_queue=query_metric("processor_queue", "processor"),
            processor_queue_depth=query_metric("processor_queue_depth", "processor"),
            processor_redis_health=query_metric("processor_redis_health", "processor"),
            processor_thread_count=query_metric("processor_thread_count", "processor"),
        )


# === Example Usage ===

if __name__ == "__main__":
    # Example: Create snapshot manually
    snapshot = SystemSnapshot(
        timestamp=datetime(2026, 2, 5, 14, 35, 0),
        trigger_channel="resource_saturation",
        
        # Notification service
        notification_cpu=45.23,
        notification_memory=256.78,
        notification_error_rate=0.02,
        notification_api_health=1.0,
        notification_delivery_success=0.98,
        notification_message_rate=150.5,
        notification_queue=25.0,
        notification_queue_depth=12.0,
        notification_thread_count=45.0,
        
        # Web API service
        web_api_cpu=67.89,
        web_api_memory=412.34,
        web_api_response_time_p95=125.67,
        web_api_db_connections=15.0,
        web_api_errors=3.0,
        web_api_queue_depth=8.0,
        web_api_redis_health=1.0,
        web_api_requests=5000.0,
        web_api_requests_per_second=200.5,
        web_api_thread_count=60.0,
        
        # Processor service
        processor_cpu=89.12,
        processor_memory=678.90,
        processor_response_time_p95=89.45,
        processor_db_connections=20.0,
        processor_memory_growth=2.5,
        processor_processing_rate=180.0,
        processor_queue=30.0,
        processor_queue_depth=15.0,
        processor_redis_health=1.0,
        processor_thread_count=70.0,
        
        # Validation
        anomaly_label=1,
        anomaly_type="cpu_spike"
    )
    
    print("=== System Snapshot ===\n")
    print(f"Timestamp: {snapshot.timestamp}")
    print(f"Trigger Channel: {snapshot.trigger_channel}\n")
    
    print("=== Model Input (29 features) ===")
    feature_vector = snapshot.to_model_input()
    feature_names = SystemSnapshot.get_feature_names()
    
    for i, (name, value) in enumerate(zip(feature_names, feature_vector)):
        print(f"[{i:2d}] {name:35s} = {value:8.2f}")
    
    print(f"\n=== CSV Row ===")
    print(snapshot.to_csv_row())
    
    print(f"\n=== JSON Export ===")
    print(json.dumps(snapshot.to_dict(), indent=2))
