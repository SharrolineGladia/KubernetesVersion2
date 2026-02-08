"""
Dual-Feature Detector: Topology-Agnostic Detection + Service-Specific RCA

This module provides unified anomaly detection that combines:
1. Scale-invariant features (27 dims) for topology-agnostic classification
2. Service-specific metrics (8 per service) for root cause analysis

Author: Anomaly Detection System
Date: February 8, 2026
"""

import os
import sys
import pickle
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add script directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from explainability_layer import ServiceMetrics, AnomalyExplainer, RCAResult


@dataclass
class DetectionSnapshot:
    """
    Unified snapshot containing both feature sets for detection + RCA.
    
    Attributes:
        features_scaleinvariant: 27-dim array of topology-agnostic features
        service_metrics: Dict mapping service name -> ServiceMetrics
        metadata: Dict with timestamp, service_count, active_services
        anomaly_type: Classification result (normal, cpu_spike, memory_leak, service_crash)
        confidence: Model confidence score (0-1)
        rca_result: Optional RCAResult from explainability layer
    """
    features_scaleinvariant: np.ndarray
    service_metrics: Dict[str, ServiceMetrics]
    metadata: Dict[str, Any]
    anomaly_type: Optional[str] = None
    confidence: Optional[float] = None
    rca_result: Optional[RCAResult] = None


class DualFeatureDetector:
    """
    Dual-feature anomaly detector combining topology-agnostic detection
    with service-specific explainability.
    
    Workflow:
        1. Transform raw metrics → scale-invariant features (27 dims)
        2. XGBoost classification → anomaly type + confidence
        3. RCA analysis → root cause service + recommendations
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.75):
        """
        Initialize detector with trained model.
        
        Args:
            model_path: Path to trained XGBoost model (.pkl)
            confidence_threshold: Minimum confidence for valid detection
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        
        # Load model (handle dict format from pickle)
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
            if isinstance(saved_data, dict):
                self.model = saved_data.get('model', saved_data)
            else:
                self.model = saved_data
        
        # Feature names (must match training data order)
        self.feature_names = [
            'cpu_utilization_mean', 'cpu_utilization_max', 'cpu_variance_coef',
            'memory_utilization_mean', 'memory_utilization_max', 'memory_variance_coef',
            'memory_pressure_max', 'memory_growth_rate',
            'thread_count_mean', 'thread_count_max', 'thread_variance_coef',
            'error_rate', 'error_spike_indicator',
            'response_time_p95_mean', 'response_time_p95_max', 'response_time_variance_coef',
            'request_rate_mean', 'request_rate_max', 'request_variance_coef',
            'queue_depth_mean', 'queue_depth_max', 'queue_variance_coef',
            'service_health_min',
            'cpu_memory_correlation', 'load_error_correlation',
            'normalized_service_count',
            'system_stress_index'
        ]
        
        # Anomaly type mapping
        self.anomaly_classes = ['normal', 'cpu_spike', 'memory_leak', 'service_crash']
        
        # Initialize explainer
        self.explainer = AnomalyExplainer(model_path=model_path)
    
    def create_snapshot(
        self,
        service_metrics: Dict[str, ServiceMetrics],
        timestamp: Optional[datetime] = None
    ) -> DetectionSnapshot:
        """
        Create detection snapshot from raw service metrics.
        
        Args:
            service_metrics: Dict mapping service name -> ServiceMetrics
            timestamp: Optional timestamp (defaults to now)
        
        Returns:
            DetectionSnapshot with both feature sets
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        # Transform to scale-invariant features
        features = self._transform_to_scaleinvariant(service_metrics)
        
        # Build metadata
        metadata = {
            'timestamp': timestamp.isoformat(),
            'service_count': len(service_metrics),
            'active_services': list(service_metrics.keys())
        }
        
        return DetectionSnapshot(
            features_scaleinvariant=features,
            service_metrics=service_metrics,
            metadata=metadata
        )
    
    def _transform_to_scaleinvariant(
        self,
        service_metrics: Dict[str, ServiceMetrics]
    ) -> np.ndarray:
        """
        Transform service-specific metrics to 27 scale-invariant features.
        
        Uses ratios, percentages, and normalized values to achieve
        topology-agnostic representation.
        
        Args:
            service_metrics: Dict of ServiceMetrics per service
        
        Returns:
            27-dimensional feature vector
        """
        if not service_metrics:
            return np.zeros(27)
        
        # Aggregate per-service metrics
        cpu_values = [m.cpu_percent for m in service_metrics.values()]
        memory_values = [m.memory_percent for m in service_metrics.values()]
        threads = [m.thread_count for m in service_metrics.values()]
        errors = [m.error_rate for m in service_metrics.values()]
        response_times = [m.response_time_p95 for m in service_metrics.values()]
        request_rates = [m.request_rate for m in service_metrics.values()]
        queue_depths = [m.queue_depth for m in service_metrics.values()]
        requests_per_sec = [m.requests_per_second for m in service_metrics.values()]
        
        n_services = len(service_metrics)
        
        # CPU features
        cpu_utilization_mean = np.mean(cpu_values) / 100.0
        cpu_utilization_max = np.max(cpu_values) / 100.0
        cpu_variance_coef = (np.std(cpu_values) / (np.mean(cpu_values) + 1e-6))
        
        # Memory features
        memory_utilization_mean = np.mean(memory_values) / 100.0
        memory_utilization_max = np.max(memory_values) / 100.0
        memory_variance_coef = (np.std(memory_values) / (np.mean(memory_values) + 1e-6))
        memory_pressure_max = np.max(memory_values) / 100.0
        memory_growth_rate = 0.0  # Would calculate from historical data
        
        # Thread features
        thread_count_mean = np.mean(threads) / 100.0
        thread_count_max = np.max(threads) / 100.0
        thread_variance_coef = (np.std(threads) / (np.mean(threads) + 1e-6))
        
        # Error features
        error_rate = np.mean(errors)
        error_spike_indicator = 1.0 if np.max(errors) > 0.1 else 0.0
        
        # Response time features
        response_time_p95_mean = np.mean(response_times) / 1000.0
        response_time_p95_max = np.max(response_times) / 1000.0
        response_time_variance_coef = (np.std(response_times) / (np.mean(response_times) + 1e-6))
        
        # Request rate features
        request_rate_mean = np.mean(request_rates) / 100.0
        request_rate_max = np.max(request_rates) / 100.0
        request_variance_coef = (np.std(request_rates) / (np.mean(request_rates) + 1e-6))
        
        # Queue features
        queue_depth_mean = np.mean(queue_depths) / 100.0
        queue_depth_max = np.max(queue_depths) / 100.0
        queue_variance_coef = (np.std(queue_depths) / (np.mean(queue_depths) + 1e-6))
        
        # Health indicator (assume healthy if no high error rates)
        service_health_min = 1.0 if np.max(errors) < 0.05 else 0.0
        
        # Correlations
        cpu_memory_correlation = np.corrcoef(cpu_values, memory_values)[0, 1] if n_services > 1 else 0.0
        load_error_correlation = np.corrcoef(cpu_values, errors)[0, 1] if n_services > 1 and np.std(errors) > 0 else 0.0
        
        # Service count (normalized)
        normalized_service_count = n_services / 10.0  # Assuming max 10 services
        
        # System stress index (weighted combination)
        system_stress_index = (
            0.4 * cpu_utilization_mean +
            0.3 * memory_utilization_mean +
            0.2 * error_rate +
            0.1 * (queue_depth_mean)
        )
        
        # Assemble feature vector
        features = np.array([
            cpu_utilization_mean, cpu_utilization_max, cpu_variance_coef,
            memory_utilization_mean, memory_utilization_max, memory_variance_coef,
            memory_pressure_max, memory_growth_rate,
            thread_count_mean, thread_count_max, thread_variance_coef,
            error_rate, error_spike_indicator,
            response_time_p95_mean, response_time_p95_max, response_time_variance_coef,
            request_rate_mean, request_rate_max, request_variance_coef,
            queue_depth_mean, queue_depth_max, queue_variance_coef,
            service_health_min,
            cpu_memory_correlation, load_error_correlation,
            normalized_service_count,
            system_stress_index
        ])
        
        return features
    
    def detect(
        self,
        snapshot: DetectionSnapshot,
        enable_rca: bool = True
    ) -> DetectionSnapshot:
        """
        Classify anomaly type and optionally perform RCA.
        
        Args:
            snapshot: DetectionSnapshot with scale-invariant features
            enable_rca: Whether to run root cause analysis
        
        Returns:
            Updated snapshot with classification results
        """
        # Get prediction probabilities
        features_reshaped = snapshot.features_scaleinvariant.reshape(1, -1)
        probabilities = self.model.predict_proba(features_reshaped)[0]
        
        # Get predicted class
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = self.anomaly_classes[predicted_class_idx]
        confidence = probabilities[predicted_class_idx]
        
        # Update snapshot
        snapshot.anomaly_type = predicted_class
        snapshot.confidence = confidence
        
        # Run RCA if enabled and anomaly detected
        if enable_rca and predicted_class != 'normal' and confidence >= self.confidence_threshold:
            snapshot.rca_result = self.explainer.explain_anomaly(
                anomaly_type=predicted_class,
                service_metrics=snapshot.service_metrics,
                confidence=confidence
            )
        
        return snapshot
    
    def detect_from_raw(
        self,
        service_metrics: Dict[str, Dict[str, float]],
        timestamp: Optional[datetime] = None,
        enable_rca: bool = True
    ) -> DetectionSnapshot:
        """
        One-shot detection from raw metric dictionaries.
        
        Args:
            service_metrics: Dict mapping service name -> dict of metrics
                Expected metrics: cpu_percent, memory_percent, error_rate,
                request_rate, response_time_p95, thread_count, queue_depth,
                requests_per_second
            timestamp: Optional timestamp
            enable_rca: Whether to run RCA
        
        Returns:
            Complete DetectionSnapshot with results
        """
        # Convert raw dicts to ServiceMetrics objects
        structured_metrics = {}
        for service_name, metrics in service_metrics.items():
            structured_metrics[service_name] = ServiceMetrics(
                cpu_percent=metrics.get('cpu_percent', 0.0),
                memory_percent=metrics.get('memory_percent', 0.0),
                error_rate=metrics.get('error_rate', 0.0),
                request_rate=metrics.get('request_rate', 0.0),
                response_time_p95=metrics.get('response_time_p95', 0.0),
                thread_count=metrics.get('thread_count', 0),
                queue_depth=metrics.get('queue_depth', 0.0),
                requests_per_second=metrics.get('requests_per_second', 0.0)
            )
        
        # Create snapshot
        snapshot = self.create_snapshot(structured_metrics, timestamp)
        
        # Run detection
        return self.detect(snapshot, enable_rca=enable_rca)


if __name__ == "__main__":
    print("Dual-Feature Detector Module")
    print("Import this module to use DualFeatureDetector class")
