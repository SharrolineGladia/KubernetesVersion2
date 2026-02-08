"""
Explainability Layer for Scale-Invariant Anomaly Detection

This module provides Root Cause Analysis (RCA) and explainability for detected anomalies
by analyzing service-specific metrics using SHAP values and statistical analysis.

Architecture:
1. XGBoost detects anomalies using scale-invariant features (topology-agnostic)
2. This layer explains WHY by analyzing raw service-specific metrics (granular)

Key Features:
- Service-level attribution (which service caused the anomaly?)
- Feature importance using SHAP
- Temporal pattern analysis
- Resource correlation analysis
- Actionable recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import pickle


@dataclass
class ServiceMetrics:
    """Raw metrics for a single service (8 key metrics)."""
    cpu_percent: float  # CPU utilization (0-100%)
    memory_percent: float  # Memory utilization (0-100%)
    error_rate: float  # Error rate (0-1)
    request_rate: float  # Request rate (requests/sec)
    response_time_p95: float  # 95th percentile response time (ms)
    thread_count: int  # Active thread count
    queue_depth: float  # Queue/backlog depth
    requests_per_second: float  # Throughput metric
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'error_rate': self.error_rate,
            'request_rate': self.request_rate,
            'response_time_p95': self.response_time_p95,
            'thread_count': self.thread_count,
            'queue_depth': self.queue_depth,
            'requests_per_second': self.requests_per_second
        }


@dataclass
class RCAResult:
    """Root cause analysis result."""
    anomaly_type: str
    root_cause_service: str
    confidence: float
    contributing_factors: List[Dict[str, Any]]
    recommendations: List[str]
    severity: str  # 'low', 'medium', 'high', 'critical'
    affected_resources: List[str]
    timestamp: str
    
    def to_dict(self) -> Dict:
        return {
            'anomaly_type': self.anomaly_type,
            'root_cause_service': self.root_cause_service,
            'confidence': self.confidence,
            'contributing_factors': self.contributing_factors,
            'recommendations': self.recommendations,
            'severity': self.severity,
            'affected_resources': self.affected_resources,
            'timestamp': self.timestamp
        }


class AnomalyExplainer:
    """
    Provides explainability for detected anomalies using service-specific metrics.
    
    This class bridges the gap between topology-agnostic detection and
    actionable, service-specific explanations.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize explainer.
        
        Args:
            model_path: Path to trained XGBoost model (for SHAP analysis)
        """
        self.model = None
        self.shap_explainer = None
        
        if model_path:
            self.load_model(model_path)
        
        # Anomaly-specific thresholds for RCA
        self.thresholds = {
            'cpu_spike': {'cpu': 85, 'cpu_growth_rate': 0.3},
            'memory_leak': {'memory': 75, 'memory_growth_rate': 0.2},
            'service_crash': {'error_rate': 0.5, 'request_drop': 0.7},
            'network_congestion': {'network_out': 80, 'latency': 500}
        }
    
    def load_model(self, model_path: str):
        """Load trained model for SHAP analysis."""
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
        
        # Handle both dict format and direct model format
        if isinstance(saved_data, dict):
            self.model = saved_data.get('model', saved_data)
        else:
            self.model = saved_data
        
        # Initialize SHAP explainer (requires shap library)
        try:
            import shap
            self.shap_explainer = shap.TreeExplainer(self.model)
        except ImportError:
            print("⚠️ SHAP library not installed. Install with: pip install shap")
            self.shap_explainer = None
        except Exception as e:
            print(f"⚠️ SHAP initialization skipped: {e}")
            self.shap_explainer = None
    
    def explain_anomaly(
        self,
        anomaly_type: str,
        service_metrics: Dict[str, ServiceMetrics],
        scale_invariant_features: Dict[str, float],
        timestamp: Optional[datetime] = None
    ) -> RCAResult:
        """
        Explain detected anomaly by analyzing service-specific metrics.
        
        Args:
            anomaly_type: Detected anomaly type from XGBoost model
            service_metrics: Raw metrics per service
            scale_invariant_features: The 27 scale-invariant features used for detection
            timestamp: Time of detection
        
        Returns:
            RCAResult with root cause, contributing factors, and recommendations
        """
        timestamp = timestamp or datetime.utcnow()
        
        # Step 1: Identify root cause service
        root_cause_service, confidence = self._identify_root_cause_service(
            anomaly_type, service_metrics
        )
        
        # Step 2: Analyze contributing factors
        contributing_factors = self._analyze_contributing_factors(
            anomaly_type,
            root_cause_service,
            service_metrics,
            scale_invariant_features
        )
        
        # Step 3: Determine severity
        severity = self._calculate_severity(
            anomaly_type,
            service_metrics[root_cause_service],
            scale_invariant_features
        )
        
        # Step 4: Generate recommendations
        recommendations = self._generate_recommendations(
            anomaly_type,
            root_cause_service,
            service_metrics,
            severity
        )
        
        # Step 5: Identify affected resources
        affected_resources = self._identify_affected_resources(
            anomaly_type,
            service_metrics[root_cause_service]
        )
        
        return RCAResult(
            anomaly_type=anomaly_type,
            root_cause_service=root_cause_service,
            confidence=confidence,
            contributing_factors=contributing_factors,
            recommendations=recommendations,
            severity=severity,
            affected_resources=affected_resources,
            timestamp=timestamp.isoformat()
        )
    
    def _identify_root_cause_service(
        self,
        anomaly_type: str,
        service_metrics: Dict[str, ServiceMetrics]
    ) -> Tuple[str, float]:
        """
        Identify which service is the root cause of the anomaly.
        
        Returns:
            (service_name, confidence_score)
        """
        if anomaly_type == 'cpu_spike':
            return self._find_cpu_hotspot(service_metrics)
        elif anomaly_type == 'memory_leak':
            return self._find_memory_leak_service(service_metrics)
        elif anomaly_type == 'service_crash':
            return self._find_crashed_service(service_metrics)
        else:
            # Default: find service with highest overall stress
            return self._find_most_stressed_service(service_metrics)
    
    def _find_cpu_hotspot(
        self,
        service_metrics: Dict[str, ServiceMetrics]
    ) -> Tuple[str, float]:
        """Find service with highest CPU usage."""
        max_cpu = -1
        hotspot_service = None
        
        for service_name, metrics in service_metrics.items():
            if metrics.cpu_percent > max_cpu:
                max_cpu = metrics.cpu_percent
                hotspot_service = service_name
        
        # Confidence based on how much higher than others
        cpu_values = [m.cpu_percent for m in service_metrics.values()]
        if len(cpu_values) > 1:
            cpu_values.remove(max_cpu)
            avg_others = np.mean(cpu_values)
            confidence = min(1.0, (max_cpu - avg_others) / 100.0)
        else:
            confidence = 0.8 if max_cpu > 85 else 0.6
        
        return hotspot_service, confidence
    
    def _find_memory_leak_service(
        self,
        service_metrics: Dict[str, ServiceMetrics]
    ) -> Tuple[str, float]:
        """Find service with memory leak symptoms."""
        max_memory = -1
        leak_service = None
        
        for service_name, metrics in service_metrics.items():
            if metrics.memory_percent > max_memory:
                max_memory = metrics.memory_percent
                leak_service = service_name
        
        # High memory + high confidence
        confidence = min(1.0, max_memory / 100.0)
        
        return leak_service, confidence
    
    def _find_crashed_service(
        self,
        service_metrics: Dict[str, ServiceMetrics]
    ) -> Tuple[str, float]:
        """Find service showing crash symptoms."""
        for service_name, metrics in service_metrics.items():
            error_rate = metrics.error_rate
            request_drop = 1.0 - min(1.0, metrics.request_rate / 100.0)
            
            # High error rate OR significant request drop
            if error_rate > 0.5 or request_drop > 0.7:
                return service_name, 0.9
        
        # Fallback: service with highest error rate
        max_error_rate = -1
        crash_service = None
        
        for service_name, metrics in service_metrics.items():
            error_rate = metrics.error_rate
            if error_rate > max_error_rate:
                max_error_rate = error_rate
                crash_service = service_name
        
        confidence = min(1.0, max_error_rate * 2)
        return crash_service, confidence
    
    def _find_most_stressed_service(
        self,
        service_metrics: Dict[str, ServiceMetrics]
    ) -> Tuple[str, float]:
        """Find service with highest overall stress."""
        max_stress = -1
        stressed_service = None
        
        for service_name, metrics in service_metrics.items():
            # Composite stress score
            stress = (
                metrics.cpu_percent / 100.0 * 0.3 +
                metrics.memory_percent / 100.0 * 0.3 +
                metrics.error_rate * 0.2 +
                min(1.0, metrics.response_time_p95 / 1000.0) * 0.2
            )
            
            if stress > max_stress:
                max_stress = stress
                stressed_service = service_name
        
        confidence = min(1.0, max_stress)
        return stressed_service, confidence
    
    def _analyze_contributing_factors(
        self,
        anomaly_type: str,
        root_cause_service: str,
        service_metrics: Dict[str, ServiceMetrics],
        scale_invariant_features: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Identify metrics contributing to the anomaly."""
        factors = []
        service = service_metrics[root_cause_service]
        
        # Analyze based on anomaly type
        if anomaly_type == 'cpu_spike':
            if service.cpu_percent > 85:
                factors.append({
                    'metric': 'cpu_utilization',
                    'value': f"{service.cpu_percent:.1f}%",
                    'severity': 'critical' if service.cpu_percent > 95 else 'high',
                    'description': f'CPU usage extremely high'
                })
            
            if service.request_rate > 100:
                factors.append({
                    'metric': 'request_rate',
                    'value': f"{service.request_rate:.0f} req/s",
                    'severity': 'medium',
                    'description': 'High request load'
                })
        
        elif anomaly_type == 'memory_leak':
            if service.memory_percent > 75:
                factors.append({
                    'metric': 'memory_usage',
                    'value': f"{service.memory_percent:.1f}%",
                    'severity': 'critical' if service.memory_percent > 90 else 'high',
                    'description': 'Memory usage critically high'
                })
            
            # Check if memory/cpu ratio is abnormal (leak pattern)
            if service.cpu_percent > 0 and service.memory_percent / service.cpu_percent > 1.5:
                factors.append({
                    'metric': 'memory_cpu_ratio',
                    'value': f"{service.memory_percent / service.cpu_percent:.2f}",
                    'severity': 'high',
                    'description': 'Memory growing disproportionate to CPU (leak pattern)'
                })
        
        elif anomaly_type == 'service_crash':
            error_rate = service.error_rate
            if error_rate > 0.3:
                factors.append({
                    'metric': 'error_rate',
                    'value': f"{error_rate * 100:.1f}%",
                    'severity': 'critical',
                    'description': 'Extremely high error rate'
                })
            
            if service.request_rate < 10:
                factors.append({
                    'metric': 'request_rate',
                    'value': f"{service.request_rate:.0f} req/s",
                    'severity': 'critical',
                    'description': 'Request rate collapsed (crash indicator)'
                })
        
        # Add scale-invariant feature insights
        if scale_invariant_features.get('memory_pressure_max', 0) > 0.9:
            factors.append({
                'metric': 'system_memory_pressure',
                'value': f"{scale_invariant_features['memory_pressure_max']:.2f}",
                'severity': 'high',
                'description': 'System-wide memory pressure detected'
            })
        
        if scale_invariant_features.get('error_variance_coef', 0) > 0.5:
            factors.append({
                'metric': 'error_variability',
                'value': f"{scale_invariant_features['error_variance_coef']:.2f}",
                'severity': 'medium',
                'description': 'Unstable error pattern (high variance)'
            })
        
        return factors
    
    def _calculate_severity(
        self,
        anomaly_type: str,
        service_metrics: ServiceMetrics,
        scale_invariant_features: Dict[str, float]
    ) -> str:
        """Calculate severity level: low, medium, high, critical."""
        severity_score = 0
        
        # Service-level severity
        if service_metrics.cpu_percent > 95:
            severity_score += 3
        elif service_metrics.cpu_percent > 85:
            severity_score += 2
        
        if service_metrics.memory_percent > 90:
            severity_score += 3
        elif service_metrics.memory_percent > 75:
            severity_score += 2
        
        error_rate = service_metrics.error_rate
        if error_rate > 0.5:
            severity_score += 3
        elif error_rate > 0.2:
            severity_score += 2
        
        # System-level severity
        if scale_invariant_features.get('system_stress', 0) > 0.8:
            severity_score += 2
        
        # Classify
        if severity_score >= 7:
            return 'critical'
        elif severity_score >= 5:
            return 'high'
        elif severity_score >= 3:
            return 'medium'
        else:
            return 'low'
    
    def _generate_recommendations(
        self,
        anomaly_type: str,
        root_cause_service: str,
        service_metrics: Dict[str, ServiceMetrics],
        severity: str
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        service = service_metrics[root_cause_service]
        
        if anomaly_type == 'cpu_spike':
            recommendations.append(
                f"🔧 Scale {root_cause_service} horizontally (add replicas)"
            )
            if service.request_rate > 100:
                recommendations.append(
                    f"⚡ Enable rate limiting to reduce load on {root_cause_service}"
                )
            recommendations.append(
                f"🔍 Profile {root_cause_service} for CPU-intensive operations"
            )
            if severity in ['high', 'critical']:
                recommendations.append(
                    f"🚨 URGENT: Consider emergency scaling or failover for {root_cause_service}"
                )
        
        elif anomaly_type == 'memory_leak':
            recommendations.append(
                f"🔧 Restart {root_cause_service} pod to clear leaked memory"
            )
            recommendations.append(
                f"🐛 Investigate memory leak in {root_cause_service} codebase"
            )
            recommendations.append(
                f"📊 Enable memory profiling for {root_cause_service}"
            )
            if service.memory_percent > 90:
                recommendations.append(
                    f"⚠️ Increase memory limits for {root_cause_service} deployment"
                )
        
        elif anomaly_type == 'service_crash':
            recommendations.append(
                f"🚨 IMMEDIATE: Check {root_cause_service} logs for crash details"
            )
            recommendations.append(
                f"🔄 Restart {root_cause_service} deployment"
            )
            recommendations.append(
                f"🔍 Review recent changes to {root_cause_service}"
            )
            recommendations.append(
                f"🛡️ Enable health checks and liveness probes for {root_cause_service}"
            )
        
        # Cross-service recommendations
        if len(service_metrics) > 1:
            recommendations.append(
                f"🌐 Check dependencies: verify services calling {root_cause_service}"
            )
        
        recommendations.append(
            f"📈 Monitor {root_cause_service} metrics for next 10 minutes"
        )
        
        return recommendations
    
    def _identify_affected_resources(
        self,
        anomaly_type: str,
        service_metrics: ServiceMetrics
    ) -> List[str]:
        """Identify which resources are affected."""
        affected = []
        
        if service_metrics.cpu > 80:
            affected.append('CPU')
        if service_metrics.memory > 70:
            affected.append('Memory')
        if service_metrics.latency > 500:
            affected.append('Network/Latency')
        if service_metrics.disk_io > 80:
            affected.append('Disk I/O')
        
        error_rate = service_metrics.errors / (service_metrics.requests + 1e-6)
        if error_rate > 0.1:
            affected.append('Error Rate')
        
        return affected if affected else ['Unknown']
    
    def explain_with_shap(
        self,
        scale_invariant_features: Dict[str, float]
    ) -> Optional[Dict[str, float]]:
        """
        Use SHAP to explain model prediction.
        
        Args:
            scale_invariant_features: The 27 features used for detection
        
        Returns:
            Dictionary of feature -> SHAP value (importance)
        """
        if self.shap_explainer is None:
            print("⚠️ SHAP explainer not initialized")
            return None
        
        try:
            import shap
            
            # Convert to DataFrame
            feature_vector = pd.DataFrame([scale_invariant_features])
            
            # Calculate SHAP values
            shap_values = self.shap_explainer.shap_values(feature_vector)
            
            # Return as dictionary (absolute values for importance)
            feature_importance = {}
            for i, feature_name in enumerate(feature_vector.columns):
                feature_importance[feature_name] = abs(shap_values[0][i])
            
            # Sort by importance
            sorted_importance = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            )
            
            return sorted_importance
        
        except Exception as e:
            print(f"⚠️ SHAP analysis failed: {e}")
            return None


def format_rca_report(rca_result: RCAResult) -> str:
    """Format RCA result as human-readable report."""
    report = []
    report.append("=" * 80)
    report.append("🔍 ROOT CAUSE ANALYSIS REPORT")
    report.append("=" * 80)
    report.append(f"Timestamp: {rca_result.timestamp}")
    report.append(f"Anomaly Type: {rca_result.anomaly_type.upper()}")
    report.append(f"Severity: {rca_result.severity.upper()}")
    report.append(f"Confidence: {rca_result.confidence * 100:.1f}%")
    report.append("")
    
    report.append(f"🎯 ROOT CAUSE SERVICE: {rca_result.root_cause_service}")
    report.append(f"📊 Affected Resources: {', '.join(rca_result.affected_resources)}")
    report.append("")
    
    report.append("⚠️ CONTRIBUTING FACTORS:")
    for i, factor in enumerate(rca_result.contributing_factors, 1):
        report.append(
            f"  {i}. [{factor['severity'].upper()}] {factor['metric']}: "
            f"{factor['value']} - {factor['description']}"
        )
    report.append("")
    
    report.append("💡 RECOMMENDATIONS:")
    for i, rec in enumerate(rca_result.recommendations, 1):
        report.append(f"  {i}. {rec}")
    
    report.append("=" * 80)
    
    return "\n".join(report)


# Example usage
if __name__ == "__main__":
    # Simulate detected anomaly
    explainer = AnomalyExplainer()
    
    # Example: Memory leak detected in 3-service system
    service_metrics = {
        'notification': ServiceMetrics(
            service_name='notification',
            cpu=45, memory=92, network_in=30, network_out=25,
            disk_io=20, requests=50, errors=2, latency=120
        ),
        'web_api': ServiceMetrics(
            service_name='web_api',
            cpu=55, memory=60, network_in=50, network_out=45,
            disk_io=15, requests=120, errors=5, latency=95
        ),
        'processor': ServiceMetrics(
            service_name='processor',
            cpu=38, memory=58, network_in=20, network_out=18,
            disk_io=40, requests=30, errors=1, latency=80
        )
    }
    
    scale_invariant_features = {
        'cpu_utilization_mean': 0.46,
        'memory_pressure_max': 0.92,
        'error_rate': 0.05,
        'system_stress': 0.65,
        # ... other features
    }
    
    # Perform RCA
    rca_result = explainer.explain_anomaly(
        anomaly_type='memory_leak',
        service_metrics=service_metrics,
        scale_invariant_features=scale_invariant_features
    )
    
    # Print report
    print(format_rca_report(rca_result))
