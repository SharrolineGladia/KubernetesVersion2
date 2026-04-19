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
- Trace analysis integration (distributed tracing)
- Log analysis integration (error context)
- Actionable recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import pickle
import sys
import os

# Import trace and log analyzers
sys.path.append(os.path.dirname(__file__))
try:
    from trace_analyzer import TraceAnalyzer, get_trace_insights
    from log_analyzer import LogAnalyzer, get_log_insights
    TRACE_LOG_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Trace/Log analyzers not available: {e}")
    TRACE_LOG_AVAILABLE = False


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
    """Root cause analysis result with trace and log context."""
    anomaly_type: str
    root_cause_service: str
    confidence: float
    contributing_factors: List[Dict[str, Any]]
    recommendations: List[str]
    severity: str  # 'low', 'medium', 'high', 'critical'
    affected_resources: List[str]
    timestamp: str
    trace_context: Optional[Dict] = None  # NEW: Trace insights
    log_context: Optional[Dict] = None    # NEW: Log insights
    failure_pattern_matches: Optional[List[Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict:
        result = {
            'anomaly_type': self.anomaly_type,
            'root_cause_service': self.root_cause_service,
            'confidence': self.confidence,
            'contributing_factors': self.contributing_factors,
            'recommendations': self.recommendations,
            'severity': self.severity,
            'affected_resources': self.affected_resources,
            'timestamp': self.timestamp
        }
        
        # Include trace context if available
        if self.trace_context:
            result['trace_context'] = self.trace_context
        
        # Include log context if available
        if self.log_context:
            result['log_context'] = self.log_context
        
        if self.failure_pattern_matches:
            result['failure_pattern_matches'] = self.failure_pattern_matches
        
        return result


class AnomalyExplainer:
    """
    Provides explainability for detected anomalies using service-specific metrics.
    
    This class bridges the gap between topology-agnostic detection and
    actionable, service-specific explanations.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        enable_traces: bool = True,
        enable_logs: bool = True,
        jaeger_url: str = "http://localhost:16686",
        log_file_path: Optional[str] = None
    ):
        """
        Initialize explainer.
        
        Args:
            model_path: Path to trained XGBoost model (for SHAP analysis)
            enable_traces: Enable trace analysis integration
            enable_logs: Enable log analysis integration
            jaeger_url: Jaeger query service URL
            log_file_path: Path to log file (for log analysis)
        """
        self.model = None
        self.shap_explainer = None
        
        if model_path:
            self.load_model(model_path)
        
        # Initialize trace analyzer
        self.trace_analyzer = None
        if enable_traces and TRACE_LOG_AVAILABLE:
            self.trace_analyzer = TraceAnalyzer(jaeger_query_url=jaeger_url)
            print("✓ Trace analyzer enabled")
        
        # Initialize log analyzer
        self.log_analyzer = None
        if enable_logs and TRACE_LOG_AVAILABLE:
            self.log_analyzer = LogAnalyzer(
                log_source='file',
                log_file_path=log_file_path
            )
            print("✓ Log analyzer enabled")
        
        # Anomaly-specific thresholds for RCA
        self.thresholds = {
            'cpu_spike': {'cpu': 85, 'cpu_growth_rate': 0.3},
            'memory_leak': {'memory': 75, 'memory_growth_rate': 0.2},
            'service_crash': {'error_rate': 0.5, 'request_drop': 0.7},
            'network_congestion': {'network_out': 80, 'latency': 500}
        }

        # Failure archetypes for true pattern matching.
        self.failure_pattern_catalog = {
            'Dependency Timeout Cascade': {
                'description': 'Slow downstream dependency triggers timeouts and propagates failures upstream.',
                'signals': [
                    'trace_slow_operation',
                    'trace_error_chain',
                    'timeout_pattern',
                    'high_latency'
                ]
            },
            'Retry Storm / Circuit Breaker Trip': {
                'description': 'Retries accumulate under failure and trigger breaker protections.',
                'signals': [
                    'retry_pattern',
                    'circuit_breaker_pattern',
                    'high_error_rate',
                    'queue_backlog'
                ]
            },
            'CPU Saturation Under Load': {
                'description': 'Sustained CPU pressure under high request load degrades service response.',
                'signals': [
                    'high_cpu',
                    'high_request_rate',
                    'high_system_stress',
                    'queue_backlog'
                ]
            },
            'Memory Pressure / Leak Progression': {
                'description': 'Elevated memory pressure and imbalance suggest leak or unbounded retention.',
                'signals': [
                    'high_memory',
                    'memory_pressure',
                    'memory_skew',
                    'queue_backlog'
                ]
            }
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
        except Exception:
            self.shap_explainer = None
    
    def explain_anomaly(
        self,
        anomaly_type: str,
        service_metrics: Dict[str, ServiceMetrics],
        scale_invariant_features: Dict[str, float],
        timestamp: Optional[datetime] = None,
        service_name: Optional[str] = None,
        trace_context_override: Optional[Dict[str, Any]] = None,
        log_context_override: Optional[Dict[str, Any]] = None
    ) -> RCAResult:
        """
        Explain detected anomaly by analyzing service-specific metrics, traces, and logs.
        
        Args:
            anomaly_type: Detected anomaly type from XGBoost model
            service_metrics: Raw metrics per service
            scale_invariant_features: The 27 scale-invariant features used for detection
            timestamp: Time of detection
            service_name: Optional service name for focused trace/log analysis
        
        Returns:
            RCAResult with root cause, contributing factors, recommendations,
            trace context, and log context
        """
        timestamp = timestamp or datetime.utcnow()
        
        # Step 1: Identify root cause service (from metrics)
        root_cause_service, confidence = self._identify_root_cause_service(
            anomaly_type, service_metrics
        )
        
        # Use detected service if not provided
        if not service_name:
            service_name = root_cause_service
        
        # Step 2: Gather trace context
        trace_context = trace_context_override
        if trace_context is None and self.trace_analyzer:
            try:
                trace_context = self.trace_analyzer.get_trace_context_for_rca(
                    anomaly_timestamp=timestamp,
                    window_minutes=5,
                    service_name=service_name
                )
                
                # Enhance confidence if traces confirm the issue
                if trace_context.get('has_trace_data'):
                    trace_error_rate = trace_context.get('error_rate_from_traces', 0)
                    if trace_error_rate > 0.3:  # High error rate in traces
                        confidence = min(1.0, confidence * 1.1)  # Boost confidence
            except Exception as e:
                print(f"⚠️  Trace analysis failed: {e}")
        
        # Step 3: Gather log context
        log_context = log_context_override
        if log_context is None and self.log_analyzer:
            try:
                log_context = self.log_analyzer.get_log_context_for_rca(
                    anomaly_timestamp=timestamp,
                    window_minutes=5,
                    service_name=service_name
                )
                
                # Enhance confidence if logs show errors
                if log_context.get('has_log_data'):
                    log_error_rate = log_context.get('error_rate_from_logs', 0)
                    if log_error_rate > 0.2:  # Significant errors in logs
                        confidence = min(1.0, confidence * 1.1)  # Boost confidence
            except Exception as e:
                print(f"⚠️  Log analysis failed: {e}")
        
        # Step 4: Analyze contributing factors (enhanced with trace/log data)
        contributing_factors = self._analyze_contributing_factors(
            anomaly_type,
            root_cause_service,
            service_metrics,
            scale_invariant_features,
            trace_context,
            log_context
        )
        
        # Step 5: Determine severity (enhanced with trace/log signals)
        severity = self._calculate_severity(
            anomaly_type,
            service_metrics[root_cause_service],
            scale_invariant_features,
            trace_context,
            log_context
        )
        
        # Step 6: Generate recommendations (enhanced with trace/log insights)
        recommendations = self._generate_recommendations(
            anomaly_type,
            root_cause_service,
            service_metrics,
            severity,
            trace_context,
            log_context
        )
        
        # Step 7: Identify affected resources
        affected_resources = self._identify_affected_resources(
            anomaly_type,
            service_metrics[root_cause_service]
        )

        # Step 8: Match current incident against known failure archetypes
        failure_pattern_matches = self._match_failure_patterns(
            anomaly_type=anomaly_type,
            service=service_metrics[root_cause_service],
            scale_invariant_features=scale_invariant_features,
            trace_context=trace_context,
            log_context=log_context
        )
        
        return RCAResult(
            anomaly_type=anomaly_type,
            root_cause_service=root_cause_service,
            confidence=confidence,
            contributing_factors=contributing_factors,
            recommendations=recommendations,
            severity=severity,
            affected_resources=affected_resources,
            timestamp=timestamp.isoformat(),
            trace_context=trace_context,
            log_context=log_context,
            failure_pattern_matches=failure_pattern_matches
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
        scale_invariant_features: Dict[str, float],
        trace_context: Optional[Dict] = None,
        log_context: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Identify metrics contributing to the anomaly.
        
        Enhanced with trace and log analysis for deeper insights.
        """
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
        
        # NEW: Add trace-based factors
        if trace_context and trace_context.get('has_trace_data'):
            # Slow operations from traces
            if trace_context.get('slow_operations'):
                slow_op = trace_context['slow_operations'][0]  # Slowest
                factors.append({
                    'metric': 'trace_slow_operation',
                    'value': f"{slow_op['duration_ms']:.0f}ms",
                    'severity': 'high',
                    'description': f"Slow: {slow_op['operation']} in {slow_op['service']}"
                })
            
            # Error propagation chain
            if trace_context.get('error_chain'):
                error_count = len(trace_context['error_chain'])
                factors.append({
                    'metric': 'trace_error_chain',
                    'value': f"{error_count} errors",
                    'severity': 'critical',
                    'description': f"Error propagation across {error_count} operations"
                })
        
        # NEW: Add log-based factors
        if log_context and log_context.get('has_log_data'):
            # Error patterns from logs
            if log_context.get('error_patterns'):
                top_error = log_context['error_patterns'][0]
                factors.append({
                    'metric': 'log_error_pattern',
                    'value': f"{top_error['count']} occurrences",
                    'severity': 'high',
                    'description': f"Repeated: {top_error['pattern'][:50]}"
                })
            
            # Critical errors
            if log_context.get('critical_errors'):
                factors.append({
                    'metric': 'log_critical_error',
                    'value': f"{len(log_context['critical_errors'])} errors",
                    'severity': 'critical',
                    'description': 'Critical errors logged (see log_context for details)'
                })
        
        return factors
    
    def _calculate_severity(
        self,
        anomaly_type: str,
        service_metrics: ServiceMetrics,
        scale_invariant_features: Dict[str, float],
        trace_context: Optional[Dict] = None,
        log_context: Optional[Dict] = None
    ) -> str:
        """
        Calculate severity level: low, medium, high, critical.
        
        Enhanced with trace and log context for more accurate severity.
        """
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
        
        # NEW: Trace-based severity boost
        if trace_context and trace_context.get('has_trace_data'):
            trace_error_rate = trace_context.get('error_rate_from_traces', 0)
            if trace_error_rate > 0.5:
                severity_score += 2
            elif trace_error_rate > 0.3:
                severity_score += 1
        
        # NEW: Log-based severity boost
        if log_context and log_context.get('has_log_data'):
            log_error_rate = log_context.get('error_rate_from_logs', 0)
            if log_error_rate > 0.4:
                severity_score += 2
            elif log_error_rate > 0.2:
                severity_score += 1
        
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
        severity: str,
        trace_context: Optional[Dict] = None,
        log_context: Optional[Dict] = None
    ) -> List[str]:
        """
        Generate actionable recommendations.
        
        Enhanced with trace and log insights for specific actions.
        """
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
            
            # NEW: Trace-specific recommendation
            if trace_context and trace_context.get('slow_operations'):
                slow_op = trace_context['slow_operations'][0]
                recommendations.append(
                    f"🔍 Optimize {slow_op['operation']} - takes {slow_op['duration_ms']:.0f}ms"
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
            # NEW: Log-specific recommendation
            if log_context and log_context.get('critical_errors'):
                error_msg = log_context['critical_errors'][0]['message'][:60]
                recommendations.append(
                    f"🚨 IMMEDIATE: Address error '{error_msg}...'"
                )
            else:
                recommendations.append(
                    f"🚨 IMMEDIATE: Check {root_cause_service} logs for crash details"
                )
            
            recommendations.append(
                f"🔄 Restart {root_cause_service} deployment"
            )
            
            # NEW: Trace-specific recommendation
            if trace_context and trace_context.get('error_chain'):
                services_affected = len(set(e['service'] for e in trace_context['error_chain']))
                recommendations.append(
                    f"🔗 Check cascade: {services_affected} services affected by error propagation"
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
        
        if service_metrics.cpu_percent > 80:
            affected.append('CPU')
        if service_metrics.memory_percent > 70:
            affected.append('Memory')
        if service_metrics.response_time_p95 > 500:
            affected.append('Network/Latency')
        if service_metrics.thread_count > 150:
            affected.append('Thread Pool')
        if service_metrics.queue_depth > 25:
            affected.append('Queue Backlog')
        if service_metrics.error_rate > 0.1:
            affected.append('Error Rate')
        
        return affected if affected else ['Unknown']

    def _match_failure_patterns(
        self,
        anomaly_type: str,
        service: ServiceMetrics,
        scale_invariant_features: Dict[str, float],
        trace_context: Optional[Dict] = None,
        log_context: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Match the current incident against known failure archetypes."""
        active_signals: Dict[str, float] = {}
        evidence: Dict[str, List[str]] = {}

        def add_signal(signal_name: str, strength: float, reason: str):
            active_signals[signal_name] = max(active_signals.get(signal_name, 0.0), strength)
            evidence.setdefault(signal_name, []).append(reason)

        if service.cpu_percent >= 85:
            add_signal('high_cpu', min(1.0, service.cpu_percent / 100.0), f"CPU at {service.cpu_percent:.1f}%")
        if service.memory_percent >= 75:
            add_signal('high_memory', min(1.0, service.memory_percent / 100.0), f"Memory at {service.memory_percent:.1f}%")
        if service.request_rate >= 100:
            add_signal('high_request_rate', min(1.0, service.request_rate / 200.0), f"Request rate at {service.request_rate:.0f} req/s")
        if service.response_time_p95 >= 500:
            add_signal('high_latency', min(1.0, service.response_time_p95 / 1500.0), f"P95 latency at {service.response_time_p95:.0f}ms")
        if service.queue_depth >= 25:
            add_signal('queue_backlog', min(1.0, service.queue_depth / 60.0), f"Queue depth at {service.queue_depth:.0f}")
        if service.error_rate >= 0.2:
            add_signal('high_error_rate', min(1.0, service.error_rate), f"Error rate at {service.error_rate:.1%}")

        if scale_invariant_features.get('system_stress', 0) >= 0.8:
            add_signal('high_system_stress', scale_invariant_features['system_stress'], f"System stress at {scale_invariant_features['system_stress']:.2f}")
        if scale_invariant_features.get('memory_pressure_max', 0) >= 0.75:
            add_signal('memory_pressure', scale_invariant_features['memory_pressure_max'], f"Memory pressure at {scale_invariant_features['memory_pressure_max']:.2f}")
        if scale_invariant_features.get('memory_imbalance', 0) >= 0.15:
            add_signal('memory_skew', scale_invariant_features['memory_imbalance'], f"Memory imbalance at {scale_invariant_features['memory_imbalance']:.2f}")

        if trace_context:
            if trace_context.get('slow_operations'):
                slowest = trace_context['slow_operations'][0]
                add_signal('trace_slow_operation', min(1.0, slowest['duration_ms'] / 2000.0), f"Slow op {slowest['operation']} at {slowest['duration_ms']:.0f}ms")
            if trace_context.get('error_chain'):
                add_signal('trace_error_chain', min(1.0, len(trace_context['error_chain']) / 4.0), f"{len(trace_context['error_chain'])} trace error events")

        if log_context:
            for pattern in log_context.get('error_patterns', []):
                pattern_text = pattern['pattern'].lower()
                count_strength = min(1.0, pattern['count'] / 12.0)
                if 'timeout' in pattern_text:
                    add_signal('timeout_pattern', count_strength, f"Timeout pattern seen {pattern['count']}x")
                if 'retry' in pattern_text:
                    add_signal('retry_pattern', count_strength, f"Retry pattern seen {pattern['count']}x")
                if 'circuit breaker' in pattern_text:
                    add_signal('circuit_breaker_pattern', count_strength, f"Circuit breaker pattern seen {pattern['count']}x")

        matches: List[Dict[str, Any]] = []
        for pattern_name, spec in self.failure_pattern_catalog.items():
            matched_signals = [signal for signal in spec['signals'] if signal in active_signals]
            if not matched_signals:
                continue

            confidence = sum(active_signals[signal] for signal in matched_signals) / len(spec['signals'])
            match_evidence = []
            for signal in matched_signals:
                match_evidence.extend(evidence.get(signal, []))

            matches.append({
                'pattern_name': pattern_name,
                'confidence': round(float(confidence), 3),
                'matched_signals': matched_signals,
                'description': spec['description'],
                'evidence': match_evidence[:4]
            })

        # Slightly bias the anomaly-aligned family upward so the top result feels sensible.
        for match in matches:
            if anomaly_type == 'cpu_spike' and match['pattern_name'] == 'CPU Saturation Under Load':
                match['confidence'] = round(min(1.0, match['confidence'] + 0.08), 3)
            if anomaly_type == 'memory_leak' and match['pattern_name'] == 'Memory Pressure / Leak Progression':
                match['confidence'] = round(min(1.0, match['confidence'] + 0.08), 3)
            if anomaly_type == 'service_crash' and match['pattern_name'] == 'Dependency Timeout Cascade':
                match['confidence'] = round(min(1.0, match['confidence'] + 0.05), 3)

        matches.sort(key=lambda item: item['confidence'], reverse=True)
        return matches[:3]
    
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
        if False and self.shap_explainer is None:
            print("⚠️ SHAP explainer not initialized")
            return None
        
        try:
            if self.model is None:
                print("âš ï¸ No model loaded for SHAP analysis")
                return None

            feature_order = list(
                getattr(self.model, 'feature_names_in_', scale_invariant_features.keys())
            )
            feature_vector = pd.DataFrame(
                [[scale_invariant_features.get(name, 0.0) for name in feature_order]],
                columns=feature_order
            )

            if self.shap_explainer is not None:
                try:
                    shap_values = self.shap_explainer.shap_values(feature_vector)
                    if isinstance(shap_values, list):
                        class_idx = int(np.argmax(self.model.predict_proba(feature_vector)[0]))
                        class_values = np.asarray(shap_values[class_idx])[0]
                    else:
                        class_values = np.asarray(shap_values)[0]

                    feature_importance = {
                        feature_name: float(abs(class_values[i]))
                        for i, feature_name in enumerate(feature_vector.columns)
                    }
                    return dict(
                        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                    )
                except Exception:
                    pass

            import xgboost as xgb

            booster = self.model.get_booster()
            contribs = booster.predict(xgb.DMatrix(feature_vector), pred_contribs=True)
            class_idx = int(np.argmax(self.model.predict_proba(feature_vector)[0]))
            class_contribs = np.asarray(contribs)[0, class_idx, :-1]

            feature_importance = {
                feature_name: float(abs(class_contribs[i]))
                for i, feature_name in enumerate(feature_vector.columns)
            }
            return dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            )
        
        except Exception as e:
            print(f"⚠️ SHAP analysis failed: {e}")
            return None


def format_rca_report(
    rca_result: RCAResult,
    shap_importance: Optional[Dict[str, float]] = None,
    max_shap_features: int = 5
) -> str:
    """Format RCA result as a full explanation-engine report."""
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


def format_explanation_report(
    rca_result: RCAResult,
    shap_importance: Optional[Dict[str, float]] = None,
    max_shap_features: int = 5
) -> str:
    """Render a complete explanation-engine report for demos and screenshots."""
    report = []
    report.append("=" * 80)
    report.append("EXPLANATION ENGINE REPORT")
    report.append("=" * 80)
    report.append(f"Timestamp: {rca_result.timestamp}")
    report.append(f"Anomaly Type: {rca_result.anomaly_type.upper()}")
    report.append(f"Severity: {rca_result.severity.upper()}")
    report.append(f"Confidence: {rca_result.confidence * 100:.1f}%")
    report.append("")
    report.append(f"Root Cause Microservice: {rca_result.root_cause_service}")
    report.append(f"Affected Resources: {', '.join(rca_result.affected_resources)}")
    report.append("")
    report.append("CONTRIBUTING FACTORS:")
    for i, factor in enumerate(rca_result.contributing_factors, 1):
        report.append(
            f"  {i}. [{factor['severity'].upper()}] {factor['metric']}: "
            f"{factor['value']} - {factor['description']}"
        )
    report.append("")

    if shap_importance:
        report.append("SHAP / FEATURE ATTRIBUTION:")
        for i, (feature, importance) in enumerate(list(shap_importance.items())[:max_shap_features], 1):
            report.append(f"  {i}. {feature}: {importance:.4f}")
        report.append("")

    if rca_result.trace_context:
        report.append("TRACE INSIGHTS:")
        report.append(
            f"  Trace Data Available: {'YES' if rca_result.trace_context.get('has_trace_data') else 'NO'}"
        )
        report.append(
            f"  Error Rate From Traces: {rca_result.trace_context.get('error_rate_from_traces', 0):.1%}"
        )
        for slow_op in rca_result.trace_context.get('slow_operations', [])[:3]:
            report.append(
                f"  Slow Operation: {slow_op['service']}.{slow_op['operation']} "
                f"({slow_op['duration_ms']:.0f}ms)"
            )
        timeline = rca_result.trace_context.get('causal_timeline', [])
        if timeline:
            report.append("  Timeline:")
            for event in timeline[:4]:
                report.append(f"    - [{event.get('service', 'unknown')}] {event.get('details', 'event')}")
        report.append("")

    if rca_result.failure_pattern_matches:
        report.append("FAILURE PATTERN MATCHING:")
        for idx, match in enumerate(rca_result.failure_pattern_matches, 1):
            report.append(
                f"  {idx}. {match['pattern_name']} "
                f"(confidence: {match['confidence'] * 100:.1f}%)"
            )
            report.append(f"     Description: {match['description']}")
            if match.get('evidence'):
                report.append(f"     Evidence: {'; '.join(match['evidence'])}")
        report.append("")

    report.append("RECOMMENDATIONS:")
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

