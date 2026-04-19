"""
Integrated Explanation Engine Demo

Shows a complete explanation report using:
1. Metrics-based RCA
2. SHAP-compatible feature attribution
3. Trace timeline context
4. Failure-pattern matching from logs

This demo is self-contained so it works even when Jaeger/log capture are not
running locally, which makes it suitable for presentation screenshots.
"""

import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / 'ml_detector' / 'scripts'))

from explainability_layer import (
    AnomalyExplainer,
    ServiceMetrics,
    format_explanation_report,
)
from trace_analyzer import TraceAnalyzer
from log_analyzer import LogAnalyzer

warnings.filterwarnings("ignore")


def print_header(title: str):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")


def print_section(title: str):
    print("\n" + "-" * 80)
    print(f" {title}")
    print("-" * 80)


def build_demo_trace_context() -> dict:
    return {
        'has_trace_data': True,
        'error_rate_from_traces': 0.35,
        'services_involved': ['api-gateway', 'notification-service', 'smtp-adapter'],
        'slow_operations': [
            {
                'service': 'notification-service',
                'operation': 'send_notification',
                'duration_ms': 1280.0,
                'threshold_ms': 1000.0,
                'trace_id': 'trace-demo-001'
            }
        ],
        'error_chain': [
            {
                'timestamp': '2026-03-26T19:39:58',
                'service': 'notification-service',
                'operation': 'send_notification',
                'error_message': 'Timeout while calling smtp-adapter'
            },
            {
                'timestamp': '2026-03-26T19:40:02',
                'service': 'smtp-adapter',
                'operation': 'deliver_email',
                'error_message': 'ConnectionError: upstream SMTP timeout'
            }
        ],
        'causal_timeline': [
            {
                'timestamp': '2026-03-26T19:39:54',
                'service': 'notification-service',
                'details': 'Latency rose from 280ms to 1280ms on send_notification'
            },
            {
                'timestamp': '2026-03-26T19:39:58',
                'service': 'notification-service',
                'details': 'Timeout while calling smtp-adapter'
            },
            {
                'timestamp': '2026-03-26T19:40:02',
                'service': 'smtp-adapter',
                'details': 'ConnectionError propagated back to notification-service'
            }
        ],
        'service_dependencies': {
            'api-gateway': ['notification-service'],
            'notification-service': ['smtp-adapter']
        }
    }


def build_demo_log_context() -> dict:
    return {
        'has_log_data': True,
        'error_rate_from_logs': 0.28,
        'warning_rate': 0.14,
        'services_with_errors': ['notification-service', 'smtp-adapter'],
        'error_patterns': [
            {'pattern': 'Connection timeout to external SMTP API', 'count': 12},
            {'pattern': 'Retry budget exhausted for notification dispatch', 'count': 8},
            {'pattern': 'Circuit breaker opened for smtp-adapter', 'count': 5}
        ],
        'critical_errors': [
            {
                'timestamp': '2026-03-26T19:40:03',
                'service': 'notification-service',
                'message': 'Connection timeout to external SMTP API'
            },
            {
                'timestamp': '2026-03-26T19:40:04',
                'service': 'smtp-adapter',
                'message': 'Retry budget exhausted for notification dispatch'
            }
        ],
        'trace_correlation': {
            'trace-demo-001': [
                {
                    'timestamp': '2026-03-26T19:40:03',
                    'level': 'ERROR',
                    'service': 'notification-service',
                    'message': 'Connection timeout to external SMTP API'
                }
            ]
        }
    }


def build_demo_shap_features() -> dict:
    return {
        'cpu_utilization_mean': 0.92,
        'cpu_utilization_max': 0.98,
        'cpu_variance_coef': 0.22,
        'cpu_imbalance': 0.31,
        'memory_pressure_mean': 0.76,
        'memory_pressure_max': 0.82,
        'memory_variance_coef': 0.19,
        'memory_imbalance': 0.18,
        'network_in_rate': 0.42,
        'network_out_rate': 0.39,
        'network_in_variance_coef': 0.11,
        'network_out_variance_coef': 0.13,
        'network_asymmetry': 0.08,
        'disk_io_rate': 0.21,
        'disk_io_variance_coef': 0.09,
        'request_rate': 0.88,
        'request_variance_coef': 0.26,
        'error_rate': 0.18,
        'error_variance_coef': 0.45,
        'latency_mean': 0.71,
        'latency_p95': 0.85,
        'latency_variance_coef': 0.24,
        'system_stress': 0.91,
        'resource_efficiency': 0.29,
        'service_density': 0.30,
        'cpu_memory_correlation': 0.64,
        'performance_degradation': 0.73
    }


def demo_integrated_rca():
    print_header("INTEGRATED EXPLANATION ENGINE DEMO")

    model_path = PROJECT_ROOT / 'ml_detector' / 'models' / 'anomaly_detector_scaleinvariant.pkl'

    explainer = AnomalyExplainer(
        model_path=model_path,
        enable_traces=False,
        enable_logs=False,
    )

    service_metrics = {
        'notification-service': ServiceMetrics(
            cpu_percent=92.3,
            memory_percent=78.5,
            error_rate=0.35,
            request_rate=150.0,
            response_time_p95=850.0,
            thread_count=225,
            queue_depth=45.0,
            requests_per_second=150.0
        )
    }
    scale_invariant_features = build_demo_shap_features()
    trace_context = build_demo_trace_context()
    log_context = build_demo_log_context()

    print_section("Input Snapshot")
    print("Service: notification-service")
    print("CPU: 92.3%")
    print("Memory: 78.5%")
    print("Error Rate: 35.0%")
    print("P95 Latency: 850ms")
    print("Queue Depth: 45")

    print_section("Explanation Engine Execution")
    rca_result = explainer.explain_anomaly(
        anomaly_type='cpu_spike',
        service_metrics=service_metrics,
        scale_invariant_features=scale_invariant_features,
        timestamp=datetime.utcnow(),
        service_name='notification-service',
        trace_context_override=trace_context,
        log_context_override=log_context
    )
    shap_importance = explainer.explain_with_shap(scale_invariant_features)

    print("RCA completed successfully")
    print(f"Root Cause Microservice: {rca_result.root_cause_service}")
    print(f"Severity: {rca_result.severity.upper()}")
    print(f"Contributing Factors: {len(rca_result.contributing_factors)}")
    print(f"SHAP Features Ranked: {len(shap_importance) if shap_importance else 0}")

    print_section("Top SHAP Features")
    if shap_importance:
        for i, (feature, score) in enumerate(list(shap_importance.items())[:6], 1):
            print(f"{i}. {feature}: {score:.4f}")
    else:
        print("SHAP attribution unavailable")

    print_section("Trace Timeline")
    for event in trace_context['causal_timeline']:
        print(f"[{event['timestamp']}] {event['service']} -> {event['details']}")

    print_section("Failure Pattern Matching")
    if rca_result.failure_pattern_matches:
        for match in rca_result.failure_pattern_matches:
            print(f"- {match['pattern_name']} ({match['confidence'] * 100:.1f}%)")
            if match.get('evidence'):
                print(f"  Evidence: {'; '.join(match['evidence'])}")
    else:
        print("No known failure archetype matched")

    print_section("Full Explanation Report")
    print(format_explanation_report(rca_result, shap_importance=shap_importance, max_shap_features=5))


def demo_trace_analyzer_standalone():
    print_header("TRACE ANALYZER STANDALONE TEST")
    analyzer = TraceAnalyzer(jaeger_query_url="http://localhost:16686")
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(minutes=5)
    print(f"Fetching traces from {start_time.strftime('%H:%M:%S')} to {end_time.strftime('%H:%M:%S')}...")
    try:
        summary = analyzer.analyze_time_window(
            start_time=start_time,
            end_time=end_time,
            service_name='notification-service'
        )
        print(f"Total traces: {summary.total_traces}")
        print(f"Error traces: {summary.error_traces}")
        print(f"Error rate: {summary.error_traces / max(1, summary.total_traces):.1%}")
    except Exception as e:
        print(f"Error: {e}")


def demo_log_analyzer_standalone():
    print_header("LOG ANALYZER STANDALONE TEST")
    analyzer = LogAnalyzer(log_source='file', log_file_path=str(PROJECT_ROOT / 'service_logs.json'))
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(minutes=5)
    try:
        summary = analyzer.analyze_time_window(
            start_time=start_time,
            end_time=end_time,
            service_name='notification-service'
        )
        print(f"Total logs: {summary.total_logs}")
        print(f"Error logs: {summary.error_logs}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("=" * 80)
    print(" INTEGRATED RCA DEMONSTRATION")
    print(" Explanation engine with SHAP + traces + failure patterns")
    print("=" * 80)
    print("\nSelect demo:")
    print("1. Full explanation engine demo (recommended)")
    print("2. Test trace analyzer only")
    print("3. Test log analyzer only")
    print("4. Run all tests")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == '1':
        demo_integrated_rca()
    elif choice == '2':
        demo_trace_analyzer_standalone()
    elif choice == '3':
        demo_log_analyzer_standalone()
    elif choice == '4':
        demo_trace_analyzer_standalone()
        demo_log_analyzer_standalone()
        demo_integrated_rca()
    else:
        print("Invalid choice")
        sys.exit(1)

    print("\n" + "=" * 80)
    print(" DEMO COMPLETE")
    print("=" * 80)
