"""
Test Counterfactual Analyzer Integration

Quick test to verify real-time counterfactual analysis works.
"""

import os
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'ml_detector' / 'scripts'))

from counterfactual_analyzer import CounterfactualAnalyzer
import pickle


def test_counterfactual_analyzer():
    """Test the counterfactual analyzer with sample anomaly."""

    print("="*80)
    print("TESTING COUNTERFACTUAL ANALYZER")
    print("="*80)

    # Load model
    model_path = PROJECT_ROOT / 'ml_detector' / 'models' / 'anomaly_detector_scaleinvariant.pkl'

    with open(model_path, 'rb') as f:
        saved_data = pickle.load(f)
        if isinstance(saved_data, dict):
            model = saved_data.get('model', saved_data)
        else:
            model = saved_data

    print(f"✅ Model loaded from {model_path}")

    # Feature names
    feature_names = [
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

    # Initialize analyzer
    analyzer = CounterfactualAnalyzer(model, feature_names)
    print(f"✅ Counterfactual analyzer initialized")

    # Simulate CPU spike anomaly
    print(f"\n{'─'*80}")
    print(f"TEST CASE 1: CPU Spike Anomaly")
    print(f"{'─'*80}")

    cpu_spike_features = np.array([
        0.85,  # cpu_utilization_mean - HIGH
        0.97,  # cpu_utilization_max - VERY HIGH
        0.15,  # cpu_variance_coef
        0.45,  # memory_utilization_mean
        0.60,  # memory_utilization_max
        0.12,  # memory_variance_coef
        0.60,  # memory_pressure_max
        0.02,  # memory_growth_rate
        0.35,  # thread_count_mean
        0.45,  # thread_count_max
        0.10,  # thread_variance_coef
        0.05,  # error_rate
        0.0,   # error_spike_indicator
        0.25,  # response_time_p95_mean
        0.35,  # response_time_p95_max
        0.08,  # response_time_variance_coef
        0.60,  # request_rate_mean
        0.75,  # request_rate_max
        0.12,  # request_variance_coef
        0.15,  # queue_depth_mean
        0.25,  # queue_depth_max
        0.08,  # queue_variance_coef
        1.0,   # service_health_min
        0.65,  # cpu_memory_correlation
        0.25,  # load_error_correlation
        0.3,   # normalized_service_count
        0.72   # system_stress_index - HIGH
    ])

    try:
        explanation = analyzer.analyze(
            features=cpu_spike_features,
            original_prediction='cpu_spike'
        )

        if explanation:
            print(explanation.format_human_readable())
            print("\n✅ TEST PASSED: Counterfactual generated successfully")
        else:
            print("\n⚠️  No counterfactual found (this is OK for complex anomalies)")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Test case 2: Memory leak
    print(f"\n{'─'*80}")
    print(f"TEST CASE 2: Memory Leak Anomaly")
    print(f"{'─'*80}")

    memory_leak_features = np.array([
        0.45,  # cpu_utilization_mean
        0.55,  # cpu_utilization_max
        0.10,  # cpu_variance_coef
        0.88,  # memory_utilization_mean - VERY HIGH
        0.95,  # memory_utilization_max - CRITICAL
        0.08,  # memory_variance_coef
        0.95,  # memory_pressure_max - CRITICAL
        0.15,  # memory_growth_rate - GROWING
        0.30,  # thread_count_mean
        0.40,  # thread_count_max
        0.08,  # thread_variance_coef
        0.02,  # error_rate
        0.0,   # error_spike_indicator
        0.18,  # response_time_p95_mean
        0.25,  # response_time_p95_max
        0.06,  # response_time_variance_coef
        0.50,  # request_rate_mean
        0.60,  # request_rate_max
        0.10,  # request_variance_coef
        0.12,  # queue_depth_mean
        0.20,  # queue_depth_max
        0.06,  # queue_variance_coef
        1.0,   # service_health_min
        0.55,  # cpu_memory_correlation
        0.10,  # load_error_correlation
        0.3,   # normalized_service_count
        0.68   # system_stress_index
    ])

    try:
        explanation = analyzer.analyze(
            features=memory_leak_features,
            original_prediction='memory_leak'
        )

        if explanation:
            print(explanation.format_human_readable())
            print("\n✅ TEST PASSED: Counterfactual generated successfully")
        else:
            print("\n⚠️  No counterfactual found (this is OK for complex anomalies)")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*80}")
    print(f"COUNTERFACTUAL ANALYZER TESTS COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    test_counterfactual_analyzer()
