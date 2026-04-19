"""
Demo: Counterfactual Analysis Output

Shows real counterfactual explanations for different anomaly types.
"""

import os
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'ml_detector' / 'scripts'))

from counterfactual_analyzer import CounterfactualAnalyzer
from dual_feature_detector import DualFeatureDetector
import pickle


def demo_counterfactual_analysis():
    """Run comprehensive counterfactual demo."""

    print("="*80)
    print("COUNTERFACTUAL ANALYSIS DEMO")
    print("="*80)
    print()

    # Load model and initialize
    model_path = PROJECT_ROOT / 'ml_detector' / 'models' / 'anomaly_detector_scaleinvariant.pkl'

    with open(model_path, 'rb') as f:
        saved_data = pickle.load(f)
        if isinstance(saved_data, dict):
            model = saved_data.get('model', saved_data)
        else:
            model = saved_data

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

    analyzer = CounterfactualAnalyzer(model, feature_names)

    # ========================================================================
    # SCENARIO 1: CPU SPIKE ANOMALY
    # ========================================================================
    print("\n" + "█"*80)
    print("█" + " SCENARIO 1: CPU SPIKE ANOMALY ".center(78) + "█")
    print("█"*80)
    print()
    print("Current Situation:")
    print("  - Average CPU: 85%")
    print("  - Max CPU: 97%")
    print("  - System Stress: 72%")
    print("  - Memory: 45%")
    print()

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

    # Predict
    probas = model.predict_proba(cpu_spike_features.reshape(1, -1))[0]
    class_names = ['cpu_spike', 'memory_leak', 'normal', 'service_crash']
    predicted_class = class_names[np.argmax(probas)]
    confidence = probas[np.argmax(probas)]

    print(f"🎯 Detection Result:")
    print(f"   Anomaly Type: {predicted_class.upper()}")
    print(f"   Confidence: {confidence:.1%}")

    # Counterfactual analysis
    explanation = analyzer.analyze(
        features=cpu_spike_features,
        original_prediction=predicted_class
    )

    if explanation:
        print(explanation.format_human_readable())

    # ========================================================================
    # SCENARIO 2: MEMORY LEAK ANOMALY
    # ========================================================================
    print("\n" + "█"*80)
    print("█" + " SCENARIO 2: MEMORY LEAK ANOMALY ".center(78) + "█")
    print("█"*80)
    print()
    print("Current Situation:")
    print("  - Memory Usage: 88%")
    print("  - Memory Pressure: 95%")
    print("  - Memory Growth Rate: 15% (growing)")
    print("  - CPU: 45%")
    print()

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

    # Predict
    probas = model.predict_proba(memory_leak_features.reshape(1, -1))[0]
    predicted_class = class_names[np.argmax(probas)]
    confidence = probas[np.argmax(probas)]

    print(f"🎯 Detection Result:")
    print(f"   Anomaly Type: {predicted_class.upper()}")
    print(f"   Confidence: {confidence:.1%}")

    # Counterfactual analysis
    explanation = analyzer.analyze(
        features=memory_leak_features,
        original_prediction=predicted_class
    )

    if explanation:
        print(explanation.format_human_readable())

    # ========================================================================
    # SCENARIO 3: MODERATE CPU STRESS (Easier to Prevent)
    # ========================================================================
    print("\n" + "█"*80)
    print("█" + " SCENARIO 3: MODERATE CPU STRESS ".center(78) + "█")
    print("█"*80)
    print()
    print("Current Situation:")
    print("  - Average CPU: 70%")
    print("  - Max CPU: 82%")
    print("  - System Stress: 65%")
    print("  - Memory: 50%")
    print()

    moderate_cpu_features = np.array([
        0.70,  # cpu_utilization_mean - MODERATE
        0.82,  # cpu_utilization_max - MODERATE HIGH
        0.12,  # cpu_variance_coef
        0.50,  # memory_utilization_mean
        0.62,  # memory_utilization_max
        0.10,  # memory_variance_coef
        0.62,  # memory_pressure_max
        0.01,  # memory_growth_rate
        0.32,  # thread_count_mean
        0.42,  # thread_count_max
        0.09,  # thread_variance_coef
        0.03,  # error_rate
        0.0,   # error_spike_indicator
        0.22,  # response_time_p95_mean
        0.30,  # response_time_p95_max
        0.07,  # response_time_variance_coef
        0.55,  # request_rate_mean
        0.68,  # request_rate_max
        0.11,  # request_variance_coef
        0.12,  # queue_depth_mean
        0.20,  # queue_depth_max
        0.07,  # queue_variance_coef
        1.0,   # service_health_min
        0.60,  # cpu_memory_correlation
        0.20,  # load_error_correlation
        0.3,   # normalized_service_count
        0.65   # system_stress_index
    ])

    # Predict
    probas = model.predict_proba(moderate_cpu_features.reshape(1, -1))[0]
    predicted_class = class_names[np.argmax(probas)]
    confidence = probas[np.argmax(probas)]

    print(f"🎯 Detection Result:")
    print(f"   Anomaly Type: {predicted_class.upper()}")
    print(f"   Confidence: {confidence:.1%}")

    # Counterfactual analysis
    explanation = analyzer.analyze(
        features=moderate_cpu_features,
        original_prediction=predicted_class
    )

    if explanation:
        print(explanation.format_human_readable())

    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print()
    print("Key Takeaways:")
    print("  ✓ Counterfactuals show EXACTLY what needs to change")
    print("  ✓ Provides specific target values (not just 'reduce CPU')")
    print("  ✓ Calculates feasibility (is this change realistic?)")
    print("  ✓ Computes in real-time (~70ms)")
    print("  ✓ Gives actionable Kubernetes-level recommendations")
    print()


if __name__ == "__main__":
    demo_counterfactual_analysis()
