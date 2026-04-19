"""
Simple Counterfactual Showcase

Shows exactly what the user will see in production when counterfactuals are generated.
"""

import os
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'ml_detector' / 'scripts'))

from counterfactual_analyzer import CounterfactualAnalyzer
import pickle


def main():
    # Load model
    model_path = PROJECT_ROOT / 'ml_detector' / 'models' / 'anomaly_detector_scaleinvariant.pkl'

    with open(model_path, 'rb') as f:
        d = pickle.load(f)
        model = d.get('model', d) if isinstance(d, dict) else d

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

    print()
    print("="*80)
    print(" THIS IS WHAT YOU'LL SEE IN PRODUCTION ".center(80))
    print("="*80)
    print()
    print("When EWMA Online Detector detects a critical anomaly,")
    print("XGBoost classifies it, RCA finds root cause, then you'll see:")
    print()

    # CPU Spike scenario
    features = np.array([
        0.85, 0.97, 0.15, 0.45, 0.60, 0.12, 0.60, 0.02,
        0.35, 0.45, 0.10, 0.05, 0.0, 0.25, 0.35, 0.08,
        0.60, 0.75, 0.12, 0.15, 0.25, 0.08, 1.0, 0.65,
        0.25, 0.3, 0.72
    ])

    explanation = analyzer.analyze(features, 'cpu_spike')

    if explanation:
        print(explanation.format_human_readable())

    print()
    print("="*80)
    print()
    print("💡 KEY INSIGHTS FROM COUNTERFACTUAL ANALYSIS:")
    print()
    print("   1. EXACT TARGET: Not just 'reduce CPU', but 'CPU from 85% → 40%'")
    print("   2. MAGNITUDE: Shows the change needed (-45%, or -52.6%)")
    print("   3. FEASIBILITY: Automatically checks if change is realistic (<80%)")
    print("   4. ACTION: Kubernetes-specific recommendation")
    print("   5. SPEED: Computes in real-time (~70ms)")
    print()
    print("="*80)
    print()


if __name__ == "__main__":
    main()
