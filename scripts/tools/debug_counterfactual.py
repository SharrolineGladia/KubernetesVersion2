"""Debug counterfactual search"""
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'ml_detector' / 'scripts'))

from counterfactual_analyzer import CounterfactualAnalyzer
import pickle

# Load model
with open(PROJECT_ROOT / 'ml_detector' / 'models' / 'anomaly_detector_scaleinvariant.pkl', 'rb') as f:
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

features = np.array([
    0.85, 0.97, 0.15, 0.45, 0.60, 0.12, 0.60, 0.02,
    0.35, 0.45, 0.10, 0.05, 0.0, 0.25, 0.35, 0.08,
    0.60, 0.75, 0.12, 0.15, 0.25, 0.08, 1.0, 0.65,
    0.25, 0.3, 0.72
])

print("Original prediction:")
pred = model.predict_proba(features.reshape(1, -1))[0]
print(f"  Probas: {pred}")
print(f"  Predicted class: {np.argmax(pred)} ({analyzer.class_names[np.argmax(pred)]})")
print()

# Test binary search directly on cpu_utilization_mean (index 0)
print('Testing binary search on cpu_utilization_mean (index 0)...')
result = analyzer._binary_search_counterfactual(features, 0, 'cpu_spike')
print(f'Result: {result}')
print()

# Test on system_stress_index (index 26)
print('Testing binary search on system_stress_index (index 26)...')
result2 = analyzer._binary_search_counterfactual(features, 26, 'cpu_spike')
print(f'Result: {result2}')
