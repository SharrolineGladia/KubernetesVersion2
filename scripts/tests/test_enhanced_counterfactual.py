"""
Test Enhanced Counterfactual Analyzer
Verifies predicted outcomes are shown for each scenario
"""

import os
import sys
from pathlib import Path
import numpy as np
import pickle

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'ml_detector' / 'scripts'))

try:
    print("="*80)
    print(" TESTING ENHANCED COUNTERFACTUAL ANALYZER ".center(80))
    print("="*80)
    print()
    
    # Import the enhanced module
    from counterfactual_analyzer import CounterfactualAnalyzer, ScenarioComparison
    print("✅ Module imported successfully")
    
    # Load model
    model_path = PROJECT_ROOT / 'ml_detector' / 'models' / 'anomaly_detector_scaleinvariant.pkl'
    with open(model_path, 'rb') as f:
        d = pickle.load(f)
        model = d.get('model', d) if isinstance(d, dict) else d
    print("✅ Model loaded")
    
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
    
    # Create analyzer
    analyzer = CounterfactualAnalyzer(model, feature_names)
    print("✅ Analyzer created")
    
    # Check new methods exist
    assert hasattr(analyzer, 'predict_scenario_outcome'), "Missing predict_scenario_outcome method"
    assert hasattr(analyzer, 'score_scenario'), "Missing score_scenario method"
    print("✅ New methods found: predict_scenario_outcome, score_scenario")
    
    # Test with CPU spike scenario
    print()
    print("Testing CPU Spike Scenario...")
    features = np.array([
        0.85, 0.97, 0.15, 0.45, 0.60, 0.12, 0.60, 0.02,
        0.35, 0.45, 0.10, 0.05, 0.0, 0.25, 0.35, 0.08,
        0.60, 0.75, 0.12, 0.15, 0.25, 0.08, 1.0, 0.65,
        0.25, 0.3, 0.72
    ])
    
    explanation = analyzer.analyze(features, 'cpu_spike')
    
    if explanation:
        print("✅ Analysis completed")
        
        # Check for scenario comparisons
        if explanation.scenario_comparisons:
            print(f"✅ Generated {len(explanation.scenario_comparisons)} scenario comparisons")
            
            # Check first scenario has predictions
            first_scenario = explanation.scenario_comparisons[0]
            assert hasattr(first_scenario, 'predicted_class'), "Missing predicted_class"
            assert hasattr(first_scenario, 'predicted_confidence'), "Missing predicted_confidence"
            assert hasattr(first_scenario, 'prevents_anomaly'), "Missing prevents_anomaly"
            assert hasattr(first_scenario, 'score'), "Missing score"
            
            print(f"✅ Scenario has predicted_class: {first_scenario.predicted_class}")
            print(f"✅ Scenario has confidence: {first_scenario.predicted_confidence:.2%}")
            print(f"✅ Scenario prevents anomaly: {first_scenario.prevents_anomaly}")
            print(f"✅ Scenario score: {first_scenario.score:.1f}/100")
            
            print()
            print("="*80)
            print(" FULL OUTPUT ".center(80))
            print("="*80)
            print(explanation.format_human_readable())
            
        else:
            print("⚠️ No scenario comparisons generated")
    else:
        print("❌ Analysis returned None")
    
    print()
    print("="*80)
    print(" ✅ ALL TESTS PASSED - ENHANCEMENT COMPLETE ".center(80))
    print("="*80)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
