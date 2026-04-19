"""
Quick syntax check for counterfactual_analyzer
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'ml_detector' / 'scripts'))

try:
    from counterfactual_analyzer import CounterfactualAnalyzer, CounterfactualExplanation, ScenarioComparison
    print("✅ Module imported successfully!")
    print(f"✅ CounterfactualAnalyzer class found")
    print(f"✅ ScenarioComparison class found")
    print(f"✅ CounterfactualExplanation class found")
    
    # Check if methods exist
    import inspect
    methods = [m for m in dir(CounterfactualAnalyzer) if not m.startswith('_')]
    print(f"\n📋 Public methods: {', '.join(methods)}")
    
except SyntaxError as e:
    print(f"❌ Syntax Error: {e}")
    import traceback
    traceback.print_exc()
except ImportError as e:
    print(f"❌ Import Error: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"❌ Unexpected Error: {e}")
    import traceback
    traceback.print_exc()
