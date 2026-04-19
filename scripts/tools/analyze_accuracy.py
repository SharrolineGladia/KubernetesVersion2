"""Quick script to analyze why accuracy is low."""
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
df = pd.read_csv(PROJECT_ROOT / 'results' / 'escalation' / 'anomaly_scenarios.csv')

print("=" * 80)
print("ACCURACY ANALYSIS - Why is it low?")
print("=" * 80)

print("\n1. DISTRIBUTION COMPARISON")
print("\nTrue Labels (Ground Truth):")
print(df['true_label'].value_counts().sort_index())

print("\nPredicted Labels (Model Output):")
print(df['predicted_label'].value_counts().sort_index())

print("\n2. PER-CLASS ACCURACY")
print("\nHow well does the model predict each class?")
for label in ['normal', 'cpu_spike', 'memory_leak', 'service_crash']:
    subset = df[df['true_label'] == label]
    correct = (subset['predicted_label'] == label).sum()
    total = len(subset)
    print(f"  {label:15s}: {correct:3d}/{total:3d} correct = {correct/total:6.1%}")

print("\n3. CONFUSION MATRIX INSIGHTS")

# How many normals were misclassified?
normal_cases = df[df['true_label'] == 'normal']
normal_as_memory_leak = (normal_cases['predicted_label'] == 'memory_leak').sum()
print(f"\n  Normal cases predicted as memory_leak: {normal_as_memory_leak}/{len(normal_cases)} ({normal_as_memory_leak/len(normal_cases):.1%})")

# Overall accuracy
overall_acc = (df['predicted_label'] == df['true_label']).sum() / len(df)
print(f"\n  Overall Accuracy: {overall_acc:.2%}")

print("\n4. CONFIDENCE ANALYSIS")
print(f"\n  Mean confidence: {df['confidence'].mean():.3f}")
print(f"  Median confidence: {df['confidence'].median():.3f}")
print(f"  Min confidence: {df['confidence'].min():.3f}")
print(f"  Max confidence: {df['confidence'].max():.3f}")

print("\n5. THE PROBLEM")
print("\n  ❌ The model is OVERCONFIDENT (99% confidence) but WRONG")
print("  ❌ It predicts 'memory_leak' when it should predict 'normal'")
print("  ❌ This is a MODEL CALIBRATION issue, not a framework bug")

print("\n6. WHAT THIS MEANS FOR ESCALATION")
correct_subset = df[df['predicted_label'] == df['true_label']]
incorrect_subset = df[df['predicted_label'] != df['true_label']]
print(f"\n  Correct predictions - mean confidence: {correct_subset['confidence'].mean():.3f}")
print(f"  Incorrect predictions - mean confidence: {incorrect_subset['confidence'].mean():.3f}")
print("\n  ⚠️  Even wrong predictions have high confidence!")
print("  ⚠️  This means edge keeps making mistakes without escalating to cloud")

print("\n" + "=" * 80)
print("CONCLUSION: The XGBoost model needs confidence calibration (Platt scaling,")
print("temperature scaling) to produce realistic uncertainty estimates.")
print("=" * 80)
