"""
Generate a polished visual report for counterfactual analysis.

Outputs:
- counterfactual_report.pdf
- slide-friendly PNG charts
"""

import os
import sys
import warnings
from datetime import datetime
from textwrap import wrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pickle

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'explanation engine')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'ml_detector', 'models', 'anomaly_detector_scaleinvariant.pkl')
PDF_PATH = os.path.join(OUTPUT_DIR, 'counterfactual_report.pdf')
COMPARISON_PNG_PATH = os.path.join(OUTPUT_DIR, 'counterfactual_visual_comparison.png')
DELTA_PNG_PATH = os.path.join(OUTPUT_DIR, 'counterfactual_required_change.png')
SUMMARY_PNG_PATH = os.path.join(OUTPUT_DIR, 'counterfactual_summary_card.png')

sys.path.append(str(PROJECT_ROOT / 'ml_detector' / 'scripts'))
from counterfactual_analyzer import CounterfactualAnalyzer


FEATURE_NAMES = [
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


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def resolve_pdf_output_path() -> str:
    if not os.path.exists(PDF_PATH):
        return PDF_PATH
    try:
        with open(PDF_PATH, 'ab'):
            pass
        return PDF_PATH
    except PermissionError:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return os.path.join(OUTPUT_DIR, f'counterfactual_report_{timestamp}.pdf')


def load_model():
    with open(MODEL_PATH, 'rb') as f:
        saved_data = pickle.load(f)
    return saved_data.get('model', saved_data) if isinstance(saved_data, dict) else saved_data


def get_demo_explanation():
    model = load_model()
    analyzer = CounterfactualAnalyzer(model, FEATURE_NAMES)

    features = np.array([
        0.85, 0.97, 0.15,
        0.45, 0.60, 0.12, 0.60, 0.02,
        0.35, 0.45, 0.10,
        0.05, 0.0,
        0.25, 0.35, 0.08,
        0.60, 0.75, 0.12,
        0.15, 0.25, 0.08,
        1.0,
        0.65, 0.25,
        0.3,
        0.72
    ])

    prediction_proba = model.predict_proba(features.reshape(1, -1))[0]
    classes = ['cpu_spike', 'memory_leak', 'normal', 'service_crash']
    predicted_class = classes[np.argmax(prediction_proba)]
    confidence = prediction_proba[np.argmax(prediction_proba)]
    explanation = analyzer.analyze(features, predicted_class)
    visual_scenarios = build_visual_scenarios(analyzer, features, predicted_class)
    return predicted_class, confidence, explanation, visual_scenarios


def build_visual_scenarios(analyzer, features, predicted_class):
    """Create a richer scenario set for visualization, including non-winning what-if cases."""
    importance = analyzer.model.feature_importances_
    top_indices = np.argsort(importance)[-5:][::-1]
    candidate_pcts = [15, 30, 45, 55]
    scenarios = []

    for feat_idx in top_indices:
        feature_name = FEATURE_NAMES[feat_idx]
        original_value = float(features[feat_idx])

        best_variant = None
        for pct in candidate_pcts:
            modified = features.copy()
            target_value = max(0.0, original_value * (1 - pct / 100.0))
            modified[feat_idx] = target_value
            predicted_outcome, pred_confidence = analyzer.predict_scenario_outcome(modified)
            prevents_anomaly = predicted_outcome == 'normal' and pred_confidence > 0.5
            score, risk_level = analyzer.score_scenario(
                prevents_anomaly=prevents_anomaly,
                confidence=pred_confidence,
                delta_percent=-pct,
                original_prediction=predicted_class
            )

            variant = {
                'scenario_name': feature_name,
                'current': original_value,
                'target': target_value,
                'delta_percent': -pct,
                'predicted_class': predicted_outcome,
                'predicted_confidence': pred_confidence,
                'prevents_anomaly': prevents_anomaly,
                'score': score,
                'risk_level': risk_level
            }

            if best_variant is None or variant['score'] > best_variant['score']:
                best_variant = variant

        if best_variant:
            scenarios.append(best_variant)

    scenarios.sort(key=lambda item: item['score'], reverse=True)
    return scenarios[:4]


def save_comparison_chart(explanation, visual_scenarios):
    scenarios = visual_scenarios or []
    labels = []
    scores = []
    confidences = []
    colors = []

    for scenario in scenarios:
        labels.append('\n'.join(wrap(scenario['scenario_name'].replace('_', ' '), 16)))
        scores.append(scenario['score'])
        confidences.append(scenario['predicted_confidence'] * 100)
        colors.append('#15803d' if scenario['prevents_anomaly'] else '#b91c1c')

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.bar(x - width / 2, scores, width, label='Scenario Score', color=colors, alpha=0.9)
    ax.bar(x + width / 2, confidences, width, label='Prediction Confidence (%)', color='#1d4ed8', alpha=0.7)

    ax.set_title('Counterfactual Scenario Comparison', fontsize=17, fontweight='bold')
    ax.set_ylabel('Score / Confidence')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.25)
    ax.legend()

    best_name = None
    if explanation and explanation.scenario_comparisons:
        best_name = explanation.scenario_comparisons[explanation.best_scenario_idx].scenario_name

    for idx, scenario in enumerate(scenarios):
        status = 'BEST' if scenario['scenario_name'] == best_name else ('PREVENTS' if scenario['prevents_anomaly'] else 'ALT')
        ax.text(x[idx] - width / 2, scores[idx] + 2, status, ha='center', va='bottom', fontsize=9, fontweight='bold')

    fig.tight_layout()
    fig.savefig(COMPARISON_PNG_PATH, dpi=220, bbox_inches='tight')
    plt.close(fig)


def save_delta_chart(visual_scenarios):
    scenarios = visual_scenarios or []
    labels = []
    deltas = []
    targets = []

    for scenario in scenarios:
        feature_name = scenario['scenario_name']
        labels.append('\n'.join(wrap(feature_name.replace('_', ' '), 16)))
        deltas.append(abs(scenario['delta_percent']))
        targets.append(scenario['target'])

    fig, ax = plt.subplots(figsize=(10, 5.8))
    bars = ax.bar(labels, deltas, color='#ea580c')
    ax.set_title('Required Change for Each Counterfactual', fontsize=17, fontweight='bold')
    ax.set_ylabel('Absolute Change Needed (%)')
    ax.grid(axis='y', alpha=0.25)

    for idx, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"Target {targets[idx]:.2f}",
            ha='center',
            va='bottom',
            fontsize=9
        )

    fig.tight_layout()
    fig.savefig(DELTA_PNG_PATH, dpi=220, bbox_inches='tight')
    plt.close(fig)


def save_summary_card(predicted_class, confidence, explanation, visual_scenarios):
    fig = plt.figure(figsize=(10, 5.8))
    fig.patch.set_facecolor('white')

    fig.text(0.07, 0.90, 'Counterfactual Summary', fontsize=22, fontweight='bold', color='#7c2d12')
    fig.text(0.07, 0.84, f'Original anomaly: {predicted_class.upper()} ({confidence * 100:.1f}% confidence)', fontsize=13, color='#334155')

    best = explanation.scenario_comparisons[explanation.best_scenario_idx] if explanation.scenario_comparisons else None
    if best:
        feature_name = list(best.feature_changes.keys())[0]
        change = best.feature_changes[feature_name]
        alternatives = ", ".join(
            s['scenario_name'] for s in visual_scenarios[1:4]
        ) if visual_scenarios and len(visual_scenarios) > 1 else "None"
        lines = [
            f"Best scenario: {feature_name}",
            f"Current -> Target: {change['current']:.2f} -> {change['target']:.2f}",
            f"Required change: {change['delta_percent']:+.1f}%",
            f"Predicted outcome: {best.predicted_class.upper()} ({best.predicted_confidence * 100:.1f}%)",
            f"Scenario score: {best.score:.1f}/100",
            f"Risk level: {best.risk_level.upper()}",
            f"Compared alternatives: {alternatives}",
            f"Recommendation: {explanation.actionable_recommendation}",
            f"Computation time: {explanation.search_time_ms:.1f}ms",
        ]
    else:
        lines = [
            "No feasible counterfactual found",
            "This anomaly likely requires multiple interventions."
        ]

    y = 0.72
    for line in lines:
        fig.text(0.09, y, '\n'.join(wrap(line, 70)), fontsize=13, color='#111827')
        y -= 0.08

    fig.savefig(SUMMARY_PNG_PATH, dpi=220, bbox_inches='tight')
    plt.close(fig)


def add_cover_page(pdf, predicted_class, confidence, explanation, visual_scenarios):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.93, 'Counterfactual Analysis Report', fontsize=24, fontweight='bold', color='#7c2d12')
    fig.text(0.08, 0.89, 'What-if scenarios showing how the anomaly could have been prevented', fontsize=11, color='#475569')

    summary = [
        f"Original anomaly: {predicted_class.upper()}",
        f"Model confidence: {confidence * 100:.1f}%",
        f"Feasible prevention found: {'YES' if explanation and explanation.is_feasible else 'NO'}",
        f"Computation time: {explanation.search_time_ms:.1f}ms" if explanation else "Computation time: N/A",
        f"Recommendation: {explanation.actionable_recommendation}" if explanation else "Recommendation: N/A",
        f"Scenarios compared: {len(visual_scenarios)}"
    ]

    y = 0.78
    for line in summary:
        fig.text(0.1, y, '\n'.join(wrap(line, 70)), fontsize=13)
        y -= 0.065

    fig.text(0.1, 0.42, 'This report visualizes counterfactual scenario comparison, required intervention magnitude, and the best recommended prevention strategy.', fontsize=12, color='#334155')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def add_image_page(pdf, image_path, title):
    img = plt.imread(image_path)
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.96, title, fontsize=20, fontweight='bold', color='#0f172a')
    ax = fig.add_axes([0.07, 0.08, 0.86, 0.84])
    ax.imshow(img)
    ax.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def add_text_page(pdf, predicted_class, confidence, explanation, visual_scenarios):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.96, 'Detailed Counterfactual Findings', fontsize=18, fontweight='bold', color='#0f172a')

    lines = [
        f"Original anomaly: {predicted_class.upper()} ({confidence * 100:.1f}%)",
        ""
    ]

    if visual_scenarios:
        best_name = None
        if explanation and explanation.scenario_comparisons:
            best_name = explanation.scenario_comparisons[explanation.best_scenario_idx].scenario_name

        for idx, scenario in enumerate(visual_scenarios, 1):
            feature_name = scenario['scenario_name']
            lines.extend([
                f"{idx}. Scenario: {feature_name}",
                f"   Change: {scenario['current']:.2f} -> {scenario['target']:.2f} ({scenario['delta_percent']:+.1f}%)",
                f"   Predicted outcome: {scenario['predicted_class'].upper()} ({scenario['predicted_confidence'] * 100:.1f}%)",
                f"   Prevents anomaly: {'YES' if scenario['prevents_anomaly'] else 'NO'}",
                f"   Score / Risk: {scenario['score']:.1f} / {scenario['risk_level'].upper()}",
                f"   Best scenario: {'YES' if scenario['scenario_name'] == best_name else 'NO'}",
                ""
            ])
        lines.extend([
            f"Best recommendation: {explanation.actionable_recommendation}",
            f"Total computation time: {explanation.search_time_ms:.1f}ms"
        ])
    else:
        lines.append("No feasible counterfactual scenarios were found.")

    y = 0.91
    for line in lines:
        fig.text(0.08, y, line, fontsize=10.5, family='monospace', color='#111827')
        y -= 0.028

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def main():
    ensure_output_dir()
    output_pdf_path = resolve_pdf_output_path()
    predicted_class, confidence, explanation, visual_scenarios = get_demo_explanation()

    save_summary_card(predicted_class, confidence, explanation, visual_scenarios)
    if visual_scenarios:
        save_comparison_chart(explanation, visual_scenarios)
        save_delta_chart(visual_scenarios)

    with PdfPages(output_pdf_path) as pdf:
        add_cover_page(pdf, predicted_class, confidence, explanation, visual_scenarios)
        add_image_page(pdf, SUMMARY_PNG_PATH, 'Counterfactual Summary')
        if visual_scenarios:
            add_image_page(pdf, COMPARISON_PNG_PATH, 'Scenario Comparison')
            add_image_page(pdf, DELTA_PNG_PATH, 'Required Change Analysis')
        add_text_page(pdf, predicted_class, confidence, explanation, visual_scenarios)

    print(f"PDF report generated: {output_pdf_path}")
    print(f"Summary card: {SUMMARY_PNG_PATH}")
    if explanation and explanation.scenario_comparisons:
        print(f"Comparison chart: {COMPARISON_PNG_PATH}")
        print(f"Delta chart: {DELTA_PNG_PATH}")


if __name__ == '__main__':
    main()
