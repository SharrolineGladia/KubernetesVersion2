"""
Generate a polished visual report for the recovery orchestrator alone.

Outputs:
- recovery_orchestrator_report.pdf
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

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'explanation engine')
PDF_PATH = os.path.join(OUTPUT_DIR, 'recovery_orchestrator_report.pdf')
SUMMARY_PNG_PATH = os.path.join(OUTPUT_DIR, 'recovery_orchestrator_summary_card.png')
RANKING_PNG_PATH = os.path.join(OUTPUT_DIR, 'recovery_orchestrator_ranking.png')
BREAKDOWN_PNG_PATH = os.path.join(OUTPUT_DIR, 'recovery_orchestrator_breakdown.png')

sys.path.append(str(PROJECT_ROOT / 'recovery-orchestrator'))
sys.path.append(str(PROJECT_ROOT / 'ml_detector' / 'scripts'))
sys.path.append(str(PROJECT_ROOT / 'scripts' / 'reports'))

from action_generator import ActionGenerator
from action_scorer import ActionScorer
from counterfactual_analyzer import ScenarioComparison
from generate_counterfactual_report_pdf import get_demo_explanation


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
        return os.path.join(OUTPUT_DIR, f'recovery_orchestrator_report_{timestamp}.pdf')


def build_actions_from_visual_scenarios(visual_scenarios):
    generator = ActionGenerator(
        deployment_name="web-api",
        namespace="production",
        root_cause_service="api-gateway"
    )

    actions = []
    seen_commands = set()

    for item in visual_scenarios:
        scenario = ScenarioComparison(
            scenario_name=item['scenario_name'],
            feature_changes={
                item['scenario_name']: {
                    'current': item['current'],
                    'target': item['target'],
                    'delta': item['target'] - item['current'],
                    'delta_percent': item['delta_percent']
                }
            },
            predicted_class=item['predicted_class'],
            predicted_confidence=item['predicted_confidence'],
            prevents_anomaly=item['prevents_anomaly'],
            score=item['score'],
            risk_level=item['risk_level']
        )

        for action in generator.generate_action_variants(scenario):
            if action and action.action_command not in seen_commands:
                actions.append(action)
                seen_commands.add(action.action_command)

    return actions


def get_recovery_data():
    predicted_class, confidence, explanation, visual_scenarios = get_demo_explanation()
    actions = build_actions_from_visual_scenarios(visual_scenarios)
    scorer = ActionScorer()
    scored_actions = scorer.score_actions(actions)
    return predicted_class, confidence, explanation, visual_scenarios, actions, scored_actions


def format_action_label(action) -> str:
    """Create a unique, human-readable label for each action variant."""
    action_type = action.action_type.value
    params = action.parameters or {}

    if action_type == 'reduce_load':
        return f"Reduce Load\n({params.get('load_reduction_percent', '?')}%)"
    if action_type == 'adjust_cpu_limits':
        return f"CPU Limits\n({params.get('new_cpu_limit', '?')})"
    if action_type == 'adjust_memory_limits':
        return f"Memory Limits\n({params.get('new_memory_limit', '?')})"
    if action_type == 'optimize_config':
        profile = params.get('config_profile', 'profile').replace('-optimized', '').title()
        return f"Config: {profile}"
    if action_type == 'scale_horizontal':
        return f"Scale Out\n(+{params.get('replicas_change', '?')})"
    if action_type == 'restart_pod':
        return "Restart Pod"

    return action.action_type.value.replace('_', ' ').title()


def save_summary_card(scored_actions):
    best = scored_actions[0]
    fig = plt.figure(figsize=(10, 5.8))
    fig.patch.set_facecolor('white')
    fig.text(0.07, 0.90, 'Recovery Orchestrator Summary', fontsize=22, fontweight='bold', color='#1d4ed8')
    fig.text(0.07, 0.84, 'Decision output after converting counterfactual scenarios into concrete recovery actions', fontsize=12, color='#475569')

    lines = [
        f"Recommended action: {format_action_label(best.action).replace(chr(10), ' ')}",
        f"Description: {best.action.action_description}",
        f"Command: {best.action.action_command}",
        f"Total score: {best.total_score:.1f}/100",
        f"Risk level: {best.risk_level.upper()}",
        f"Estimated duration: {best.estimated_duration_seconds} seconds",
        f"Rollback difficulty: {best.rollback_difficulty.upper()}",
        f"Expected outcome: {best.action.expected_outcome}",
    ]

    y = 0.72
    for line in lines:
        fig.text(0.09, y, '\n'.join(wrap(line, 80)), fontsize=12.5, color='#111827')
        y -= 0.08

    fig.savefig(SUMMARY_PNG_PATH, dpi=220, bbox_inches='tight')
    plt.close(fig)


def save_ranking_chart(scored_actions):
    labels = [format_action_label(s.action) for s in scored_actions]
    scores = [s.total_score for s in scored_actions]
    colors = ['#1d4ed8' if i == 0 else '#93c5fd' for i in range(len(scored_actions))]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    bars = ax.bar(labels, scores, color=colors)
    ax.set_title('Recovery Action Ranking', fontsize=17, fontweight='bold')
    ax.set_ylabel('Overall Score')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.25)

    for idx, bar in enumerate(bars):
        risk = scored_actions[idx].risk_level.upper()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{scores[idx]:.1f}\n{risk}",
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold'
        )

    fig.tight_layout()
    fig.savefig(RANKING_PNG_PATH, dpi=220, bbox_inches='tight')
    plt.close(fig)


def save_breakdown_chart(scored_actions):
    best = scored_actions[0]
    categories = ['Effectiveness', 'Safety', 'Speed', 'Cost', 'Simplicity']
    values = [
        best.effectiveness * 100,
        best.safety * 100,
        best.speed * 100,
        best.cost * 100,
        best.simplicity * 100
    ]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    bars = ax.barh(categories, values, color=['#1d4ed8', '#0ea5e9', '#10b981', '#f59e0b', '#64748b'])
    ax.set_title('Recommended Action Score Breakdown', fontsize=17, fontweight='bold')
    ax.set_xlabel('Score (%)')
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.25)

    for bar, value in zip(bars, values):
        ax.text(value + 1.5, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va='center', fontsize=10)

    fig.tight_layout()
    fig.savefig(BREAKDOWN_PNG_PATH, dpi=220, bbox_inches='tight')
    plt.close(fig)


def add_cover_page(pdf, predicted_class, confidence, scored_actions):
    best = scored_actions[0]
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.93, 'Recovery Orchestrator Report', fontsize=24, fontweight='bold', color='#1d4ed8')
    fig.text(0.08, 0.89, 'Autonomous action generation, scoring, and recommendation for the detected anomaly', fontsize=11, color='#475569')

    lines = [
        f"Upstream anomaly: {predicted_class.upper()} ({confidence * 100:.1f}% confidence)",
        f"Candidate actions scored: {len(scored_actions)}",
        f"Recommended action: {format_action_label(best.action).replace(chr(10), ' ')}",
        f"Risk / Duration: {best.risk_level.upper()} / {best.estimated_duration_seconds}s",
        f"Command: {best.action.action_command}",
    ]

    y = 0.78
    for line in lines:
        fig.text(0.1, y, '\n'.join(wrap(line, 78)), fontsize=13)
        y -= 0.07

    fig.text(0.1, 0.45, 'This report focuses only on the recovery orchestrator stage: action synthesis from counterfactual insights, multi-factor ranking, and final remediation choice.', fontsize=12, color='#334155')
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


def add_text_page(pdf, scored_actions):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.96, 'Detailed Recovery Decisions', fontsize=18, fontweight='bold', color='#0f172a')

    lines = []
    for idx, scored in enumerate(scored_actions, 1):
        lines.extend([
            f"{idx}. Action: {format_action_label(scored.action).replace(chr(10), ' ')}",
            f"   Description: {scored.action.action_description}",
            f"   Command: {scored.action.action_command}",
            f"   Score: {scored.total_score:.1f}/100",
            f"   Breakdown: eff={scored.effectiveness * 100:.1f}, safe={scored.safety * 100:.1f}, speed={scored.speed * 100:.1f}, cost={scored.cost * 100:.1f}, simp={scored.simplicity * 100:.1f}",
            f"   Risk / Rollback: {scored.risk_level.upper()} / {scored.rollback_difficulty.upper()}",
            ""
        ])

    y = 0.91
    for line in lines:
        fig.text(0.08, y, '\n'.join(wrap(line, 92)), fontsize=10.2, family='monospace', color='#111827')
        y -= 0.026
        if y < 0.08:
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.08, 0.96, 'Detailed Recovery Decisions (cont.)', fontsize=18, fontweight='bold', color='#0f172a')
            y = 0.91

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def main():
    ensure_output_dir()
    output_pdf_path = resolve_pdf_output_path()
    predicted_class, confidence, explanation, visual_scenarios, actions, scored_actions = get_recovery_data()

    save_summary_card(scored_actions)
    save_ranking_chart(scored_actions)
    save_breakdown_chart(scored_actions)

    with PdfPages(output_pdf_path) as pdf:
        add_cover_page(pdf, predicted_class, confidence, scored_actions)
        add_image_page(pdf, SUMMARY_PNG_PATH, 'Recovery Summary')
        add_image_page(pdf, RANKING_PNG_PATH, 'Recovery Action Ranking')
        add_image_page(pdf, BREAKDOWN_PNG_PATH, 'Recommended Action Breakdown')
        add_text_page(pdf, scored_actions)

    print(f"PDF report generated: {output_pdf_path}")
    print(f"Summary card: {SUMMARY_PNG_PATH}")
    print(f"Ranking chart: {RANKING_PNG_PATH}")
    print(f"Breakdown chart: {BREAKDOWN_PNG_PATH}")


if __name__ == '__main__':
    main()
