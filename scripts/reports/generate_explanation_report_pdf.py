"""
Generate a polished PDF report for the explanation engine demo.

Outputs:
- explanation_report.pdf
- supporting PNG figures for slides/reuse

The data path intentionally mirrors the current explanation-engine demo so the
PDF is consistent with the terminal output the user already reviewed.
"""

import os
import sys
import warnings
from textwrap import wrap
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / 'ml_detector' / 'scripts'))
sys.path.append(str(PROJECT_ROOT / 'scripts' / 'demos'))

from explainability_layer import AnomalyExplainer, ServiceMetrics, format_explanation_report
from demo_integrated_rca import (
    build_demo_trace_context,
    build_demo_log_context,
    build_demo_shap_features,
)


OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'explanation engine')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'ml_detector', 'models', 'anomaly_detector_scaleinvariant.pkl')
PDF_PATH = os.path.join(OUTPUT_DIR, 'explanation_engine_report.pdf')
SHAP_PNG_PATH = os.path.join(OUTPUT_DIR, 'explanation_report_shap.png')
TIMELINE_PNG_PATH = os.path.join(OUTPUT_DIR, 'explanation_report_timeline.png')
PATTERN_PNG_PATH = os.path.join(OUTPUT_DIR, 'explanation_report_patterns.png')


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def resolve_pdf_output_path() -> str:
    """Use the standard report path, but fall back if the file is locked/open."""
    if not os.path.exists(PDF_PATH):
        return PDF_PATH

    try:
        with open(PDF_PATH, 'ab'):
            pass
        return PDF_PATH
    except PermissionError:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return os.path.join(OUTPUT_DIR, f'explanation_engine_report_{timestamp}.pdf')


def generate_report_data():
    explainer = AnomalyExplainer(
        model_path=MODEL_PATH,
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

    rca_result = explainer.explain_anomaly(
        anomaly_type='cpu_spike',
        service_metrics=service_metrics,
        scale_invariant_features=scale_invariant_features,
        timestamp=datetime.utcnow(),
        service_name='notification-service',
        trace_context_override=trace_context,
        log_context_override=log_context,
    )
    shap_importance = explainer.explain_with_shap(scale_invariant_features)
    full_report = format_explanation_report(
        rca_result,
        shap_importance=shap_importance,
        max_shap_features=5,
    )
    return rca_result, shap_importance, trace_context, log_context, full_report


def save_shap_chart(shap_importance):
    items = list(shap_importance.items())[:6]
    features = [name for name, _ in items][::-1]
    values = [score for _, score in items][::-1]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(features, values, color=['#b91c1c', '#dc2626', '#ea580c', '#f59e0b', '#0284c7', '#475569'])
    ax.set_title('Top Feature Attributions', fontsize=16, fontweight='bold')
    ax.set_xlabel('Attribution Magnitude')
    ax.grid(axis='x', alpha=0.25)
    fig.tight_layout()
    fig.savefig(SHAP_PNG_PATH, dpi=220, bbox_inches='tight')
    plt.close(fig)


def save_failure_pattern_chart(pattern_matches):
    patterns = pattern_matches[:3]
    labels = [p['pattern_name'] for p in patterns]
    labels = ['\n'.join(wrap(label, 20)) for label in labels]
    counts = [p['confidence'] * 100 for p in patterns]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(labels, counts, color=['#7c2d12', '#9a3412', '#c2410c'])
    ax.set_title('Failure Pattern Matching', fontsize=16, fontweight='bold')
    ax.set_ylabel('Match Confidence (%)')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.25)
    for idx, pattern in enumerate(patterns):
        evidence = '; '.join(pattern.get('evidence', [])[:2])
        ax.text(
            idx,
            counts[idx] + 2,
            '\n'.join(wrap(evidence, 24)),
            ha='center',
            va='bottom',
            fontsize=8.5,
            color='#374151'
        )
    fig.tight_layout()
    fig.savefig(PATTERN_PNG_PATH, dpi=220, bbox_inches='tight')
    plt.close(fig)


def save_timeline_chart(trace_context):
    timeline = trace_context['causal_timeline']
    times = [datetime.fromisoformat(event['timestamp']) for event in timeline]
    start_time = min(times)
    x_seconds = [(t - start_time).total_seconds() for t in times]
    y = [1] * len(timeline)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    if x_seconds:
        ax.hlines(y=1, xmin=min(x_seconds), xmax=max(x_seconds), color='#94a3b8', linewidth=2.5, zorder=1)

    ax.scatter(x_seconds, y, s=220, color='#0f766e', zorder=3)

    for idx, event in enumerate(timeline):
        timestamp_label = times[idx].strftime('%H:%M:%S')
        service_label = event['service']
        detail_label = '\n'.join(wrap(event['details'], 28))
        offset_y = 0.14 if idx % 2 == 0 else -0.22
        va = 'bottom' if idx % 2 == 0 else 'top'

        ax.annotate(
            f"{timestamp_label}\n{service_label}",
            (x_seconds[idx], 1),
            xytext=(0, 18 if idx % 2 == 0 else -24),
            textcoords='offset points',
            ha='center',
            va=va,
            fontsize=9.5,
            fontweight='bold',
            color='#0f172a',
            arrowprops=dict(arrowstyle='-', color='#94a3b8', lw=1),
        )
        ax.text(
            x_seconds[idx],
            1 + offset_y,
            detail_label,
            ha='center',
            va=va,
            fontsize=9.5,
            color='#334155'
        )

    ax.set_title('Trace-Based Causal Timeline', fontsize=16, fontweight='bold')
    ax.set_xlabel('Time Since First Trace Event (seconds)')
    ax.set_yticks([])
    ax.set_ylim(0.55, 1.45)
    if x_seconds:
        padding = max(0.8, (max(x_seconds) - min(x_seconds)) * 0.12)
        ax.set_xlim(min(x_seconds) - padding, max(x_seconds) + padding)
    ax.grid(axis='x', alpha=0.2, linestyle='--')
    for spine in ['left', 'right', 'top']:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(TIMELINE_PNG_PATH, dpi=220, bbox_inches='tight')
    plt.close(fig)


def add_cover_page(pdf, rca_result):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')
    fig.text(0.08, 0.93, 'Explanation Engine Report', fontsize=24, fontweight='bold', color='#7f1d1d')
    fig.text(0.08, 0.89, 'Anomaly explanation with root cause, SHAP attribution, trace timeline, and failure patterns', fontsize=11, color='#334155')

    summary_lines = [
        f"Anomaly Type: {rca_result.anomaly_type.upper()}",
        f"Root Cause Microservice: {rca_result.root_cause_service}",
        f"Severity: {rca_result.severity.upper()}",
        f"Affected Resources: {', '.join(rca_result.affected_resources)}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]

    y = 0.79
    for line in summary_lines:
        fig.text(0.1, y, line, fontsize=13)
        y -= 0.05

    info = [
        ('Metrics', 'CPU 92.3%, memory 78.5%, error rate 35%, P95 latency 850ms'),
        ('Root Cause', 'notification-service identified as the anomalous microservice'),
        ('SHAP', 'Top features ranked directly from the trained XGBoost model'),
        ('Traces', 'Timeline shows latency escalation and downstream timeout propagation'),
        ('Failure Patterns', 'Current incident is matched against known failure archetypes using metrics, traces, and logs'),
    ]
    y = 0.54
    for title, desc in info:
        fig.text(0.1, y, title, fontsize=13, fontweight='bold', color='#111827')
        fig.text(0.27, y, '\n'.join(wrap(desc, 60)), fontsize=12, color='#374151')
        y -= 0.1

    fig.text(0.1, 0.08, 'Generated from the explanation engine demo pipeline.', fontsize=10, color='#64748b')
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


def add_text_report_page(pdf, report_text):
    lines = report_text.splitlines()
    chunk_size = 42

    for start in range(0, len(lines), chunk_size):
        chunk = lines[start:start + chunk_size]
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.96, 'Detailed Explanation Report', fontsize=18, fontweight='bold', color='#0f172a')
        y = 0.92
        for line in chunk:
            fig.text(0.08, y, line, fontsize=9.4, family='monospace', color='#111827')
            y -= 0.02
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)


def main():
    ensure_output_dir()
    rca_result, shap_importance, trace_context, log_context, full_report = generate_report_data()
    output_pdf_path = resolve_pdf_output_path()

    save_shap_chart(shap_importance)
    save_timeline_chart(trace_context)
    save_failure_pattern_chart(rca_result.failure_pattern_matches or [])

    with PdfPages(output_pdf_path) as pdf:
        add_cover_page(pdf, rca_result)
        add_image_page(pdf, SHAP_PNG_PATH, 'SHAP Feature Attribution')
        add_image_page(pdf, TIMELINE_PNG_PATH, 'Trace Timeline')
        add_image_page(pdf, PATTERN_PNG_PATH, 'Failure Pattern Matching')
        add_text_report_page(pdf, full_report)

    print(f"PDF report generated: {output_pdf_path}")
    print(f"SHAP chart: {SHAP_PNG_PATH}")
    print(f"Timeline chart: {TIMELINE_PNG_PATH}")
    print(f"Pattern chart: {PATTERN_PNG_PATH}")


if __name__ == '__main__':
    main()
