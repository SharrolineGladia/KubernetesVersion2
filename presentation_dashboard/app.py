from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard_data import (
    artifact_paths,
    counterfactual_bundle,
    ewma_series,
    ewma_snapshot,
    explainability_bundle,
    injection_steps,
    read_bytes,
    recovery_bundle,
    services_dataframe,
)


st.set_page_config(
    page_title="Autonomous Anomaly Response Console",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(14, 165, 233, 0.10), transparent 28%),
                linear-gradient(180deg, #f6fbff 0%, #eef4f8 46%, #f7fafc 100%);
            color: #0f172a;
        }
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 3rem;
            max-width: 1460px;
        }
        .hero {
            padding: 1.25rem 1.5rem;
            border-radius: 22px;
            background: linear-gradient(135deg, #0f172a 0%, #102f49 46%, #0f766e 100%);
            color: #f8fafc;
            border: 1px solid rgba(148, 163, 184, 0.20);
            box-shadow: 0 24px 60px rgba(15, 23, 42, 0.22);
            margin-bottom: 1.1rem;
        }
        .hero-kicker {
            color: #7dd3fc;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.76rem;
            font-weight: 700;
        }
        .hero-title {
            font-size: 2rem;
            font-weight: 800;
            margin: 0.3rem 0 0.55rem 0;
        }
        .hero-subtitle {
            font-size: 1rem;
            color: #dbeafe;
            max-width: 70rem;
        }
        .card {
            background: rgba(255, 255, 255, 0.76);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 18px;
            padding: 1rem 1.05rem;
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.07);
            min-height: 122px;
        }
        .card-title {
            color: #475569;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.72rem;
            font-weight: 700;
            margin-bottom: 0.45rem;
        }
        .card-value {
            font-size: 1.8rem;
            line-height: 1.1;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.2rem;
        }
        .card-note {
            color: #475569;
            font-size: 0.92rem;
        }
        .section-title {
            font-size: 1.35rem;
            font-weight: 800;
            color: #0f172a;
            margin-top: 0.35rem;
            margin-bottom: 0.15rem;
        }
        .section-copy {
            color: #475569;
            margin-bottom: 0.8rem;
        }
        .pill {
            display: inline-block;
            padding: 0.22rem 0.65rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 700;
            margin-right: 0.45rem;
            margin-bottom: 0.35rem;
        }
        .pill-ok {
            background: rgba(34, 197, 94, 0.12);
            color: #166534;
        }
        .pill-warn {
            background: rgba(245, 158, 11, 0.14);
            color: #92400e;
        }
        .pill-bad {
            background: rgba(239, 68, 68, 0.12);
            color: #991b1b;
        }
        .code-panel {
            background: #0f172a;
            border-radius: 16px;
            padding: 1rem;
            color: #dbeafe;
            border: 1px solid rgba(125, 211, 252, 0.14);
            font-size: 0.92rem;
        }
        .report-box {
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 18px;
            padding: 1rem;
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.05);
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.80);
            border: 1px solid rgba(148, 163, 184, 0.20);
            padding: 0.9rem 1rem;
            border-radius: 16px;
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.05);
        }
        .small-note {
            color: #64748b;
            font-size: 0.88rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def card(title: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">{title}</div>
            <div class="card-value">{value}</div>
            <div class="card-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def image_if_exists(path: Path, caption: str | None = None) -> None:
    if path.exists():
        st.image(str(path), use_container_width=True, caption=caption)
    else:
        st.info(f"Artifact not found: {path.name}")


def download_button(label: str, path: Path, mime: str) -> None:
    if path.exists():
        st.download_button(label, data=read_bytes(path), file_name=path.name, mime=mime, use_container_width=True)
    else:
        st.button(f"{label} unavailable", disabled=True, use_container_width=True)


def report_preview(title: str, text: str, path: Path, image_paths: list[Path] | None = None) -> None:
    st.markdown(f"**{title}**")
    st.markdown('<div class="report-box">', unsafe_allow_html=True)
    if text:
        st.code(text[:4000], language="text")
    else:
        st.info("Report content becomes available once an anomaly is detected.")
    st.markdown('</div>', unsafe_allow_html=True)
    if image_paths:
        cols = st.columns(len(image_paths))
        for col, img_path in zip(cols, image_paths):
            with col:
                image_if_exists(img_path)
    download_button(f"Download {title}", path, "application/pdf")


def build_service_chart(df: pd.DataFrame) -> go.Figure:
    melted = df.melt(
        id_vars=["service", "status"],
        value_vars=["cpu_percent", "memory_percent", "error_rate_percent"],
        var_name="metric",
        value_name="value",
    )
    metric_labels = {
        "cpu_percent": "CPU %",
        "memory_percent": "Memory %",
        "error_rate_percent": "Error Rate %",
    }
    melted["metric"] = melted["metric"].map(metric_labels)
    fig = px.bar(
        melted,
        x="service",
        y="value",
        color="metric",
        barmode="group",
        color_discrete_map={"CPU %": "#ef4444", "Memory %": "#0ea5e9", "Error Rate %": "#f59e0b"},
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=18, b=10), legend_title_text="", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.65)")
    return fig


def build_ewma_chart(df: pd.DataFrame, threshold: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["stress_score"], mode="lines+markers", name="Stress score", line=dict(color="#dc2626", width=3), marker=dict(size=9)))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["resource_pressure"], mode="lines+markers", name="Resource pressure", line=dict(color="#0284c7", width=3), marker=dict(size=8)))
    fig.add_hline(y=threshold, line_dash="dash", line_color="#0f172a", annotation_text="classification threshold")
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=18, b=10), yaxis_title="Score", xaxis_title="Observation window", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.65)", legend_title_text="")
    return fig


def build_trace_timeline_chart(df: pd.DataFrame) -> go.Figure:
    timeline = df.copy()
    timeline["event_time"] = pd.to_datetime(timeline["timestamp"])
    timeline = timeline.sort_values("event_time").reset_index(drop=True)
    timeline["step"] = range(1, len(timeline) + 1)
    timeline["label"] = timeline["service"] + "<br>" + timeline["details"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=timeline["event_time"],
            y=timeline["step"],
            mode="lines+markers+text",
            text=timeline["service"],
            textposition="top center",
            line=dict(color="#0f766e", width=4),
            marker=dict(size=18, color="#0ea5e9", line=dict(color="#0f172a", width=1)),
            hovertext=timeline["details"],
            hovertemplate="%{text}<br>%{hovertext}<br>%{x|%H:%M:%S}<extra></extra>",
            name="Trace timeline",
        )
    )
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=18, b=10),
        xaxis_title="Trace event time",
        yaxis_title="Sequence",
        yaxis=dict(tickmode="linear", dtick=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.65)",
        showlegend=False,
    )
    return fig


def build_counterfactual_chart(df: pd.DataFrame) -> go.Figure:
    fig = px.bar(df, x="scenario_name", y="score", color="prevents_anomaly", text="predicted_confidence_percent", color_discrete_map={True: "#16a34a", False: "#dc2626"}, labels={"scenario_name": "Scenario", "score": "Scenario score", "prevents_anomaly": "Prevents anomaly"})
    fig.update_traces(texttemplate="%{text:.1f}% conf", textposition="outside")
    fig.update_layout(height=390, margin=dict(l=10, r=10, t=18, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.65)")
    return fig


def build_recovery_breakdown_chart(df: pd.DataFrame) -> go.Figure:
    top = df.iloc[0]
    metrics = pd.DataFrame([
        {"dimension": "Effectiveness", "score": top["effectiveness"]},
        {"dimension": "Safety", "score": top["safety"]},
        {"dimension": "Speed", "score": top["speed"]},
        {"dimension": "Cost", "score": top["cost"]},
        {"dimension": "Simplicity", "score": top["simplicity"]},
    ])
    fig = px.bar(metrics, x="dimension", y="score", color="dimension", color_discrete_sequence=["#1d4ed8", "#0ea5e9", "#10b981", "#f59e0b", "#64748b"])
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=18, b=10), showlegend=False, yaxis_title="Percent", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.65)")
    return fig


inject_styles()
artifacts = artifact_paths()
if "anomaly_injected" not in st.session_state:
    st.session_state.anomaly_injected = False

is_injected = st.session_state.anomaly_injected
services = services_dataframe(is_injected)
injection = injection_steps(is_injected)
ewma = ewma_series(is_injected)
snapshot = ewma_snapshot(is_injected)
explain = explainability_bundle(is_injected)
counterfactual = counterfactual_bundle(is_injected)
recovery = recovery_bundle(is_injected)

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Operational Console</div>
        <div class="hero-title">Autonomous Anomaly Response Console</div>
        <div class="hero-subtitle">
            End-to-end dashboard for system health, anomaly injection, frozen EWMA snapshots,
            explainability, counterfactual analysis, and recovery orchestration.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

action_col1, action_col2, action_col3 = st.columns([1.2, 1.2, 2.4])
with action_col1:
    if st.button("Inject Anomaly", use_container_width=True, type="primary", disabled=is_injected):
        st.session_state.anomaly_injected = True
        st.rerun()
with action_col2:
    if st.button("Restore Normal State", use_container_width=True, disabled=not is_injected):
        st.session_state.anomaly_injected = False
        st.rerun()
with action_col3:
    state_pill = "<div class='pill pill-bad'>Incident active</div>" if is_injected else "<div class='pill pill-ok'>System healthy</div>"
    st.markdown(state_pill, unsafe_allow_html=True)
    st.caption("The dashboard replays the same local demo flow and dataset-backed outputs used by the project scripts.")

if "executed_recovery_action" not in st.session_state:
    st.session_state.executed_recovery_action = None

status_label = "Critical" if is_injected else "Normal"
status_note = "EWMA crossed the 0.60 threshold and froze the incident context" if is_injected else "All monitored channels remain within steady-state bounds"

top_col1, top_col2, top_col3, top_col4 = st.columns(4)
with top_col1:
    card("Services online", "5", "Multi-service environment available for detection and orchestration")
with top_col2:
    card("System state", status_label, status_note)
with top_col3:
    card("Explainability", "Ready" if is_injected else "Standby", "Root-cause reports activate once a frozen anomaly snapshot exists")
with top_col4:
    card("Pipeline", "Active", "EWMA -> XGBoost -> RCA -> Counterfactuals -> Recovery")

tabs = st.tabs([
    "1. Services Running",
    "2. Anomaly Injection",
    "3. EWMA Snapshot",
    "4. Explainability Report",
    "5. Counterfactuals",
    "6. Recovery Orchestration",
])

with tabs[0]:
    st.markdown('<div class="section-title">Services Running</div>', unsafe_allow_html=True)
    subtitle = "Current system health after anomaly injection." if is_injected else "Current steady-state system health before any anomaly injection."
    st.markdown(f'<div class="section-copy">{subtitle}</div>', unsafe_allow_html=True)
    summary1, summary2, summary3 = st.columns(3)
    with summary1:
        st.metric("Healthy services", int((services["status"] == "Healthy").sum()))
    with summary2:
        st.metric("Stressed services", int((services["status"] == "Stressed").sum()))
    with summary3:
        st.metric("Average p95 latency", f'{services["p95_latency_ms"].mean():.0f} ms')
    st.plotly_chart(build_service_chart(services), use_container_width=True)

    left, right = st.columns([1.3, 1.2])
    with left:
        st.dataframe(services.rename(columns={"service": "Service", "status": "Status", "cpu_percent": "CPU %", "memory_percent": "Memory %", "error_rate_percent": "Error %", "p95_latency_ms": "P95 latency ms", "queue_depth": "Queue depth", "requests_per_second": "Req/s"}), use_container_width=True, hide_index=True)
    with right:
        explanation_copy = (
            "notification and notification-worker are now carrying the anomaly load while the rest of the topology stays comparatively healthy."
            if is_injected else
            "all services remain within normal operating thresholds, which gives us a clean baseline before any incident is introduced."
        )
        st.markdown(f"""
            <div class="card">
                <div class="card-title">Operational interpretation</div>
                <div class="card-note">{explanation_copy}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("")
        st.markdown("""
            <div class="code-panel">
            Environment: local demo workflow backed by project datasets and generated reports<br>
            Signals available: metrics, traces, structured logs<br>
            Incident path: online monitoring -> snapshot freeze -> explanation -> remediation
            </div>
            """, unsafe_allow_html=True)

with tabs[1]:
    st.markdown('<div class="section-title">Anomaly Injection</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">Trigger the same anomaly progression used by the local project demos without requiring live backend services.</div>', unsafe_allow_html=True)

    a1, a2, a3 = st.columns(3)
    with a1:
        st.metric("Triggered channels", "3" if is_injected else "0", "resource, performance, backpressure")
    with a2:
        st.metric("Injected requests", "1000" if is_injected else "0", "notification flood to create queue pressure")
    with a3:
        st.metric("Pipeline state", "Incident running" if is_injected else "Baseline only")

    st.dataframe(injection.rename(columns={"time": "Time", "operation": "Operation", "channel": "Channel", "impact": "Impact", "evidence": "Evidence"}), use_container_width=True, hide_index=True)

    col_left, col_right = st.columns([1.15, 1])
    with col_left:
        st.markdown("**Demo command flow**")
        st.code("python anomaly-trigger/trigger_anomaly.py --resource --performance --backpressure", language="bash")
        st.code("python scripts/demos/demo_full_pipeline.py", language="bash")
    with col_right:
        copy = (
            "The anomaly has been injected, so the remaining tabs now show the stressed-state EWMA, explainability artifacts, counterfactual proof, and recovery plan."
            if is_injected else
            "Press `Inject Anomaly` to move the system from baseline monitoring into the stressed incident flow."
        )
        st.markdown(f"""
            <div class="card">
                <div class="card-title">Flow state</div>
                <div class="card-note">{copy}</div>
            </div>
            """, unsafe_allow_html=True)

with tabs[2]:
    st.markdown('<div class="section-title">Online EWMA Snapshot</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">EWMA monitoring remains low in the normal path and freezes a snapshot only after sustained pressure during the incident path.</div>', unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Stress score", f'{snapshot["stress_score"]:.2f}')
    with s2:
        st.metric("Threshold", f'{snapshot["threshold"]:.2f}')
    with s3:
        st.metric("Peak stress memory", f'{snapshot["peak_stress_memory"]:.2f}')
    with s4:
        st.metric("Resource pressure", f'{snapshot["resource_pressure"]:.2f}')

    st.plotly_chart(build_ewma_chart(ewma, snapshot["threshold"]), use_container_width=True)
    left, right = st.columns([1.15, 1])
    with left:
        st.dataframe(ewma.rename(columns={"timestamp": "Timestamp", "stress_score": "Stress score", "resource_pressure": "Resource pressure", "state": "State", "cpu_percent": "CPU %", "memory_mb": "Memory MB", "threads": "Threads"}), use_container_width=True, hide_index=True)
    with right:
        st.markdown(f"""
            <div class="card">
                <div class="card-title">Snapshot summary</div>
                <div class="card-value">{snapshot["state"].title()}</div>
                <div class="card-note">
                    Observed at <strong>{snapshot["trigger_time"]}</strong><br>
                    Reason: {snapshot["reason"]}<br>
                    Used for: {snapshot["captured_for"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("")
        image_if_exists(artifacts["ewma_spike_plot"] if is_injected else artifacts["ewma_sustained_plot"], "Stored EWMA evaluation artifact")

with tabs[3]:
    xgboost = explain["xgboost"]
    rca = explain.get("rca_result")

    st.markdown('<div class="section-title">Complete Explainability Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">XGBoost classification, SHAP importance, trace evidence, and failure-pattern context for the frozen incident snapshot.</div>', unsafe_allow_html=True)

    e1, e2, e3, e4 = st.columns(4)
    with e1:
        st.metric("XGBoost result", xgboost["anomaly_type"].replace("_", " ").title(), f'{xgboost["confidence"] * 100:.1f}% confidence')
    with e2:
        st.metric("Feature vector", str(xgboost["feature_count"]), "scale-invariant features")
    with e3:
        severity = getattr(rca, "severity", "Normal" if not is_injected else "Critical").title()
        st.metric("Severity", severity)
    with e4:
        root_cause = getattr(rca, "root_cause_service", "None" if not is_injected else "notification-service")
        st.metric("Root cause", root_cause)

    if not is_injected:
        st.info("Explainability artifacts are generated after anomaly injection and snapshot freeze. The download buttons remain available for the generated reports.")

    dl1, dl2 = st.columns(2)
    with dl1:
        download_button("Download explainability PDF", artifacts["explanation_pdf"], "application/pdf")
    with dl2:
        if artifacts["shap_metrics"].exists():
            download_button("Download SHAP metrics JSON", artifacts["shap_metrics"], "application/json")
        else:
            st.button("Download SHAP metrics JSON unavailable", disabled=True, use_container_width=True)

    if is_injected and rca:
        c1, c2 = st.columns([1.05, 1])
        with c1:
            st.markdown("**Contributing factors**")
            factors_df = explain["contributing_factors_df"].copy()
            if not factors_df.empty:
                st.dataframe(factors_df, use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**Recommended next actions**")
            for idx, rec in enumerate(rca.recommendations[:5], 1):
                st.markdown(f"{idx}. {rec}")

    visual_tab1, visual_tab2, visual_tab3, visual_tab4 = st.tabs(["SHAP", "Trace timeline", "Failure patterns", "Report preview"])
    with visual_tab1:
        left, right = st.columns([1.2, 1])
        with left:
            image_if_exists(artifacts["explanation_shap"], "SHAP attribution figure")
        with right:
            top_shap_df = explain.get("shap_top_df", pd.DataFrame())
            if not top_shap_df.empty:
                st.dataframe(top_shap_df, use_container_width=True, hide_index=True)
            shap_metrics = explain.get("shap_metrics", {})
            if shap_metrics:
                st.markdown("**SHAP quality metrics**")
                st.write({"top1_match_rate": f'{shap_metrics.get("anomaly_only_top1_match_rate", 0):.2f}%', "top3_coverage": f'{shap_metrics.get("anomaly_only_top3_coverage_rate", 0):.2f}%', "mean_stability": round(shap_metrics.get("mean_shap_stability", 0), 4), "mean_time_ms": round(shap_metrics.get("mean_explanation_time", 0), 2)})
    with visual_tab2:
        if not explain["trace_timeline_df"].empty:
            st.plotly_chart(build_trace_timeline_chart(explain["trace_timeline_df"]), use_container_width=True)
            top, bottom = st.columns([1.1, 1])
            with top:
                st.markdown("**Trace event table**")
                st.dataframe(explain["trace_timeline_df"], use_container_width=True, hide_index=True)
            with bottom:
                if not explain["error_chain_df"].empty:
                    st.markdown("**Error propagation chain**")
                    st.dataframe(explain["error_chain_df"], use_container_width=True, hide_index=True)
                image_if_exists(artifacts["explanation_timeline"], "Generated timeline artifact")
        elif not is_injected:
            st.info("Trace timeline will appear after anomaly injection.")
        else:
            st.warning("Trace timeline data is not available for this incident.")
    with visual_tab3:
        top, bottom = st.columns([1.2, 1])
        with top:
            image_if_exists(artifacts["explanation_patterns"], "Failure-pattern matching output")
        with bottom:
            if not explain["log_patterns_df"].empty:
                st.markdown("**Log-derived patterns**")
                st.dataframe(explain["log_patterns_df"], use_container_width=True, hide_index=True)
            elif not is_injected:
                st.info("Failure patterns are populated during incident analysis.")
            if not explain["failure_patterns_df"].empty:
                st.markdown("**Matched failure archetypes**")
                st.dataframe(explain["failure_patterns_df"], use_container_width=True, hide_index=True)
    with visual_tab4:
        report_preview("Explainability report", explain.get("full_report_text", ""), artifacts["explanation_pdf"], [artifacts["explanation_shap"], artifacts["explanation_timeline"], artifacts["explanation_patterns"]])

with tabs[4]:
    best_scenario = counterfactual.get("best_scenario") or {}
    scenarios_df = counterfactual["scenarios_df"]
    proof_df = counterfactual["proof_df"]

    st.markdown('<div class="section-title">Counterfactuals With Proof</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">What-if analysis shows which targeted change would have prevented the anomaly and backs it up with scenario comparisons plus evaluation evidence.</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Original prediction", counterfactual["predicted_class"].replace("_", " ").title(), f'{counterfactual["confidence"] * 100:.1f}% confidence')
    with c2:
        change_label = f'{best_scenario.get("delta_percent", 0):+.1f}%' if best_scenario else "N/A"
        st.metric("Best change", change_label)
    with c3:
        st.metric("Best scenario score", f'{best_scenario.get("score", 0):.1f}/100' if best_scenario else "N/A")
    with c4:
        st.metric("Prevents anomaly", "Yes" if best_scenario.get("prevents_anomaly") else "No")

    if not is_injected:
        st.info("Counterfactual search activates after a non-normal prediction is produced.")

    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        download_button("Download counterfactual PDF", artifacts["counterfactual_pdf"], "application/pdf")
    with dl2:
        download_button("Download counterfactual CSV", artifacts["counterfactual_csv"], "text/csv")
    with dl3:
        if artifacts["counterfactual_confidence"].exists():
            download_button("Download confidence chart", artifacts["counterfactual_confidence"], "image/png")
        else:
            st.button("Download confidence chart unavailable", disabled=True, use_container_width=True)

    if not scenarios_df.empty:
        st.plotly_chart(build_counterfactual_chart(scenarios_df), use_container_width=True)

    upper, lower = st.columns([1.15, 1])
    with upper:
        st.markdown("**Scenario proof table**")
        if not scenarios_df.empty:
            st.dataframe(scenarios_df.rename(columns={"scenario_name": "Scenario", "current": "Current", "target": "Target", "delta_percent": "Delta %", "predicted_class": "Predicted class", "predicted_confidence_percent": "Confidence %", "prevents_anomaly": "Prevents anomaly", "score": "Score", "risk_level": "Risk"}), use_container_width=True, hide_index=True)
    with lower:
        st.markdown("**Best prevention strategy**")
        if best_scenario:
            st.markdown(f"""
                <div class="card">
                    <div class="card-title">Recommended intervention</div>
                    <div class="card-value">{best_scenario["feature_name"].replace("_", " ")}</div>
                    <div class="card-note">
                        Current -> Target: <strong>{best_scenario["current"]:.2f} -> {best_scenario["target"]:.2f}</strong><br>
                        New predicted state: <strong>{best_scenario["predicted_class"].upper()}</strong><br>
                        Confidence: <strong>{best_scenario["predicted_confidence"] * 100:.1f}%</strong><br>
                        Risk level: <strong>{best_scenario["risk_level"].upper()}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    vis1, vis2, vis3, vis4 = st.tabs(["Summary card", "Scenario comparison", "Required change", "Evaluation evidence"])
    with vis1:
        image_if_exists(artifacts["counterfactual_summary"], "Counterfactual summary card")
    with vis2:
        image_if_exists(artifacts["counterfactual_comparison"], "Visual comparison across what-if scenarios")
    with vis3:
        image_if_exists(artifacts["counterfactual_delta"], "Required change per candidate scenario")
    with vis4:
        report_preview("counterfactual report", "Evaluation evidence and detailed scenario outcomes are provided below.", artifacts["counterfactual_pdf"], [artifacts["counterfactual_comparison"], artifacts["counterfactual_delta"], artifacts["counterfactual_summary"]])
        if not proof_df.empty:
            st.dataframe(proof_df, use_container_width=True, hide_index=True)

with tabs[5]:
    recommended = recovery["recommended_action"]
    ranked_actions_df = recovery["ranked_actions_df"]

    st.markdown('<div class="section-title">Recovery Orchestration</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">Counterfactual outcomes are turned into concrete Kubernetes actions, scored on five dimensions, and ranked for operator execution.</div>', unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.metric("Recommended action", recommended.get("action_type", "N/A").replace("_", " ").title())
    with r2:
        st.metric("Action score", f'{recommended.get("score", 0):.1f}/100' if recommended else "N/A")
    with r3:
        st.metric("Risk level", str(recommended.get("risk_level", "N/A")).title())
    with r4:
        st.metric("Duration", f'{recommended.get("duration_seconds", 0)} sec' if recommended else "N/A")

    if not is_injected:
        st.info("Recovery orchestration is triggered after counterfactual analysis identifies viable remediation paths.")

    dl1, dl2 = st.columns(2)
    with dl1:
        download_button("Download recovery PDF", artifacts["recovery_pdf"], "application/pdf")
    with dl2:
        if not ranked_actions_df.empty:
            st.download_button("Download ranked actions CSV", ranked_actions_df.to_csv(index=False).encode("utf-8"), file_name="ranked_recovery_actions.csv", mime="text/csv", use_container_width=True)
        else:
            st.button("Download ranked actions CSV unavailable", disabled=True, use_container_width=True)

    main_col, side_col = st.columns([1.25, 0.95])
    with main_col:
        if not ranked_actions_df.empty:
            st.dataframe(ranked_actions_df.rename(columns={"action_type": "Action type", "description": "Description", "command": "Command", "target_service": "Target service", "expected_outcome": "Model note", "scenario_predicted_class": "Residual class", "scenario_predicted_confidence": "Residual confidence %", "scenario_prevents_anomaly": "Clears anomaly", "score": "Score", "risk_level": "Risk", "duration_seconds": "Duration s", "rollback": "Rollback", "effectiveness": "Effectiveness %", "safety": "Safety %", "speed": "Speed %", "cost": "Cost %", "simplicity": "Simplicity %"}), use_container_width=True, hide_index=True)
    with side_col:
        st.markdown("**Recommended command**")
        st.code(recommended.get("command", "No command available"), language="bash")
        execute_disabled = (not is_injected) or (not recommended)
        if st.button("Execute Action", use_container_width=True, disabled=execute_disabled, key="execute_recovery_action"):
            st.session_state.executed_recovery_action = {
                "action_type": recommended.get("action_type", "unknown"),
                "command": recommended.get("command", ""),
                "target_service": recommended.get("target_service", ""),
                "predicted_class": recommended.get("scenario_predicted_class"),
                "predicted_confidence": recommended.get("scenario_predicted_confidence"),
                "clears_anomaly": recommended.get("scenario_prevents_anomaly", False),
            }
        if st.session_state.executed_recovery_action and is_injected:
            executed = st.session_state.executed_recovery_action
            st.success(f"Executed demo action: {str(executed['action_type']).replace('_', ' ').title()} on {executed['target_service']}")
        residual_class = recommended.get("scenario_predicted_class")
        residual_confidence = recommended.get("scenario_predicted_confidence")
        clears_anomaly = recommended.get("scenario_prevents_anomaly")
        st.markdown("**Predicted result after this action**")
        if residual_class:
            st.write(f"Residual model state: {str(residual_class).replace('_', ' ').upper()} ({residual_confidence:.0f}% confidence)")
            if clears_anomaly:
                st.success("This action is predicted to clear the anomaly and return the system to a normal state.")
            else:
                st.warning("This action is a mitigation step, but the linked scenario still predicts a residual anomaly state. It is not a full guaranteed recovery on its own.")
        else:
            st.write(recommended.get("expected_outcome", "No outcome available"))
        st.markdown(f"""
            <div class="small-note">
                Rollback difficulty: <strong>{recommended.get("rollback", "N/A")}</strong><br>
                Target service: <strong>{recommended.get("target_service", "N/A")}</strong>
            </div>
            """, unsafe_allow_html=True)

    viz1, viz2, viz3, viz4 = st.tabs(["Ranking", "Score breakdown", "Summary card", "Report preview"])
    with viz1:
        image_if_exists(artifacts["recovery_ranking"], "Recovery action ranking from the orchestrator")
    with viz2:
        if not ranked_actions_df.empty:
            st.plotly_chart(build_recovery_breakdown_chart(ranked_actions_df), use_container_width=True)
        image_if_exists(artifacts["recovery_breakdown"], "Detailed breakdown for the selected action")
    with viz3:
        image_if_exists(artifacts["recovery_summary"], "Recovery orchestrator summary card")
    with viz4:
        report_preview("recovery report", "Detailed recovery ranking and score breakdown are available in the generated report artifact.", artifacts["recovery_pdf"], [artifacts["recovery_ranking"], artifacts["recovery_breakdown"], artifacts["recovery_summary"]])
