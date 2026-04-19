from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from live_utils import (
    artifact_paths,
    command_presets,
    launch_background_process,
    process_running,
    read_bytes,
    read_log_tail,
    run_command,
    service_status_snapshot,
    stop_process,
)


st.set_page_config(
    page_title="Live Operations Console",
    page_icon="L",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(8, 145, 178, 0.08), transparent 28%),
                linear-gradient(180deg, #f6fbff 0%, #eef5f9 44%, #f8fafc 100%);
        }
        .block-container {
            max-width: 1480px;
            padding-top: 1.2rem;
            padding-bottom: 3rem;
        }
        .hero {
            padding: 1.3rem 1.5rem;
            border-radius: 22px;
            background: linear-gradient(135deg, #0f172a 0%, #0b3b5a 45%, #0f766e 100%);
            color: white;
            box-shadow: 0 24px 60px rgba(15, 23, 42, 0.2);
            margin-bottom: 1rem;
        }
        .hero-kicker {
            color: #93c5fd;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.76rem;
            font-weight: 700;
        }
        .hero-title {
            font-size: 2rem;
            font-weight: 800;
            margin-top: 0.25rem;
        }
        .hero-copy {
            color: #dbeafe;
            max-width: 72rem;
            margin-top: 0.45rem;
        }
        .panel {
            background: rgba(255,255,255,0.8);
            border: 1px solid rgba(148,163,184,0.2);
            border-radius: 18px;
            padding: 1rem;
            box-shadow: 0 14px 30px rgba(15,23,42,0.06);
        }
        .section-title {
            font-size: 1.25rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.2rem;
        }
        .section-copy {
            color: #475569;
            margin-bottom: 0.8rem;
        }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.82);
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 16px;
            padding: 0.9rem 1rem;
            box-shadow: 0 14px 30px rgba(15,23,42,0.05);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    if "live_processes" not in st.session_state:
        st.session_state.live_processes = {}
    if "live_command_runs" not in st.session_state:
        st.session_state.live_command_runs = []


def record_run(name: str, result: dict) -> None:
    st.session_state.live_command_runs.insert(0, {"name": name, **result})
    st.session_state.live_command_runs = st.session_state.live_command_runs[:12]


def launch_preset(preset_key: str) -> None:
    preset = command_presets()[preset_key]
    if preset["kind"] == "background":
        process_info = launch_background_process(
            preset["process_name"],
            preset["command"],
            cwd=preset["cwd"],
        )
        st.session_state.live_processes[preset["process_name"]] = process_info
    else:
        result = run_command(preset["command"], cwd=preset["cwd"], timeout=preset.get("timeout", 240))
        record_run(preset_key, result)


def process_rows() -> list[dict]:
    rows = []
    for name, info in st.session_state.live_processes.items():
        running = process_running(info["pid"])
        rows.append(
            {
                "process": name,
                "pid": info["pid"],
                "running": running,
                "started_at": info["started_at"],
                "command": " ".join(info["command"]),
                "log_path": info["log_path"],
            }
        )
    return rows


inject_styles()
init_state()
artifacts = artifact_paths()
status_rows = service_status_snapshot()
status_df = pd.DataFrame(status_rows)
process_df = pd.DataFrame(process_rows())

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Live Operations</div>
        <div class="hero-title">Live Operations Console</div>
        <div class="hero-copy">
            Local control center for service health, anomaly triggers, detector launch, integration checks,
            and real execution of the existing project demo scripts.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Ports reachable", int(status_df["port_open"].sum()), f"/{len(status_df)} monitored")
with m2:
    st.metric("HTTP healthy", int(status_df["http_ok"].sum()), f"/{len(status_df)} responding")
with m3:
    st.metric("Background processes", len(process_df), "launched from this console")
with m4:
    st.metric("Artifacts ready", sum(1 for p in artifacts.values() if p.exists()), f"/{len(artifacts)} found")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Service Status",
    "2. Launch & Control",
    "3. Anomaly Trigger",
    "4. Live Script Runs",
    "5. Artifacts & Logs",
])

with tab1:
    st.markdown('<div class="section-title">Service Status</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">Current reachability of the local services and observability endpoints used by the live workflow.</div>', unsafe_allow_html=True)
    st.dataframe(
        status_df.rename(columns={
            "service": "Service",
            "endpoint": "Endpoint",
            "port": "Port",
            "port_open": "Port open",
            "http_ok": "HTTP ok",
            "details": "Details",
        }),
        use_container_width=True,
        hide_index=True,
    )
    if st.button("Refresh status", key="refresh_status", use_container_width=False):
        st.rerun()

with tab2:
    st.markdown('<div class="section-title">Launch & Control</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">Start the local services and detector processes from the console. Background launches write to log files under `live_dashboard/runtime`.</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("Start Jaeger", use_container_width=True):
            launch_preset("start_jaeger")
            st.rerun()
    with c2:
        if st.button("Launch Notification Service", use_container_width=True):
            launch_preset("launch_notification")
            st.rerun()
    with c3:
        if st.button("Launch Online Detector", use_container_width=True):
            launch_preset("launch_detector")
            st.rerun()
    with c4:
        if st.button("Verify Integration", use_container_width=True):
            launch_preset("verify_integration")
            st.rerun()

    if not process_df.empty:
        st.markdown("**Managed processes**")
        for row in process_df.to_dict("records"):
            cols = st.columns([1.4, 1.2, 1.2, 0.8])
            with cols[0]:
                st.write(f"{row['process']} (PID {row['pid']})")
            with cols[1]:
                st.write("Running" if row["running"] else "Stopped")
            with cols[2]:
                st.write(row["started_at"])
            with cols[3]:
                if st.button("Stop", key=f"stop_{row['pid']}", use_container_width=True):
                    stop_process(int(row["pid"]))
                    st.rerun()

with tab3:
    st.markdown('<div class="section-title">Anomaly Trigger</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">Trigger and stop the actual anomaly flow against the local notification service endpoints.</div>', unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    with a1:
        if st.button("Trigger All Channels", use_container_width=True, type="primary"):
            launch_preset("trigger_all")
            st.rerun()
    with a2:
        if st.button("Stop Active Anomaly", use_container_width=True):
            launch_preset("stop_anomaly")
            st.rerun()
    with a3:
        st.code("python anomaly-trigger/trigger_anomaly.py --all", language="bash")

with tab4:
    st.markdown('<div class="section-title">Live Script Runs</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">Execute the real project demos and inspection scripts, then inspect their captured output here.</div>', unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    with r1:
        if st.button("Run Integrated RCA", use_container_width=True):
            launch_preset("run_rca")
            st.rerun()
    with r2:
        if st.button("Run Full Pipeline", use_container_width=True):
            launch_preset("run_full_pipeline")
            st.rerun()
    with r3:
        if st.button("Run Recovery Pipeline", use_container_width=True):
            launch_preset("run_recovery")
            st.rerun()

    runs = st.session_state.live_command_runs
    if runs:
        for idx, run in enumerate(runs):
            with st.expander(f"{run['name']} | rc={run.get('returncode')} | {run.get('started_at')}", expanded=(idx == 0)):
                st.code(" ".join(run.get("command", [])), language="bash")
                output = (run.get("stdout") or "") + ("\n" + run.get("stderr") if run.get("stderr") else "")
                st.code(output or "No output captured.", language="text")
    else:
        st.info("No script runs recorded yet.")

with tab5:
    st.markdown('<div class="section-title">Artifacts & Logs</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">Open the generated reports and inspect the log tails for processes started from this console.</div>', unsafe_allow_html=True)
    left, right = st.columns([1, 1.2])
    with left:
        st.markdown("**Generated artifacts**")
        for label, path in artifacts.items():
            cols = st.columns([1.7, 0.9])
            with cols[0]:
                st.write(f"{label}: {path.name}")
            with cols[1]:
                if path.exists():
                    st.download_button(f"Download {label}", data=read_bytes(path), file_name=path.name, use_container_width=True, key=f"dl_{label}")
                else:
                    st.button("Missing", disabled=True, key=f"missing_{label}", use_container_width=True)
    with right:
        st.markdown("**Process log tail**")
        if not process_df.empty:
            process_names = process_df["process"].tolist()
            selected = st.selectbox("Select process log", process_names)
            selected_info = st.session_state.live_processes[selected]
            st.code(read_log_tail(selected_info["log_path"]) or "Log file is empty.", language="text")
        else:
            st.info("No background processes launched yet, so there are no live logs to show.")
