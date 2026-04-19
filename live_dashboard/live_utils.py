from __future__ import annotations

import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil
import requests


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
RUNTIME_DIR = APP_DIR / "runtime"
DEMOS_DIR = PROJECT_ROOT / "scripts" / "demos"
REPORTS_DIR = PROJECT_ROOT / "scripts" / "reports"
TESTS_DIR = PROJECT_ROOT / "scripts" / "tests"
RUNTIME_DIR.mkdir(exist_ok=True)


def is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def http_status(url: str, timeout: float = 1.0) -> dict[str, Any]:
    try:
        response = requests.get(url, timeout=timeout)
        return {"ok": True, "status_code": response.status_code, "text": response.text[:300]}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def service_status_snapshot() -> list[dict[str, Any]]:
    checks = [
        {"name": "Jaeger UI", "host": "127.0.0.1", "port": 16686, "url": "http://localhost:16686"},
        {"name": "Notification Service", "host": "127.0.0.1", "port": 8003, "url": "http://localhost:8003/health"},
        {"name": "Prometheus", "host": "127.0.0.1", "port": 9090, "url": "http://localhost:9090/-/healthy"},
    ]

    rows: list[dict[str, Any]] = []
    for check in checks:
        port_open = is_port_open(check["host"], check["port"])
        http_info = http_status(check["url"]) if port_open else {"ok": False, "error": "port closed"}
        rows.append(
            {
                "service": check["name"],
                "endpoint": check["url"],
                "port": check["port"],
                "port_open": port_open,
                "http_ok": http_info.get("ok", False),
                "details": http_info.get("status_code", http_info.get("error", "unknown")),
            }
        )
    return rows


def run_command(command: list[str], cwd: Path | None = None, timeout: int = 240) -> dict[str, Any]:
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd or PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "command": command,
            "started_at": started_at,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": f"Timed out after {timeout}s",
            "command": command,
            "started_at": started_at,
        }
    except Exception as exc:
        return {
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "command": command,
            "started_at": started_at,
        }


def launch_background_process(name: str, command: list[str], cwd: Path | None = None) -> dict[str, Any]:
    log_path = RUNTIME_DIR / f"{name}.log"
    log_handle = open(log_path, "a", encoding="utf-8")
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    process = subprocess.Popen(
        command,
        cwd=str(cwd or PROJECT_ROOT),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=creationflags,
    )
    return {
        "name": name,
        "pid": process.pid,
        "command": command,
        "cwd": str(cwd or PROJECT_ROOT),
        "log_path": str(log_path),
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def process_running(pid: int) -> bool:
    try:
        return psutil.pid_exists(pid) and psutil.Process(pid).is_running()
    except Exception:
        return False


def stop_process(pid: int) -> bool:
    try:
        process = psutil.Process(pid)
        process.terminate()
        process.wait(timeout=5)
        return True
    except Exception:
        try:
            process = psutil.Process(pid)
            process.kill()
            return True
        except Exception:
            return False


def read_log_tail(path: str | Path, max_chars: int = 12000) -> str:
    log_path = Path(path)
    if not log_path.exists():
        return ""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def artifact_paths() -> dict[str, Path]:
    base = PROJECT_ROOT / "results" / "explanation engine"
    return {
        "explanation_pdf": base / "explanation_engine_report.pdf",
        "counterfactual_pdf": base / "counterfactual_report.pdf",
        "recovery_pdf": base / "recovery_orchestrator_report.pdf",
        "timeline_png": base / "explanation_report_timeline.png",
        "shap_png": base / "explanation_report_shap.png",
        "recovery_png": base / "recovery_orchestrator_ranking.png",
    }


def read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def command_presets() -> dict[str, dict[str, Any]]:
    python_exe = sys.executable
    return {
        "start_jaeger": {
            "label": "Start Jaeger",
            "kind": "run",
            "command": ["docker-compose", "-f", "docker-compose-jaeger.yml", "up", "-d"],
            "cwd": PROJECT_ROOT,
            "timeout": 120,
        },
        "launch_notification": {
            "label": "Launch Notification Service",
            "kind": "background",
            "command": [python_exe, "notification_service.py"],
            "cwd": PROJECT_ROOT / "services" / "notification-service",
            "process_name": "notification_service",
        },
        "launch_detector": {
            "label": "Launch Online Detector",
            "kind": "background",
            "command": [python_exe, "-m", "online_detector.main"],
            "cwd": PROJECT_ROOT,
            "process_name": "online_detector",
        },
        "verify_integration": {
            "label": "Verify Integration",
            "kind": "run",
            "command": [python_exe, str(TESTS_DIR / "test_integration_verification.py")],
            "cwd": PROJECT_ROOT,
            "timeout": 180,
        },
        "trigger_all": {
            "label": "Trigger All Channels",
            "kind": "run",
            "command": [python_exe, "anomaly-trigger/trigger_anomaly.py", "--all"],
            "cwd": PROJECT_ROOT,
            "timeout": 240,
        },
        "stop_anomaly": {
            "label": "Stop Anomaly",
            "kind": "run",
            "command": [python_exe, "anomaly-trigger/trigger_anomaly.py", "--stop"],
            "cwd": PROJECT_ROOT,
            "timeout": 120,
        },
        "run_rca": {
            "label": "Run Integrated RCA",
            "kind": "run",
            "command": [python_exe, str(DEMOS_DIR / "demo_integrated_rca.py")],
            "cwd": PROJECT_ROOT,
            "timeout": 240,
        },
        "run_full_pipeline": {
            "label": "Run Full Pipeline",
            "kind": "run",
            "command": [python_exe, str(DEMOS_DIR / "demo_full_pipeline.py")],
            "cwd": PROJECT_ROOT,
            "timeout": 240,
        },
        "run_recovery": {
            "label": "Run Recovery Pipeline",
            "kind": "run",
            "command": [python_exe, str(DEMOS_DIR / "demo_recovery_pipeline.py")],
            "cwd": PROJECT_ROOT,
            "timeout": 240,
        },
    }
