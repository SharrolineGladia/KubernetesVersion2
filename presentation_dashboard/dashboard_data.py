from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
EXPLANATION_DIR = RESULTS_DIR / "explanation engine"
EWMA_DIR = RESULTS_DIR / "ewma"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ML_SCRIPTS = PROJECT_ROOT / "ml_detector" / "scripts"
RECOVERY_DIR = PROJECT_ROOT / "recovery-orchestrator"
DEMOS_DIR = PROJECT_ROOT / "scripts" / "demos"
REPORTS_DIR = PROJECT_ROOT / "scripts" / "reports"

for path in (ML_SCRIPTS, RECOVERY_DIR, DEMOS_DIR, REPORTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def artifact_paths() -> dict[str, Path]:
    return {
        "explanation_pdf": EXPLANATION_DIR / "explanation_engine_report.pdf",
        "explanation_shap": EXPLANATION_DIR / "explanation_report_shap.png",
        "explanation_timeline": EXPLANATION_DIR / "explanation_report_timeline.png",
        "explanation_patterns": EXPLANATION_DIR / "explanation_report_patterns.png",
        "counterfactual_pdf": EXPLANATION_DIR / "counterfactual_report.pdf",
        "counterfactual_summary": EXPLANATION_DIR / "counterfactual_summary_card.png",
        "counterfactual_comparison": EXPLANATION_DIR / "counterfactual_visual_comparison.png",
        "counterfactual_delta": EXPLANATION_DIR / "counterfactual_required_change.png",
        "counterfactual_confidence": EXPLANATION_DIR / "counterfactual_confidence_distribution.png",
        "counterfactual_csv": EXPLANATION_DIR / "counterfactual_details_v2.csv",
        "recovery_pdf": EXPLANATION_DIR / "recovery_orchestrator_report.pdf",
        "recovery_summary": EXPLANATION_DIR / "recovery_orchestrator_summary_card.png",
        "recovery_ranking": EXPLANATION_DIR / "recovery_orchestrator_ranking.png",
        "recovery_breakdown": EXPLANATION_DIR / "recovery_orchestrator_breakdown.png",
        "shap_metrics": EXPLANATION_DIR / "shap_evaluation_metrics_v2.json",
        "shap_eval_csv": EXPLANATION_DIR / "shap_evaluation_detailed_v2.csv",
        "ewma_spike_plot": EWMA_DIR / "plot_spike.png",
        "ewma_sustained_plot": EWMA_DIR / "plot_sustained_high.png",
    }


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def stressed_service_metrics() -> dict[str, dict[str, float]]:
    return {
        "web-api": {
            "cpu_percent": 25.0,
            "memory_percent": 45.0,
            "error_rate": 0.005,
            "request_rate": 80.0,
            "response_time_p95": 120.0,
            "thread_count": 25,
            "queue_depth": 5.0,
            "requests_per_second": 16.0,
        },
        "processor": {
            "cpu_percent": 35.0,
            "memory_percent": 55.0,
            "error_rate": 0.008,
            "request_rate": 60.0,
            "response_time_p95": 180.0,
            "thread_count": 30,
            "queue_depth": 8.0,
            "requests_per_second": 12.0,
        },
        "cache": {
            "cpu_percent": 15.0,
            "memory_percent": 38.0,
            "error_rate": 0.002,
            "request_rate": 120.0,
            "response_time_p95": 50.0,
            "thread_count": 10,
            "queue_depth": 2.0,
            "requests_per_second": 24.0,
        },
        "notification": {
            "cpu_percent": 92.0,
            "memory_percent": 78.0,
            "error_rate": 0.018,
            "request_rate": 150.0,
            "response_time_p95": 450.0,
            "thread_count": 85,
            "queue_depth": 45.0,
            "requests_per_second": 30.0,
        },
        "notification-worker": {
            "cpu_percent": 88.0,
            "memory_percent": 82.0,
            "error_rate": 0.022,
            "request_rate": 140.0,
            "response_time_p95": 520.0,
            "thread_count": 90,
            "queue_depth": 52.0,
            "requests_per_second": 28.0,
        },
    }


def normal_service_metrics() -> dict[str, dict[str, float]]:
    return {
        "web-api": {
            "cpu_percent": 24.0,
            "memory_percent": 42.0,
            "error_rate": 0.003,
            "request_rate": 78.0,
            "response_time_p95": 110.0,
            "thread_count": 22,
            "queue_depth": 3.0,
            "requests_per_second": 15.5,
        },
        "processor": {
            "cpu_percent": 29.0,
            "memory_percent": 48.0,
            "error_rate": 0.004,
            "request_rate": 58.0,
            "response_time_p95": 150.0,
            "thread_count": 26,
            "queue_depth": 4.0,
            "requests_per_second": 11.5,
        },
        "cache": {
            "cpu_percent": 12.0,
            "memory_percent": 35.0,
            "error_rate": 0.001,
            "request_rate": 118.0,
            "response_time_p95": 45.0,
            "thread_count": 9,
            "queue_depth": 1.0,
            "requests_per_second": 23.5,
        },
        "notification": {
            "cpu_percent": 27.0,
            "memory_percent": 44.0,
            "error_rate": 0.004,
            "request_rate": 84.0,
            "response_time_p95": 130.0,
            "thread_count": 24,
            "queue_depth": 4.0,
            "requests_per_second": 16.8,
        },
        "notification-worker": {
            "cpu_percent": 23.0,
            "memory_percent": 40.0,
            "error_rate": 0.003,
            "request_rate": 74.0,
            "response_time_p95": 118.0,
            "thread_count": 22,
            "queue_depth": 3.0,
            "requests_per_second": 14.8,
        },
    }


def services_dataframe(injected: bool) -> pd.DataFrame:
    source = stressed_service_metrics() if injected else normal_service_metrics()
    rows: list[dict[str, Any]] = []
    for service_name, metrics in source.items():
        rows.append(
            {
                "service": service_name,
                "status": "Stressed" if metrics["cpu_percent"] > 80 else "Healthy",
                "cpu_percent": metrics["cpu_percent"],
                "memory_percent": metrics["memory_percent"],
                "error_rate_percent": metrics["error_rate"] * 100,
                "p95_latency_ms": metrics["response_time_p95"],
                "queue_depth": metrics["queue_depth"],
                "requests_per_second": metrics["requests_per_second"],
            }
        )
    return pd.DataFrame(rows)


def injection_steps(injected: bool) -> pd.DataFrame:
    if not injected:
        return pd.DataFrame(
            [
                {
                    "time": "14:22:55",
                    "operation": "Steady-state monitoring",
                    "channel": "online detector",
                    "impact": "All services remain within normal utilization bands",
                    "evidence": "stress_score < 0.30",
                }
            ]
        )
    return pd.DataFrame(
        [
            {
                "time": "14:23:10",
                "operation": "Critical mode enabled",
                "channel": "resource_saturation",
                "impact": "CPU spike, memory growth, and thread explosion initiated",
                "evidence": "POST /simulate-critical",
            },
            {
                "time": "14:23:14",
                "operation": "Queue flood started",
                "channel": "performance + backpressure",
                "impact": "1000 notification requests injected to build backlog",
                "evidence": "POST /send-notification x1000",
            },
            {
                "time": "14:23:28",
                "operation": "Pipeline response",
                "channel": "online detector",
                "impact": "EWMA stress climbs above threshold and snapshot freezes",
                "evidence": "stress_score > 0.60",
            },
        ]
    )


def ewma_series(injected: bool) -> pd.DataFrame:
    if not injected:
        return pd.DataFrame(
            [
                {"timestamp": "14:22:35", "stress_score": 0.14, "resource_pressure": 0.18, "state": "normal", "cpu_percent": 21, "memory_mb": 205, "threads": 24},
                {"timestamp": "14:22:40", "stress_score": 0.16, "resource_pressure": 0.20, "state": "normal", "cpu_percent": 23, "memory_mb": 210, "threads": 25},
                {"timestamp": "14:22:45", "stress_score": 0.18, "resource_pressure": 0.21, "state": "normal", "cpu_percent": 24, "memory_mb": 212, "threads": 26},
                {"timestamp": "14:22:50", "stress_score": 0.17, "resource_pressure": 0.22, "state": "normal", "cpu_percent": 22, "memory_mb": 214, "threads": 25},
                {"timestamp": "14:22:55", "stress_score": 0.19, "resource_pressure": 0.23, "state": "normal", "cpu_percent": 24, "memory_mb": 215, "threads": 26},
                {"timestamp": "14:23:00", "stress_score": 0.20, "resource_pressure": 0.24, "state": "normal", "cpu_percent": 25, "memory_mb": 218, "threads": 27},
            ]
        )
    return pd.DataFrame(
        [
            {"timestamp": "14:23:00", "stress_score": 0.18, "resource_pressure": 0.21, "state": "normal", "cpu_percent": 24, "memory_mb": 210, "threads": 28},
            {"timestamp": "14:23:05", "stress_score": 0.27, "resource_pressure": 0.31, "state": "normal", "cpu_percent": 31, "memory_mb": 236, "threads": 34},
            {"timestamp": "14:23:10", "stress_score": 0.41, "resource_pressure": 0.47, "state": "stressed", "cpu_percent": 54, "memory_mb": 288, "threads": 48},
            {"timestamp": "14:23:15", "stress_score": 0.58, "resource_pressure": 0.63, "state": "stressed", "cpu_percent": 69, "memory_mb": 332, "threads": 58},
            {"timestamp": "14:23:20", "stress_score": 0.67, "resource_pressure": 0.73, "state": "critical", "cpu_percent": 83, "memory_mb": 380, "threads": 72},
            {"timestamp": "14:23:25", "stress_score": 0.75, "resource_pressure": 0.81, "state": "critical", "cpu_percent": 91, "memory_mb": 421, "threads": 88},
        ]
    )


def ewma_snapshot(injected: bool) -> dict[str, Any]:
    if not injected:
        return {
            "trigger_time": "2026-03-25T14:23:00Z",
            "channel": "resource_saturation",
            "state": "normal",
            "stress_score": 0.20,
            "threshold": 0.60,
            "reason": "Detector remains below escalation thresholds",
            "peak_stress_memory": 0.24,
            "resource_pressure": 0.24,
            "service_count": 5,
            "captured_for": "Continuous monitoring only",
        }
    return {
        "trigger_time": "2026-03-25T14:23:25Z",
        "channel": "resource_saturation",
        "state": "critical",
        "stress_score": 0.75,
        "threshold": 0.60,
        "reason": "CPU and memory pressure sustained for 3 EWMA windows",
        "peak_stress_memory": 0.75,
        "resource_pressure": 0.81,
        "service_count": 5,
        "captured_for": "XGBoost classification and explainability pipeline",
    }


def normal_detection_summary() -> dict[str, Any]:
    services = normal_service_metrics()
    return {
        "anomaly_type": "normal",
        "confidence": 0.972,
        "active_services": list(services.keys()),
        "service_count": len(services),
        "feature_count": 27,
        "features": {},
    }


def stressed_detection_summary() -> dict[str, Any]:
    from generate_counterfactual_report_pdf import get_demo_explanation

    predicted_class, confidence, _, _ = get_demo_explanation()
    services = stressed_service_metrics()
    return {
        "anomaly_type": predicted_class,
        "confidence": float(confidence),
        "active_services": list(services.keys()),
        "service_count": len(services),
        "feature_count": 27,
        "features": {},
    }


def explainability_bundle(injected: bool) -> dict[str, Any]:
    bundle: dict[str, Any] = {"artifacts": artifact_paths()}
    bundle["xgboost"] = stressed_detection_summary() if injected else normal_detection_summary()

    shap_metrics_path = artifact_paths()["shap_metrics"]
    if shap_metrics_path.exists():
        bundle["shap_metrics"] = read_json(shap_metrics_path)
    shap_eval_path = artifact_paths()["shap_eval_csv"]
    if shap_eval_path.exists():
        bundle["shap_evaluation"] = pd.read_csv(shap_eval_path)

    if not injected:
        bundle["rca_result"] = None
        bundle["full_report_text"] = "System is healthy. Explainability report is generated when an anomaly snapshot is frozen."
        bundle["contributing_factors_df"] = pd.DataFrame()
        bundle["trace_timeline_df"] = pd.DataFrame()
        bundle["error_chain_df"] = pd.DataFrame()
        bundle["log_patterns_df"] = pd.DataFrame()
        bundle["critical_errors_df"] = pd.DataFrame()
        bundle["failure_patterns_df"] = pd.DataFrame()
        bundle["shap_top_df"] = pd.DataFrame()
        return bundle

    from demo_integrated_rca import build_demo_log_context, build_demo_trace_context
    from generate_explanation_report_pdf import generate_report_data

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            rca_result, shap_importance, trace_context, log_context, full_report = generate_report_data()
        bundle["rca_result"] = rca_result
        bundle["full_report_text"] = full_report
        bundle["contributing_factors_df"] = pd.DataFrame(rca_result.contributing_factors)
        bundle["trace_timeline_df"] = pd.DataFrame(trace_context["causal_timeline"])
        bundle["error_chain_df"] = pd.DataFrame(trace_context["error_chain"])
        bundle["log_patterns_df"] = pd.DataFrame(log_context["error_patterns"])
        bundle["critical_errors_df"] = pd.DataFrame(log_context["critical_errors"])
        bundle["failure_patterns_df"] = pd.DataFrame(rca_result.failure_pattern_matches or [])
        bundle["shap_top_df"] = pd.DataFrame(
            [{"feature": feature, "importance": value} for feature, value in list((shap_importance or {}).items())[:8]]
        )
    except Exception:
        trace_context = build_demo_trace_context()
        log_context = build_demo_log_context()
        bundle["rca_result"] = None
        bundle["full_report_text"] = "Explainability report available through downloadable artifact."
        bundle["contributing_factors_df"] = pd.DataFrame()
        bundle["trace_timeline_df"] = pd.DataFrame(trace_context["causal_timeline"])
        bundle["error_chain_df"] = pd.DataFrame(trace_context["error_chain"])
        bundle["log_patterns_df"] = pd.DataFrame(log_context["error_patterns"])
        bundle["critical_errors_df"] = pd.DataFrame(log_context["critical_errors"])
        bundle["failure_patterns_df"] = pd.DataFrame()
        bundle["shap_top_df"] = pd.DataFrame()

    return bundle


def counterfactual_bundle(injected: bool) -> dict[str, Any]:
    if not injected:
        return {
            "predicted_class": "normal",
            "confidence": 0.972,
            "explanation": None,
            "best_scenario": None,
            "scenarios_df": pd.DataFrame(),
            "proof_df": pd.DataFrame(),
            "artifacts": artifact_paths(),
        }

    from generate_counterfactual_report_pdf import get_demo_explanation

    predicted_class, confidence, explanation, visual_scenarios = get_demo_explanation()
    scenarios_df = pd.DataFrame(visual_scenarios)
    if not scenarios_df.empty:
        scenarios_df["predicted_confidence_percent"] = scenarios_df["predicted_confidence"] * 100

    best_scenario: dict[str, Any] | None = None
    if explanation and explanation.scenario_comparisons and explanation.best_scenario_idx is not None:
        best = explanation.scenario_comparisons[explanation.best_scenario_idx]
        feature_name = list(best.feature_changes.keys())[0]
        change_info = best.feature_changes[feature_name]
        best_scenario = {
            "scenario_name": best.scenario_name,
            "feature_name": feature_name,
            "current": change_info["current"],
            "target": change_info["target"],
            "delta_percent": change_info["delta_percent"],
            "predicted_class": best.predicted_class,
            "predicted_confidence": best.predicted_confidence,
            "prevents_anomaly": best.prevents_anomaly,
            "score": best.score,
            "risk_level": best.risk_level,
        }

    counterfactual_csv = artifact_paths()["counterfactual_csv"]
    proof_df = pd.read_csv(counterfactual_csv) if counterfactual_csv.exists() else pd.DataFrame()

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "explanation": explanation,
        "best_scenario": best_scenario,
        "scenarios_df": scenarios_df,
        "proof_df": proof_df,
        "artifacts": artifact_paths(),
    }


def recovery_bundle(injected: bool) -> dict[str, Any]:
    if not injected:
        return {
            "predicted_class": "normal",
            "confidence": 0.972,
            "explanation": None,
            "visual_scenarios": [],
            "actions": [],
            "scored_actions": [],
            "ranked_actions_df": pd.DataFrame(),
            "recommended_action": {},
            "artifacts": artifact_paths(),
        }

    from generate_recovery_orchestrator_report_pdf import get_recovery_data

    predicted_class, confidence, explanation, visual_scenarios, actions, scored_actions = get_recovery_data()

    rows: list[dict[str, Any]] = []
    for scored in scored_actions:
        rows.append(
            {
                "action_type": scored.action.action_type.value,
                "description": scored.action.action_description,
                "command": scored.action.action_command,
                "target_service": scored.action.target_service,
                "expected_outcome": scored.action.expected_outcome,
                "scenario_predicted_class": getattr(scored.action.from_scenario, "predicted_class", None),
                "scenario_predicted_confidence": round(float(getattr(scored.action.from_scenario, "predicted_confidence", 0.0)) * 100, 1) if getattr(scored.action, "from_scenario", None) else None,
                "scenario_prevents_anomaly": getattr(scored.action.from_scenario, "prevents_anomaly", False) if getattr(scored.action, "from_scenario", None) else False,
                "score": round(scored.total_score, 1),
                "risk_level": scored.risk_level,
                "duration_seconds": scored.estimated_duration_seconds,
                "rollback": scored.rollback_difficulty,
                "effectiveness": round(scored.effectiveness * 100, 1),
                "safety": round(scored.safety * 100, 1),
                "speed": round(scored.speed * 100, 1),
                "cost": round(scored.cost * 100, 1),
                "simplicity": round(scored.simplicity * 100, 1),
            }
        )

    ranked_actions_df = pd.DataFrame(rows)
    if not ranked_actions_df.empty:
        ranked_actions_df = ranked_actions_df.sort_values(
            by=["scenario_prevents_anomaly", "score"],
            ascending=[False, False],
            kind="stable",
        ).reset_index(drop=True)
        recommended_action = ranked_actions_df.iloc[0].to_dict()
    else:
        recommended_action = {}

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "explanation": explanation,
        "visual_scenarios": visual_scenarios,
        "actions": actions,
        "scored_actions": scored_actions,
        "ranked_actions_df": ranked_actions_df,
        "recommended_action": recommended_action,
        "artifacts": artifact_paths(),
    }
