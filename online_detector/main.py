# main.py

import time
from datetime import datetime

from config import (
    SCRAPE_INTERVAL_SECONDS,
    CPU_WEIGHT,
    MEMORY_WEIGHT,
    THREAD_WEIGHT,
    NORMAL_THRESHOLD,
    ANOMALY_THRESHOLD
    ,CPU_SOFT_LIMIT
    ,MEMORY_SOFT_LIMIT_MB
    ,THREAD_SOFT_LIMIT
)

from metrics_reader import PrometheusClient
from detector import ResourceSaturationDetector


def main():
    prom = PrometheusClient()
    detector = ResourceSaturationDetector()

    print("🚀 EWMA Online Detector started (Resource Saturation)")
    print("📡 Sampling every", SCRAPE_INTERVAL_SECONDS, "seconds\n")

    while True:
        try:
            cpu = prom.get_cpu()
            memory = prom.get_memory()
            threads = prom.get_threads()

            stresses = detector.update(cpu, memory, threads)

            ewma_stress_score = (
                CPU_WEIGHT * stresses["cpu"] +
                MEMORY_WEIGHT * stresses["memory"] +
                THREAD_WEIGHT * stresses["threads"]
            )

            cpu_pressure = min(1.0, cpu / CPU_SOFT_LIMIT) if CPU_SOFT_LIMIT > 0 else 0.0
            memory_pressure = min(1.0, memory / MEMORY_SOFT_LIMIT_MB) if MEMORY_SOFT_LIMIT_MB > 0 else 0.0
            thread_pressure = min(1.0, threads / THREAD_SOFT_LIMIT) if THREAD_SOFT_LIMIT > 0 else 0.0

            resource_pressure_weighted = (
                CPU_WEIGHT * cpu_pressure +
                MEMORY_WEIGHT * memory_pressure +
                THREAD_WEIGHT * thread_pressure
            )

            # A single saturated resource (e.g., memory at limit) is operationally dangerous
            # even if others are normal. Using a peak pressure prevents a weighted average
            # from hiding that saturation.
            resource_pressure_peak = max(cpu_pressure, memory_pressure, thread_pressure)
            resource_pressure = max(resource_pressure_weighted, resource_pressure_peak)

            # Improved scoring: Use additive combination with ceiling instead of multiplicative
            # This prevents low pressure from dampening high EWMA stress signals
            # Formula: Take the maximum of:
            # 1) EWMA stress (detects abnormal behavior) - PRIMARY SIGNAL
            # 2) Resource pressure (detects saturation) - SECONDARY SIGNAL
            # 3) Combined weighted score (balanced view)
            
            # EWMA-dominant: pressure should not trigger alone at baseline
            combined_weighted = 0.75 * ewma_stress_score + 0.25 * resource_pressure
            
            # Use maximum to ensure either signal can trigger detection
            channel_risk_score = max(
                ewma_stress_score * 0.85,  # EWMA alone (primary signal)
                resource_pressure * 0.35,  # Pressure alone (minimal weight - only for extreme saturation)
                combined_weighted          # Weighted combination
            )
            
            # Boost score if pressure is extremely high (saturation scenario)
            if resource_pressure >= 0.92:
                channel_risk_score = max(channel_risk_score, 0.70)
            
            channel_risk_score = max(0.0, min(1.0, channel_risk_score))

            # FSM must only see the combined risk score.
            channel_state = detector.update_channel_state(channel_risk_score)

            output = {
                "timestamp": datetime.utcnow().isoformat(),
                "channel": "resource_saturation",
                # Preserve the existing field name, but make it the final score fed to the FSM.
                "stress_score": round(channel_risk_score, 3),
                "ewma_stress_score": round(ewma_stress_score, 3),
                "resource_pressure": round(resource_pressure, 3),
                "resource_pressure_weighted": round(resource_pressure_weighted, 3),
                "resource_pressure_peak": round(resource_pressure_peak, 3),
                "channel_risk_score": round(channel_risk_score, 3),
                "state": channel_state["state"],
                "state_duration": channel_state["state_duration"],
                "transition_reason": channel_state["transition_reason"],
                "raw": {
                    "cpu_percent": round(cpu, 3),
                    "memory_mb": round(memory, 3),
                    "threads": round(threads, 3)
                },
                "pressures": {
                    "cpu": round(cpu_pressure, 3),
                    "memory": round(memory_pressure, 3),
                    "threads": round(thread_pressure, 3)
                },
                "signals": {
                    "cpu": round(stresses["cpu"], 3),
                    "memory": round(stresses["memory"], 3),
                    "threads": round(stresses["threads"], 3)
                }
            }

            print(output)

        except Exception as e:
            print("⚠️ detector error:", str(e))

        time.sleep(SCRAPE_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
