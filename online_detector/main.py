# main.py

import time
from datetime import datetime

from .config import (
    SCRAPE_INTERVAL_SECONDS,
    CPU_WEIGHT,
    MEMORY_WEIGHT,
    THREAD_WEIGHT,
    NORMAL_THRESHOLD,
    ANOMALY_THRESHOLD,
    CPU_SOFT_LIMIT,
    MEMORY_SOFT_LIMIT_MB,
    THREAD_SOFT_LIMIT
)

from .metrics_reader import PrometheusClient
from .detector import (
    ResourceSaturationDetector,
    PerformanceDegradationChannel,
    BackpressureOverloadChannel
)


def main():
    prom = PrometheusClient()
    
    # Initialize all three channels
    resource_saturation = ResourceSaturationDetector()
    performance_degradation = PerformanceDegradationChannel()
    backpressure_overload = BackpressureOverloadChannel()

    # Independent tracking for each channel's next poll time
    next_poll = {
        "resource_saturation": time.time(),
        "performance_degradation": time.time(),
        "backpressure_overload": time.time()
    }

    print("🚀 Multi-Channel EWMA Online Detector started")
    print(f"📡 Resource Saturation: every {SCRAPE_INTERVAL_SECONDS}s")
    print(f"📡 Performance Degradation: every {performance_degradation.scrape_interval}s")
    print(f"📡 Backpressure Overload: every {backpressure_overload.scrape_interval}s\n")

    while True:
        try:
            now = time.time()
            timestamp = datetime.utcnow()

            # Poll Resource Saturation Channel
            if now >= next_poll["resource_saturation"]:
                cpu = prom.get_cpu()
                memory = prom.get_memory()
                threads = prom.get_threads()

                stresses = resource_saturation.update(cpu, memory, threads, timestamp)

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

                resource_pressure_peak = max(cpu_pressure, memory_pressure, thread_pressure)
                resource_pressure = max(resource_pressure_weighted, resource_pressure_peak)

                combined_weighted = 0.75 * ewma_stress_score + 0.25 * resource_pressure
                
                channel_risk_score = max(
                    ewma_stress_score * 0.85,
                    resource_pressure * 0.35,
                    combined_weighted
                )
                
                if resource_pressure >= 0.92:
                    channel_risk_score = max(channel_risk_score, 0.70)
                
                channel_risk_score = max(0.0, min(1.0, channel_risk_score))

                raw_metrics = {"cpu": cpu, "memory": memory, "threads": threads}
                channel_state = resource_saturation.update_channel_state(
                    channel_risk_score, raw_metrics, timestamp
                )

                output = {
                    "timestamp": timestamp.isoformat(),
                    "channel": "resource_saturation",
                    "stress_score": round(channel_risk_score, 3),
                    "ewma_stress_score": round(ewma_stress_score, 3),
                    "resource_pressure": round(resource_pressure, 3),
                    "state": channel_state["state"],
                    "state_duration": channel_state["state_duration"],
                    "transition_reason": channel_state["transition_reason"],
                    "raw": {
                        "cpu_percent": round(cpu, 3),
                        "memory_mb": round(memory, 3),
                        "threads": round(threads, 3)
                    }
                }

                print(output)

                # Check for frozen snapshot
                snapshot = resource_saturation.get_frozen_snapshot()
                if snapshot:
                    print(f"📸 Snapshot frozen for {snapshot['channel']}: {snapshot['trigger_time']}")

                next_poll["resource_saturation"] = now + SCRAPE_INTERVAL_SECONDS

            # Poll Performance Degradation Channel
            if now >= next_poll["performance_degradation"]:
                try:
                    p95_response_time = prom.get_p95_response_time()
                    state_info = performance_degradation.update(p95_response_time, timestamp)

                    output = {
                        "timestamp": timestamp.isoformat(),
                        "channel": "performance_degradation",
                        "raw_metric": round(p95_response_time, 3),
                        "state": state_info["state"],
                        "state_duration": state_info["state_duration"],
                        "transition_reason": state_info["transition_reason"]
                    }

                    print(output)

                    # Check for frozen snapshot
                    snapshot = performance_degradation.get_frozen_snapshot()
                    if snapshot:
                        print(f"📸 Snapshot frozen for {snapshot['channel']}: {snapshot['trigger_time']}")

                except Exception as e:
                    print(f"⚠️ performance_degradation channel error: {str(e)}")

                next_poll["performance_degradation"] = now + performance_degradation.scrape_interval

            # Poll Backpressure Overload Channel
            if now >= next_poll["backpressure_overload"]:
                try:
                    queue_depth = prom.get_queue_depth()
                    state_info = backpressure_overload.update(queue_depth, timestamp)

                    output = {
                        "timestamp": timestamp.isoformat(),
                        "channel": "backpressure_overload",
                        "raw_metric": round(queue_depth, 3),
                        "state": state_info["state"],
                        "state_duration": state_info["state_duration"],
                        "transition_reason": state_info["transition_reason"]
                    }

                    print(output)

                    # Check for frozen snapshot
                    snapshot = backpressure_overload.get_frozen_snapshot()
                    if snapshot:
                        print(f"📸 Snapshot frozen for {snapshot['channel']}: {snapshot['trigger_time']}")

                except Exception as e:
                    print(f"⚠️ backpressure_overload channel error: {str(e)}")

                next_poll["backpressure_overload"] = now + backpressure_overload.scrape_interval

            # Sleep until the next scheduled poll
            next_event = min(next_poll.values())
            sleep_duration = max(0.1, next_event - time.time())
            time.sleep(sleep_duration)

        except Exception as e:
            print("⚠️ detector error:", str(e))
            time.sleep(1)


if __name__ == "__main__":
    main()
