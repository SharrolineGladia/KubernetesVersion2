# metrics_reader.py

import requests
from .config import PROMETHEUS_URL, SERVICE_NAME, NAMESPACE

class PrometheusClient:
    def __init__(self, base_url=PROMETHEUS_URL):
        self.base_url = base_url.rstrip("/")

    def query(self, promql: str) -> float:
        response = requests.get(
            f"{self.base_url}/api/v1/query",
            params={"query": promql},
            timeout=5
        )
        response.raise_for_status()
        data = response.json()

        if data["status"] != "success":
            raise RuntimeError("Prometheus query failed")

        result = data["data"]["result"]
        if not result:
            return 0.0

        return float(result[0]["value"][1])

    def get_cpu(self) -> float:
        return self.query(
            f'notification_service_cpu_percent{{namespace="{NAMESPACE}",service="{SERVICE_NAME}"}}'
        )

    def get_memory(self) -> float:
        return self.query(
            f'notification_service_memory_mb{{namespace="{NAMESPACE}",service="{SERVICE_NAME}"}}'
        )

    def get_threads(self) -> float:
        return self.query(
            f'notification_service_thread_count{{namespace="{NAMESPACE}",service="{SERVICE_NAME}"}}'
        )

    def get_p95_response_time(self) -> float:
        """Get p95 response time in milliseconds."""
        return self.query(
            f'notification_service_response_time_p95_ms{{namespace="{NAMESPACE}",service="{SERVICE_NAME}"}}'
        )

    def get_queue_depth(self) -> float:
        """Get current queue depth."""
        return self.query(
            f'notification_service_internal_queue_depth{{namespace="{NAMESPACE}",service="{SERVICE_NAME}"}}'
        )

