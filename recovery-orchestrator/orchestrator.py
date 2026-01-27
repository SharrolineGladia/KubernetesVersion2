import datetime
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException
import time


# -----------------------------
# Configuration (hardcoded for v1)
# -----------------------------
NAMESPACE = "journal-implementation"
APP_LABEL_KEY = "app"
APP_LABEL_VALUE = "notification-service"
RECOVERY_REASON = "cpu_spike_detected"  # dummy reason for now


def log(message, **kwargs):
    """Simple structured logger"""
    timestamp = datetime.datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "component": "recovery-orchestrator",
        "message": message,
        **kwargs
    }
    print(log_entry)


def load_kubernetes_config():
    """
    Load Kubernetes configuration.
    This works when running the script locally with kubectl configured.
    """
    try:
        config.load_kube_config()
        log("Kubernetes config loaded successfully")
    except Exception as e:
        log("Failed to load Kubernetes config", error=str(e))
        raise


def get_target_pod(v1_api):
    """
    Find one running pod for the target service using label selector.
    """
    label_selector = f"{APP_LABEL_KEY}={APP_LABEL_VALUE}"

    try:
        pods = v1_api.list_namespaced_pod(
            namespace=NAMESPACE,
            label_selector=label_selector
        )
    except ApiException as e:
        log("Failed to list pods", error=str(e))
        raise

    if not pods.items:
        raise RuntimeError("No pods found for target service")

    # Prefer a running pod
    for pod in pods.items:
        if pod.status.phase == "Running":
            return pod.metadata.name

    # Fallback: return the first pod
    return pods.items[0].metadata.name


def restart_pod(v1_api, pod_name):
    """
    Delete the pod to trigger Kubernetes self-healing.
    """
    try:
        v1_api.delete_namespaced_pod(
            name=pod_name,
            namespace=NAMESPACE
        )
        log(
            "Pod deletion requested",
            pod_name=pod_name,
            namespace=NAMESPACE,
            reason=RECOVERY_REASON
        )
    except ApiException as e:
        log("Failed to delete pod", pod_name=pod_name, error=str(e))
        raise


def main():
    trigger_time = time.time()
    log(
        "Recovery triggered",
        trigger_time=trigger_time,
        reason=RECOVERY_REASON
    )

    # Step 1: Load Kubernetes config
    load_kubernetes_config()

    # Step 2: Create Kubernetes API client
    v1_api = client.CoreV1Api()

    # Step 3: Identify target pod
    try:
        pod_name = get_target_pod(v1_api)
        log("Target pod identified", pod_name=pod_name)
    except Exception as e:
        log("Pod selection failed", error=str(e))
        return

    # Step 4: Trigger recovery action
    restart_pod(v1_api, pod_name)

    log("Recovery action completed", action="restart_pod")


if __name__ == "__main__":
    main()
