"""
Online Detector Integration with XGBoost Classification & Explainability

This module extends the online_detector to:
1. Collect raw service-specific metrics from EWMA detectors
2. Transform to scale-invariant features for XGBoost classification
3. Perform root cause analysis when anomalies detected
4. Generate actionable recommendations

Integration Points:
- Captures snapshots from ResourceSaturationDetector
- Enriches with service-specific metrics (CPU, memory, network, etc.)
- Classifies anomaly type using XGBoost
- Explains root cause using service-level analysis
"""

import sys
import os
from typing import Dict, Optional
from datetime import datetime
import json

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from online_detector.metrics_reader import PrometheusClient
from ml_detector.scripts.dual_feature_detector import DualFeatureDetector, DetectionSnapshot


class EnrichedDetector:
    """
    Enriched detector that combines EWMA stress detection with
    XGBoost classification and RCA.
    
    Workflow:
    1. EWMA detector identifies potential anomaly (stress > threshold)
    2. Capture full service metrics snapshot
    3. Classify anomaly type with XGBoost (cpu_spike, memory_leak, etc.)
    4. Perform RCA if confidence high enough
    5. Generate actionable report
    """
    
    def __init__(
        self,
        model_path: str = "ml_detector/models/anomaly_detector_scaleinvariant.pkl",
        stress_threshold: float = 0.6,
        confidence_threshold: float = 0.80
    ):
        """
        Initialize enriched detector.
        
        Args:
            model_path: Path to trained XGBoost model
            stress_threshold: EWMA stress threshold to trigger classification
            confidence_threshold: Min confidence for RCA (0-1)
        """
        self.model_path = model_path
        self.stress_threshold = stress_threshold
        self.confidence_threshold = confidence_threshold
        
        # Initialize components
        self.prom = PrometheusClient()
        self.detector = DualFeatureDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            enable_explainability=True
        )
        
        # State tracking
        self.last_classification_time = None
        self.classification_cooldown = 60  # seconds between classifications
        
        print(f"✅ Enriched Detector initialized")
        print(f"   Model: {model_path}")
        print(f"   Stress threshold: {stress_threshold}")
        print(f"   Confidence threshold: {confidence_threshold}")
    
    def should_classify(
        self,
        stress_score: float,
        current_time: datetime
    ) -> bool:
        """
        Determine if we should trigger XGBoost classification.
        
        Args:
            stress_score: Current EWMA stress score
            current_time: Current timestamp
        
        Returns:
            True if should classify, False otherwise
        """
        # Check stress threshold
        if stress_score < self.stress_threshold:
            return False
        
        # Check cooldown (avoid spamming classifications)
        if self.last_classification_time is not None:
            elapsed = (current_time - self.last_classification_time).total_seconds()
            if elapsed < self.classification_cooldown:
                return False
        
        return True
    
    def capture_service_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Capture current metrics for all services.
        
        In production, this would query Prometheus for all service metrics.
        For demo, we'll use the notification-service as example.
        
        Returns:
            Dictionary of service_name -> metrics
        """
        # Get metrics from Prometheus
        cpu = self.prom.get_cpu()
        memory = self.prom.get_memory()
        threads = self.prom.get_threads()
        
        try:
            latency = self.prom.get_p95_response_time()
        except:
            latency = 0
        
        try:
            queue_depth = self.prom.get_queue_depth()
        except:
            queue_depth = 0
        
        # For demo: single service (notification)
        # In production, query for multiple services
        service_metrics = {
            'notification': {
                'cpu': cpu,
                'memory': memory,
                'network_in': threads * 0.5,  # Approximation
                'network_out': threads * 0.4,
                'disk_io': 20,  # Would query from Prometheus
                'requests': threads,  # Approximation (threads ≈ request load)
                'errors': max(0, queue_depth - 50),  # Approximation
                'latency': latency
            }
        }
        
        return service_metrics
    
    def classify_and_explain(
        self,
        stress_score: float,
        current_time: datetime
    ) -> Optional[DetectionSnapshot]:
        """
        Classify anomaly type and perform RCA.
        
        Args:
            stress_score: EWMA stress score that triggered classification
            current_time: Current timestamp
        
        Returns:
            DetectionSnapshot with classification and RCA results, or None if threshold not met
        """
        # Check if should classify
        if not self.should_classify(stress_score, current_time):
            return None
        
        print(f"\n🔍 TRIGGERING CLASSIFICATION (stress: {stress_score:.3f})")
        
        # Capture service metrics
        service_metrics = self.capture_service_metrics()
        
        # Perform detection + RCA
        snapshot = self.detector.detect_from_raw(
            raw_service_data=service_metrics,
            perform_rca=True
        )
        
        # Update last classification time
        self.last_classification_time = current_time
        
        # Print report
        print(self.detector.format_detection_report(snapshot))
        
        # Log to file
        self._log_detection(snapshot, stress_score)
        
        return snapshot
    
    def _log_detection(
        self,
        snapshot: DetectionSnapshot,
        stress_score: float
    ):
        """Log detection result to file."""
        log_entry = {
            'timestamp': snapshot.timestamp,
            'ewma_stress_score': stress_score,
            'anomaly_type': snapshot.anomaly_type,
            'confidence': snapshot.detection_confidence,
            'service_count': snapshot.service_count,
            'active_services': snapshot.active_services,
        }
        
        if snapshot.rca_result:
            log_entry['rca'] = {
                'root_cause_service': snapshot.rca_result.root_cause_service,
                'severity': snapshot.rca_result.severity,
                'affected_resources': snapshot.rca_result.affected_resources,
                'recommendations': snapshot.rca_result.recommendations
            }
        
        # Append to log file
        log_path = "ml_detector/results/detection_log.jsonl"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')


def integrate_with_online_detector():
    """
    Example: How to integrate with online_detector/main.py
    
    Add this code to the main() function in online_detector/main.py:
    """
    
    example_code = '''
# In online_detector/main.py, add at the top:
from ml_detector.scripts.online_detector_integration import EnrichedDetector

def main():
    prom = PrometheusClient()
    
    # Initialize EWMA detector (existing code)
    resource_saturation = ResourceSaturationDetector()
    
    # NEW: Initialize XGBoost classifier with RCA
    enriched_detector = EnrichedDetector(
        model_path="../ml_detector/models/anomaly_detector_scaleinvariant.pkl",
        stress_threshold=0.6,  # Trigger classification when stress > 0.6
        confidence_threshold=0.80  # Perform RCA when confidence > 80%
    )
    
    print("🚀 Enriched Online Detector started")
    print("   EWMA stress detection + XGBoost classification + RCA\\n")
    
    while True:
        try:
            now = time.time()
            timestamp = datetime.utcnow()
            
            # Poll Resource Saturation Channel (existing code)
            cpu = prom.get_cpu()
            memory = prom.get_memory()
            threads = prom.get_threads()
            
            stresses = resource_saturation.update(cpu, memory, threads, timestamp)
            ewma_stress_score = (
                CPU_WEIGHT * stresses["cpu"] +
                MEMORY_WEIGHT * stresses["memory"] +
                THREAD_WEIGHT * stresses["threads"]
            )
            
            # ... (existing stress calculation code) ...
            
            # NEW: Classify and explain if stress high enough
            detection_snapshot = enriched_detector.classify_and_explain(
                stress_score=channel_risk_score,
                current_time=timestamp
            )
            
            # If anomaly detected with high confidence, trigger escalation
            if detection_snapshot and detection_snapshot.anomaly_type != 'normal':
                if detection_snapshot.detection_confidence >= 0.85:
                    print(f"🚨 ESCALATING: {detection_snapshot.anomaly_type} "
                          f"in {detection_snapshot.rca_result.root_cause_service}")
                    # Trigger escalation logic here
            
            # ... (rest of existing code) ...
            
        except Exception as e:
            print("⚠️ detector error:", str(e))
            time.sleep(1)
    '''
    
    print("=" * 80)
    print("INTEGRATION GUIDE")
    print("=" * 80)
    print(example_code)
    print("=" * 80)


# Standalone demo
if __name__ == "__main__":
    import time
    
    print("🚀 Online Detector Integration Demo")
    print("=" * 80)
    
    # Show integration guide
    integrate_with_online_detector()
    
    print("\n📊 Running Demo Detection...\n")
    
    # Initialize enriched detector
    try:
        enriched = EnrichedDetector(
            model_path="../models/anomaly_detector_scaleinvariant.pkl",
            stress_threshold=0.5,  # Lower threshold for demo
            confidence_threshold=0.75
        )
        
        # Simulate monitoring loop
        print("Monitoring services (press Ctrl+C to stop)...\n")
        
        for i in range(5):
            current_time = datetime.utcnow()
            
            # Simulate varying stress levels
            stress_score = 0.4 + (i * 0.15)  # 0.4 → 0.55 → 0.7 → 0.85 → 1.0
            
            print(f"[{current_time.isoformat()}] EWMA Stress: {stress_score:.3f}")
            
            # Try classification
            snapshot = enriched.classify_and_explain(stress_score, current_time)
            
            if snapshot is None:
                print("   → Below threshold, skipping classification\n")
            
            time.sleep(2)
        
        print("\n✅ Demo complete. Check ml_detector/results/detection_log.jsonl")
        
    except FileNotFoundError:
        print("\n⚠️ Model file not found. Please ensure:")
        print("   1. Model exists at: ml_detector/models/anomaly_detector_scaleinvariant.pkl")
        print("   2. Run from project root directory")
        print("\nIntegration guide printed above - refer to that for production setup.")
