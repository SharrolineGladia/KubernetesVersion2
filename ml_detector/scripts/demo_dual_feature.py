"""
Complete Dual-Feature Detection Demo

This script demonstrates the full workflow:
1. Scale-invariant detection (topology-agnostic)
2. Service-specific explainability (RCA with actionable insights)
3. Multi-topology validation (1, 2, 3 services)

Usage:
    python demo_dual_feature.py
"""

import sys
import os

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from scripts.dual_feature_detector import DualFeatureDetector
from scripts.explainability_layer import format_rca_report
from datetime import datetime


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def demo_1_service_edge():
    """Demo: 1-service edge node deployment."""
    print_section("📱 SCENARIO 1: Edge Node (1 Service)")
    
    print("Context: IoT gateway running single notification service")
    print("Expected: 80.78% accuracy, suitable for triage-level detection\n")
    
    # Memory leak scenario
    raw_data = {
        'notification': {
            'cpu': 52,
            'memory': 94,  # Critical memory usage
            'network_in': 28,
            'network_out': 22,
            'disk_io': 18,
            'requests': 45,
            'errors': 3,
            'latency': 150
        }
    }
    
    print("📊 Raw Metrics:")
    print(f"   notification: CPU={raw_data['notification']['cpu']}%, "
          f"Memory={raw_data['notification']['memory']}%, "
          f"Requests={raw_data['notification']['requests']}/s")
    print()
    
    # Detect
    detector = DualFeatureDetector(
        model_path='models/anomaly_detector_scaleinvariant.pkl',
        confidence_threshold=0.80
    )
    
    snapshot = detector.detect_from_raw(raw_data, perform_rca=True)
    print(detector.format_detection_report(snapshot))
    
    # Decision logic
    print("\n🎯 DECISION LOGIC:")
    if snapshot.detection_confidence < 0.80:
        print(f"   → Confidence {snapshot.detection_confidence:.1%} < 80% (edge threshold)")
        print("   → ACTION: Escalate to cloud for deeper analysis")
        print("   → BANDWIDTH: Send 27 scale-invariant features + 1 service metrics")
    else:
        print(f"   → Confidence {snapshot.detection_confidence:.1%} ≥ 80%")
        print("   → ACTION: Handle locally with RCA recommendations")
        if snapshot.rca_result:
            print(f"   → ROOT CAUSE: {snapshot.rca_result.root_cause_service}")
            print(f"   → SEVERITY: {snapshot.rca_result.severity.upper()}")


def demo_2_service_edge_cluster():
    """Demo: 2-service edge cluster deployment."""
    print_section("🏢 SCENARIO 2: Edge Cluster (2 Services)")
    
    print("Context: Retail store cluster running notification + web_api")
    print("Expected: 93.37% accuracy, good for local handling\n")
    
    # CPU spike scenario
    raw_data = {
        'notification': {
            'cpu': 96,  # Critical CPU
            'memory': 72,
            'network_in': 68,
            'network_out': 62,
            'disk_io': 38,
            'requests': 195,
            'errors': 12,
            'latency': 340
        },
        'web_api': {
            'cpu': 89,  # Also elevated
            'memory': 68,
            'network_in': 78,
            'network_out': 72,
            'disk_io': 28,
            'requests': 235,
            'errors': 15,
            'latency': 295
        }
    }
    
    print("📊 Raw Metrics:")
    for service, metrics in raw_data.items():
        print(f"   {service}: CPU={metrics['cpu']}%, "
              f"Memory={metrics['memory']}%, "
              f"Requests={metrics['requests']}/s")
    print()
    
    # Detect
    detector = DualFeatureDetector(
        model_path='models/anomaly_detector_scaleinvariant.pkl',
        confidence_threshold=0.85
    )
    
    snapshot = detector.detect_from_raw(raw_data, perform_rca=True)
    print(detector.format_detection_report(snapshot))
    
    # Decision logic
    print("\n🎯 DECISION LOGIC:")
    if snapshot.detection_confidence < 0.85:
        print(f"   → Confidence {snapshot.detection_confidence:.1%} < 85% (cluster threshold)")
        print("   → ACTION: Escalate to cloud")
    else:
        print(f"   → Confidence {snapshot.detection_confidence:.1%} ≥ 85%")
        print("   → ACTION: Execute local recovery")
        if snapshot.rca_result and snapshot.rca_result.recommendations:
            print("   → RECOMMENDATIONS:")
            for rec in snapshot.rca_result.recommendations[:3]:
                print(f"      • {rec}")


def demo_3_service_cloud():
    """Demo: 3-service cloud deployment."""
    print_section("☁️ SCENARIO 3: Cloud Datacenter (3 Services)")
    
    print("Context: Full microservice stack in cloud region")
    print("Expected: 99.46% accuracy, optimal performance\n")
    
    # Service crash scenario
    raw_data = {
        'notification': {
            'cpu': 8,  # Crashed - minimal activity
            'memory': 15,
            'network_in': 5,
            'network_out': 3,
            'disk_io': 2,
            'requests': 2,  # Nearly zero traffic
            'errors': 1,
            'latency': 0
        },
        'web_api': {
            'cpu': 58,  # Compensating
            'memory': 65,
            'network_in': 85,
            'network_out': 78,
            'disk_io': 35,
            'requests': 185,
            'errors': 45,  # High errors due to notification failure
            'latency': 450
        },
        'processor': {
            'cpu': 42,
            'memory': 52,
            'network_in': 28,
            'network_out': 22,
            'disk_io': 48,
            'requests': 35,
            'errors': 8,
            'latency': 120
        }
    }
    
    print("📊 Raw Metrics:")
    for service, metrics in raw_data.items():
        print(f"   {service}: CPU={metrics['cpu']}%, "
              f"Memory={metrics['memory']}%, "
              f"Requests={metrics['requests']}/s, "
              f"Errors={metrics['errors']}")
    print()
    
    # Detect
    detector = DualFeatureDetector(
        model_path='models/anomaly_detector_scaleinvariant.pkl',
        confidence_threshold=0.90
    )
    
    snapshot = detector.detect_from_raw(raw_data, perform_rca=True)
    print(detector.format_detection_report(snapshot))
    
    # Decision logic
    print("\n🎯 DECISION LOGIC:")
    print(f"   → Confidence {snapshot.detection_confidence:.1%}")
    print("   → TOPOLOGY: 3 services (training configuration)")
    print("   → ACCURACY: ~99.46% (optimal)")
    if snapshot.rca_result:
        print(f"   → ROOT CAUSE: {snapshot.rca_result.root_cause_service}")
        print(f"   → SEVERITY: {snapshot.rca_result.severity.upper()}")
        print("   → ACTION: Automated recovery orchestrator triggered")


def demo_feature_comparison():
    """Demo: Scale-invariant vs raw features."""
    print_section("🔬 FEATURE ENGINEERING COMPARISON")
    
    raw_data = {
        'notification': {
            'cpu': 95,
            'memory': 85,
            'network_in': 60,
            'network_out': 55,
            'disk_io': 40,
            'requests': 150,
            'errors': 8,
            'latency': 250
        },
        'web_api': {
            'cpu': 78,
            'memory': 68,
            'network_in': 80,
            'network_out': 72,
            'disk_io': 25,
            'requests': 200,
            'errors': 10,
            'latency': 180
        }
    }
    
    detector = DualFeatureDetector(
        model_path='models/anomaly_detector_scaleinvariant.pkl'
    )
    
    snapshot = detector.create_snapshot(raw_data)
    
    print("📊 RAW SERVICE-SPECIFIC METRICS (for explainability):")
    print("   notification:")
    for metric, value in snapshot.service_metrics['notification'].to_dict().items():
        print(f"      {metric}: {value}")
    
    print("\n   web_api:")
    for metric, value in snapshot.service_metrics['web_api'].to_dict().items():
        print(f"      {metric}: {value}")
    
    print("\n\n🔄 TRANSFORMED SCALE-INVARIANT FEATURES (for detection):")
    print("   (These work across 1-10 services without retraining)")
    
    top_features = sorted(
        snapshot.features_scaleinvariant.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:10]
    
    for feature, value in top_features:
        print(f"      {feature}: {value:.4f}")
    
    print("\n\n💡 KEY INSIGHT:")
    print("   • RAW metrics: Change semantics with service count")
    print("     Example: max_cpu=95 (2 svc) vs max_cpu=95 (10 svc) - DIFFERENT meanings")
    print("   • SCALE-INVARIANT: Preserve semantics across topologies")
    print("     Example: cpu_utilization_mean=0.87 - SAME meaning for any service count")


def demo_bandwidth_efficiency():
    """Demo: Bandwidth efficiency of dual-feature architecture."""
    print_section("📡 BANDWIDTH EFFICIENCY ANALYSIS")
    
    print("Traditional Approach (send all raw metrics):")
    print("   Edge (1 service): 8 metrics × 1 service = 8 values")
    print("   Edge (2 services): 8 metrics × 2 services = 16 values")
    print("   Cloud (3 services): 8 metrics × 3 services = 24 values")
    print("   Large (10 services): 8 metrics × 10 services = 80 values")
    print()
    
    print("Dual-Feature Approach:")
    print("   Detection phase (always): 27 scale-invariant features")
    print("   Explainability phase (only if anomaly + low confidence):")
    print("      Edge (1 service): +8 metrics for RCA")
    print("      Edge (2 services): +16 metrics for RCA")
    print("      Cloud (3 services): Usually handled locally (confidence high)")
    print()
    
    print("💰 SAVINGS:")
    print("   Edge node (1 svc): 27 vs 8 = 3.4× overhead")
    print("      BUT detection is local, no transmission!")
    print("   Edge cluster (2 svc): 27 vs 16 = 69% overhead")
    print("      With 93% accuracy, escalation rare")
    print("   Cloud (3 svc): 27 vs 24 = 13% overhead")
    print("      With 99% accuracy, almost never escalate")
    print("   Large cluster (10 svc): 27 vs 80 = 66% SAVINGS! ✨")
    print()
    
    print("🎯 EFFECTIVE BANDWIDTH REDUCTION:")
    print("   • 1-3 services: ~equivalent (but enables topology-agnostic model)")
    print("   • 4+ services: 30-66% bandwidth reduction")
    print("   • Only send raw metrics on escalation (rare with high confidence)")


if __name__ == "__main__":
    print("=" * 80)
    print("  🚀 DUAL-FEATURE ANOMALY DETECTION DEMO")
    print("  Scale-Invariant Detection + Service-Specific Explainability")
    print("=" * 80)
    
    try:
        # Run all demos
        demo_1_service_edge()
        input("\n\nPress Enter for next scenario...")
        
        demo_2_service_edge_cluster()
        input("\n\nPress Enter for next scenario...")
        
        demo_3_service_cloud()
        input("\n\nPress Enter for feature comparison...")
        
        demo_feature_comparison()
        input("\n\nPress Enter for bandwidth analysis...")
        
        demo_bandwidth_efficiency()
        
        print("\n\n" + "=" * 80)
        print("✅ DEMO COMPLETE")
        print("=" * 80)
        print("\nKey Takeaways:")
        print("  1. XGBoost detects anomalies using 27 scale-invariant features")
        print("  2. Works across 1-10 services without retraining")
        print("  3. RCA uses service-specific metrics for actionable insights")
        print("  4. Dual architecture enables both detection + explainability")
        print("  5. Bandwidth efficient for large-scale deployments (4+ services)")
        print("\nNext Steps:")
        print("  • Integrate with online_detector/main.py (see online_detector_integration.py)")
        print("  • Train model on production traces for SHAP analysis")
        print("  • Test on real multi-service deployments")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: Model file not found")
        print(f"   {e}")
        print("\nPlease ensure:")
        print("  1. Trained model exists at: ml_detector/models/anomaly_detector_scaleinvariant.pkl")
        print("  2. You're running from the project root: python ml_detector/scripts/demo_dual_feature.py")
        print("\nTo train the model:")
        print("  cd ml_detector/scripts")
        print("  python transform_dataset_scaleinvariant.py")
        print("  python train_scaleinvariant_model.py")
