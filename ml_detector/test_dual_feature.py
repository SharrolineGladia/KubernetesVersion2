"""
Quick Test - Dual-Feature Detection with RCA

This demonstrates the complete workflow:
1. Raw service metrics → 2. Detection → 3. Root Cause Analysis
"""

import sys
import os

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from scripts.dual_feature_detector import DualFeatureDetector

def test_memory_leak_edge():
    """Test memory leak detection on 1-service edge node."""
    print("=" * 80)
    print("TEST 1: Memory Leak - Edge Node (1 Service)")
    print("=" * 80)
    print()
    
    # Initialize detector
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'models', 'anomaly_detector_scaleinvariant.pkl')
    detector = DualFeatureDetector(
        model_path=model_path,
        confidence_threshold=0.75,  # Lower threshold for edge
        enable_explainability=True
    )
    
    # Simulate memory leak
    raw_data = {
        'notification': {
            'cpu': 48,
            'memory': 94,  # Critical!
            'network_in': 30,
            'network_out': 25,
            'disk_io': 20,
            'requests': 50,
            'errors': 3,
            'latency': 150
        }
    }
    
    print("📊 Input Metrics:")
    print(f"   notification-service:")
    print(f"      CPU: {raw_data['notification']['cpu']}%")
    print(f"      Memory: {raw_data['notification']['memory']}% ⚠️ CRITICAL")
    print(f"      Requests: {raw_data['notification']['requests']}/s")
    print(f"      Errors: {raw_data['notification']['errors']}")
    print()
    
    # Detect + RCA
    snapshot = detector.detect_from_raw(raw_data, perform_rca=True)
    
    print("🔍 Detection Results:")
    print(f"   Anomaly Type: {snapshot.anomaly_type.upper()}")
    print(f"   Confidence: {snapshot.detection_confidence * 100:.1f}%")
    print(f"   Service Count: {snapshot.service_count}")
    print()
    
    if snapshot.rca_result:
        print("🎯 Root Cause Analysis:")
        print(f"   Root Cause Service: {snapshot.rca_result.root_cause_service}")
        print(f"   Severity: {snapshot.rca_result.severity.upper()}")
        print(f"   Affected Resources: {', '.join(snapshot.rca_result.affected_resources)}")
        print()
        
        print("⚠️  Contributing Factors:")
        for i, factor in enumerate(snapshot.rca_result.contributing_factors[:3], 1):
            print(f"   {i}. [{factor['severity'].upper()}] {factor['metric']}: {factor['value']}")
            print(f"      → {factor['description']}")
        print()
        
        print("💡 Recommendations:")
        for i, rec in enumerate(snapshot.rca_result.recommendations[:3], 1):
            print(f"   {i}. {rec}")
    else:
        print("ℹ️  Confidence too low for RCA (would escalate to cloud)")
    
    print()


def test_cpu_spike_cluster():
    """Test CPU spike detection on 2-service edge cluster."""
    print("=" * 80)
    print("TEST 2: CPU Spike - Edge Cluster (2 Services)")
    print("=" * 80)
    print()
    
    # Initialize detector
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'models', 'anomaly_detector_scaleinvariant.pkl')
    detector = DualFeatureDetector(
        model_path=model_path,
        confidence_threshold=0.80,
        enable_explainability=True
    )
    
    # Simulate CPU spike
    raw_data = {
        'notification': {
            'cpu': 97,  # Critical!
            'memory': 72,
            'network_in': 70,
            'network_out': 65,
            'disk_io': 40,
            'requests': 200,
            'errors': 15,
            'latency': 350
        },
        'web_api': {
            'cpu': 89,  # Also high
            'memory': 68,
            'network_in': 80,
            'network_out': 72,
            'disk_io': 30,
            'requests': 240,
            'errors': 18,
            'latency': 290
        }
    }
    
    print("📊 Input Metrics:")
    for service, metrics in raw_data.items():
        print(f"   {service}:")
        print(f"      CPU: {metrics['cpu']}% {'⚠️ CRITICAL' if metrics['cpu'] > 90 else ''}")
        print(f"      Memory: {metrics['memory']}%")
        print(f"      Requests: {metrics['requests']}/s")
    print()
    
    # Detect + RCA
    snapshot = detector.detect_from_raw(raw_data, perform_rca=True)
    
    print("🔍 Detection Results:")
    print(f"   Anomaly Type: {snapshot.anomaly_type.upper()}")
    print(f"   Confidence: {snapshot.detection_confidence * 100:.1f}%")
    print(f"   Service Count: {snapshot.service_count}")
    print()
    
    if snapshot.rca_result:
        print("🎯 Root Cause Analysis:")
        print(f"   Root Cause Service: {snapshot.rca_result.root_cause_service}")
        print(f"   Severity: {snapshot.rca_result.severity.upper()}")
        print()
        
        print("💡 Top Recommendations:")
        for i, rec in enumerate(snapshot.rca_result.recommendations[:4], 1):
            print(f"   {i}. {rec}")
    
    print()


def test_service_crash():
    """Test service crash detection on 3-service cloud."""
    print("=" * 80)
    print("TEST 3: Service Crash - Cloud (3 Services)")
    print("=" * 80)
    print()
    
    # Initialize detector
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'models', 'anomaly_detector_scaleinvariant.pkl')
    detector = DualFeatureDetector(
        model_path=model_path,
        confidence_threshold=0.85,
        enable_explainability=True
    )
    
    # Simulate service crash
    raw_data = {
        'notification': {
            'cpu': 5,  # Crashed - minimal activity
            'memory': 10,
            'network_in': 2,
            'network_out': 1,
            'disk_io': 1,
            'requests': 1,  # Nearly zero
            'errors': 0,
            'latency': 0
        },
        'web_api': {
            'cpu': 65,
            'memory': 70,
            'network_in': 85,
            'network_out': 78,
            'disk_io': 38,
            'requests': 180,
            'errors': 50,  # High errors due to notification crash
            'latency': 480
        },
        'processor': {
            'cpu': 45,
            'memory': 55,
            'network_in': 30,
            'network_out': 25,
            'disk_io': 50,
            'requests': 40,
            'errors': 10,
            'latency': 120
        }
    }
    
    print("📊 Input Metrics:")
    for service, metrics in raw_data.items():
        status = ""
        if metrics['cpu'] < 10:
            status = " 🔴 CRASHED"
        elif metrics['errors'] > 40:
            status = " ⚠️ HIGH ERRORS"
        print(f"   {service}:{status}")
        print(f"      CPU: {metrics['cpu']}%, Requests: {metrics['requests']}/s, Errors: {metrics['errors']}")
    print()
    
    # Detect + RCA
    snapshot = detector.detect_from_raw(raw_data, perform_rca=True)
    
    print("🔍 Detection Results:")
    print(f"   Anomaly Type: {snapshot.anomaly_type.upper()}")
    print(f"   Confidence: {snapshot.detection_confidence * 100:.1f}%")
    print()
    
    if snapshot.rca_result:
        print("🎯 Root Cause Analysis:")
        print(f"   Root Cause Service: {snapshot.rca_result.root_cause_service} 🔴")
        print(f"   Severity: {snapshot.rca_result.severity.upper()}")
        print()
        
        print("🚨 Urgent Recommendations:")
        for i, rec in enumerate(snapshot.rca_result.recommendations[:4], 1):
            print(f"   {i}. {rec}")
    
    print()


if __name__ == "__main__":
    print()
    print("🚀 DUAL-FEATURE DETECTION TEST SUITE")
    print("   Scale-Invariant Detection + Service-Specific RCA")
    print()
    
    try:
        # Test 1: Memory leak (1 service)
        test_memory_leak_edge()
        
        # Test 2: CPU spike (2 services)
        test_cpu_spike_cluster()
        
        # Test 3: Service crash (3 services)
        test_service_crash()
        
        print("=" * 80)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print()
        print("Key Takeaways:")
        print("  • Detection uses 27 scale-invariant features (works with 1-10 services)")
        print("  • RCA uses service-specific metrics (identifies exact service)")
        print("  • System provides actionable recommendations for recovery")
        print("  • Confidence thresholds determine local vs cloud handling")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure the trained model exists at:")
        print("  ml_detector/models/anomaly_detector_scaleinvariant.pkl")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
