"""
Quick test to verify trace and log integration is working.

Tests:
1. Trace analyzer can connect to Jaeger
2. Log analyzer can parse log files
3. RCA layer properly integrates both

Run this AFTER starting your microservice to verify integration.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / 'ml_detector' / 'scripts'))

from trace_analyzer import TraceAnalyzer
from log_analyzer import LogAnalyzer
from explainability_layer import AnomalyExplainer, ServiceMetrics


def test_trace_analyzer():
    """Test 1: Can we connect to Jaeger?"""
    print("Test 1: Trace Analyzer Connection")
    print("-" * 40)
    
    try:
        analyzer = TraceAnalyzer(jaeger_query_url="http://localhost:16686")
        
        # Try to fetch traces from last minute
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=1)
        
        summary = analyzer.analyze_time_window(start_time, end_time)
        
        if summary.total_traces > 0:
            print(f"✅ SUCCESS - Found {summary.total_traces} traces")
            print(f"   Services: {summary.services_involved}")
            return True
        else:
            print("⚠️  WARNING - Jaeger connected but no traces found")
            print("   Start your microservice and generate traffic")
            return False
    
    except Exception as e:
        print(f"❌ FAILED - {e}")
        print("   Make sure Jaeger is running: docker-compose -f docker-compose-jaeger.yml up -d")
        return False


def test_log_analyzer():
    """Test 2: Can we parse log files?"""
    print("\nTest 2: Log Analyzer")
    print("-" * 40)
    
    log_file = PROJECT_ROOT / 'service_logs.json'
    
    if not os.path.exists(log_file):
        print(f"⚠️  WARNING - Log file not found: {log_file}")
        print("   Capture logs with: python scripts/tools/capture_service_logs.py notification-service 1")
        return False
    
    try:
        analyzer = LogAnalyzer(log_source='file', log_file_path=log_file)
        
        # Analyze last 10 minutes
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=10)
        
        summary = analyzer.analyze_time_window(start_time, end_time)
        
        if summary.total_logs > 0:
            print(f"✅ SUCCESS - Found {summary.total_logs} log entries")
            print(f"   Errors: {summary.error_logs}, Warnings: {summary.warning_logs}")
            return True
        else:
            print("⚠️  WARNING - Log file exists but no entries in time window")
            print("   Logs may be too old. Capture new logs.")
            return False
    
    except Exception as e:
        print(f"❌ FAILED - {e}")
        return False


def test_rca_integration():
    """Test 3: Does RCA properly integrate traces and logs?"""
    print("\nTest 3: RCA Integration")
    print("-" * 40)
    
    try:
        explainer = AnomalyExplainer(
            model_path=None,
            enable_traces=True,
            enable_logs=True,
            jaeger_url="http://localhost:16686",
            log_file_path=str(PROJECT_ROOT / 'service_logs.json')
        )
        
        # Check if analyzers were initialized
        trace_enabled = explainer.trace_analyzer is not None
        log_enabled = explainer.log_analyzer is not None
        
        print(f"Trace analyzer: {'✅ Enabled' if trace_enabled else '❌ Disabled'}")
        print(f"Log analyzer: {'✅ Enabled' if log_enabled else '❌ Disabled'}")
        
        if trace_enabled and log_enabled:
            print("\n✅ SUCCESS - RCA integration ready")
            print("   Run: python scripts/demos/demo_integrated_rca.py")
            return True
        else:
            print("\n⚠️  PARTIAL - Some analyzers not available")
            return False
    
    except Exception as e:
        print(f"❌ FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print(" TRACE & LOG INTEGRATION VERIFICATION")
    print("=" * 60)
    print()
    
    results = {
        'trace_analyzer': test_trace_analyzer(),
        'log_analyzer': test_log_analyzer(),
        'rca_integration': test_rca_integration()
    }
    
    # Summary
    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name.replace('_', ' ').title()}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All checks passed! Integration is working.")
        print("\nNext: Run the integrated demo:")
        print("   python scripts/demos/demo_integrated_rca.py")
    else:
        print("\n⚠️  Some checks failed. Review errors above.")
        print("\nCommon fixes:")
        print("  1. Start Jaeger: docker-compose -f docker-compose-jaeger.yml up -d")
        print("  2. Start service: cd services/notification-service && python notification_service.py")
        print("  3. Capture logs: python scripts/tools/capture_service_logs.py notification-service 1")
    
    print()


if __name__ == "__main__":
    main()
