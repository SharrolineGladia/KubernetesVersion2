#!/usr/bin/env python3
"""
Trigger anomalies for testing the online detector.
Allows selective triggering of individual channels or all at once.

Usage:
  python trigger_anomaly.py --all                    # Trigger all 3 channels
  python trigger_anomaly.py --resource               # Resource saturation only
  python trigger_anomaly.py --performance            # Performance degradation only
  python trigger_anomaly.py --backpressure           # Backpressure overload only
  python trigger_anomaly.py --resource --performance # Multiple channels
  python trigger_anomaly.py --stop                   # Stop all anomalies
"""

import argparse
import requests
import time
import threading

SERVICE_URL = "http://localhost:8003"

def trigger_resource_saturation():
    """Trigger CPU/Memory/Thread spike."""
    print("🔥 Triggering Resource Saturation anomaly...")
    print("   - CPU spike")
    print("   - Memory growth")
    print("   - Thread explosion")
    response = requests.post(f"{SERVICE_URL}/simulate-critical")
    if response.status_code == 200:
        print("✅ Resource saturation triggered")
    else:
        print(f"❌ Failed: {response.status_code}")
    return response.status_code == 200

def trigger_performance_degradation_and_backpressure():
    """Send notifications to build queue and increase response time."""
    print("🔥 Triggering Performance Degradation & Backpressure Overload...")
    print("   - Flooding with 1000 notification requests")
    print("   - This will increase queue depth and response time")
    
    success_count = 0
    for i in range(1000):
        try:
            response = requests.post(
                f"{SERVICE_URL}/send-notification",
                json={"user_id": f"test_{i}", "message": "load test"},
                timeout=2
            )
            if response.status_code == 200:
                success_count += 1
        except Exception as e:
            pass  # Continue even on errors
        
        if (i + 1) % 100 == 0:
            print(f"   Progress: {i + 1}/1000 requests sent...")
    
    print(f"✅ Sent {success_count}/1000 requests successfully")
    return success_count > 0

def stop_all_anomalies():
    """Stop all triggered anomalies."""
    print("🛑 Stopping all anomalies...")
    
    try:
        response = requests.post(f"{SERVICE_URL}/stop-critical")
        if response.status_code == 200:
            print("✅ Critical mode stopped (Resource saturation will decrease)")
        else:
            print(f"⚠️  Stop critical failed: {response.status_code}")
    except Exception as e:
        print(f"⚠️  Error stopping critical: {e}")
    
    # Note: Queue will drain naturally as workers process messages
    print("ℹ️  Queue will drain as notification workers process backlog")
    print("ℹ️  All channels should return to normal within 1-2 minutes")

def main():
    parser = argparse.ArgumentParser(
        description="Trigger anomalies for testing the online detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python trigger_anomaly.py --all                    # All channels
  python trigger_anomaly.py --resource --performance # Two channels
  python trigger_anomaly.py --stop                   # Stop anomalies
        """
    )
    
    parser.add_argument('--all', action='store_true', 
                       help='Trigger all three channels (resource, performance, backpressure)')
    parser.add_argument('--resource', action='store_true',
                       help='Trigger resource saturation (CPU/Memory/Threads)')
    parser.add_argument('--performance', action='store_true',
                       help='Trigger performance degradation (p95 response time)')
    parser.add_argument('--backpressure', action='store_true',
                       help='Trigger backpressure overload (queue depth)')
    parser.add_argument('--stop', action='store_true',
                       help='Stop all triggered anomalies')
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if not any([args.all, args.resource, args.performance, args.backpressure, args.stop]):
        parser.print_help()
        return
    
    print("=" * 70)
    print("🧪 ANOMALY TRIGGER TOOL")
    print("=" * 70)
    print()
    
    # Handle stop command
    if args.stop:
        stop_all_anomalies()
        print()
        print("=" * 70)
        print("✅ Done! Monitor your detector output to see recovery.")
        print("=" * 70)
        return
    
    # Determine which channels to trigger
    trigger_resource = args.all or args.resource
    trigger_performance = args.all or args.performance
    trigger_backpressure = args.all or args.backpressure
    
    # Note: performance and backpressure use the same mechanism (queue flooding)
    trigger_queue = trigger_performance or trigger_backpressure
    
    print("📋 Triggering channels:")
    if trigger_resource:
        print("   ✓ Resource Saturation")
    if trigger_performance:
        print("   ✓ Performance Degradation")
    if trigger_backpressure:
        print("   ✓ Backpressure Overload")
    print()
    
    # Trigger selected anomalies
    results = []
    
    if trigger_resource:
        print("-" * 70)
        results.append(trigger_resource_saturation())
        print()
        time.sleep(1)
    
    if trigger_queue:
        print("-" * 70)
        results.append(trigger_performance_degradation_and_backpressure())
        print()
    
    # Summary
    print("=" * 70)
    if all(results):
        print("🎉 SUCCESS: All selected anomalies triggered!")
        print()
        print("👀 Watch your detector output:")
        print("   - States should transition: normal → stressed → critical")
        print("   - Snapshots should freeze when critical is reached")
        print()
        print("🛑 To stop anomalies:")
        print("   python trigger_anomaly.py --stop")
    else:
        print("⚠️  Some anomalies failed to trigger. Check service status.")
    print("=" * 70)

if __name__ == "__main__":
    main()
