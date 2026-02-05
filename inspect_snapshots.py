#!/usr/bin/env python3
"""
Inspect frozen snapshots from the detector.
Run this while the detector is in critical state.

Usage: python inspect_snapshots.py
"""

import json
from online_detector.main import resource_saturation, performance_degradation, backpressure_overload

def inspect_snapshots():
    print("=" * 80)
    print("📸 FROZEN SNAPSHOT INSPECTOR")
    print("=" * 80)
    print()
    
    channels = [
        ("Resource Saturation", resource_saturation),
        ("Performance Degradation", performance_degradation),
        ("Backpressure Overload", backpressure_overload)
    ]
    
    for name, detector in channels:
        snapshot = detector.get_frozen_snapshot()
        
        print(f"{'=' * 80}")
        print(f"Channel: {name}")
        print(f"{'=' * 80}")
        
        if snapshot:
            print(f"✅ SNAPSHOT FROZEN")
            print(f"   Trigger Time: {snapshot['trigger_time']}")
            print(f"   Window: {snapshot['snapshot_window_seconds']}s ({snapshot['snapshot_window_seconds']/60:.1f} minutes)")
            print(f"   Data Points: {len(snapshot['data'])} observations")
            print()
            print(f"First 3 observations:")
            for i, obs in enumerate(snapshot['data'][:3]):
                print(f"   [{i+1}] {json.dumps(obs, indent=6)}")
            
            print()
            print(f"Last 3 observations:")
            for i, obs in enumerate(snapshot['data'][-3:]):
                print(f"   [{len(snapshot['data'])-2+i}] {json.dumps(obs, indent=6)}")
            
            print()
            print(f"Full snapshot saved to: {name.lower().replace(' ', '_')}_snapshot.json")
            with open(f"{name.lower().replace(' ', '_')}_snapshot.json", 'w') as f:
                json.dump(snapshot, f, indent=2)
        else:
            print(f"❌ NO SNAPSHOT - Channel not in critical state")
        
        print()
    
    print("=" * 80)
    print("💡 Next Steps:")
    print("   1. These snapshots capture the last 10 minutes before critical state")
    print("   2. Send to recovery orchestrator for root cause analysis")
    print("   3. Use for automated remediation decisions")
    print("=" * 80)

if __name__ == "__main__":
    inspect_snapshots()
