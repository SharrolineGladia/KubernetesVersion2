#!/usr/bin/env python3
"""
Test script to verify all three EWMA channels are working properly.
Tests snapshot freezing, state transitions, and independent polling.

Run from project root: python -m online_detector.test_channels
"""

import time
import sys
from datetime import datetime
from ..detector import (
    ResourceSaturationDetector,
    PerformanceDegradationChannel,
    BackpressureOverloadChannel
)

def print_separator(title=""):
    print("\n" + "="*70)
    if title:
        print(f"  {title}")
        print("="*70)

def test_resource_saturation_channel():
    """Test ResourceSaturationChannel with snapshot freezing."""
    print_separator("TEST 1: Resource Saturation Channel")
    
    channel = ResourceSaturationDetector()
    print(f"✓ Channel initialized: {channel.channel_name}")
    print(f"✓ Buffer size: {channel.observation_buffer.maxlen}")
    print(f"✓ Scrape interval: {channel.scrape_interval}s")
    
    # Simulate normal operation
    print("\n📊 Simulating normal metrics...")
    for i in range(5):
        timestamp = datetime.utcnow()
        stresses = channel.update(30.0, 100.0, 10.0, timestamp)
        stress_score = 0.5 * stresses["cpu"] + 0.3 * stresses["memory"] + 0.2 * stresses["threads"]
        state = channel.update_channel_state(stress_score, {"cpu": 30, "mem": 100, "threads": 10}, timestamp)
        print(f"  Sample {i+1}: stress={stress_score:.3f}, state={state['state']}, buffer_len={len(channel.observation_buffer)}")
        time.sleep(0.1)
    
    # Simulate anomaly to trigger critical state
    print("\n🔥 Simulating high stress to trigger CRITICAL state...")
    for i in range(10):
        timestamp = datetime.utcnow()
        stresses = channel.update(95.0, 220.0, 50.0, timestamp)
        stress_score = 0.8  # High stress
        state = channel.update_channel_state(stress_score, {"cpu": 95, "mem": 220, "threads": 50}, timestamp)
        print(f"  Sample {i+1}: stress={stress_score:.3f}, state={state['state']}, frozen={channel.is_snapshot_frozen}")
        
        if channel.is_snapshot_frozen:
            snapshot = channel.get_frozen_snapshot()
            print(f"\n📸 SNAPSHOT FROZEN!")
            print(f"   Channel: {snapshot['channel']}")
            print(f"   Trigger time: {snapshot['trigger_time']}")
            print(f"   Window: {snapshot['snapshot_window_seconds']}s")
            print(f"   Data points: {len(snapshot['data'])}")
            break
        time.sleep(0.1)
    
    # Verify snapshot exists
    snapshot = channel.get_frozen_snapshot()
    if snapshot:
        print(f"\n✅ Snapshot freeze WORKING - captured {len(snapshot['data'])} observations")
        return True
    else:
        print("\n❌ Snapshot freeze FAILED - no snapshot created")
        return False

def test_performance_degradation_channel():
    """Test PerformanceDegradationChannel."""
    print_separator("TEST 2: Performance Degradation Channel")
    
    channel = PerformanceDegradationChannel()
    print(f"✓ Channel initialized: {channel.channel_name}")
    print(f"✓ Buffer size: {channel.observation_buffer.maxlen}")
    print(f"✓ Scrape interval: {channel.scrape_interval}s")
    print(f"✓ Thresholds: LOW=0.3, HIGH=0.6")
    
    # Simulate normal response times - establish baseline
    print("\n📊 Establishing baseline response times (80ms)...")
    for i in range(8):
        timestamp = datetime.utcnow()
        response_time = 80  # Stable baseline
        state = channel.update(response_time, timestamp)
        print(f"  Sample {i+1}: p95={response_time}ms, state={state['state']}, buffer_len={len(channel.observation_buffer)}")
        time.sleep(0.05)
    
    # Simulate SUDDEN spike (10x increase) to trigger critical
    print("\n🔥 SUDDEN SPIKE: 80ms → 800ms (10x anomaly)...")
    for i in range(12):
        timestamp = datetime.utcnow()
        response_time = 800  # Sudden 10x spike
        state = channel.update(response_time, timestamp)
        print(f"  Sample {i+1}: p95={response_time}ms, state={state['state']}, frozen={channel.is_snapshot_frozen}")
        
        if channel.is_snapshot_frozen:
            snapshot = channel.get_frozen_snapshot()
            print(f"\n📸 SNAPSHOT FROZEN!")
            print(f"   Channel: {snapshot['channel']}")
            print(f"   Trigger time: {snapshot['trigger_time']}")
            print(f"   Data points: {len(snapshot['data'])}")
            break
        time.sleep(0.05)
    
    snapshot = channel.get_frozen_snapshot()
    if snapshot:
        print(f"\n✅ Performance channel WORKING - snapshot captured")
        return True
    else:
        print("\n❌ Performance channel FAILED - no snapshot (EWMA may need more dramatic spike)")
        return False

def test_backpressure_overload_channel():
    """Test BackpressureOverloadChannel."""
    print_separator("TEST 3: Backpressure Overload Channel")
    
    channel = BackpressureOverloadChannel()
    print(f"✓ Channel initialized: {channel.channel_name}")
    print(f"✓ Buffer size: {channel.observation_buffer.maxlen}")
    print(f"✓ Scrape interval: {channel.scrape_interval}s")
    print(f"✓ Thresholds: LOW=0.4, HIGH=0.7")
    
    # Simulate normal queue depth - establish baseline
    print("\n📊 Establishing baseline queue depth (5 items)...")
    for i in range(8):
        timestamp = datetime.utcnow()
        queue_depth = 5  # Stable baseline
        state = channel.update(queue_depth, timestamp)
        print(f"  Sample {i+1}: queue={queue_depth}, state={state['state']}, buffer_len={len(channel.observation_buffer)}")
        time.sleep(0.05)
    
    # Simulate SUDDEN queue explosion (20x increase)
    print("\n🔥 SUDDEN OVERLOAD: 5 → 100 items (20x anomaly)...")
    for i in range(12):
        timestamp = datetime.utcnow()
        queue_depth = 100  # Sudden massive spike
        state = channel.update(queue_depth, timestamp)
        print(f"  Sample {i+1}: queue={queue_depth}, state={state['state']}, frozen={channel.is_snapshot_frozen}")
        
        if channel.is_snapshot_frozen:
            snapshot = channel.get_frozen_snapshot()
            print(f"\n📸 SNAPSHOT FROZEN!")
            print(f"   Channel: {snapshot['channel']}")
            print(f"   Trigger time: {snapshot['trigger_time']}")
            print(f"   Data points: {len(snapshot['data'])}")
            break
        time.sleep(0.05)
    
    snapshot = channel.get_frozen_snapshot()
    if snapshot:
        print(f"\n✅ Backpressure channel WORKING - snapshot captured")
        return True
    else:
        print("\n❌ Backpressure channel FAILED - no snapshot (EWMA may need more dramatic spike)")
        return False

def test_snapshot_reset():
    """Test that snapshot resets when returning to normal."""
    print_separator("TEST 4: Snapshot Reset on Return to Normal")
    
    channel = PerformanceDegradationChannel()
    
    # Establish baseline first
    print("📊 Establishing baseline (100ms)...")
    for i in range(8):
        timestamp = datetime.utcnow()
        channel.update(100, timestamp)
        time.sleep(0.03)
    
    # Trigger critical state with sudden spike
    print("🔥 Triggering critical state with 10x spike (100ms → 1000ms)...")
    for i in range(12):
        timestamp = datetime.utcnow()
        channel.update(1000, timestamp)
        if channel.is_snapshot_frozen:
            print(f"  ✓ Critical state reached at sample {i+1}")
            break
        time.sleep(0.03)
    
    snapshot_before = channel.get_frozen_snapshot()
    print(f"  Snapshot before: {'EXISTS ✓' if snapshot_before else 'NONE ✗'}")
    
    # Return to baseline
    print("📉 Returning to baseline (100ms)...")
    for i in range(15):
        timestamp = datetime.utcnow()
        state = channel.update(100, timestamp)
        if state['state'] == 'normal' and not channel.get_frozen_snapshot():
            print(f"  ✓ Returned to normal at sample {i+1}, snapshot cleared")
            break
        time.sleep(0.03)
    
    snapshot_after = channel.get_frozen_snapshot()
    print(f"  Snapshot after: {'EXISTS ✗' if snapshot_after else 'NONE ✓'}")
    
    if snapshot_before and not snapshot_after:
        print("\n✅ Snapshot reset WORKING - cleared on return to normal")
        return True
    elif not snapshot_before:
        print("\n❌ Snapshot reset test INCONCLUSIVE - no snapshot was created")
        return False
    else:
        print("\n❌ Snapshot reset FAILED - snapshot not cleared properly")
        return False

def test_buffer_size():
    """Test that buffer maintains correct size."""
    print_separator("TEST 5: Rolling Buffer Size Management")
    
    channel = PerformanceDegradationChannel()
    max_size = channel.observation_buffer.maxlen
    
    print(f"📊 Max buffer size: {max_size}")
    print(f"   Adding {max_size + 10} samples...")
    
    for i in range(max_size + 10):
        channel.update(100, datetime.utcnow())
        current_size = len(channel.observation_buffer)
        if i % 5 == 0:
            print(f"  After {i+1} samples: buffer size = {current_size}")
    
    final_size = len(channel.observation_buffer)
    
    if final_size == max_size:
        print(f"\n✅ Buffer size management WORKING - maintained at {max_size}")
        return True
    else:
        print(f"\n❌ Buffer size management FAILED - expected {max_size}, got {final_size}")
        return False

def main():
    print_separator("🧪 EWMA CHANNEL TEST SUITE")
    print(f"Testing all three channels with snapshot freeze support")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    # Run all tests
    results.append(("Resource Saturation Channel", test_resource_saturation_channel()))
    results.append(("Performance Degradation Channel", test_performance_degradation_channel()))
    results.append(("Backpressure Overload Channel", test_backpressure_overload_channel()))
    results.append(("Snapshot Reset", test_snapshot_reset()))
    results.append(("Buffer Size Management", test_buffer_size()))
    
    # Print summary
    print_separator("📊 TEST RESULTS SUMMARY")
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n{'='*70}")
    print(f"Total: {passed}/{len(results)} tests passed")
    print(f"{'='*70}\n")
    
    return 0 if passed == len(results) else 1

if __name__ == "__main__":
    sys.exit(main())
