#!/usr/bin/env python3
"""
Generate heavy load on notification service to trigger real anomalies.
Run from project root: python generate_load.py
"""

import requests
import threading
import time

SERVICE_URL = "http://localhost:8003/health"
NUM_THREADS = 100  # Concurrent threads (increased from 50)
DURATION_SECONDS = 120  # Run for 2 minutes
REQUESTS_PER_SECOND = 200  # Increased from 100

stop_flag = False

def send_requests():
    """Continuously send requests until stop_flag is set."""
    while not stop_flag:
        try:
            # Add CPU-intensive work to stress the service more
            for _ in range(100):
                _ = sum(i**2 for i in range(1000))  # Burn some CPU locally
            response = requests.get(SERVICE_URL, timeout=1)
        except Exception as e:
            pass  # Ignore errors, keep hammering
        time.sleep(1.0 / REQUESTS_PER_SECOND)

def main():
    global stop_flag
    
    print(f"🔥 Starting load generation...")
    print(f"   Target: {SERVICE_URL}")
    print(f"   Threads: {NUM_THREADS}")
    print(f"   Rate: ~{REQUESTS_PER_SECOND * NUM_THREADS} req/sec")
    print(f"   Duration: {DURATION_SECONDS}s")
    print()
    print("⚠️  Watch your detector output for state transitions!")
    print("=" * 60)
    
    # Start threads
    threads = []
    for i in range(NUM_THREADS):
        t = threading.Thread(target=send_requests, daemon=True)
        t.start()
        threads.append(t)
    
    # Run for duration
    start_time = time.time()
    try:
        while time.time() - start_time < DURATION_SECONDS:
            elapsed = int(time.time() - start_time)
            remaining = DURATION_SECONDS - elapsed
            print(f"⏱️  Running... {elapsed}s elapsed, {remaining}s remaining", end='\r')
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    
    # Stop
    stop_flag = True
    print("\n\n✅ Load generation complete!")
    print("   Check detector output for anomalies detected.")

if __name__ == "__main__":
    main()
