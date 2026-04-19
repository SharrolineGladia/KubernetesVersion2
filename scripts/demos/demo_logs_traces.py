"""
Quick Demo Script to Generate Logs and Traces
Run this after starting the notification service to generate activity
"""
import time
import requests
import json

SERVICE_URL = "http://localhost:8003"

def print_section(title):
    """Print a section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def send_request(method, endpoint, data=None):
    """Send a request and show the result"""
    url = f"{SERVICE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=5)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=5)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"✓ {method} {endpoint}")
        print(f"  Status: {response.status_code}")
        if response.headers.get('x-trace-id'):
            print(f"  Trace ID: {response.headers['x-trace-id']}")
        
        return response
    except Exception as e:
        print(f"✗ {method} {endpoint}")
        print(f"  Error: {e}")
        return None

def main():
    """Run the demo"""
    print_section("Log & Trace Demo - Generating Activity")
    
    print("This script will generate various requests to create logs and traces.")
    print(f"Make sure the notification service is running on {SERVICE_URL}")
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    input()
    
    # Test 1: Health check
    print_section("1. Health Check Request")
    send_request("GET", "/health")
    time.sleep(1)
    
    # Test 2: Multiple notification requests
    print_section("2. Processing Notifications")
    notifications = [
        {"type": "order_confirmation", "message": "Order #12345 confirmed", "order_id": "12345"},
        {"type": "alert", "message": "System alert: High CPU usage", "priority": "high"},
        {"type": "reminder", "message": "Reminder: Meeting at 3 PM"},
        {"type": "order_confirmation", "message": "Order #67890 confirmed", "order_id": "67890"},
    ]
    
    for notif in notifications:
        send_request("POST", "/notify", notif)
        time.sleep(0.5)
    
    # Test 3: Get metrics
    print_section("3. Fetching Metrics")
    send_request("GET", "/metrics")
    time.sleep(1)
    
    # Test 4: Trigger anomaly (optional)
    print_section("4. Triggering Anomaly (CPU spike)")
    response = send_request("POST", "/anomaly/trigger/cpu")
    if response:
        print("  Waiting 5 seconds for anomaly to be visible...")
        time.sleep(5)
    
    # Test 5: More requests to show distributed tracing
    print_section("5. Rapid Fire Requests (for trace visualization)")
    for i in range(5):
        send_request("POST", "/notify", {
            "type": "alert",
            "message": f"Rapid test {i+1}",
            "test_id": f"test-{i+1}"
        })
        time.sleep(0.2)
    
    # Final summary
    print_section("Demo Complete!")
    print("✓ Generated multiple HTTP requests with tracing")
    print("✓ Created structured JSON logs with trace correlation")
    print("✓ Triggered various operations for visualization")
    print("\n📸 Now you can take screenshots of:")
    print("   1. Console logs (structured JSON with trace_id)")
    print("   2. Jaeger UI at http://localhost:16686")
    print("   3. Service metrics at http://localhost:8003/metrics")
    print("\n💡 To view traces in Jaeger:")
    print("   1. Open http://localhost:16686")
    print("   2. Select 'notification-service' from the dropdown")
    print("   3. Click 'Find Traces'")
    print("   4. Click on any trace to see details")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo cancelled by user.")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        print("Make sure the notification service is running!")
