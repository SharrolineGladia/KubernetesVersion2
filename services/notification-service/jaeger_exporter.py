"""
Jaeger Trace Exporter
Exports traces to Jaeger for visualization
"""
import json
import requests
import time
from typing import Dict, Any, List
from datetime import datetime

class JaegerExporter:
    """Export traces to Jaeger in Zipkin JSON format"""
    
    def __init__(self, jaeger_endpoint: str = "http://localhost:14268/api/traces"):
        self.jaeger_endpoint = jaeger_endpoint
        self.service_name = "notification-service"
        
    def export_span(self, span_dict: Dict[str, Any]) -> bool:
        """
        Export a single span to Jaeger
        """
        try:
            # Convert our span format to Zipkin JSON v2 format (compatible with Jaeger)
            zipkin_span = self._convert_to_zipkin_format(span_dict)
            
            # Send to Jaeger
            response = requests.post(
                self.jaeger_endpoint,
                json=[zipkin_span],
                headers={'Content-Type': 'application/json'},
                timeout=2
            )
            
            return response.status_code == 202 or response.status_code == 200
        except Exception as e:
            print(f"Failed to export span to Jaeger: {e}")
            return False
    
    def export_trace(self, spans: List[Dict[str, Any]]) -> bool:
        """
        Export multiple spans (a complete trace) to Jaeger
        """
        try:
            zipkin_spans = [self._convert_to_zipkin_format(span) for span in spans]
            
            response = requests.post(
                self.jaeger_endpoint,
                json=zipkin_spans,
                headers={'Content-Type': 'application/json'},
                timeout=2
            )
            
            return response.status_code == 202 or response.status_code == 200
        except Exception as e:
            print(f"Failed to export trace to Jaeger: {e}")
            return False
    
    def _convert_to_zipkin_format(self, span: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert our internal span format to Zipkin JSON v2 format
        """
        # Convert timestamps to microseconds
        start_timestamp = int(span['start_time'] * 1_000_000)
        duration = int(span.get('duration', 0) * 1_000_000) if span.get('duration') else None
        
        zipkin_span = {
            "traceId": span['trace_id'].replace('-', '')[:32],  # Zipkin wants hex, max 32 chars
            "id": span['span_id'].replace('-', '')[:16],  # Zipkin span ID, max 16 chars
            "name": span.get('operation_name', 'unknown'),
            "timestamp": start_timestamp,
            "localEndpoint": {
                "serviceName": span.get('service_name', self.service_name)
            },
            "tags": span.get('tags', {}),
        }
        
        # Add parent span if exists
        if span.get('parent_span_id'):
            zipkin_span["parentId"] = span['parent_span_id'].replace('-', '')[:16]
        
        # Add duration if span is finished
        if duration:
            zipkin_span["duration"] = duration
        
        # Add annotations/logs
        if span.get('logs'):
            zipkin_span["annotations"] = []
            for log in span['logs']:
                annotation = {
                    "timestamp": int(log['timestamp'] * 1_000_000),
                    "value": log.get('message', 'log')
                }
                zipkin_span["annotations"].append(annotation)
        
        # Add status as tag
        if span.get('status'):
            zipkin_span["tags"]["status"] = span['status']
        
        return zipkin_span


def test_jaeger_connection(endpoint: str = "http://localhost:14268/api/traces") -> bool:
    """Test if Jaeger is reachable"""
    try:
        # Try to reach Jaeger UI
        ui_endpoint = endpoint.replace(':14268/api/traces', ':16686')
        response = requests.get(ui_endpoint, timeout=2)
        print(f"✓ Jaeger UI is accessible at {ui_endpoint}")
        return True
    except Exception as e:
        print(f"✗ Cannot reach Jaeger: {e}")
        print(f"  Make sure Jaeger is running and accessible")
        return False


if __name__ == "__main__":
    # Test connection
    print("Testing Jaeger connection...")
    test_jaeger_connection()
    
    # Create a test span
    exporter = JaegerExporter()
    test_span = {
        'trace_id': 'test-trace-123',
        'span_id': 'span-456',
        'parent_span_id': None,
        'service_name': 'notification-service',
        'operation_name': 'test_operation',
        'start_time': time.time(),
        'end_time': time.time() + 0.1,
        'duration': 0.1,
        'status': 'ok',
        'tags': {'test': 'true', 'environment': 'demo'},
        'logs': [
            {'timestamp': time.time(), 'message': 'Processing started'},
            {'timestamp': time.time() + 0.05, 'message': 'Processing completed'}
        ]
    }
    
    print("\nSending test span to Jaeger...")
    success = exporter.export_span(test_span)
    if success:
        print("✓ Test span sent successfully!")
        print("  Check Jaeger UI at http://localhost:16686")
    else:
        print("✗ Failed to send test span")
