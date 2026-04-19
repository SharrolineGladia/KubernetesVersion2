"""
Trace Analyzer for Root Cause Analysis

Integrates distributed traces (Jaeger/OpenTelemetry) into the RCA pipeline.

Capabilities:
- Fetch traces from Jaeger API for specific time windows
- Reconstruct causal timeline of events
- Identify slow operations (high latency spans)
- Detect error propagation across services
- Build service dependency graph from trace data

Integrates with: explainability_layer.py for enhanced RCA

Author: Anomaly Detection System
Date: March 2026
"""

import requests
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import time


@dataclass
class TraceSpan:
    """Represents a single span in a distributed trace."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    service_name: str
    operation_name: str
    start_time: float  # Unix timestamp in seconds
    duration_ms: float
    tags: Dict[str, any]
    logs: List[Dict]
    status: str  # 'ok', 'error'
    
    def is_error(self) -> bool:
        """Check if span has error."""
        return (
            self.status == 'error' or
            self.tags.get('error', False) or
            self.tags.get('http.status_code', 0) >= 400
        )


@dataclass
class TraceSummary:
    """Summary of traces for a time window."""
    time_window_start: datetime
    time_window_end: datetime
    total_traces: int
    error_traces: int
    services_involved: List[str]
    slow_operations: List[Dict]  # Operations exceeding latency threshold
    error_chain: List[Dict]  # Error propagation sequence
    dependency_graph: Dict[str, List[str]]  # service -> [downstream services]
    
    def to_dict(self) -> Dict:
        return {
            'time_window_start': self.time_window_start.isoformat(),
            'time_window_end': self.time_window_end.isoformat(),
            'total_traces': self.total_traces,
            'error_traces': self.error_traces,
            'error_rate': self.error_traces / max(1, self.total_traces),
            'services_involved': self.services_involved,
            'slow_operations': self.slow_operations,
            'error_chain': self.error_chain,
            'dependency_graph': self.dependency_graph
        }


class TraceAnalyzer:
    """
    Analyzes distributed traces from Jaeger to enhance RCA.
    
    Workflow:
        1. Fetch traces for anomaly time window from Jaeger API
        2. Parse spans and reconstruct call chains
        3. Identify slow operations and errors
        4. Build service dependency graph
        5. Create causal timeline
    """
    
    def __init__(
        self,
        jaeger_query_url: str = "http://localhost:16686",
        slow_threshold_ms: float = 1000.0
    ):
        """
        Initialize trace analyzer.
        
        Args:
            jaeger_query_url: Jaeger query service URL
            slow_threshold_ms: Latency threshold for "slow" operations
        """
        self.jaeger_query_url = jaeger_query_url.rstrip('/')
        self.slow_threshold_ms = slow_threshold_ms
    
    def analyze_time_window(
        self,
        start_time: datetime,
        end_time: datetime,
        service_name: Optional[str] = None
    ) -> TraceSummary:
        """
        Analyze traces for a specific time window.
        
        Args:
            start_time: Start of time window
            end_time: End of time window
            service_name: Optional service to focus on
        
        Returns:
            TraceSummary with analysis results
        """
        # Fetch traces from Jaeger
        traces = self._fetch_traces(start_time, end_time, service_name)
        
        if not traces:
            return TraceSummary(
                time_window_start=start_time,
                time_window_end=end_time,
                total_traces=0,
                error_traces=0,
                services_involved=[],
                slow_operations=[],
                error_chain=[],
                dependency_graph={}
            )
        
        # Parse and analyze traces
        all_spans = []
        error_traces = 0
        services = set()
        
        for trace in traces:
            spans = self._parse_trace(trace)
            all_spans.extend(spans)
            
            # Track services
            for span in spans:
                services.add(span.service_name)
            
            # Count errors
            if any(span.is_error() for span in spans):
                error_traces += 1
        
        # Identify slow operations
        slow_operations = self._identify_slow_operations(all_spans)
        
        # Reconstruct error propagation chain
        error_chain = self._reconstruct_error_chain(all_spans)
        
        # Build service dependency graph
        dependency_graph = self._build_dependency_graph(all_spans)
        
        return TraceSummary(
            time_window_start=start_time,
            time_window_end=end_time,
            total_traces=len(traces),
            error_traces=error_traces,
            services_involved=list(services),
            slow_operations=slow_operations,
            error_chain=error_chain,
            dependency_graph=dependency_graph
        )
    
    def _fetch_traces(
        self,
        start_time: datetime,
        end_time: datetime,
        service_name: Optional[str] = None
    ) -> List[Dict]:
        """
        Fetch traces from Jaeger API.
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            service_name: Optional service filter
        
        Returns:
            List of trace JSON objects from Jaeger
        """
        # Convert to microseconds (Jaeger API format)
        start_us = int(start_time.timestamp() * 1_000_000)
        end_us = int(end_time.timestamp() * 1_000_000)
        
        # Build query parameters
        params = {
            'start': start_us,
            'end': end_us,
            'limit': 100  # Fetch up to 100 traces
        }
        
        if service_name:
            params['service'] = service_name
        
        try:
            # Query Jaeger API
            url = f"{self.jaeger_query_url}/api/traces"
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            traces = data.get('data', [])
            
            return traces
        
        except requests.RequestException as e:
            print(f"⚠️  Failed to fetch traces from Jaeger: {e}")
            print(f"   URL: {url}")
            print(f"   Make sure Jaeger is running on {self.jaeger_query_url}")
            return []
        except Exception as e:
            print(f"⚠️  Error parsing Jaeger response: {e}")
            return []
    
    def _parse_trace(self, trace_json: Dict) -> List[TraceSpan]:
        """
        Parse a single trace JSON into TraceSpan objects.
        
        Args:
            trace_json: Trace data from Jaeger API
        
        Returns:
            List of TraceSpan objects
        """
        spans = []
        
        # Jaeger format: trace has 'spans' array
        for span_data in trace_json.get('spans', []):
            # Extract basic fields
            trace_id = span_data.get('traceID', '')
            span_id = span_data.get('spanID', '')
            
            # Parent span (references array)
            parent_span_id = None
            for ref in span_data.get('references', []):
                if ref.get('refType') == 'CHILD_OF':
                    parent_span_id = ref.get('spanID')
                    break
            
            # Service and operation
            service_name = span_data.get('process', {}).get('serviceName', 'unknown')
            operation_name = span_data.get('operationName', 'unknown')
            
            # Timing
            start_time = span_data.get('startTime', 0) / 1_000_000  # Convert µs to seconds
            duration_ms = span_data.get('duration', 0) / 1000  # Convert µs to ms
            
            # Tags
            tags = {}
            for tag in span_data.get('tags', []):
                tags[tag['key']] = tag['value']
            
            # Logs/events
            logs = span_data.get('logs', [])
            
            # Status
            status = 'error' if tags.get('error') or tags.get('http.status_code', 0) >= 400 else 'ok'
            
            span = TraceSpan(
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                service_name=service_name,
                operation_name=operation_name,
                start_time=start_time,
                duration_ms=duration_ms,
                tags=tags,
                logs=logs,
                status=status
            )
            
            spans.append(span)
        
        return spans
    
    def _identify_slow_operations(self, spans: List[TraceSpan]) -> List[Dict]:
        """
        Identify operations that exceeded latency threshold.
        
        Returns:
            List of dicts with slow operation details
        """
        slow_ops = []
        
        for span in spans:
            if span.duration_ms > self.slow_threshold_ms:
                slow_ops.append({
                    'service': span.service_name,
                    'operation': span.operation_name,
                    'duration_ms': span.duration_ms,
                    'threshold_ms': self.slow_threshold_ms,
                    'slowdown_factor': span.duration_ms / self.slow_threshold_ms,
                    'trace_id': span.trace_id
                })
        
        # Sort by duration (slowest first)
        slow_ops.sort(key=lambda x: x['duration_ms'], reverse=True)
        
        return slow_ops
    
    def _reconstruct_error_chain(self, spans: List[TraceSpan]) -> List[Dict]:
        """
        Reconstruct error propagation chain.
        
        Shows: Service A failed → Service B timed out → Service C returned error
        
        Returns:
            List of error events in chronological order
        """
        # Filter error spans
        error_spans = [s for s in spans if s.is_error()]
        
        if not error_spans:
            return []
        
        # Sort by start time
        error_spans.sort(key=lambda s: s.start_time)
        
        # Build error chain
        error_chain = []
        for span in error_spans:
            error_chain.append({
                'timestamp': datetime.fromtimestamp(span.start_time).isoformat(),
                'service': span.service_name,
                'operation': span.operation_name,
                'error_type': span.tags.get('error.type', 'unknown'),
                'error_message': span.tags.get('error.message', 'No message'),
                'http_status': span.tags.get('http.status_code'),
                'trace_id': span.trace_id,
                'span_id': span.span_id
            })
        
        return error_chain
    
    def _build_dependency_graph(self, spans: List[TraceSpan]) -> Dict[str, List[str]]:
        """
        Build service dependency graph from traces.
        
        Returns:
            Dict mapping service -> list of downstream services it calls
        """
        dependencies = {}
        
        # Group spans by trace
        traces = {}
        for span in spans:
            if span.trace_id not in traces:
                traces[span.trace_id] = []
            traces[span.trace_id].append(span)
        
        # For each trace, find parent-child relationships
        for trace_id, trace_spans in traces.items():
            # Build span lookup
            span_map = {s.span_id: s for s in trace_spans}
            
            # Find dependencies
            for span in trace_spans:
                if span.parent_span_id and span.parent_span_id in span_map:
                    parent_span = span_map[span.parent_span_id]
                    
                    # Parent service calls child service
                    if parent_span.service_name != span.service_name:
                        if parent_span.service_name not in dependencies:
                            dependencies[parent_span.service_name] = set()
                        dependencies[parent_span.service_name].add(span.service_name)
        
        # Convert sets to lists
        return {k: list(v) for k, v in dependencies.items()}
    
    def get_trace_context_for_rca(
        self,
        anomaly_timestamp: datetime,
        window_minutes: int = 5,
        service_name: Optional[str] = None
    ) -> Dict:
        """
        Get trace context for RCA report.
        
        Args:
            anomaly_timestamp: When anomaly was detected
            window_minutes: How many minutes before anomaly to analyze
            service_name: Optional service filter
        
        Returns:
            Dict with trace insights for RCA
        """
        start_time = anomaly_timestamp - timedelta(minutes=window_minutes)
        end_time = anomaly_timestamp
        
        summary = self.analyze_time_window(start_time, end_time, service_name)
        
        # Format for RCA integration
        rca_context = {
            'has_trace_data': summary.total_traces > 0,
            'error_rate_from_traces': summary.error_traces / max(1, summary.total_traces),
            'services_involved': summary.services_involved,
            'causal_timeline': self._format_causal_timeline(summary),
            'slow_operations': summary.slow_operations[:5],  # Top 5 slowest
            'error_propagation': summary.error_chain,
            'error_chain': summary.error_chain,
            'service_dependencies': summary.dependency_graph
        }
        
        return rca_context
    
    def _format_causal_timeline(self, summary: TraceSummary) -> List[Dict]:
        """
        Format timeline showing what happened leading to anomaly.
        
        Combines slow operations and errors in chronological order.
        """
        timeline = []
        
        # Add slow operations
        for slow_op in summary.slow_operations:
            timeline.append({
                'timestamp': 'from_trace',  # Would need to extract from trace
                'event_type': 'performance_degradation',
                'service': slow_op['service'],
                'operation': slow_op['operation'],
                'details': f"Slow operation: {slow_op['duration_ms']:.0f}ms (threshold: {slow_op['threshold_ms']:.0f}ms)"
            })
        
        # Add errors
        for error in summary.error_chain:
            timeline.append({
                'timestamp': error['timestamp'],
                'event_type': 'error',
                'service': error['service'],
                'operation': error['operation'],
                'details': error.get('error_message', 'Error occurred')
            })
        
        # Sort by timestamp (if available)
        timeline.sort(key=lambda x: x.get('timestamp', ''))
        
        return timeline


# Convenience function for integration
def get_trace_insights(
    anomaly_timestamp: datetime,
    jaeger_url: str = "http://localhost:16686",
    service_name: Optional[str] = None
) -> Dict:
    """
    Quick function to get trace insights for RCA.
    
    Usage in RCA:
        trace_insights = get_trace_insights(
            anomaly_timestamp=datetime.now(),
            service_name='notification-service'
        )
        
        if trace_insights['has_trace_data']:
            print(f"Error rate: {trace_insights['error_rate_from_traces']}")
            print(f"Slow ops: {trace_insights['slow_operations']}")
    """
    analyzer = TraceAnalyzer(jaeger_url)
    return analyzer.get_trace_context_for_rca(anomaly_timestamp, service_name=service_name)


if __name__ == "__main__":
    print("Trace Analyzer Module")
    print("Import this module to use TraceAnalyzer class")
    print()
    print("Example usage:")
    print("  from trace_analyzer import get_trace_insights")
    print("  insights = get_trace_insights(datetime.now(), service_name='notification-service')")
