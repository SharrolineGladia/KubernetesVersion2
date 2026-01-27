"""
Lightweight Distributed Tracing Utility
Provides trace ID propagation and span tracking for the microservices
"""
import uuid
import time
import json
import threading
from datetime import datetime
from typing import Optional, Dict, Any

# Global trace storage
_traces = {}
_trace_lock = threading.Lock()

class TraceSpan:
    """Represents a single span in a distributed trace"""
    
    def __init__(self, trace_id: str, span_id: str, parent_span_id: Optional[str] = None,
                 service_name: str = "", operation_name: str = ""):
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self.service_name = service_name
        self.operation_name = operation_name
        self.start_time = time.time()
        self.end_time = None
        self.duration = None
        self.status = "active"
        self.tags = {}
        self.logs = []
        
    def set_tag(self, key: str, value: Any):
        """Add a tag to the span"""
        self.tags[key] = value
        
    def log(self, message: str, **kwargs):
        """Add a log entry to the span"""
        log_entry = {
            'timestamp': time.time(),
            'message': message,
            **kwargs
        }
        self.logs.append(log_entry)
        
    def finish(self, status: str = "ok"):
        """Finish the span"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.status = status
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary"""
        return {
            'trace_id': self.trace_id,
            'span_id': self.span_id,
            'parent_span_id': self.parent_span_id,
            'service_name': self.service_name,
            'operation_name': self.operation_name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'status': self.status,
            'tags': self.tags,
            'logs': self.logs
        }

class DistributedTracer:
    """Lightweight distributed tracer"""
    
    @staticmethod
    def generate_trace_id() -> str:
        """Generate a new trace ID"""
        return str(uuid.uuid4())
    
    @staticmethod
    def generate_span_id() -> str:
        """Generate a new span ID"""
        return str(uuid.uuid4())[:8]
    
    @staticmethod
    def start_span(service_name: str, operation_name: str, 
                   trace_id: Optional[str] = None, 
                   parent_span_id: Optional[str] = None) -> TraceSpan:
        """Start a new span"""
        if trace_id is None:
            trace_id = DistributedTracer.generate_trace_id()
        
        span_id = DistributedTracer.generate_span_id()
        span = TraceSpan(trace_id, span_id, parent_span_id, service_name, operation_name)
        
        with _trace_lock:
            if trace_id not in _traces:
                _traces[trace_id] = {}
            _traces[trace_id][span_id] = span
        
        return span
    
    @staticmethod
    def get_trace(trace_id: str) -> Optional[Dict[str, TraceSpan]]:
        """Get all spans for a trace"""
        with _trace_lock:
            return _traces.get(trace_id, {}).copy()
    
    @staticmethod
    def get_all_traces() -> Dict[str, Dict[str, TraceSpan]]:
        """Get all traces"""
        with _trace_lock:
            return {tid: {sid: span for sid, span in spans.items()} 
                   for tid, spans in _traces.items()}
    
    @staticmethod
    def extract_trace_context(headers: Dict[str, str]) -> tuple[Optional[str], Optional[str]]:
        """Extract trace context from HTTP headers"""
        trace_id = headers.get('x-trace-id')
        parent_span_id = headers.get('x-span-id')
        return trace_id, parent_span_id
    
    @staticmethod
    def inject_trace_context(headers: Dict[str, str], trace_id: str, span_id: str):
        """Inject trace context into HTTP headers"""
        headers['x-trace-id'] = trace_id
        headers['x-span-id'] = span_id
    
    @staticmethod
    def export_traces_to_file(filename: str = None):
        """Export all traces to a JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"traces_{timestamp}.json"
        
        with _trace_lock:
            export_data = {}
            for trace_id, spans in _traces.items():
                export_data[trace_id] = {
                    span_id: span.to_dict() 
                    for span_id, span in spans.items()
                }
        
        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            return filename
        except Exception as e:
            print(f"Error exporting traces: {e}")
            return None
    
    @staticmethod
    def clear_traces():
        """Clear all stored traces (useful for testing)"""
        with _trace_lock:
            _traces.clear()

# Context-local storage for current span
_current_span = threading.local()

def get_current_span() -> Optional[TraceSpan]:
    """Get the current active span in this thread"""
    return getattr(_current_span, 'span', None)

def set_current_span(span: Optional[TraceSpan]):
    """Set the current active span in this thread"""
    _current_span.span = span

class SpanContext:
    """Context manager for spans"""
    
    def __init__(self, span: TraceSpan):
        self.span = span
        self.previous_span = None
    
    def __enter__(self):
        self.previous_span = get_current_span()
        set_current_span(self.span)
        return self.span
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.span.set_tag("error", True)
            self.span.set_tag("error.type", exc_type.__name__)
            self.span.set_tag("error.message", str(exc_val))
            self.span.finish("error")
        else:
            self.span.finish("ok")
        set_current_span(self.previous_span)

def trace_function(service_name: str, operation_name: str):
    """Decorator to automatically trace a function"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            current = get_current_span()
            parent_span_id = current.span_id if current else None
            trace_id = current.trace_id if current else None
            
            span = DistributedTracer.start_span(
                service_name, operation_name, trace_id, parent_span_id
            )
            
            with SpanContext(span):
                span.set_tag("function.name", func.__name__)
                span.set_tag("function.args_count", len(args))
                return func(*args, **kwargs)
        
        return wrapper
    return decorator