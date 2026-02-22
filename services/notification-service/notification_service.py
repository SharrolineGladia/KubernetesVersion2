import time
import json
import threading
import psutil
import os
import sys
import random
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# FastAPI imports for HTTP endpoints
from fastapi import FastAPI, HTTPException, Request
from starlette.responses import Response

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import distributed tracing
from tracing_utils import DistributedTracer, SpanContext, get_current_span, set_jaeger_exporter

# Try to import Jaeger exporter (optional)
try:
    from jaeger_exporter import JaegerExporter
    JAEGER_ENABLED = True
except ImportError:
    JAEGER_ENABLED = False
    print("Note: Jaeger exporter not available. Install 'requests' package to enable.")

# Simple logging function
def log_message(level, message, **kwargs):
    """Simple logging without external dependencies"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_data = {
        'timestamp': timestamp,
        'service': 'notification-service',
        'level': level,
        'message': message,
        'pid': os.getpid(),
    }
    
    # Add trace context if available
    current_span = get_current_span()
    if current_span:
        log_data['trace_id'] = current_span.trace_id
        log_data['span_id'] = current_span.span_id
    
    # Add additional context
    log_data.update(kwargs)
    
    print(json.dumps(log_data))

# Prometheus metrics
NOTIFICATIONS_PROCESSED = Counter('notification_service_processed_total', 'Notifications processed', ['type', 'status'])
PROCESSING_DURATION = Histogram('notification_service_duration_seconds', 'Processing duration')
QUEUE_SIZE = Gauge('notification_service_queue_size', 'Queue size')
WORKER_THREADS = Gauge('notification_service_workers', 'Active worker threads')
CPU_USAGE = Gauge('notification_service_cpu_percent', 'CPU usage percentage')
MEMORY_USAGE = Gauge('notification_service_memory_mb', 'Memory usage in MB')

# ROOT CAUSE ANALYSIS METRICS
MESSAGE_RATE = Gauge('notification_service_message_rate_per_sec', 'Messages processed per second')
DELIVERY_SUCCESS_RATE = Gauge('notification_service_delivery_success_rate', 'Successful delivery rate (0-1)')
THREAD_COUNT = Gauge('notification_service_thread_count', 'Number of active threads')
EXTERNAL_API_HEALTH = Gauge('notification_service_external_api_health', 'External API health (1=ok, 0=fail)')
QUEUE_DEPTH = Gauge('notification_service_internal_queue_depth', 'Internal queue depth')
ERROR_RATE = Gauge('notification_service_error_rate', 'Current error rate')
RESPONSE_TIME_P95 = Gauge('notification_service_response_time_p95_ms', '95th percentile response time')

# Global variables
try:
    process = psutil.Process()
except:
    process = None

running = True
cpu_spike_active = False

# Baseline / anomaly controls
memory_leak_active = False
thread_spike_active = False

# Experiment modes (mutually exclusive)
stress_mode_enabled = False
critical_mode_enabled = False

_mode_lock = threading.Lock()

# Memory allocators (used to create small baseline variation and leak anomalies)
_baseline_memory_chunks = []
_leak_memory_chunks = []
_memory_lock = threading.Lock()

# Prevent psutil sampling from being called too frequently from multiple threads.
_metrics_sample_lock = threading.Lock()
_last_metrics_sample_time = 0.0

# Stress mode state (real threads + moderate memory)
_stress_threads = []
_stress_thread_stop_events = []
_stress_memory_chunks = []

# Simulate message queue (in-memory)
message_queue = []
queue_lock = threading.Lock()
processed_messages_timestamps = []

# FastAPI app for HTTP endpoints  
app = FastAPI(title="Notification Service")

# Anomaly control flags
leak_active = False
crash_enabled = False


def memory_baseline_variation_loop():
    """Create small, realistic memory usage fluctuations during normal operation."""
    global _baseline_memory_chunks

    # Target a small rolling footprint so RSS wiggles a few MB.
    # Example: 60 chunks * ~80KB ≈ 4.8MB.
    target_chunks = 60

    while running:
        try:
            if stress_mode_enabled or critical_mode_enabled:
                time.sleep(1)
                continue

            if memory_leak_active:
                time.sleep(1)
                continue

            chunk_kb = random.randint(50, 100)
            chunk = bytearray(chunk_kb * 1024)

            with _memory_lock:
                _baseline_memory_chunks.append(chunk)

                # Keep a rolling window; occasionally drop a few older chunks.
                if len(_baseline_memory_chunks) > target_chunks and random.random() < 0.7:
                    drop = random.randint(1, 5)
                    del _baseline_memory_chunks[:drop]

            time.sleep(0.7)

        except Exception as e:
            log_message('ERROR', 'Memory baseline variation error', error=str(e))
            time.sleep(1)


def memory_leak_loop():
    """Allocate large chunks while memory_leak_active is True (no frees while active)."""
    global _leak_memory_chunks

    while running:
        try:
            if not memory_leak_active:
                time.sleep(0.5)
                continue

            # ~1MB allocations in bursts to create step-like growth between scrapes.
            burst = 5
            new_chunks = [bytearray(1024 * 1024) for _ in range(burst)]
            with _memory_lock:
                _leak_memory_chunks.extend(new_chunks)

            time.sleep(0.15)

        except Exception as e:
            log_message('ERROR', 'Memory leak simulation error', error=str(e))
            time.sleep(1)


def _short_lived_thread_worker(lifetime_seconds: float = 5.0):
    try:
        time.sleep(lifetime_seconds)
    except Exception:
        pass


def thread_baseline_variation_loop():
    """Occasionally spawn short-lived threads to create small thread-count variation."""
    while running:
        try:
            if stress_mode_enabled or critical_mode_enabled:
                time.sleep(1)
                continue

            if thread_spike_active:
                time.sleep(1)
                continue

            # Low, steady rate: most of the time do nothing; sometimes spawn 1 thread.
            if random.random() < 0.35:
                t = threading.Thread(target=_short_lived_thread_worker, args=(5.0,), daemon=True)
                t.start()

            time.sleep(2.0)

        except Exception as e:
            log_message('ERROR', 'Thread baseline variation error', error=str(e))
            time.sleep(1)


def _long_lived_thread_worker(lifetime_seconds: float = 30.0):
    try:
        time.sleep(lifetime_seconds)
    except Exception:
        pass


def thread_spike_loop():
    """When thread_spike_active, spawn long-lived threads faster than they terminate."""
    while running:
        try:
            if not thread_spike_active:
                time.sleep(0.5)
                continue

            # Spawn in bursts to create step-like increases that are easy to detect.
            threads_per_burst = 15
            lifetime_seconds = 60.0
            for _ in range(threads_per_burst):
                t = threading.Thread(target=_long_lived_thread_worker, args=(lifetime_seconds,), daemon=True)
                t.start()

            time.sleep(1.0)

        except Exception as e:
            log_message('ERROR', 'Thread spike simulation error', error=str(e))
            time.sleep(1)


def _set_exclusive_mode(mode_name: str | None):
    """Ensure only one injection mode is active at a time.

    mode_name can be one of: None, 'stress', 'critical', 'cpu_spike', 'memory_leak', 'thread_spike'.
    """
    global stress_mode_enabled
    global critical_mode_enabled
    global cpu_spike_active
    global memory_leak_active
    global thread_spike_active

    with _mode_lock:
        stress_mode_enabled = (mode_name == 'stress')
        critical_mode_enabled = (mode_name == 'critical')
        cpu_spike_active = (mode_name == 'cpu_spike')
        memory_leak_active = (mode_name == 'memory_leak')
        thread_spike_active = (mode_name == 'thread_spike')


def _stress_idle_worker(stop_event: threading.Event):
    while running and not stop_event.is_set():
        time.sleep(0.25)


def stress_mode_loop():
    """Moderate, sustained degradation: gradual threads/memory, moderate CPU.

    Designed to push stress_score above LOW_STRESS_LEVEL but below ANOMALY_THRESHOLD.
    When disabled, resources decay gradually (no abrupt reset).
    """
    global _stress_threads
    global _stress_thread_stop_events
    global _stress_memory_chunks

    # Sustained elevation with minimal oscillation.
    # Create a step-change that persists long enough for detector to recognize it.
    # No rapid cycling - maintain elevated state for ~60 seconds to ensure detection.
    
    # Target: Moderate sustained load that crosses stress threshold
    # Threads: ~35-40 (above baseline but not extreme)
    # Memory: ~70-90MB (crosses the 100MB soft limit at ~70-90%)
    target_threads = 38
    thread_adjust_step = 5

    target_mem_chunks = 80  # ~80MB sustained
    mem_adjust_step = 8

    while running:
        try:
            if stress_mode_enabled:
                # Ramp up to target levels and maintain
                if len(_stress_threads) < target_threads:
                    to_add = min(thread_adjust_step, target_threads - len(_stress_threads))
                    for _ in range(to_add):
                        stop_event = threading.Event()
                        t = threading.Thread(target=_stress_idle_worker, args=(stop_event,), daemon=True)
                        t.start()
                        _stress_threads.append(t)
                        _stress_thread_stop_events.append(stop_event)

                # Ramp up memory to target and maintain
                with _memory_lock:
                    if len(_stress_memory_chunks) < target_mem_chunks:
                        to_add = min(mem_adjust_step, target_mem_chunks - len(_stress_memory_chunks))
                        _stress_memory_chunks.extend([bytearray(1024 * 1024) for _ in range(to_add)])

                # Once at target, maintain stable count with minimal variation
                if len(_stress_threads) >= target_threads:
                    # Remove excess threads if we've gone over target
                    while len(_stress_threads) > target_threads:
                        stop_event = _stress_thread_stop_events.pop()
                        stop_event.set()
                        _stress_threads.pop()
                
                # Maintain memory at target
                with _memory_lock:
                    if len(_stress_memory_chunks) > target_mem_chunks:
                        excess = len(_stress_memory_chunks) - target_mem_chunks
                        del _stress_memory_chunks[:excess]

                time.sleep(1.5)

            else:
                # Gradual decay: stop at most one stress thread per cycle.
                if _stress_thread_stop_events:
                    stop_event = _stress_thread_stop_events.pop(0)
                    stop_event.set()
                    _stress_threads.pop(0)

                # Gradual decay: release a few memory chunks per cycle.
                with _memory_lock:
                    if _stress_memory_chunks:
                        del _stress_memory_chunks[:3]

                time.sleep(1.0)

        except Exception as e:
            log_message('ERROR', 'Stress mode loop error', error=str(e))
            time.sleep(1)


def critical_mode_loop():
    """Failure-risk conditions: sustained CPU burn, aggressive memory growth, and thread explosion."""
    global _leak_memory_chunks
    global _stress_threads
    global _stress_thread_stop_events

    # Cap memory to avoid crashing the container but make it severe
    max_leak_chunks = 250  # 250MB
    max_threads = 70  # High thread count

    while running:
        try:
            if not critical_mode_enabled:
                time.sleep(0.5)
                continue

            # Aggressive memory growth in bursts; bounded but high
            with _memory_lock:
                if len(_leak_memory_chunks) < max_leak_chunks:
                    burst = min(15, max_leak_chunks - len(_leak_memory_chunks))
                    _leak_memory_chunks.extend([bytearray(1024 * 1024) for _ in range(burst)])

            # Also spawn many threads during critical
            if len(_stress_threads) < max_threads:
                to_add = min(8, max_threads - len(_stress_threads))
                for _ in range(to_add):
                    stop_event = threading.Event()
                    t = threading.Thread(target=_stress_idle_worker, args=(stop_event,), daemon=True)
                    t.start()
                    _stress_threads.append(t)
                    _stress_thread_stop_events.append(stop_event)

            time.sleep(0.15)

        except Exception as e:
            log_message('ERROR', 'Critical mode loop error', error=str(e))
            time.sleep(1)

def update_system_metrics():
    """Update system metrics"""
    try:
        # Throttle psutil sampling so multiple worker threads don't create
        # extremely short sampling intervals (which can lead to huge cpu_percent values).
        global _last_metrics_sample_time
        now = time.time()

        with _metrics_sample_lock:
            if process and (now - _last_metrics_sample_time) >= 0.9:
                cpu_percent = process.cpu_percent()
                memory_mb = process.memory_info().rss / (1024 * 1024)

                CPU_USAGE.set(cpu_percent)
                MEMORY_USAGE.set(memory_mb)

                # RCA METRICS: Thread count
                try:
                    THREAD_COUNT.set(process.num_threads())
                except:
                    THREAD_COUNT.set(0)

                _last_metrics_sample_time = now
        
        with queue_lock:
            queue_size = len(message_queue)
        QUEUE_SIZE.set(queue_size)
        
        # RCA METRICS: Calculate notification metrics
        current_time = time.time()
        
        # Message processing rate
        global processed_messages_timestamps
        if 'processed_messages_timestamps' not in globals():
            processed_messages_timestamps = []
        
        recent_messages = [t for t in processed_messages_timestamps if current_time - t < 60]
        message_rate = len(recent_messages) / 60.0
        MESSAGE_RATE.set(message_rate)
        
        # Delivery success rate simulation
        success_rate = 0.99 if queue_size < 50 else max(0.8, 0.99 - (queue_size / 1000))
        DELIVERY_SUCCESS_RATE.set(success_rate)
        
        # External API health simulation
        api_health = 1 if queue_size < 100 and success_rate > 0.95 else 0
        EXTERNAL_API_HEALTH.set(api_health)
        
        # Queue depth (backlog)
        queue_depth = max(0, queue_size - 10)  # Backlog beyond normal capacity
        QUEUE_DEPTH.set(queue_depth)
        
        # Error rate
        error_rate = 1.0 - success_rate
        ERROR_RATE.set(error_rate)
        
        # Response time P95 simulation
        base_time = 30 + (queue_size * 2)  # Base 30ms + 2ms per queue item
        p95_time = base_time * 1.3  # P95 is ~30% higher
        RESPONSE_TIME_P95.set(p95_time)
        
    except Exception as e:
        log_message('ERROR', 'Failed to update metrics', error=str(e))

# HTTP endpoints
@app.middleware("http")
async def tracing_middleware(request: Request, call_next):
    """Middleware for distributed tracing"""
    # Extract trace context from request headers
    trace_id, parent_span_id = DistributedTracer.extract_trace_context(dict(request.headers))
    
    # Start a new span for this request
    operation_name = f"{request.method} {request.url.path}"
    span = DistributedTracer.start_span(
        service_name="notification-service", 
        operation_name=operation_name,
        trace_id=trace_id,
        parent_span_id=parent_span_id
    )
    
    # Add request details to span
    span.set_tag("http.method", request.method)
    span.set_tag("http.url", str(request.url))
    span.set_tag("http.path", request.url.path)
    
    try:
        with SpanContext(span):
            response = await call_next(request)
            
            # Add response details to span
            span.set_tag("http.status_code", response.status_code)
            
            # Inject trace context into response headers
            DistributedTracer.inject_trace_context(response.headers, span.trace_id, span.span_id)
            
            return response
            
    except Exception as e:
        span.set_tag("error", True)
        span.set_tag("error.message", str(e))
        span.log("request_exception", error=str(e))
        raise

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "notification-service"}

@app.get("/metrics")
async def metrics():
    from prometheus_client import generate_latest
    return Response(generate_latest(), media_type="text/plain")

@app.post("/send-notification")
async def send_notification(notification_data: dict):
    """Send notification endpoint with tracing"""
    current_span = get_current_span()
    
    try:
        notification_id = notification_data.get("order_id", f"notification_{int(time.time())}")
        
        if current_span:
            current_span.set_tag("notification.id", notification_id)
            current_span.set_tag("notification.type", notification_data.get("type", "unknown"))
            current_span.log("notification_received", notification_id=notification_id)
        
        # Add to processing queue
        with queue_lock:
            message_queue.append({
                **notification_data,
                "received_at": time.time(),
                "trace_id": current_span.trace_id if current_span else None,
                "span_id": current_span.span_id if current_span else None
            })
        
        if current_span:
            current_span.log("notification_queued", notification_id=notification_id)
        
        log_message('INFO', 'Notification queued', 
                   notification_id=notification_id,
                   type=notification_data.get("type"))
        
        return {"status": "queued", "notification_id": notification_id}
        
    except Exception as e:
        if current_span:
            current_span.log("notification_failed", error=str(e))
        
        log_message('ERROR', 'Notification processing failed', error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/queue-status")
async def get_queue_status():
    """Get queue status"""
    with queue_lock:
        queue_size = len(message_queue)
    
    return {"queue_size": queue_size}

@app.post("/simulate-cpu-spike")
async def start_cpu_spike():
    """Start CPU spike simulation"""
    _set_exclusive_mode('cpu_spike')
    log_message('WARNING', 'CPU spike simulation started')
    return {"status": "cpu_spike_started"}

@app.post("/stop-cpu-spike")
async def stop_cpu_spike():
    """Stop CPU spike simulation"""
    _set_exclusive_mode(None)
    log_message('INFO', 'CPU spike simulation stopped')
    return {"status": "cpu_spike_stopped"}


@app.post("/simulate-memory-leak")
async def start_memory_leak():
    """Start memory leak simulation"""
    _set_exclusive_mode('memory_leak')
    log_message('WARNING', 'Memory leak simulation started')
    return {"status": "memory_leak_started"}


@app.post("/stop-memory-leak")
async def stop_memory_leak():
    """Stop memory leak simulation and release leaked chunks"""
    global _leak_memory_chunks
    _set_exclusive_mode(None)

    # Once disabled, allow memory to return to baseline by dropping references.
    with _memory_lock:
        _leak_memory_chunks = []

    log_message('INFO', 'Memory leak simulation stopped')
    return {"status": "memory_leak_stopped"}


@app.post("/simulate-thread-spike")
async def start_thread_spike():
    """Start thread spike simulation"""
    _set_exclusive_mode('thread_spike')
    log_message('WARNING', 'Thread spike simulation started')
    return {"status": "thread_spike_started"}


@app.post("/stop-thread-spike")
async def stop_thread_spike():
    """Stop thread spike simulation"""
    _set_exclusive_mode(None)
    log_message('INFO', 'Thread spike simulation stopped')
    return {"status": "thread_spike_stopped"}


@app.post("/simulate-stress")
async def start_stress_mode():
    """Moderate, sustained degradation (should reach stressed, not critical)."""
    _set_exclusive_mode('stress')
    log_message('WARNING', 'Stress mode started')
    return {"status": "stress_started"}


@app.post("/stop-stress")
async def stop_stress_mode():
    """Stop moderate stress mode (decays naturally)."""
    _set_exclusive_mode(None)
    log_message('INFO', 'Stress mode stopped')
    return {"status": "stress_stopped"}


@app.post("/simulate-critical")
async def start_critical_mode():
    """Severe failure-risk conditions (should reach critical with persistence)."""
    _set_exclusive_mode('critical')
    log_message('WARNING', 'Critical mode started')
    return {"status": "critical_started"}


@app.post("/stop-critical")
async def stop_critical_mode():
    """Stop critical mode and allow metrics to stabilize."""
    global _leak_memory_chunks
    _set_exclusive_mode(None)

    # Release the critical-mode allocations gradually by dropping references.
    with _memory_lock:
        _leak_memory_chunks = []

    log_message('INFO', 'Critical mode stopped')
    return {"status": "critical_stopped"}


@app.get("/mode-status")
async def mode_status():
    """Debug endpoint to confirm which injection mode is active and current injected levels."""
    with _memory_lock:
        stress_mem_mb = round(len(_stress_memory_chunks) * 1.0, 1)
        leak_mem_mb = round(len(_leak_memory_chunks) * 1.0, 1)

    return {
        "stress_mode_enabled": stress_mode_enabled,
        "critical_mode_enabled": critical_mode_enabled,
        "cpu_spike_active": cpu_spike_active,
        "memory_leak_active": memory_leak_active,
        "thread_spike_active": thread_spike_active,
        "total_threads": len(_stress_threads),
        "stress_memory_mb_allocated": stress_mem_mb,
        "leak_memory_mb_allocated": leak_mem_mb,
        "mode": "critical" if critical_mode_enabled else ("stress" if stress_mode_enabled else "normal"),
        "targets": {
            "stress_threads": 38,
            "stress_memory_mb": 80,
            "critical_threads": 70,
            "critical_memory_mb": 250
        }
    }

def simulate_cpu_spike():
    """CPU spike simulation thread"""
    global cpu_spike_active
    while running:
        try:
            if cpu_spike_active or critical_mode_enabled:
                # Aggressive sustained CPU burn for critical mode
                # Burn continuously for 95% of the time
                end_time = time.time() + 0.95
                x = 0
                while (cpu_spike_active or critical_mode_enabled) and running and time.time() < end_time:
                    x = (x * 1664525 + 1013904223) & 0xFFFFFFFF
                time.sleep(0.05)  # Brief pause

            elif stress_mode_enabled:
                # Moderate sustained CPU elevation (not pulsing)
                # Burn for 60% of each second to create persistent elevation
                # This ensures CPU stays elevated between 5-second scrapes
                burn_seconds = 0.70
                end_time = time.time() + burn_seconds
                x = 0
                while stress_mode_enabled and running and time.time() < end_time:
                    x = (x * 1103515245 + 12345) & 0x7FFFFFFF

                time.sleep(max(0.0, 1.0 - burn_seconds))

            else:
                time.sleep(0.1)
        except Exception as e:
            log_message('ERROR', 'CPU spike simulation error', error=str(e))
            time.sleep(1)

def process_notification(notification_data):
    """Process a single notification with distributed tracing"""
    start_time = time.time()
    
    try:
        notification = notification_data
        if isinstance(notification_data, str):
            notification = json.loads(notification_data)
        
        notification_type = notification.get('type', 'unknown')
        notification_id = notification.get('order_id', 'unknown')
        
        # Create span for notification processing
        trace_id = notification.get('trace_id')
        parent_span_id = notification.get('span_id')
        
        span = DistributedTracer.start_span(
            service_name="notification-service",
            operation_name=f"process_{notification_type}",
            trace_id=trace_id,
            parent_span_id=parent_span_id
        )
        
        with SpanContext(span):
            span.set_tag("notification.id", notification_id)
            span.set_tag("notification.type", notification_type)
            span.log("notification_processing_started", notification_id=notification_id)
            
            # Simulate processing time
            span.log("processing_delay_started")
            time.sleep(0.2)
            span.log("processing_delay_completed")
            
            # Simulate external API call for order confirmations
            if notification_type == 'order_confirmation':
                span.log("external_api_call_started")
                span.set_tag("external.api", "email_service")
                time.sleep(0.3)
                span.log("external_api_call_completed")
            
            # Track processed message timestamp
            with queue_lock:
                processed_messages_timestamps.append(time.time())
                # Keep only recent timestamps (last 5 minutes)  
                cutoff = time.time() - 300
                processed_messages_timestamps[:] = [t for t in processed_messages_timestamps if t > cutoff]
            
            duration = time.time() - start_time
            PROCESSING_DURATION.observe(duration)
            NOTIFICATIONS_PROCESSED.labels(type=notification_type, status='success').inc()
            
            span.set_tag("processing.duration_ms", duration * 1000)
            span.log("notification_processing_completed", 
                   notification_id=notification_id,
                   duration=duration)
            
            log_message('INFO', 'Notification processed',
                       notification_id=notification_id,
                       type=notification_type,
                       duration=duration,
                       trace_id=span.trace_id,
                       span_id=span.span_id)
        
        return True
        
    except Exception as e:
        NOTIFICATIONS_PROCESSED.labels(type='unknown', status='error').inc()
        log_message('ERROR', 'Notification processing failed', error=str(e))
        return False

def notification_worker(worker_id):
    """Worker thread to process notifications"""
    log_message('INFO', 'Worker started', worker_id=worker_id)
    WORKER_THREADS.inc()
    
    try:
        while running:
            try:
                notification_data = None
                
                # Get message from queue
                with queue_lock:
                    if message_queue:
                        notification_data = message_queue.pop(0)
                
                if notification_data:
                    process_notification(notification_data)
                else:
                    # Simulate getting messages from order processor
                    try:
                        import requests
                        response = requests.get('http://localhost:8002/queue-status', timeout=1)
                        if response.status_code == 200:
                            data = response.json()
                            if data.get('queue_size', 0) > 0:
                                # Simulate processing a message
                                fake_notification = {
                                    'order_id': f'simulated_{int(time.time())}',
                                    'type': 'order_confirmation',
                                    'timestamp': datetime.now().isoformat()
                                }
                                process_notification(fake_notification)
                    except:
                        pass  # Ignore connection errors
                    
                    time.sleep(1)  # Wait if no messages
                
                update_system_metrics()
                
            except Exception as e:
                log_message('ERROR', 'Worker error', worker_id=worker_id, error=str(e))
                time.sleep(1)
                
    finally:
        WORKER_THREADS.dec()
        log_message('INFO', 'Worker stopped', worker_id=worker_id)

def main():
    """Main function that runs both FastAPI and worker threads"""
    global running
    
    log_message('INFO', 'Starting Notification Service')
    print("🚀 Starting Notification Service...")
    print("   HTTP API: http://localhost:8003")
    print("   Health: http://localhost:8003/health") 
    print("   Metrics: http://localhost:8003/metrics")
    print("   Press Ctrl+C to stop")
    
    # Initialize Jaeger exporter if available
    if JAEGER_ENABLED:
        jaeger_endpoint = os.getenv('JAEGER_ENDPOINT', 'http://localhost:14268/api/traces')
        try:
            exporter = JaegerExporter(jaeger_endpoint)
            set_jaeger_exporter(exporter)
            print(f"   Jaeger Tracing: ENABLED ({jaeger_endpoint})")
            log_message('INFO', 'Jaeger tracing enabled', endpoint=jaeger_endpoint)
        except Exception as e:
            print(f"   Jaeger Tracing: FAILED ({e})")
            log_message('WARNING', 'Failed to initialize Jaeger', error=str(e))
    else:
        print("   Jaeger Tracing: DISABLED (install 'requests' package to enable)")
        log_message('INFO', 'Jaeger tracing disabled')

    # psutil CPU sampling warm-up so the first reading isn't misleading.
    try:
        if process:
            process.cpu_percent(interval=None)
    except Exception as e:
        log_message('WARNING', 'CPU warm-up failed', error=str(e))
    
    # Start CPU spike simulation thread
    cpu_thread = threading.Thread(target=simulate_cpu_spike, daemon=True)
    cpu_thread.start()

    # Start baseline variation threads
    mem_baseline_thread = threading.Thread(target=memory_baseline_variation_loop, daemon=True)
    mem_baseline_thread.start()

    mem_leak_thread = threading.Thread(target=memory_leak_loop, daemon=True)
    mem_leak_thread.start()

    thread_baseline_thread = threading.Thread(target=thread_baseline_variation_loop, daemon=True)
    thread_baseline_thread.start()

    thread_spike_thread = threading.Thread(target=thread_spike_loop, daemon=True)
    thread_spike_thread.start()

    # Start experiment mode controllers
    stress_mode_thread = threading.Thread(target=stress_mode_loop, daemon=True)
    stress_mode_thread.start()

    critical_mode_thread = threading.Thread(target=critical_mode_loop, daemon=True)
    critical_mode_thread.start()
    
    # Start worker threads
    num_workers = 2
    workers = []
    
    for i in range(num_workers):
        worker = threading.Thread(target=notification_worker, args=(i,), daemon=True)
        worker.start()
        workers.append(worker)
    
    print("   ✅ Service started with 2 workers")
    print("   ✅ CPU spike simulation ready")
    print("   ✅ Memory baseline variation active")
    print("   ✅ Thread baseline variation active")
    print("   ✅ Processing queue messages")
    
    # Run FastAPI server
    try:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8003, log_level="warning")
    except KeyboardInterrupt:
        running = False
        log_message('INFO', 'Shutting down notification service')
        print("\n⏹️  Notification Service stopped")
        
        # Wait for workers to finish
        for worker in workers:
            worker.join(timeout=2)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error starting service: {e}")
        log_message('ERROR', 'Service startup failed', error=str(e))
        input("Press Enter to exit...")