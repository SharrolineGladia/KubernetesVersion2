# detector.py

from collections import deque
from datetime import datetime

from .config import EWMA_ALPHA, EPSILON, Z_MAX


class PersistenceStateMachine:
    """Lightweight persistence-based FSM for channel states.

    Uses consecutive windows (counters) rather than timers.
    """

    def __init__(
        self,
        low_stress_level: float,
        high_stress_level: float,
        normal_to_stressed_windows: int,
        stressed_to_critical_windows: int,
        stressed_to_normal_windows: int,
        critical_to_stressed_windows: int,
    ):
        self.low_stress_level = low_stress_level
        self.high_stress_level = high_stress_level

        self.normal_to_stressed_windows = normal_to_stressed_windows
        self.stressed_to_critical_windows = stressed_to_critical_windows
        self.stressed_to_normal_windows = stressed_to_normal_windows
        self.critical_to_stressed_windows = critical_to_stressed_windows

        self.current_state = "normal"
        self.state_duration = 0
        self.last_transition_timestamp = None
        self.transition_reason = None

        self._above_low_count = 0
        self._below_low_count = 0
        self._above_high_count = 0
        self._below_high_count = 0

    def _transition(self, new_state: str, reason: str, now_iso: str):
        self.current_state = new_state
        self.state_duration = 1
        self.last_transition_timestamp = now_iso
        self.transition_reason = reason

        self._above_low_count = 0
        self._below_low_count = 0
        self._above_high_count = 0
        self._below_high_count = 0

    def update(self, stress_score: float, now: datetime | None = None) -> dict:
        now = now or datetime.utcnow()
        now_iso = now.isoformat()

        self.transition_reason = None

        if self.current_state == "normal":
            if stress_score >= self.low_stress_level:
                self._above_low_count += 1
            else:
                self._above_low_count = 0

            if self._above_low_count >= self.normal_to_stressed_windows:
                self._transition(
                    "stressed",
                    (
                        f"stress_score {stress_score:.3f} >= LOW_STRESS_LEVEL {self.low_stress_level} "
                        f"for {self.normal_to_stressed_windows} consecutive windows"
                    ),
                    now_iso,
                )
            else:
                self.state_duration += 1

        elif self.current_state == "stressed":
            if stress_score >= self.high_stress_level:
                self._above_high_count += 1
            else:
                self._above_high_count = 0

            if stress_score < self.low_stress_level:
                self._below_low_count += 1
            else:
                self._below_low_count = 0

            if self._above_high_count >= self.stressed_to_critical_windows:
                self._transition(
                    "critical",
                    (
                        f"stress_score {stress_score:.3f} >= HIGH_STRESS_LEVEL {self.high_stress_level} "
                        f"for {self.stressed_to_critical_windows} consecutive windows"
                    ),
                    now_iso,
                )
            elif self._below_low_count >= self.stressed_to_normal_windows:
                self._transition(
                    "normal",
                    (
                        f"stress_score {stress_score:.3f} < LOW_STRESS_LEVEL {self.low_stress_level} "
                        f"for {self.stressed_to_normal_windows} consecutive windows"
                    ),
                    now_iso,
                )
            else:
                self.state_duration += 1

        else:  # critical
            if stress_score < self.high_stress_level:
                self._below_high_count += 1
            else:
                self._below_high_count = 0

            if self._below_high_count >= self.critical_to_stressed_windows:
                self._transition(
                    "stressed",
                    (
                        f"stress_score {stress_score:.3f} < HIGH_STRESS_LEVEL {self.high_stress_level} "
                        f"for {self.critical_to_stressed_windows} consecutive windows"
                    ),
                    now_iso,
                )
            else:
                self.state_duration += 1

        return {
            "state": self.current_state,
            "state_duration": int(self.state_duration),
            "last_transition_timestamp": self.last_transition_timestamp,
            "transition_reason": self.transition_reason,
        }

class EWMAMetric:
    def __init__(self):
        self.mean = None
        self.variance = 0.1  # Initialize with small positive value instead of 0
        self.n_samples = 0

    def update(self, value: float):
        self.n_samples += 1
        
        if self.mean is None:
            self.mean = value
            # Start with reasonable variance estimate
            self.variance = 0.1
            return 0.0

        # Calculate deviation before updating mean
        delta = value - self.mean
        
        # Update mean with EWMA
        self.mean = EWMA_ALPHA * value + (1 - EWMA_ALPHA) * self.mean
        
        # Update variance with EWMA (using squared deviation)
        self.variance = (
            EWMA_ALPHA * (delta ** 2) + (1 - EWMA_ALPHA) * self.variance
        )
        
        # Ensure variance doesn't collapse to zero
        self.variance = max(self.variance, 0.01)
        
        # Calculate standardized score
        std_dev = (self.variance ** 0.5)
        z_score = abs(delta) / (std_dev + EPSILON)
        
        # Normalize to [0, 1] range with smoother scaling
        normalized_score = min(1.0, z_score / Z_MAX)
        
        return normalized_score


class ResourceSaturationDetector:
    def __init__(self):
        self.cpu = EWMAMetric()
        self.memory = EWMAMetric()
        self.threads = EWMAMetric()

        # Persistence FSM for the resource_saturation channel.
        # Uses the existing score semantics (0..1) and conservative window counts.
        from .config import (
            NORMAL_THRESHOLD,
            ANOMALY_THRESHOLD,
            NORMAL_TO_STRESSED_WINDOWS,
            STRESSED_TO_CRITICAL_WINDOWS,
            STRESSED_TO_NORMAL_WINDOWS,
            CRITICAL_TO_STRESSED_WINDOWS,
            SCRAPE_INTERVAL_SECONDS,
        )

        self.channel_fsm = PersistenceStateMachine(
            low_stress_level=NORMAL_THRESHOLD,
            high_stress_level=ANOMALY_THRESHOLD,
            normal_to_stressed_windows=NORMAL_TO_STRESSED_WINDOWS,
            stressed_to_critical_windows=STRESSED_TO_CRITICAL_WINDOWS,
            stressed_to_normal_windows=STRESSED_TO_NORMAL_WINDOWS,
            critical_to_stressed_windows=CRITICAL_TO_STRESSED_WINDOWS,
        )

        # Snapshot freeze support: rolling buffer for last 10 minutes
        self.channel_name = "resource_saturation"
        self.scrape_interval = SCRAPE_INTERVAL_SECONDS
        buffer_size = int(600 / SCRAPE_INTERVAL_SECONDS)  # 10 minutes / interval
        self.observation_buffer = deque(maxlen=buffer_size)
        self.frozen_snapshot = None
        self.is_snapshot_frozen = False

    def update(self, cpu_val, mem_val, thread_val, timestamp=None):
        timestamp = timestamp or datetime.utcnow()
        
        cpu_stress = self.cpu.update(cpu_val)
        mem_stress = self.memory.update(mem_val)
        thread_stress = self.threads.update(thread_val)

        return {
            "cpu": cpu_stress,
            "memory": mem_stress,
            "threads": thread_stress,
            "timestamp": timestamp
        }

    def update_channel_state(self, stress_score: float, raw_metric=None, timestamp=None) -> dict:
        timestamp = timestamp or datetime.utcnow()
        
        # Store observation in rolling buffer
        observation = {
            "timestamp": timestamp.isoformat(),
            "raw_metric": raw_metric,
            "ewma_signal": None,  # Placeholder for EWMA signal if needed
            "stress_score": stress_score,
            "state": self.channel_fsm.current_state
        }
        self.observation_buffer.append(observation)
        
        # Get previous state
        prev_state = self.channel_fsm.current_state
        
        # Update FSM
        state_info = self.channel_fsm.update(stress_score, timestamp)
        
        # Check for transition into critical
        if prev_state != "critical" and state_info["state"] == "critical":
            if not self.is_snapshot_frozen:
                self._freeze_snapshot(timestamp)
        
        # Reset snapshot when returning to normal
        if state_info["state"] == "normal":
            self.frozen_snapshot = None
            self.is_snapshot_frozen = False
        
        return state_info

    def _freeze_snapshot(self, trigger_time: datetime):
        """Freeze the current observation buffer as an immutable snapshot."""
        self.frozen_snapshot = {
            "channel": self.channel_name,
            "trigger_time": trigger_time.isoformat(),
            "snapshot_window_seconds": int(self.scrape_interval * len(self.observation_buffer)),
            "data": list(self.observation_buffer)  # Create a copy
        }
        self.is_snapshot_frozen = True

    def get_frozen_snapshot(self):
        """Return the frozen snapshot if available, otherwise None."""
        return self.frozen_snapshot


class PerformanceDegradationChannel:
    """EWMA-based channel for p95_response_time_ms with snapshot freeze support.
    
    Uses hybrid detection:
    - EWMA: Detects transient spikes (unexpected deviations)
    - Absolute thresholds: Detects sustained degradation
    """
    
    def __init__(self):
        self.metric = EWMAMetric()
        
        # Channel-specific configuration
        self.channel_name = "performance_degradation"
        self.scrape_interval = 45  # seconds
        LOW_STRESS_LEVEL = 0.3
        HIGH_STRESS_LEVEL = 0.6
        
        # Import absolute thresholds
        from .config import (
            P95_RESPONSE_TIME_BASELINE,
            P95_RESPONSE_TIME_STRESSED,
            P95_RESPONSE_TIME_CRITICAL
        )
        self.baseline_threshold = P95_RESPONSE_TIME_BASELINE
        self.stressed_threshold = P95_RESPONSE_TIME_STRESSED
        self.critical_threshold = P95_RESPONSE_TIME_CRITICAL
        
        # FSM with window counts based on scrape interval
        # 10 minutes = 600s / 45s = ~13 samples
        self.channel_fsm = PersistenceStateMachine(
            low_stress_level=LOW_STRESS_LEVEL,
            high_stress_level=HIGH_STRESS_LEVEL,
            normal_to_stressed_windows=3,
            stressed_to_critical_windows=3,
            stressed_to_normal_windows=5,
            critical_to_stressed_windows=3,
        )
        
        # Rolling buffer for 10 minutes
        buffer_size = int(600 / self.scrape_interval)
        self.observation_buffer = deque(maxlen=buffer_size)
        self.frozen_snapshot = None
        self.is_snapshot_frozen = False
    
    def update(self, raw_metric_value: float, timestamp=None):
        """Update metric and return current state using hybrid detection.
        
        Combines EWMA (transient spikes) + absolute thresholds (sustained degradation).
        """
        timestamp = timestamp or datetime.utcnow()
        
        # Calculate EWMA stress signal (0-1 range)
        ewma_signal = self.metric.update(raw_metric_value)
        
        # Calculate absolute threshold stress (0-1 range)
        # Maps raw value to stress level based on configured thresholds
        if raw_metric_value <= self.baseline_threshold:
            absolute_stress = 0.0
        elif raw_metric_value >= self.critical_threshold:
            absolute_stress = 1.0
        elif raw_metric_value <= self.stressed_threshold:
            # Linear interpolation between baseline and stressed
            absolute_stress = 0.5 * (raw_metric_value - self.baseline_threshold) / \
                             (self.stressed_threshold - self.baseline_threshold)
        else:
            # Linear interpolation between stressed and critical
            absolute_stress = 0.5 + 0.5 * (raw_metric_value - self.stressed_threshold) / \
                             (self.critical_threshold - self.stressed_threshold)
        
        # Hybrid stress: Take maximum of EWMA and absolute
        # This catches BOTH transient spikes AND sustained degradation
        combined_stress = max(ewma_signal, absolute_stress)
        
        # Store observation in rolling buffer
        observation = {
            "timestamp": timestamp.isoformat(),
            "raw_metric": raw_metric_value,
            "ewma_signal": ewma_signal,
            "absolute_stress": absolute_stress,
            "stress_score": combined_stress,
            "state": self.channel_fsm.current_state
        }
        self.observation_buffer.append(observation)
        
        # Get previous state
        prev_state = self.channel_fsm.current_state
        
        # Update FSM with combined stress score
        state_info = self.channel_fsm.update(combined_stress, timestamp)
        
        # Check for transition into critical
        if prev_state != "critical" and state_info["state"] == "critical":
            if not self.is_snapshot_frozen:
                self._freeze_snapshot(timestamp)
        
        # Reset snapshot when returning to normal
        if state_info["state"] == "normal":
            self.frozen_snapshot = None
            self.is_snapshot_frozen = False
        
        return state_info
    
    def _freeze_snapshot(self, trigger_time: datetime):
        """Freeze the current observation buffer as an immutable snapshot."""
        self.frozen_snapshot = {
            "channel": self.channel_name,
            "trigger_time": trigger_time.isoformat(),
            "snapshot_window_seconds": int(self.scrape_interval * len(self.observation_buffer)),
            "data": list(self.observation_buffer)
        }
        self.is_snapshot_frozen = True
    
    def get_state(self):
        """Return current FSM state."""
        return self.channel_fsm.current_state
    
    def get_frozen_snapshot(self):
        """Return the frozen snapshot if available, otherwise None."""
        return self.frozen_snapshot


class BackpressureOverloadChannel:
    """EWMA-based channel for queue_depth with snapshot freeze support.
    
    Uses hybrid detection:
    - EWMA: Detects transient spikes (unexpected deviations)
    - Absolute thresholds: Detects sustained degradation
    """
    
    def __init__(self):
        self.metric = EWMAMetric()
        
        # Channel-specific configuration
        self.channel_name = "backpressure_overload"
        self.scrape_interval = 30  # seconds
        LOW_STRESS_LEVEL = 0.4
        HIGH_STRESS_LEVEL = 0.7
        
        # Import absolute thresholds
        from .config import (
            QUEUE_DEPTH_BASELINE,
            QUEUE_DEPTH_STRESSED,
            QUEUE_DEPTH_CRITICAL
        )
        self.baseline_threshold = QUEUE_DEPTH_BASELINE
        self.stressed_threshold = QUEUE_DEPTH_STRESSED
        self.critical_threshold = QUEUE_DEPTH_CRITICAL
        
        # FSM with window counts based on scrape interval
        # 10 minutes = 600s / 30s = 20 samples
        self.channel_fsm = PersistenceStateMachine(
            low_stress_level=LOW_STRESS_LEVEL,
            high_stress_level=HIGH_STRESS_LEVEL,
            normal_to_stressed_windows=3,
            stressed_to_critical_windows=3,
            stressed_to_normal_windows=5,
            critical_to_stressed_windows=3,
        )
        
        # Rolling buffer for 10 minutes
        buffer_size = int(600 / self.scrape_interval)
        self.observation_buffer = deque(maxlen=buffer_size)
        self.frozen_snapshot = None
        self.is_snapshot_frozen = False
    
    def update(self, raw_metric_value: float, timestamp=None):
        """Update metric and return current state using hybrid detection.
        
        Combines EWMA (transient spikes) + absolute thresholds (sustained degradation).
        """
        timestamp = timestamp or datetime.utcnow()
        
        # Calculate EWMA stress signal (0-1 range)
        ewma_signal = self.metric.update(raw_metric_value)
        
        # Calculate absolute threshold stress (0-1 range)
        if raw_metric_value <= self.baseline_threshold:
            absolute_stress = 0.0
        elif raw_metric_value >= self.critical_threshold:
            absolute_stress = 1.0
        elif raw_metric_value <= self.stressed_threshold:
            # Linear interpolation between baseline and stressed
            absolute_stress = 0.5 * (raw_metric_value - self.baseline_threshold) / \
                             (self.stressed_threshold - self.baseline_threshold)
        else:
            # Linear interpolation between stressed and critical
            absolute_stress = 0.5 + 0.5 * (raw_metric_value - self.stressed_threshold) / \
                             (self.critical_threshold - self.stressed_threshold)
        
        # Hybrid stress: Take maximum of EWMA and absolute
        # This catches BOTH transient spikes AND sustained degradation
        combined_stress = max(ewma_signal, absolute_stress)
        
        # Store observation in rolling buffer
        observation = {
            "timestamp": timestamp.isoformat(),
            "raw_metric": raw_metric_value,
            "ewma_signal": ewma_signal,
            "absolute_stress": absolute_stress,
            "stress_score": combined_stress,
            "state": self.channel_fsm.current_state
        }
        self.observation_buffer.append(observation)
        
        # Get previous state
        prev_state = self.channel_fsm.current_state
        
        # Update FSM with combined stress score
        state_info = self.channel_fsm.update(combined_stress, timestamp)
        
        # Check for transition into critical
        if prev_state != "critical" and state_info["state"] == "critical":
            if not self.is_snapshot_frozen:
                self._freeze_snapshot(timestamp)
        
        # Reset snapshot when returning to normal
        if state_info["state"] == "normal":
            self.frozen_snapshot = None
            self.is_snapshot_frozen = False
        
        return state_info
    
    def _freeze_snapshot(self, trigger_time: datetime):
        """Freeze the current observation buffer as an immutable snapshot."""
        self.frozen_snapshot = {
            "channel": self.channel_name,
            "trigger_time": trigger_time.isoformat(),
            "snapshot_window_seconds": int(self.scrape_interval * len(self.observation_buffer)),
            "data": list(self.observation_buffer)
        }
        self.is_snapshot_frozen = True
    
    def get_state(self):
        """Return current FSM state."""
        return self.channel_fsm.current_state
    
    def get_frozen_snapshot(self):
        """Return the frozen snapshot if available, otherwise None."""
        return self.frozen_snapshot


