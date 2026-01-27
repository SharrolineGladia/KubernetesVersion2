# detector.py

from datetime import datetime

from config import EWMA_ALPHA, EPSILON, Z_MAX


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
        from config import (
            NORMAL_THRESHOLD,
            ANOMALY_THRESHOLD,
            NORMAL_TO_STRESSED_WINDOWS,
            STRESSED_TO_CRITICAL_WINDOWS,
            STRESSED_TO_NORMAL_WINDOWS,
            CRITICAL_TO_STRESSED_WINDOWS,
        )

        self.channel_fsm = PersistenceStateMachine(
            low_stress_level=NORMAL_THRESHOLD,
            high_stress_level=ANOMALY_THRESHOLD,
            normal_to_stressed_windows=NORMAL_TO_STRESSED_WINDOWS,
            stressed_to_critical_windows=STRESSED_TO_CRITICAL_WINDOWS,
            stressed_to_normal_windows=STRESSED_TO_NORMAL_WINDOWS,
            critical_to_stressed_windows=CRITICAL_TO_STRESSED_WINDOWS,
        )

    def update(self, cpu_val, mem_val, thread_val):
        cpu_stress = self.cpu.update(cpu_val)
        mem_stress = self.memory.update(mem_val)
        thread_stress = self.threads.update(thread_val)

        return {
            "cpu": cpu_stress,
            "memory": mem_stress,
            "threads": thread_stress
        }

    def update_channel_state(self, stress_score: float) -> dict:
        return self.channel_fsm.update(stress_score)
