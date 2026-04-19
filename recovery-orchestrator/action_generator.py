"""
Recovery Action Generator

Converts counterfactual scenarios into concrete Kubernetes recovery actions.

Takes the output from CounterfactualAnalyzer and generates actionable K8s commands:
- Scale horizontally (add/remove replicas)
- Adjust resource limits (CPU/memory)
- Restart pods
- Apply optimized configurations

Author: Anomaly Detection System
Date: March 2026
"""

import sys
import os
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

# Add ml_detector to path for importing counterfactual analyzer
script_dir = os.path.dirname(os.path.abspath(__file__))
ml_detector_path = os.path.join(os.path.dirname(script_dir), 'ml_detector', 'scripts')
sys.path.insert(0, ml_detector_path)

from counterfactual_analyzer import CounterfactualExplanation, ScenarioComparison


class ActionType(Enum):
    """Types of recovery actions."""
    SCALE_HORIZONTAL = "scale_horizontal"
    ADJUST_CPU_LIMITS = "adjust_cpu_limits"
    ADJUST_MEMORY_LIMITS = "adjust_memory_limits"
    RESTART_POD = "restart_pod"
    OPTIMIZE_CONFIG = "optimize_config"
    REDUCE_LOAD = "reduce_load"


@dataclass
class RecoveryAction:
    """
    Container for a single recovery action.
    
    Attributes:
        action_type: Type of action (scale, adjust resources, restart)
        action_command: Kubernetes command to execute
        action_description: Human-readable description
        target_service: Service to apply action to
        parameters: Action-specific parameters
        expected_outcome: What this action should achieve
        from_scenario: Reference to counterfactual scenario
    """
    action_type: ActionType
    action_command: str
    action_description: str
    target_service: str
    parameters: Dict
    expected_outcome: str
    from_scenario: Optional[ScenarioComparison] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/API."""
        return {
            'action_type': self.action_type.value,
            'action_command': self.action_command,
            'action_description': self.action_description,
            'target_service': self.target_service,
            'parameters': self.parameters,
            'expected_outcome': self.expected_outcome,
            'scenario_score': self.from_scenario.score if self.from_scenario else None
        }


class ActionGenerator:
    """
    Generates recovery actions from counterfactual explanations.
    
    Workflow:
        1. Takes CounterfactualExplanation with scenario comparisons
        2. For each scenario, determines appropriate K8s action
        3. Generates concrete kubectl commands
        4. Returns list of RecoveryAction objects
    """
    
    # Default deployment name (can be overridden)
    DEFAULT_DEPLOYMENT = "web-api"
    DEFAULT_NAMESPACE = "default"
    
    # Feature to action mapping
    FEATURE_ACTION_MAP = {
        'cpu_utilization_mean': ActionType.SCALE_HORIZONTAL,
        'cpu_utilization_max': ActionType.ADJUST_CPU_LIMITS,
        'memory_utilization_mean': ActionType.ADJUST_MEMORY_LIMITS,
        'memory_pressure_max': ActionType.ADJUST_MEMORY_LIMITS,
        'memory_utilization_max': ActionType.ADJUST_MEMORY_LIMITS,
        'error_rate': ActionType.RESTART_POD,
        'queue_depth_mean': ActionType.SCALE_HORIZONTAL,
        'system_stress_index': ActionType.SCALE_HORIZONTAL,
        'thread_count_mean': ActionType.OPTIMIZE_CONFIG,
        'response_time_p95_mean': ActionType.OPTIMIZE_CONFIG,
    }
    
    def __init__(
        self,
        deployment_name: str = None,
        namespace: str = None,
        root_cause_service: str = None
    ):
        """
        Initialize action generator.
        
        Args:
            deployment_name: K8s deployment name
            namespace: K8s namespace
            root_cause_service: Service identified as root cause
        """
        self.deployment_name = deployment_name or self.DEFAULT_DEPLOYMENT
        self.namespace = namespace or self.DEFAULT_NAMESPACE
        self.root_cause_service = root_cause_service or self.deployment_name
    
    def generate_actions(
        self,
        counterfactual: CounterfactualExplanation,
        max_actions: int = 5
    ) -> List[RecoveryAction]:
        """
        Generate recovery actions from counterfactual explanation.
        
        Args:
            counterfactual: CounterfactualExplanation with scenario comparisons
            max_actions: Maximum number of actions to generate
        
        Returns:
            List of RecoveryAction objects, sorted by scenario score
        """
        actions = []
        seen_commands = set()
        
        # Process scenario comparisons (already sorted by score)
        scenarios_to_process = counterfactual.scenario_comparisons[:max_actions]
        
        for scenario in scenarios_to_process:
            # Only generate actions for scenarios that prevent anomaly
            if scenario.prevents_anomaly:
                for action in self.generate_action_variants(scenario):
                    if action and action.action_command not in seen_commands:
                        actions.append(action)
                        seen_commands.add(action.action_command)
        
        # If no preventing scenarios, generate fallback restart action
        if not actions:
            actions.append(self._generate_fallback_action(counterfactual))
        
        return actions[:max_actions]

    def generate_action_variants(
        self,
        scenario: ScenarioComparison
    ) -> List[RecoveryAction]:
        """Generate multiple viable recovery options for one scenario."""
        if not scenario.feature_changes:
            return []

        feature_name = list(scenario.feature_changes.keys())[0]
        change_info = scenario.feature_changes[feature_name]
        actions: List[RecoveryAction] = []

        primary_action = self._scenario_to_action(scenario)
        if primary_action:
            actions.append(primary_action)

        # Add secondary variants so the orchestrator can compare different remediations.
        if feature_name in {'cpu_utilization_mean', 'cpu_utilization_max', 'system_stress_index', 'queue_depth_mean'}:
            actions.append(self._generate_cpu_limit_action(scenario, feature_name, change_info))
            actions.append(self._generate_reduce_load_action(scenario, feature_name, change_info))
            actions.append(self._generate_optimize_config_action(scenario, feature_name, change_info))
        elif feature_name in {'memory_utilization_mean', 'memory_utilization_max', 'memory_pressure_max'}:
            actions.append(self._generate_memory_limit_action(scenario, feature_name, change_info))
            actions.append(self._generate_restart_action(scenario, feature_name, change_info))
            actions.append(self._generate_optimize_config_action(scenario, feature_name, change_info))
        elif feature_name in {'error_rate', 'response_time_p95_mean'}:
            actions.append(self._generate_restart_action(scenario, feature_name, change_info))
            actions.append(self._generate_reduce_load_action(scenario, feature_name, change_info))
            actions.append(self._generate_optimize_config_action(scenario, feature_name, change_info))

        # Remove duplicate commands while preserving order.
        unique_actions: List[RecoveryAction] = []
        seen_commands = set()
        for action in actions:
            if action and action.action_command not in seen_commands:
                unique_actions.append(action)
                seen_commands.add(action.action_command)

        return unique_actions
    
    def _scenario_to_action(
        self,
        scenario: ScenarioComparison
    ) -> Optional[RecoveryAction]:
        """
        Convert a counterfactual scenario to a recovery action.
        
        Args:
            scenario: ScenarioComparison with predicted outcome
        
        Returns:
            RecoveryAction or None if cannot convert
        """
        # Get the feature that needs to change
        if not scenario.feature_changes:
            return None
        
        feature_name = list(scenario.feature_changes.keys())[0]
        change_info = scenario.feature_changes[feature_name]
        
        # Map feature to action type
        action_type = self.FEATURE_ACTION_MAP.get(
            feature_name,
            ActionType.RESTART_POD  # Default fallback
        )
        
        # Generate action based on type
        if action_type == ActionType.SCALE_HORIZONTAL:
            return self._generate_scale_action(scenario, feature_name, change_info)
        
        elif action_type == ActionType.ADJUST_CPU_LIMITS:
            return self._generate_cpu_limit_action(scenario, feature_name, change_info)
        
        elif action_type == ActionType.ADJUST_MEMORY_LIMITS:
            return self._generate_memory_limit_action(scenario, feature_name, change_info)
        
        elif action_type == ActionType.RESTART_POD:
            return self._generate_restart_action(scenario, feature_name, change_info)
        
        else:
            return self._generate_restart_action(scenario, feature_name, change_info)
    
    def _generate_scale_action(
        self,
        scenario: ScenarioComparison,
        feature_name: str,
        change_info: Dict
    ) -> RecoveryAction:
        """Generate horizontal scaling action."""
        delta_percent = change_info['delta_percent']
        
        # Negative delta = need to reduce metric = scale UP (add replicas)
        # Positive delta = metric increasing = scale DOWN (less likely)
        if delta_percent < 0:
            # Scale up to reduce per-pod load
            replicas_to_add = max(1, int(abs(delta_percent) / 30))  # ~30% reduction per replica
            new_replicas = f"+{replicas_to_add}"
            action_verb = "up"
        else:
            replicas_to_add = 1
            new_replicas = "+1"
            action_verb = "up"
        
        command = f"kubectl scale deployment {self.deployment_name} --replicas={new_replicas} -n {self.namespace}"
        
        description = (
            f"Scale {action_verb} {self.deployment_name} by {replicas_to_add} replica(s) "
            f"to reduce {feature_name} by ~{abs(delta_percent):.1f}%"
        )
        
        expected_outcome = (
            f"Predicted: {scenario.predicted_class.upper()} "
            f"(confidence: {scenario.predicted_confidence*100:.0f}%)"
        )
        
        return RecoveryAction(
            action_type=ActionType.SCALE_HORIZONTAL,
            action_command=command,
            action_description=description,
            target_service=self.root_cause_service,
            parameters={
                'replicas_change': replicas_to_add,
                'feature': feature_name,
                'target_reduction': abs(delta_percent)
            },
            expected_outcome=expected_outcome,
            from_scenario=scenario
        )
    
    def _generate_cpu_limit_action(
        self,
        scenario: ScenarioComparison,
        feature_name: str,
        change_info: Dict
    ) -> RecoveryAction:
        """Generate CPU limit adjustment action."""
        delta_percent = change_info['delta_percent']
        
        # Calculate new CPU limit (assuming current is 1000m)
        current_cpu = 1000  # millicores
        new_cpu = int(current_cpu * (1 + delta_percent / 100))
        new_cpu = max(500, min(4000, new_cpu))  # Clamp to reasonable range
        
        command = (
            f"kubectl set resources deployment {self.deployment_name} "
            f"--limits=cpu={new_cpu}m -n {self.namespace}"
        )
        
        description = (
            f"Adjust CPU limits for {self.deployment_name} to {new_cpu}m "
            f"to address {feature_name} spike"
        )
        
        expected_outcome = (
            f"Predicted: {scenario.predicted_class.upper()} "
            f"(confidence: {scenario.predicted_confidence*100:.0f}%)"
        )
        
        return RecoveryAction(
            action_type=ActionType.ADJUST_CPU_LIMITS,
            action_command=command,
            action_description=description,
            target_service=self.root_cause_service,
            parameters={
                'new_cpu_limit': f"{new_cpu}m",
                'feature': feature_name,
                'change_percent': delta_percent
            },
            expected_outcome=expected_outcome,
            from_scenario=scenario
        )
    
    def _generate_memory_limit_action(
        self,
        scenario: ScenarioComparison,
        feature_name: str,
        change_info: Dict
    ) -> RecoveryAction:
        """Generate memory limit adjustment action."""
        delta_percent = change_info['delta_percent']
        
        # Calculate new memory limit (assuming current is 2Gi)
        current_mem = 2048  # Mi
        new_mem = int(current_mem * (1 + delta_percent / 100))
        new_mem = max(512, min(8192, new_mem))  # Clamp to reasonable range
        
        command = (
            f"kubectl set resources deployment {self.deployment_name} "
            f"--limits=memory={new_mem}Mi -n {self.namespace}"
        )
        
        description = (
            f"Adjust memory limits for {self.deployment_name} to {new_mem}Mi "
            f"to address {feature_name} pressure"
        )
        
        expected_outcome = (
            f"Predicted: {scenario.predicted_class.upper()} "
            f"(confidence: {scenario.predicted_confidence*100:.0f}%)"
        )
        
        return RecoveryAction(
            action_type=ActionType.ADJUST_MEMORY_LIMITS,
            action_command=command,
            action_description=description,
            target_service=self.root_cause_service,
            parameters={
                'new_memory_limit': f"{new_mem}Mi",
                'feature': feature_name,
                'change_percent': delta_percent
            },
            expected_outcome=expected_outcome,
            from_scenario=scenario
        )
    
    def _generate_restart_action(
        self,
        scenario: ScenarioComparison,
        feature_name: str,
        change_info: Dict
    ) -> RecoveryAction:
        """Generate pod restart action."""
        command = f"kubectl rollout restart deployment {self.deployment_name} -n {self.namespace}"
        
        description = (
            f"Restart {self.deployment_name} to clear {feature_name} issues "
            f"(e.g., memory leaks, stuck threads)"
        )
        
        expected_outcome = (
            f"Predicted: {scenario.predicted_class.upper()} "
            f"(confidence: {scenario.predicted_confidence*100:.0f}%)"
        )
        
        return RecoveryAction(
            action_type=ActionType.RESTART_POD,
            action_command=command,
            action_description=description,
            target_service=self.root_cause_service,
            parameters={
                'feature': feature_name,
                'change_type': 'restart'
            },
            expected_outcome=expected_outcome,
            from_scenario=scenario
        )

    def _generate_reduce_load_action(
        self,
        scenario: ScenarioComparison,
        feature_name: str,
        change_info: Dict
    ) -> RecoveryAction:
        """Generate a traffic reduction / rate limiting action."""
        reduction_pct = max(10, min(50, int(abs(change_info['delta_percent']))))
        command = (
            f"kubectl annotate deployment {self.deployment_name} "
            f"rate-limit={reduction_pct}pct -n {self.namespace} --overwrite"
        )
        description = (
            f"Reduce inbound load on {self.deployment_name} by ~{reduction_pct}% "
            f"to relieve {feature_name}"
        )
        expected_outcome = (
            f"Predicted: {scenario.predicted_class.upper()} "
            f"(confidence: {scenario.predicted_confidence*100:.0f}%)"
        )
        return RecoveryAction(
            action_type=ActionType.REDUCE_LOAD,
            action_command=command,
            action_description=description,
            target_service=self.root_cause_service,
            parameters={
                'load_reduction_percent': reduction_pct,
                'feature': feature_name
            },
            expected_outcome=expected_outcome,
            from_scenario=scenario
        )

    def _generate_optimize_config_action(
        self,
        scenario: ScenarioComparison,
        feature_name: str,
        change_info: Dict
    ) -> RecoveryAction:
        """Generate a configuration tuning action."""
        profile = 'throughput-optimized' if abs(change_info['delta_percent']) >= 30 else 'latency-optimized'
        command = (
            f"kubectl annotate deployment {self.deployment_name} "
            f"config-profile={profile} -n {self.namespace} --overwrite"
        )
        description = (
            f"Apply {profile} configuration on {self.deployment_name} "
            f"to mitigate {feature_name}"
        )
        expected_outcome = (
            f"Predicted: {scenario.predicted_class.upper()} "
            f"(confidence: {scenario.predicted_confidence*100:.0f}%)"
        )
        return RecoveryAction(
            action_type=ActionType.OPTIMIZE_CONFIG,
            action_command=command,
            action_description=description,
            target_service=self.root_cause_service,
            parameters={
                'config_profile': profile,
                'feature': feature_name
            },
            expected_outcome=expected_outcome,
            from_scenario=scenario
        )
    
    def _generate_fallback_action(
        self,
        counterfactual: CounterfactualExplanation
    ) -> RecoveryAction:
        """Generate fallback restart action when no good scenario found."""
        command = f"kubectl rollout restart deployment {self.deployment_name} -n {self.namespace}"
        
        description = (
            f"Restart {self.deployment_name} as fallback recovery action "
            f"(no counterfactual scenario reliably prevents anomaly)"
        )
        
        expected_outcome = "Uncertain - restart may help transient issues"
        
        return RecoveryAction(
            action_type=ActionType.RESTART_POD,
            action_command=command,
            action_description=description,
            target_service=self.root_cause_service,
            parameters={
                'fallback': True
            },
            expected_outcome=expected_outcome,
            from_scenario=None
        )
    
    def format_actions(self, actions: List[RecoveryAction]) -> str:
        """Format actions as human-readable text."""
        lines = []
        lines.append("")
        lines.append("╔" + "═" * 78 + "╗")
        lines.append("║" + " RECOVERY ACTIONS GENERATED ".center(78) + "║")
        lines.append("╚" + "═" * 78 + "╝")
        lines.append("")
        
        for i, action in enumerate(actions, 1):
            lines.append(f"🔧 ACTION {i}: {action.action_type.value.upper().replace('_', ' ')}")
            lines.append(f"   Target: {action.target_service}")
            lines.append(f"   Description: {action.action_description}")
            lines.append(f"   Command: {action.action_command}")
            lines.append(f"   Expected: {action.expected_outcome}")
            if action.from_scenario:
                lines.append(f"   Scenario Score: {action.from_scenario.score:.0f}/100")
            lines.append("")
        
        return "\n".join(lines)


# Convenience function for use in main pipeline
def generate_recovery_actions(
    counterfactual: CounterfactualExplanation,
    deployment_name: str = None,
    namespace: str = None,
    root_cause_service: str = None,
    max_actions: int = 3
) -> List[RecoveryAction]:
    """
    Quick function to generate recovery actions.
    
    Usage in online detector:
        actions = generate_recovery_actions(
            counterfactual=explanation,
            deployment_name="web-api",
            namespace="production",
            root_cause_service="api-gateway"
        )
        
        for action in actions:
            print(action.action_command)
    """
    generator = ActionGenerator(deployment_name, namespace, root_cause_service)
    return generator.generate_actions(counterfactual, max_actions)


if __name__ == "__main__":
    print("Action Generator Module")
    print("Import this module to use ActionGenerator class")
