"""
Recovery Action Scorer

Scores and ranks recovery actions based on multiple criteria:
- Effectiveness: Will this solve the problem?
- Safety: Risk of downtime or cascading failures
- Speed: How fast can it execute?
- Cost: Resource impact (money and compute)
- Simplicity: How easy to implement and rollback?

Author: Anomaly Detection System
Date: March 2026
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

from action_generator import RecoveryAction, ActionType


class ScoreCategory(Enum):
    """Scoring categories."""
    EFFECTIVENESS = "effectiveness"
    SAFETY = "safety"
    SPEED = "speed"
    COST = "cost"
    SIMPLICITY = "simplicity"


@dataclass
class ActionScore:
    """
    Complete scoring breakdown for a recovery action.
    
    All scores are 0.0 to 1.0 (higher is better).
    """
    action: RecoveryAction
    
    # Individual scores
    effectiveness: float
    safety: float
    speed: float
    cost: float
    simplicity: float
    
    # Weighted total (0-100)
    total_score: float
    
    # Additional metadata
    risk_level: str  # 'low', 'medium', 'high'
    estimated_duration_seconds: int
    rollback_difficulty: str  # 'easy', 'medium', 'hard'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'action_type': self.action.action_type.value,
            'action_description': self.action.action_description,
            'effectiveness': round(self.effectiveness, 3),
            'safety': round(self.safety, 3),
            'speed': round(self.speed, 3),
            'cost': round(self.cost, 3),
            'simplicity': round(self.simplicity, 3),
            'total_score': round(self.total_score, 1),
            'risk_level': self.risk_level,
            'estimated_duration_seconds': self.estimated_duration_seconds,
            'rollback_difficulty': self.rollback_difficulty
        }


class ActionScorer:
    """
    Scores recovery actions on multiple dimensions.
    
    Scoring Weights (default):
    - Effectiveness: 40% (most important - does it work?)
    - Safety: 30% (avoid making things worse)
    - Speed: 15% (faster recovery = less downtime)
    - Cost: 10% (resource efficiency)
    - Simplicity: 5% (ease of implementation)
    """
    
    # Default scoring weights (sum to 100)
    DEFAULT_WEIGHTS = {
        ScoreCategory.EFFECTIVENESS: 0.40,
        ScoreCategory.SAFETY: 0.30,
        ScoreCategory.SPEED: 0.15,
        ScoreCategory.COST: 0.10,
        ScoreCategory.SIMPLICITY: 0.05
    }
    
    # Action type characteristics
    ACTION_CHARACTERISTICS = {
        ActionType.SCALE_HORIZONTAL: {
            'base_safety': 0.9,  # Very safe - K8s handles gradually
            'base_speed': 0.7,   # Moderate - need to wait for pods
            'base_cost': 0.5,    # Adds resources = higher cost
            'base_simplicity': 0.9,  # Simple command
            'duration_seconds': 30,
            'rollback': 'easy'
        },
        ActionType.ADJUST_CPU_LIMITS: {
            'base_safety': 0.7,  # Moderate risk - may OOM
            'base_speed': 0.8,   # Fast - just config change
            'base_cost': 0.7,    # May use more CPU
            'base_simplicity': 0.8,  # Straightforward
            'duration_seconds': 20,
            'rollback': 'easy'
        },
        ActionType.ADJUST_MEMORY_LIMITS: {
            'base_safety': 0.7,  # Moderate risk - may OOM
            'base_speed': 0.8,   # Fast - just config change
            'base_cost': 0.6,    # May use more memory
            'base_simplicity': 0.8,  # Straightforward
            'duration_seconds': 20,
            'rollback': 'easy'
        },
        ActionType.RESTART_POD: {
            'base_safety': 0.8,  # Generally safe with rolling restart
            'base_speed': 0.9,   # Very fast - immediate action
            'base_cost': 1.0,    # No additional cost
            'base_simplicity': 1.0,  # Simplest action
            'duration_seconds': 10,
            'rollback': 'easy'
        },
        ActionType.OPTIMIZE_CONFIG: {
            'base_safety': 0.6,  # Higher risk - config changes
            'base_speed': 0.6,   # Slower - need testing
            'base_cost': 0.9,    # Usually no cost
            'base_simplicity': 0.5,  # More complex
            'duration_seconds': 60,
            'rollback': 'medium'
        },
        ActionType.REDUCE_LOAD: {
            'base_safety': 0.9,  # Very safe
            'base_speed': 0.5,   # Depends on traffic patterns
            'base_cost': 1.0,    # No cost
            'base_simplicity': 0.7,  # Moderate complexity
            'duration_seconds': 45,
            'rollback': 'medium'
        }
    }
    
    def __init__(self, weights: Dict[ScoreCategory, float] = None):
        """
        Initialize action scorer.
        
        Args:
            weights: Custom scoring weights (optional)
        """
        self.weights = weights or self.DEFAULT_WEIGHTS
        
        # Validate weights sum to ~1.0
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
    
    def score_actions(
        self,
        actions: List[RecoveryAction]
    ) -> List[ActionScore]:
        """
        Score all actions and return sorted by total score.
        
        Args:
            actions: List of RecoveryAction objects
        
        Returns:
            List of ActionScore objects, sorted by total_score (descending)
        """
        scored_actions = []
        
        for action in actions:
            score = self._score_single_action(action)
            scored_actions.append(score)
        
        # Sort by total score (highest first)
        scored_actions.sort(key=lambda x: x.total_score, reverse=True)
        
        return scored_actions
    
    def _score_single_action(self, action: RecoveryAction) -> ActionScore:
        """Score a single action on all dimensions."""
        
        # Get base characteristics for this action type
        characteristics = self.ACTION_CHARACTERISTICS.get(
            action.action_type,
            self._get_default_characteristics()
        )
        
        # Score effectiveness (based on scenario confidence)
        effectiveness = self._score_effectiveness(action)
        
        # Score safety
        safety = self._score_safety(action, characteristics)
        
        # Score speed
        speed = self._score_speed(action, characteristics)
        
        # Score cost
        cost = self._score_cost(action, characteristics)
        
        # Score simplicity
        simplicity = self._score_simplicity(action, characteristics)
        
        # Calculate weighted total (0-100 scale)
        total_score = (
            effectiveness * self.weights[ScoreCategory.EFFECTIVENESS] +
            safety * self.weights[ScoreCategory.SAFETY] +
            speed * self.weights[ScoreCategory.SPEED] +
            cost * self.weights[ScoreCategory.COST] +
            simplicity * self.weights[ScoreCategory.SIMPLICITY]
        ) * 100
        
        # Determine risk level
        risk_level = self._determine_risk_level(action, safety, effectiveness)
        
        return ActionScore(
            action=action,
            effectiveness=effectiveness,
            safety=safety,
            speed=speed,
            cost=cost,
            simplicity=simplicity,
            total_score=total_score,
            risk_level=risk_level,
            estimated_duration_seconds=characteristics['duration_seconds'],
            rollback_difficulty=characteristics['rollback']
        )
    
    def _score_effectiveness(self, action: RecoveryAction) -> float:
        """
        Score effectiveness: Will this solve the problem?
        
        Based on:
        - Scenario confidence (if available)
        - Whether scenario prevents anomaly
        - Action type appropriateness
        """
        if action.from_scenario:
            # Use scenario confidence as primary indicator
            effectiveness = action.from_scenario.predicted_confidence
            
            # Boost if it definitely prevents anomaly
            if action.from_scenario.prevents_anomaly:
                effectiveness = min(1.0, effectiveness * 1.1)
            
            # Adjust based on scenario score
            scenario_score_factor = action.from_scenario.score / 100.0
            effectiveness = (effectiveness * 0.7) + (scenario_score_factor * 0.3)
        else:
            # Fallback action - lower effectiveness
            effectiveness = 0.4
        
        return max(0.0, min(1.0, effectiveness))
    
    def _score_safety(
        self,
        action: RecoveryAction,
        characteristics: Dict
    ) -> float:
        """
        Score safety: Risk of downtime or cascading failures.
        
        Factors:
        - Base safety of action type
        - Magnitude of change (if applicable)
        - Risk level from scenario
        """
        base_safety = characteristics['base_safety']
        
        # Adjust for scenario risk if available
        if action.from_scenario:
            risk_level = action.from_scenario.risk_level
            if risk_level == 'high':
                safety = base_safety * 0.7
            elif risk_level == 'medium':
                safety = base_safety * 0.85
            else:  # low
                safety = base_safety
        else:
            # Fallback actions are generally safe (restart)
            safety = base_safety
        
        return max(0.0, min(1.0, safety))
    
    def _score_speed(
        self,
        action: RecoveryAction,
        characteristics: Dict
    ) -> float:
        """
        Score speed: How fast can it execute?
        
        Based on:
        - Action type speed
        - Estimated duration
        """
        base_speed = characteristics['base_speed']
        
        # Faster is better, but not the most important factor
        return base_speed
    
    def _score_cost(
        self,
        action: RecoveryAction,
        characteristics: Dict
    ) -> float:
        """
        Score cost: Resource impact.
        
        Factors:
        - Does it add replicas? (costs money)
        - Does it increase resource limits? (costs money)
        - Restart = free
        """
        base_cost = characteristics['base_cost']
        
        # Adjust based on specific parameters
        if action.action_type == ActionType.SCALE_HORIZONTAL:
            replicas_change = action.parameters.get('replicas_change', 1)
            # More replicas = higher cost
            cost_factor = max(0.3, 1.0 - (replicas_change * 0.1))
            cost = base_cost * cost_factor
        else:
            cost = base_cost
        
        return max(0.0, min(1.0, cost))
    
    def _score_simplicity(
        self,
        action: RecoveryAction,
        characteristics: Dict
    ) -> float:
        """
        Score simplicity: Ease of implementation and rollback.
        
        Simpler actions are preferred when effectiveness is similar.
        """
        return characteristics['base_simplicity']
    
    def _determine_risk_level(
        self,
        action: RecoveryAction,
        safety: float,
        effectiveness: float
    ) -> str:
        """
        Determine overall risk level.
        
        Combines safety score with effectiveness to classify risk.
        """
        # Low effectiveness + low safety = high risk
        # High effectiveness + high safety = low risk
        risk_score = (safety + effectiveness) / 2.0
        
        if risk_score >= 0.75:
            return 'low'
        elif risk_score >= 0.5:
            return 'medium'
        else:
            return 'high'
    
    def _get_default_characteristics(self) -> Dict:
        """Fallback characteristics for unknown action types."""
        return {
            'base_safety': 0.7,
            'base_speed': 0.7,
            'base_cost': 0.7,
            'base_simplicity': 0.7,
            'duration_seconds': 30,
            'rollback': 'medium'
        }
    
    def format_scores(self, scored_actions: List[ActionScore]) -> str:
        """Format scored actions as human-readable comparison table."""
        lines = []
        lines.append("")
        lines.append("╔" + "═" * 78 + "╗")
        lines.append("║" + " RECOVERY ACTION SCORING & RANKING ".center(78) + "║")
        lines.append("╚" + "═" * 78 + "╝")
        lines.append("")
        
        # Header
        lines.append(f"{'RANK':<6} {'ACTION':<20} {'SCORE':<8} {'RISK':<8} {'DURATION':<10}")
        lines.append("─" * 78)
        
        # Actions
        for rank, scored in enumerate(scored_actions, 1):
            action_name = scored.action.action_type.value.replace('_', ' ').title()
            rank_icon = "⭐" if rank == 1 else f"{rank}."
            
            lines.append(
                f"{rank_icon:<6} "
                f"{action_name[:19]:<20} "
                f"{scored.total_score:>5.1f}/100 "
                f"{scored.risk_level.upper():<8} "
                f"{scored.estimated_duration_seconds}s"
            )
        
        lines.append("")
        
        # Best action details
        if scored_actions:
            best = scored_actions[0]
            lines.append("🏆 RECOMMENDED ACTION:")
            lines.append(f"   {best.action.action_description}")
            lines.append(f"   Command: {best.action.action_command}")
            lines.append("")
            lines.append("📊 Score Breakdown:")
            lines.append(f"   Effectiveness: {best.effectiveness*100:>5.1f}% (will it work?)")
            lines.append(f"   Safety:        {best.safety*100:>5.1f}% (is it safe?)")
            lines.append(f"   Speed:         {best.speed*100:>5.1f}% (how fast?)")
            lines.append(f"   Cost:          {best.cost*100:>5.1f}% (resource efficiency)")
            lines.append(f"   Simplicity:    {best.simplicity*100:>5.1f}% (ease of rollback)")
            lines.append("")
            lines.append(f"⚠️  Risk Level: {best.risk_level.upper()}")
            lines.append(f"⏱️  Estimated Duration: {best.estimated_duration_seconds} seconds")
            lines.append(f"↩️  Rollback: {best.rollback_difficulty.upper()}")
        
        lines.append("")
        lines.append("─" * 80)
        
        return "\n".join(lines)


# Convenience function for use in main pipeline
def score_and_rank_actions(
    actions: List[RecoveryAction],
    custom_weights: Dict[ScoreCategory, float] = None
) -> List[ActionScore]:
    """
    Quick function to score and rank recovery actions.
    
    Usage:
        scored = score_and_rank_actions(actions)
        best_action = scored[0].action
        print(best_action.action_command)
    """
    scorer = ActionScorer(weights=custom_weights)
    return scorer.score_actions(actions)


if __name__ == "__main__":
    print("Action Scorer Module")
    print("Import this module to use ActionScorer class")
