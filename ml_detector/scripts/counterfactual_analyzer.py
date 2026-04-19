"""
Real-Time Counterfactual Analyzer

Provides live "what-if" explanations during anomaly detection:
- What changes would have prevented this anomaly?
- How much adjustment is needed?
- Is the counterfactual actionable?

Integrates with dual_feature_detector for real-time use.

Author: Anomaly Detection System
Date: March 2026
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time


@dataclass
class ScenarioComparison:
    """Detailed comparison of a single counterfactual scenario."""
    
    scenario_name: str
    feature_changes: Dict[str, float]  # {feature: change_value}
    predicted_class: str
    predicted_confidence: float
    prevents_anomaly: bool
    score: float  # 0-100 ranking score
    risk_level: str  # 'low', 'medium', 'high'
    
    def to_dict(self) -> Dict:
        return {
            'scenario_name': self.scenario_name,
            'feature_changes': self.feature_changes,
            'predicted_class': self.predicted_class,
            'predicted_confidence': round(self.predicted_confidence, 3),
            'prevents_anomaly': self.prevents_anomaly,
            'score': round(self.score, 1),
            'risk_level': self.risk_level
        }


@dataclass
class CounterfactualExplanation:
    """Container for counterfactual analysis results."""

    # What needs to change
    target_feature: str
    current_value: float
    target_value: float
    delta: float
    delta_percent: float

    # Alternative scenarios
    alternative_actions: List[Dict[str, any]]

    # Feasibility
    is_feasible: bool
    confidence: float
    search_time_ms: float

    # Actionable recommendation
    actionable_recommendation: str
    
    # Fields with defaults must come last
    scenario_comparisons: List[ScenarioComparison] = field(default_factory=list)
    best_scenario_idx: Optional[int] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/display."""
        return {
            'target_feature': self.target_feature,
            'current_value': round(self.current_value, 3),
            'target_value': round(self.target_value, 3),
            'delta': round(self.delta, 3),
            'delta_percent': round(self.delta_percent, 1),
            'is_feasible': self.is_feasible,
            'confidence': round(self.confidence, 2),
            'search_time_ms': round(self.search_time_ms, 2),
            'actionable_recommendation': self.actionable_recommendation,
            'alternative_actions': self.alternative_actions
        }

    def format_human_readable(self) -> str:
        """Format as human-readable text with scenario comparison."""
        lines = []
        lines.append("")
        lines.append("╔" + "═" * 78 + "╗")
        lines.append("║" + " COUNTERFACTUAL ANALYSIS: What Could Have Prevented This? ".center(78) + "║")
        lines.append("╚" + "═" * 78 + "╝")
        lines.append("")

        if self.is_feasible:
            lines.append(f"✅ PREVENTION POSSIBLE: This anomaly could have been avoided!")
            lines.append("")
            
            # Show scenario comparison table if available
            if self.scenario_comparisons:
                lines.append("📊 SCENARIO COMPARISON:")
                lines.append("")
                lines.append(f"{'SCENARIO':<25} {'PREDICTION':<15} {'CONFIDENCE':<12} {'STATUS':<15}")
                lines.append("─" * 78)
                
                for i, scenario in enumerate(self.scenario_comparisons):
                    status_icon = "✅ PREVENTS" if scenario.prevents_anomaly else "❌ STILL ANOMALOUS"
                    if scenario.prevents_anomaly and i == self.best_scenario_idx:
                        status_icon = "⭐ BEST"
                    
                    # Get feature name for scenario
                    feature_name = list(scenario.feature_changes.keys())[0] if scenario.feature_changes else scenario.scenario_name
                    scenario_display = feature_name[:24]
                    
                    lines.append(
                        f"{scenario_display:<25} "
                        f"{scenario.predicted_class.upper():<15} "
                        f"{scenario.predicted_confidence*100:>5.1f}%      "
                        f"{status_icon}"
                    )
                
                lines.append("")
                
                # Show best scenario details
                if self.best_scenario_idx is not None and self.best_scenario_idx < len(self.scenario_comparisons):
                    best = self.scenario_comparisons[self.best_scenario_idx]
                    lines.append(f"🎯 RECOMMENDED SCENARIO:")
                    for feature, change_info in best.feature_changes.items():
                        lines.append(f"   {feature}: {change_info['current']:.2f} → {change_info['target']:.2f} "
                                   f"({change_info['delta_percent']:+.1f}%)")
                    lines.append(f"   New State: {best.predicted_class.upper()} ({best.predicted_confidence*100:.0f}% confidence)")
                    lines.append(f"   Score: {best.score:.0f}/100 | Risk: {best.risk_level.upper()}")
                    lines.append("")
            else:
                # Fallback to old format if no scenario comparisons
                lines.append(f"🎯 Primary Cause:")
                lines.append(f"   {self.target_feature}")
                lines.append(f"   Current: {self.current_value:.2f} → Target: {self.target_value:.2f}")
                lines.append(f"   Change needed: {self.delta:+.2f} ({self.delta_percent:+.1f}%)")
                lines.append("")

            lines.append(f"💡 Actionable Recommendation:")
            lines.append(f"   {self.actionable_recommendation}")
            lines.append("")

            lines.append(f"⚡ Computation Time: {self.search_time_ms:.1f}ms")
        else:
            lines.append(f"⚠️ COMPLEX ANOMALY: Simple prevention not found")
            lines.append(f"   This anomaly likely requires multiple interventions")
            lines.append(f"   or is caused by external factors.")

        lines.append("")
        lines.append("─" * 80)

        return "\n".join(lines)


class CounterfactualAnalyzer:
    """
    Real-time counterfactual analyzer for anomaly detection.

    Answers: "What changes would have prevented this anomaly?"
    """

    # Feature name mapping to human-readable explanations
    FEATURE_EXPLANATIONS = {
        'cpu_utilization_mean': 'Average CPU usage across services',
        'cpu_utilization_max': 'Maximum CPU usage (hotspot)',
        'memory_pressure_max': 'Maximum memory pressure',
        'error_rate': 'Error rate across services',
        'response_time_p95_mean': 'Average 95th percentile response time',
        'queue_depth_mean': 'Average queue backlog',
        'system_stress_index': 'Overall system stress level',
        'cpu_memory_correlation': 'CPU-Memory correlation pattern',
        'thread_count_mean': 'Average thread count',
    }

    # Actionable recommendations based on feature
    FEATURE_RECOMMENDATIONS = {
        'cpu_utilization_mean': 'Reduce overall workload or scale horizontally (add replicas)',
        'cpu_utilization_max': 'Identify and optimize the CPU hotspot service',
        'memory_pressure_max': 'Increase memory limits or fix memory leaks',
        'error_rate': 'Fix error-prone code paths or improve error handling',
        'response_time_p95_mean': 'Optimize slow operations or add caching',
        'queue_depth_mean': 'Increase processing capacity or add queue workers',
        'system_stress_index': 'Reduce overall load or scale system resources',
        'thread_count_mean': 'Optimize thread usage or increase thread pool limits',
    }

    def __init__(self, model, feature_names: List[str], max_search_iterations: int = 30):
        """
        Initialize counterfactual analyzer.

        Args:
            model: Trained XGBoost model
            feature_names: List of 27 feature names
            max_search_iterations: Maximum binary search iterations
        """
        self.model = model
        self.feature_names = feature_names
        self.max_search_iterations = max_search_iterations

        # Determine class order from model
        # XGBoost typically uses alphabetical order: cpu_spike, memory_leak, normal, service_crash
        self.class_names = ['cpu_spike', 'memory_leak', 'normal', 'service_crash']
        self.normal_class_idx = 2  # Position of 'normal' in class list

    def predict_scenario_outcome(
        self,
        modified_features: np.ndarray
    ) -> Tuple[str, float]:
        """
        Predict the outcome of a counterfactual scenario.

        Args:
            modified_features: Feature vector with counterfactual changes applied

        Returns:
            Tuple of (predicted_class, confidence)
        """
        try:
            prediction_proba = self.model.predict_proba(modified_features.reshape(1, -1))[0]
            predicted_class_idx = np.argmax(prediction_proba)
            predicted_class = self.class_names[predicted_class_idx]
            confidence = prediction_proba[predicted_class_idx]
            return predicted_class, confidence
        except Exception as e:
            # Fallback to unknown
            return 'unknown', 0.0

    def score_scenario(
        self,
        prevents_anomaly: bool,
        confidence: float,
        delta_percent: float,
        original_prediction: str
    ) -> Tuple[float, str]:
        """
        Score a counterfactual scenario (0-100).

        Factors:
        - Prevents anomaly: 50 points
        - Confidence level: 30 points
        - Feasibility (smaller change = better): 20 points

        Args:
            prevents_anomaly: Whether scenario predicts normal state
            confidence: Model confidence in prediction
            delta_percent: Percentage change required
            original_prediction: Original anomaly type

        Returns:
            Tuple of (score, risk_level)
        """
        score = 0.0

        # Factor 1: Does it prevent the anomaly? (50 points)
        if prevents_anomaly:
            score += 50.0

        # Factor 2: Confidence in prediction (30 points)
        score += confidence * 30.0

        # Factor 3: Feasibility - smaller changes are better (20 points)
        abs_delta = abs(delta_percent)
        if abs_delta < 20:
            feasibility_score = 20.0
        elif abs_delta < 40:
            feasibility_score = 15.0
        elif abs_delta < 60:
            feasibility_score = 10.0
        elif abs_delta < 80:
            feasibility_score = 5.0
        else:
            feasibility_score = 0.0
        
        score += feasibility_score

        # Determine risk level based on change magnitude
        if abs_delta < 30:
            risk_level = 'low'
        elif abs_delta < 60:
            risk_level = 'medium'
        else:
            risk_level = 'high'

        return score, risk_level

    def analyze(
        self,
        features: np.ndarray,
        original_prediction: str,
        shap_values: Optional[np.ndarray] = None,
        top_k: int = 5  # Test top 5 features (increased from 3)
    ) -> Optional[CounterfactualExplanation]:
        """
        Perform real-time counterfactual analysis.

        Args:
            features: 27-dimensional feature vector (current snapshot)
            original_prediction: Original anomaly prediction
            shap_values: Optional SHAP importance values (if available)
            top_k: Number of top features to test

        Returns:
            CounterfactualExplanation if successful, None otherwise
        """
        start_time = time.time()

        # Don't analyze normal predictions
        if original_prediction == 'normal':
            return None

        # Get feature importance (use SHAP if available, else use model feature_importances)
        if shap_values is not None and len(shap_values.shape) > 0:
            importance = np.abs(shap_values)
        else:
            try:
                # Get feature importances from model
                importance = self.model.feature_importances_
            except:
                # Fallback: use uniform importance
                importance = np.ones(len(features))

        # Get top-k most important features
        top_indices = np.argsort(importance)[-top_k:][::-1]

        # Try to find counterfactual for each top feature
        best_result = None
        best_delta = float('inf')
        alternative_actions = []

        for feature_idx in top_indices:
            result = self._binary_search_counterfactual(
                features,
                feature_idx,
                original_prediction
            )

            if result['success']:
                # Track alternative actions
                alternative_actions.append({
                    'feature': self.feature_names[feature_idx],
                    'current': result['original_value'],
                    'target': result['target_value'],
                    'delta': result['delta'],
                    'delta_percent': result['delta_percent']
                })

                # Keep best (smallest delta)
                if abs(result['delta']) < abs(best_delta):
                    best_delta = result['delta']
                    best_result = result

        # Build explanation with scenario comparisons
        if best_result is not None:
            feature_name = self.feature_names[best_result['feature_idx']]

            # Get human-readable explanation
            feature_explanation = self.FEATURE_EXPLANATIONS.get(
                feature_name,
                feature_name.replace('_', ' ').title()
            )

            # Get actionable recommendation
            recommendation = self.FEATURE_RECOMMENDATIONS.get(
                feature_name,
                f"Adjust {feature_name} to avoid this anomaly"
            )

            # Check feasibility (is the change realistic?)
            is_feasible = self._check_feasibility(
                best_result['delta_percent']
            )

            search_time_ms = (time.time() - start_time) * 1000

            # Remove best result from alternatives
            alternative_actions = [
                a for a in alternative_actions
                if a['feature'] != feature_name
            ]

            # Generate scenario comparisons for all successful counterfactuals
            scenario_comparisons = []
            best_score = 0.0
            best_scenario_idx = 0
            
            # Include the best result as first scenario
            all_results = [best_result] + [
                {'feature_idx': self.feature_names.index(alt['feature']),
                 'original_value': alt['current'],
                 'target_value': alt['target'],
                 'delta': alt['delta'],
                 'delta_percent': alt['delta_percent'],
                 'confidence': 0.0}  # Will be recalculated
                for alt in alternative_actions
            ]
            
            for idx, result in enumerate(all_results):
                if isinstance(result['feature_idx'], str):
                    continue  # Skip malformed entries
                    
                feat_idx = result['feature_idx']
                feat_name = self.feature_names[feat_idx]
                
                # Create modified feature vector
                modified_features = features.copy()
                modified_features[feat_idx] = result['target_value']
                
                # Predict outcome
                predicted_class, pred_confidence = self.predict_scenario_outcome(modified_features)
                
                # Check if it prevents anomaly
                prevents_anomaly = (predicted_class == 'normal' and pred_confidence > 0.5)
                
                # Score the scenario
                score, risk_level = self.score_scenario(
                    prevents_anomaly,
                    pred_confidence,
                    result['delta_percent'],
                    original_prediction
                )
                
                # Track best scenario
                if score > best_score:
                    best_score = score
                    best_scenario_idx = idx
                
                # Create scenario comparison
                scenario = ScenarioComparison(
                    scenario_name=feat_name,
                    feature_changes={
                        feat_name: {
                            'current': result['original_value'],
                            'target': result['target_value'],
                            'delta': result['delta'],
                            'delta_percent': result['delta_percent']
                        }
                    },
                    predicted_class=predicted_class,
                    predicted_confidence=pred_confidence,
                    prevents_anomaly=prevents_anomaly,
                    score=score,
                    risk_level=risk_level
                )
                
                scenario_comparisons.append(scenario)
            
            # Sort scenarios by score (highest first)
            scenario_comparisons.sort(key=lambda x: x.score, reverse=True)
            
            # Update best scenario index after sorting
            best_scenario_idx = 0  # Highest score is now first

            explanation = CounterfactualExplanation(
                target_feature=feature_explanation,
                current_value=best_result['original_value'],
                target_value=best_result['target_value'],
                delta=best_result['delta'],
                delta_percent=best_result['delta_percent'],
                alternative_actions=alternative_actions,
                scenario_comparisons=scenario_comparisons,
                is_feasible=is_feasible,
                confidence=best_result['confidence'],
                search_time_ms=search_time_ms,
                actionable_recommendation=recommendation,
                best_scenario_idx=best_scenario_idx
            )

            return explanation

        return None

    def _binary_search_counterfactual(
        self,
        features: np.ndarray,
        feature_idx: int,
        original_prediction: str
    ) -> Dict:
        """
        Binary search to find minimum change that flips prediction to 'normal'.

        Uses a simpler linear search approach for reliability.
        """
        original_value = features[feature_idx]

        # Try decreasing the value in steps
        best_result = None

        for direction in [-1, 1]:  # -1 = decrease, +1 = increase
            # Test progressively larger changes
            for delta_multiplier in np.linspace(0.1, 1.0, 20):
                delta = direction * original_value * delta_multiplier
                test_value = original_value + delta

                # Clamp to valid range
                test_value = max(0.0, min(1.5, test_value))

                # Skip if no change
                if abs(test_value - original_value) < 1e-6:
                    continue

                # Create modified feature vector
                modified_features = features.copy()
                modified_features[feature_idx] = test_value

                # Predict
                try:
                    prediction_proba = self.model.predict_proba(modified_features.reshape(1, -1))[0]
                    predicted_class_idx = np.argmax(prediction_proba)
                    predicted_class = self.class_names[predicted_class_idx]
                    confidence = prediction_proba[predicted_class_idx]

                    # Check if flipped to normal
                    if predicted_class == 'normal' and confidence > 0.5:
                        actual_delta = test_value - original_value
                        delta_percent = (actual_delta / (abs(original_value) + 1e-6)) * 100

                        result = {
                            'success': True,
                            'feature_idx': feature_idx,
                            'original_value': original_value,
                            'target_value': test_value,
                            'delta': actual_delta,
                            'delta_percent': delta_percent,
                            'confidence': confidence
                        }

                        # Keep best (smallest absolute delta)
                        if best_result is None or abs(actual_delta) < abs(best_result['delta']):
                            best_result = result

                except Exception as e:
                    continue

        if best_result is not None:
            return best_result

        # No counterfactual found
        return {
            'success': False,
            'feature_idx': feature_idx,
            'original_value': original_value
        }

    def _check_feasibility(self, delta_percent: float) -> bool:
        """
        Check if the required change is feasible in practice.

        Args:
            delta_percent: Percentage change required

        Returns:
            True if feasible (change < 80%), False otherwise
        """
        # Consider feasible if change is less than 80% (relaxed for Kubernetes environments)
        return abs(delta_percent) < 80.0


# Convenience function for use in main pipeline
def generate_counterfactual_explanation(
    model,
    feature_names: List[str],
    features: np.ndarray,
    anomaly_type: str,
    shap_values: Optional[np.ndarray] = None
) -> Optional[CounterfactualExplanation]:
    """
    Quick function to generate counterfactual explanation.

    Usage in online detector:
        explanation = generate_counterfactual_explanation(
            model=xgboost_model,
            feature_names=feature_names,
            features=snapshot.features_scaleinvariant,
            anomaly_type=snapshot.anomaly_type
        )

        if explanation:
            print(explanation.format_human_readable())
    """
    analyzer = CounterfactualAnalyzer(model, feature_names)
    return analyzer.analyze(features, anomaly_type, shap_values)


if __name__ == "__main__":
    print("Counterfactual Analyzer Module")
    print("Import this module to use CounterfactualAnalyzer class")
