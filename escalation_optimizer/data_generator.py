"""
Anomaly Data Generator for Escalation Evaluation

Generates realistic anomaly detection scenarios using the trained XGBoost model
to obtain genuine confidence scores for different anomaly types.
"""

import sys
import os
import pickle
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from datetime import datetime, timedelta

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ml_detector', 'scripts'))
from dual_feature_detector import DualFeatureDetector, ServiceMetrics


@dataclass
class AnomalyScenario:
    """Represents a single anomaly detection scenario."""
    scenario_id: int
    timestamp: datetime
    anomaly_type: str  # 'normal', 'cpu_spike', 'memory_leak', 'service_crash'
    true_label: str    # Ground truth
    predicted_label: str  # Model prediction
    confidence: float  # Model confidence score (0-1)
    service_count: int  # Number of services in scenario
    service_metrics: Dict[str, ServiceMetrics]
    features_scaleinvariant: np.ndarray  # 27-dim feature vector
    is_correct: bool  # Whether prediction matches ground truth
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV export."""
        return {
            'scenario_id': self.scenario_id,
            'timestamp': self.timestamp.isoformat(),
            'true_label': self.true_label,
            'predicted_label': self.predicted_label,
            'confidence': self.confidence,
            'service_count': self.service_count,
            'is_correct': self.is_correct
        }


class AnomalyDataGenerator:
    """
    Generates realistic anomaly scenarios using trained XGBoost model.
    
    Loads samples from the training dataset and uses the model to generate
    real confidence scores (not synthetic).
    """
    
    def __init__(self, model_path: str, dataset_path: str):
        """
        Initialize generator.
        
        Args:
            model_path: Path to trained XGBoost model
            dataset_path: Path to scale-invariant dataset CSV
        """
        self.model_path = model_path
        self.dataset_path = dataset_path
        
        # Load detector
        self.detector = DualFeatureDetector(model_path=model_path, confidence_threshold=0.0)
        
        # Load dataset
        self.dataset = pd.read_csv(dataset_path)
        
        # DO NOT strip whitespace - the model was trained with labels that have leading spaces!
        # (e.g., ' normal', ' cpu_spike', ' memory_leak', ' service_crash')
        
        print(f"✅ Loaded dataset: {len(self.dataset)} samples")
        
        # Separate by anomaly type (note: labels have leading space in the dataset!)
        self.samples_by_type = {
            'normal': self.dataset[self.dataset['anomaly_type'] == ' normal'],
            'cpu_spike': self.dataset[self.dataset['anomaly_type'] == ' cpu_spike'],
            'memory_leak': self.dataset[self.dataset['anomaly_type'] == ' memory_leak'],
            'service_crash': self.dataset[self.dataset['anomaly_type'] == ' service_crash']
        }
        
        print(f"   Normal:        {len(self.samples_by_type['normal'])} samples")
        print(f"   CPU Spike:     {len(self.samples_by_type['cpu_spike'])} samples")
        print(f"   Memory Leak:   {len(self.samples_by_type['memory_leak'])} samples")
        print(f"   Service Crash: {len(self.samples_by_type['service_crash'])} samples")
        
        # Feature columns (27 scale-invariant features)
        self.feature_columns = [
            'cpu_utilization_mean', 'cpu_utilization_max', 'cpu_variance_coef',
            'cpu_imbalance', 'memory_pressure_mean', 'memory_pressure_max',
            'memory_variance_coef', 'memory_imbalance', 'network_in_rate',
            'network_out_rate', 'network_in_variance_coef', 'network_out_variance_coef',
            'network_asymmetry', 'disk_io_rate', 'disk_io_variance_coef',
            'request_rate', 'request_variance_coef', 'error_rate',
            'error_variance_coef', 'latency_mean', 'latency_p95',
            'latency_variance_coef', 'system_stress', 'resource_efficiency',
            'service_density', 'cpu_memory_correlation', 'performance_degradation'
        ]
        
        # Anomaly class mapping
        self.label_mapping = {'normal': 0, 'cpu_spike': 1, 'memory_leak': 2, 'service_crash': 3}
    
    def generate_scenarios(
        self,
        n_samples: int = 1000,
        distribution: Dict[str, float] = None
    ) -> List[AnomalyScenario]:
        """
        Generate anomaly scenarios using real model predictions.
        
        Args:
            n_samples: Total number of scenarios to generate
            distribution: Dict mapping anomaly type -> proportion
                        Default: {'normal': 0.5, 'cpu_spike': 0.2, 'memory_leak': 0.15, 'service_crash': 0.15}
        
        Returns:
            List of AnomalyScenario objects with real confidence scores
        """
        if distribution is None:
            distribution = {
                'normal': 0.5,
                'cpu_spike': 0.2,
                'memory_leak': 0.15,
                'service_crash': 0.15
            }
        
        print(f"\n🔄 Generating {n_samples} anomaly scenarios...")
        print(f"   Distribution: {distribution}")
        
        scenarios = []
        base_time = datetime(2026, 2, 23, 10, 0, 0)
        
        for anomaly_type, proportion in distribution.items():
            n_type = int(n_samples * proportion)
            samples = self.samples_by_type[anomaly_type].sample(n=n_type, replace=True, random_state=42)
            
            print(f"\n   Processing {anomaly_type}: {n_type} samples")
            
            for idx, (_, row) in enumerate(samples.iterrows()):
                # Extract true label from dataset (has leading space!)
                true_label = row['anomaly_type']
                
                # Extract features
                features = row[self.feature_columns].values.astype(float)
                
                # Get model prediction and confidence using the label encoder
                features_reshaped = features.reshape(1, -1)
                probabilities = self.detector.model.predict_proba(features_reshaped)[0]
                predicted_class_idx = np.argmax(probabilities)
                # Use the label encoder to get the actual label (preserves spaces!)
                predicted_label = self.detector.label_encoder.inverse_transform([predicted_class_idx])[0]
                confidence = probabilities[predicted_class_idx]
                
                # Create service metrics (simulated based on features)
                service_metrics = self._create_service_metrics_from_features(features, row)
                
                # Create scenario
                scenario = AnomalyScenario(
                    scenario_id=len(scenarios) + 1,
                    timestamp=base_time + timedelta(seconds=len(scenarios) * 5),
                    anomaly_type=true_label,
                    true_label=true_label,
                    predicted_label=predicted_label,
                    confidence=confidence,
                    service_count=3,  # Dataset trained on 3 services
                    service_metrics=service_metrics,
                    features_scaleinvariant=features,
                    is_correct=(predicted_label == true_label)
                )
                
                scenarios.append(scenario)
        
        print(f"\n✅ Generated {len(scenarios)} scenarios")
        
        # Print confidence distribution
        confidences = [s.confidence for s in scenarios]
        print(f"\n📊 Confidence Distribution:")
        print(f"   Mean:   {np.mean(confidences):.3f}")
        print(f"   Median: {np.median(confidences):.3f}")
        print(f"   Std:    {np.std(confidences):.3f}")
        print(f"   Min:    {np.min(confidences):.3f}")
        print(f"   Max:    {np.max(confidences):.3f}")
        
        # Print accuracy
        accuracy = sum(s.is_correct for s in scenarios) / len(scenarios)
        print(f"\n📈 Overall Accuracy: {accuracy:.2%}")
        
        return scenarios
    
    def _create_service_metrics_from_features(
        self,
        features: np.ndarray,
        row: pd.Series
    ) -> Dict[str, ServiceMetrics]:
        """
        Create ServiceMetrics from feature vector.
        
        Since we only have aggregate features, we simulate 3 services with
        variations around the mean values.
        """
        # Extract key metrics
        cpu_mean = features[0] * 100  # Convert to percentage
        memory_mean = features[4] * 100
        error_rate = features[17]
        latency_mean = features[19] * 1000  # Convert to ms
        request_rate = features[15] * 100
        
        # Create 3 services with variations
        services = {}
        for i in range(3):
            service_name = f"service-{i+1}"
            
            # Add random variation (±10%)
            variation = 1.0 + np.random.uniform(-0.1, 0.1)
            
            services[service_name] = ServiceMetrics(
                cpu_percent=cpu_mean * variation,
                memory_percent=memory_mean * variation,
                error_rate=error_rate * variation,
                request_rate=request_rate * variation,
                response_time_p95=latency_mean * variation,
                thread_count=int(20 * variation),  # Simulated thread count
                queue_depth=features[19] * 10 * variation,  # Simulated queue depth
                requests_per_second=request_rate * variation
            )
        
        return services
    
    def save_scenarios(self, scenarios: List[AnomalyScenario], output_path: str):
        """Save scenarios to CSV."""
        df = pd.DataFrame([s.to_dict() for s in scenarios])
        df.to_csv(output_path, index=False)
        print(f"\n💾 Saved {len(scenarios)} scenarios to {output_path}")


def main():
    """Generate evaluation dataset."""
    
    print("=" * 80)
    print("  ANOMALY SCENARIO GENERATOR")
    print("  Bandwidth-Aware Escalation Evaluation")
    print("=" * 80)
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, 'ml_detector', 'models', 'anomaly_detector_scaleinvariant.pkl')
    dataset_path = os.path.join(base_dir, 'ml_detector', 'datasets', 'metrics_dataset_scaleinvariant.csv')
    output_path = os.path.join(base_dir, 'results', 'escalation', 'anomaly_scenarios.csv')
    
    # Initialize generator
    generator = AnomalyDataGenerator(model_path=model_path, dataset_path=dataset_path)
    
    # Generate scenarios (1000 samples for robust evaluation)
    scenarios = generator.generate_scenarios(n_samples=1000)
    
    # Save to CSV
    generator.save_scenarios(scenarios, output_path)
    
    print("\n" + "=" * 80)
    print("✅ GENERATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
