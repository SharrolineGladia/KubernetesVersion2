"""
EWMA-based Anomaly Detection Evaluation Script
Tests the EWMA detector with various synthetic patterns to measure performance.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import online_detector.config as config
from online_detector.detector import EWMAMetric, ResourceSaturationDetector


class EWMAEvaluator:
    def __init__(self):
        self.results = {}
        
    def generate_normal_data(self, n_samples: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """Generate normal operational pattern."""
        data = np.random.normal(0.3, 0.1, n_samples)
        data = np.clip(data, 0, 1)
        labels = np.zeros(n_samples)
        return data, labels
    
    def generate_spike_pattern(self, n_samples: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """Generate sudden spike pattern."""
        data = np.random.normal(0.3, 0.05, n_samples)
        labels = np.zeros(n_samples)
        
        # Inject spike
        spike_start = 200
        spike_duration = 20
        data[spike_start:spike_start + spike_duration] = np.random.normal(0.85, 0.05, spike_duration)
        labels[spike_start:spike_start + spike_duration] = 1
        
        data = np.clip(data, 0, 1)
        return data, labels
    
    def generate_gradual_increase_pattern(self, n_samples: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """Generate gradual degradation pattern (e.g., memory leak)."""
        data = np.zeros(n_samples)
        labels = np.zeros(n_samples)
        
        # Normal phase
        data[:150] = np.random.normal(0.3, 0.05, 150)
        
        # Gradual increase
        increase_start = 150
        increase_duration = 200
        baseline = 0.3
        target = 0.8
        
        for i in range(increase_duration):
            progress = i / increase_duration
            mean_value = baseline + (target - baseline) * progress
            data[increase_start + i] = np.random.normal(mean_value, 0.05)
            if mean_value > 0.55:
                labels[increase_start + i] = 1
        
        # Sustained high
        data[increase_start + increase_duration:] = np.random.normal(0.8, 0.05, 
                                                                      n_samples - increase_start - increase_duration)
        labels[increase_start + increase_duration:] = 1
        
        data = np.clip(data, 0, 1)
        return data, labels
    
    def generate_sustained_high_pattern(self, n_samples: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """Generate sustained high utilization."""
        data = np.random.normal(0.3, 0.05, n_samples)
        labels = np.zeros(n_samples)
        
        # Sustained high period
        high_start = 200
        high_duration = 150
        data[high_start:high_start + high_duration] = np.random.normal(0.75, 0.05, high_duration)
        labels[high_start:high_start + high_duration] = 1
        
        data = np.clip(data, 0, 1)
        return data, labels
    
    def generate_oscillating_pattern(self, n_samples: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """Generate oscillating pattern."""
        data = np.zeros(n_samples)
        labels = np.zeros(n_samples)
        
        # Normal phase
        data[:100] = np.random.normal(0.3, 0.05, 100)
        
        # Oscillating phase
        osc_start = 100
        osc_duration = 300
        period = 50
        
        for i in range(osc_duration):
            t = i / period
            base_value = 0.3 + 0.4 * (np.sin(2 * np.pi * t) + 1) / 2
            data[osc_start + i] = np.random.normal(base_value, 0.05)
            if base_value > 0.55:
                labels[osc_start + i] = 1
        
        # Return to normal
        data[osc_start + osc_duration:] = np.random.normal(0.3, 0.05, 
                                                            n_samples - osc_start - osc_duration)
        
        data = np.clip(data, 0, 1)
        return data, labels
    
    def generate_multiple_spikes_pattern(self, n_samples: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """Generate multiple spike pattern."""
        data = np.random.normal(0.3, 0.05, n_samples)
        labels = np.zeros(n_samples)
        
        spike_positions = [100, 200, 300, 400]
        spike_duration = 15
        
        for pos in spike_positions:
            if pos + spike_duration < n_samples:
                data[pos:pos + spike_duration] = np.random.normal(0.85, 0.05, spike_duration)
                labels[pos:pos + spike_duration] = 1
        
        data = np.clip(data, 0, 1)
        return data, labels
    
    def generate_recovery_pattern(self, n_samples: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """Generate anomaly with recovery pattern."""
        data = np.zeros(n_samples)
        labels = np.zeros(n_samples)
        
        # Normal
        data[:150] = np.random.normal(0.3, 0.05, 150)
        
        # Spike
        spike_start = 150
        spike_duration = 50
        data[spike_start:spike_start + spike_duration] = np.random.normal(0.85, 0.05, spike_duration)
        labels[spike_start:spike_start + spike_duration] = 1
        
        # Gradual recovery
        recovery_start = spike_start + spike_duration
        recovery_duration = 100
        for i in range(recovery_duration):
            progress = i / recovery_duration
            mean_value = 0.85 - 0.55 * progress
            data[recovery_start + i] = np.random.normal(mean_value, 0.05)
            if mean_value > 0.55:
                labels[recovery_start + i] = 1
        
        # Back to normal
        data[recovery_start + recovery_duration:] = np.random.normal(0.3, 0.05, 
                                                                       n_samples - recovery_start - recovery_duration)
        
        data = np.clip(data, 0, 1)
        return data, labels
    
    def evaluate_pattern(self, data: np.ndarray, labels: np.ndarray, pattern_name: str) -> Dict:
        """Evaluate EWMA detector on a pattern."""
        detector = ResourceSaturationDetector()
        predictions = []
        stress_scores = []  # Track stress scores for visualization
        detection_times = []
        anomaly_started = False
        first_detection_time = None
        
        for i, value in enumerate(data):
            # Scale from 0-1 to 0-100 (CPU percentage)
            cpu_percent = value * 100.0
            mem_percent = 30.0  # Constant baseline
            thread_count = 20.0  # Constant baseline
            
            # Update detector with CPU value (memory and threads are constant)
            component_stresses = detector.update(
                cpu_val=cpu_percent,
                mem_val=mem_percent,
                thread_val=thread_count
            )
            
            # Calculate weighted stress score (same as in production)
            stress_score = (
                0.5 * component_stresses['cpu'] +
                0.3 * component_stresses['memory'] +
                0.2 * component_stresses['threads']
            )
            stress_scores.append(stress_score)
            
            # Update channel state with stress score
            result = detector.update_channel_state(stress_score)
            
            # Detector is anomalous if in STRESSED or CRITICAL state
            is_anomaly = result['state'] in ['stressed', 'critical']
            predictions.append(1 if is_anomaly else 0)
            
            # Track detection latency
            if labels[i] == 1 and not anomaly_started:
                anomaly_started = True
                first_detection_time = None
            
            if anomaly_started and is_anomaly and first_detection_time is None:
                first_detection_time = i
                if np.any(labels[:i] == 1):
                    actual_start = np.where(labels == 1)[0][0]
                    detection_times.append(i - actual_start)
            
            if labels[i] == 0:
                anomaly_started = False
        
        predictions = np.array(predictions)
        
        # Calculate metrics
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        tn = np.sum((predictions == 0) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # False Alarm Rate
        total_normal = np.sum(labels == 0)
        far = (fp / total_normal * 100) if total_normal > 0 else 0
        
        # Average detection latency
        avg_latency = np.mean(detection_times) * 5 if detection_times else 0  # *5 for 5-second intervals
        
        results = {
            'pattern': pattern_name,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'false_alarm_rate': float(far),
            'detection_latency_seconds': float(avg_latency),
            'total_samples': len(data),
            'anomaly_samples': int(np.sum(labels)),
            'normal_samples': int(np.sum(labels == 0))
        }
        
        return results, predictions, np.array(stress_scores)
    
    def plot_pattern(self, data: np.ndarray, labels: np.ndarray, predictions: np.ndarray, 
                    stress_scores: np.ndarray, pattern_name: str, metrics: Dict):
        """Generate publication-quality visualization for a pattern."""
        
        # Conceptual titles for each pattern
        titles = {
            'normal': 'Normal Operation Baseline',
            'spike': 'Rapid Detection of Sudden Resource Spike',
            'gradual_increase': 'Detection of Gradual Resource Degradation',
            'sustained_high': 'Sustained High Utilization Detection',
            'oscillating': 'Oscillating Load Pattern Detection',
            'multiple_spikes': 'Multiple Spike Event Detection',
            'recovery': 'Anomaly Detection with System Recovery'
        }
        
        # Single plot for publication quality
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        
        time = np.arange(len(data)) * 5  # 5-second intervals
        
        # Ground truth anomaly regions (light red shading - bottom layer)
        ax.fill_between(time, 0, 1, where=labels == 1, alpha=0.3, color='#E63946', 
                       label='Ground Truth Anomaly', zorder=1)
        
        # Detected anomaly regions (light blue shading - top layer)
        ax.fill_between(time, 0, 1, where=predictions == 1, alpha=0.3, color='#457B9D',
                       label='Detected Anomaly', zorder=2)
        
        # Plot CPU usage as primary line on top
        ax.plot(time, data, label='CPU Utilization', color='#1D3557', alpha=1.0, linewidth=1.0, zorder=3)
        
        # Optional: Overlay combined stress score as faint secondary line
        # This shows the hybrid mechanism in action
        ax2 = ax.twinx()
        ax2.plot(time, stress_scores, label='Combined Stress', color='#F77F00', 
                alpha=0.3, linewidth=1, linestyle='--')
        ax2.set_ylabel('Stress Score', fontsize=11, color='#F77F00')
        ax2.tick_params(axis='y', labelcolor='#F77F00')
        ax2.set_ylim(0, 1.0)
        ax2.spines['right'].set_color('#F77F00')
        
        # Axis labels and title
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('CPU Utilization', fontsize=12)
        ax.set_title(titles.get(pattern_name, pattern_name.replace('_', ' ').title()), 
                    fontsize=14, fontweight='bold', pad=15)
        
        # Legend (simplified, outside plot area)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', 
                 bbox_to_anchor=(1.15, 1.0), framealpha=0.95, fontsize=9)
        
        # Clean grid
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        ax.set_ylim(0, 1.0)
        ax.set_xlim(time[0], time[-1])
        
        # Professional styling
        ax.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        
        plt.tight_layout()
        
        output_path = os.path.join(os.path.dirname(__file__), 'ewma', f'plot_{pattern_name}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  ✓ Saved visualization: plot_{pattern_name}.png")
    
    def run_all_evaluations(self):
        """Run evaluations on all patterns."""
        patterns = {
            'normal': self.generate_normal_data,
            'spike': self.generate_spike_pattern,
            'gradual_increase': self.generate_gradual_increase_pattern,
            'sustained_high': self.generate_sustained_high_pattern,
            'oscillating': self.generate_oscillating_pattern,
            'multiple_spikes': self.generate_multiple_spikes_pattern,
            'recovery': self.generate_recovery_pattern
        }
        
        all_results = {}
        
        print("\n" + "="*70)
        print("EWMA Anomaly Detector Evaluation")
        print("="*70 + "\n")
        
        for pattern_name, generator_func in patterns.items():
            print(f"Evaluating pattern: {pattern_name}")
            data, labels = generator_func()
            metrics, predictions, stress_scores = self.evaluate_pattern(data, labels, pattern_name)
            all_results[pattern_name] = metrics
            
            # Generate visualization
            self.plot_pattern(data, labels, predictions, stress_scores, pattern_name, metrics)
            
            # Print results
            print(f"  Precision: {metrics['precision']:.2%}")
            print(f"  Recall: {metrics['recall']:.2%}")
            print(f"  F1-Score: {metrics['f1_score']:.2%}")
            print(f"  False Alarm Rate: {metrics['false_alarm_rate']:.2f}%")
            print(f"  Detection Latency: {metrics['detection_latency_seconds']:.2f}s")
            print()
        
        # Calculate overall metrics
        total_tp = sum(r['true_positives'] for r in all_results.values() if r['pattern'] != 'normal')
        total_fp = sum(r['false_positives'] for r in all_results.values() if r['pattern'] != 'normal')
        total_fn = sum(r['false_negatives'] for r in all_results.values() if r['pattern'] != 'normal')
        
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) \
                     if (overall_precision + overall_recall) > 0 else 0
        
        # Average FAR from normal pattern
        avg_far = all_results['normal']['false_alarm_rate']
        
        # Average detection latency
        latencies = [r['detection_latency_seconds'] for r in all_results.values() 
                    if r['pattern'] != 'normal' and r['detection_latency_seconds'] > 0]
        avg_latency = np.mean(latencies) if latencies else 0
        
        all_results['overall'] = {
            'precision': float(overall_precision),
            'recall': float(overall_recall),
            'f1_score': float(overall_f1),
            'false_alarm_rate': float(avg_far),
            'detection_latency_seconds': float(avg_latency)
        }
        
        # Save results
        output_path = os.path.join(os.path.dirname(__file__), 'ewma', 'results.json')
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print("="*70)
        print("Overall Performance")
        print("="*70)
        print(f"Precision: {overall_precision:.2%}")
        print(f"Recall: {overall_recall:.2%}")
        print(f"F1-Score: {overall_f1:.2%}")
        print(f"False Alarm Rate: {avg_far:.2f}%")
        print(f"Average Detection Latency: {avg_latency:.2f}s")
        print("\n✓ Results saved to: ewma/results.json")
        print("="*70 + "\n")
        
        return all_results


if __name__ == '__main__':
    evaluator = EWMAEvaluator()
    results = evaluator.run_all_evaluations()
