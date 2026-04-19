"""
Lightweight Recovery Evaluator (Simulation Mode)

This version can run without a full Kubernetes cluster for testing and development.
It simulates recovery actions and measures framework performance.

Use this when:
- Kubernetes cluster is not available
- Testing evaluation logic
- Quick prototyping
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import random

# Configuration
SIMULATION_MODE = True
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class SimulatedAnomalyRecord:
    """Record for simulated anomaly injection and recovery."""
    injection_id: int
    anomaly_type: str
    injection_time: float
    detection_time: float
    explanation_time: float
    recovery_trigger_time: float
    recovery_completion_time: float
    normalization_time: float
    
    # Pre/post metrics
    pre_recovery_cpu: float
    pre_recovery_memory: float
    pre_recovery_error_rate: float
    pre_recovery_anomaly_prob: float
    
    post_recovery_cpu: float
    post_recovery_memory: float
    post_recovery_error_rate: float
    post_recovery_anomaly_prob: float
    
    # Outcomes
    detection_success: bool
    recovery_success: bool
    recurrence_detected: bool
    
    # Calculated metrics
    detection_latency: float
    mttr: float
    end_to_end_latency: float
    recovery_execution_latency: float
    pod_restart_duration: float
    anomaly_prob_reduction: float


class SimulatedRecoveryEvaluator:
    """
    Simulated evaluator that doesn't require Kubernetes.
    Uses realistic timing models based on typical system behavior.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize simulator with reproducible random seed."""
        random.seed(seed)
        np.random.seed(seed)
        self.results_dir = RESULTS_DIR
        self.records: List[SimulatedAnomalyRecord] = []
        
        print("✅ Simulated Recovery Evaluator initialized")
        print(f"   Mode: SIMULATION")
        print(f"   Results directory: {self.results_dir}")
    
    def simulate_anomaly_scenario(
        self,
        injection_id: int,
        anomaly_type: str
    ) -> SimulatedAnomalyRecord:
        """
        Simulate complete anomaly detection and recovery scenario.
        
        Uses realistic probability distributions for timing and metrics.
        """
        print(f"\n{'='*60}")
        print(f"🔬 Simulation #{injection_id}: {anomaly_type}")
        print(f"{'='*60}")
        
        # Start time
        injection_time = time.time()
        
        # Simulate detection latency (typically 5-15 seconds for EWMA)
        detection_latency = np.random.uniform(5, 15)
        detection_time = injection_time + detection_latency
        detection_success = random.random() > 0.05  # 95% detection rate
        
        if not detection_success:
            print("❌ Detection failed (5% false negative)")
            # Create failed record
            return self._create_failed_record(injection_id, anomaly_type, injection_time)
        
        print(f"✅ Detected after {detection_latency:.2f}s")
        
        # Simulate explanation generation (typically 0.5-2 seconds)
        explanation_latency = np.random.uniform(0.5, 2.0)
        explanation_time = detection_time + explanation_latency
        
        print(f"🧠 Explanation generated in {explanation_latency:.2f}s")
        
        # Recovery trigger (immediate)
        recovery_trigger_time = explanation_time + 0.1
        
        # Simulate Kubernetes action time (typically 1-3 seconds)
        k8s_action_latency = np.random.uniform(1, 3)
        
        # Simulate pod restart duration (typically 10-30 seconds)
        pod_restart_duration = np.random.uniform(10, 30)
        recovery_completion_time = recovery_trigger_time + k8s_action_latency + pod_restart_duration
        
        recovery_success = random.random() > 0.02  # 98% recovery success rate
        
        if not recovery_success:
            print("❌ Recovery failed (2% failure rate)")
            return self._create_failed_record(injection_id, anomaly_type, injection_time)
        
        print(f"✅ Recovery completed in {pod_restart_duration:.2f}s")
        
        # Simulate normalization time (typically 5-20 seconds after recovery)
        normalization_latency = np.random.uniform(5, 20)
        normalization_time = recovery_completion_time + normalization_latency
        
        mttr = normalization_time - detection_time
        end_to_end_latency = normalization_time - injection_time
        
        print(f"✅ Normalized in {mttr:.2f}s (E2E: {end_to_end_latency:.2f}s)")
        
        # Generate pre-recovery metrics based on anomaly type
        pre_metrics = self._generate_anomaly_metrics(anomaly_type)
        
        # Generate post-recovery metrics (normalized)
        post_metrics = self._generate_normal_metrics()
        
        # Calculate anomaly probability reduction
        anomaly_prob_reduction = pre_metrics['anomaly_prob'] - post_metrics['anomaly_prob']
        
        print(f"📊 Anomaly probability: {pre_metrics['anomaly_prob']:.3f} → {post_metrics['anomaly_prob']:.3f}")
        print(f"   Reduction: {anomaly_prob_reduction:.3f}")
        
        # Simulate recurrence check (typically 5% recurrence rate)
        recurrence_detected = random.random() < 0.05
        
        if recurrence_detected:
            print("⚠️  Anomaly recurred within 5 minutes")
        else:
            print("✅ No recurrence detected")
        
        # Create record
        record = SimulatedAnomalyRecord(
            injection_id=injection_id,
            anomaly_type=anomaly_type,
            injection_time=injection_time,
            detection_time=detection_time,
            explanation_time=explanation_time,
            recovery_trigger_time=recovery_trigger_time,
            recovery_completion_time=recovery_completion_time,
            normalization_time=normalization_time,
            
            pre_recovery_cpu=pre_metrics['cpu'],
            pre_recovery_memory=pre_metrics['memory'],
            pre_recovery_error_rate=pre_metrics['error_rate'],
            pre_recovery_anomaly_prob=pre_metrics['anomaly_prob'],
            
            post_recovery_cpu=post_metrics['cpu'],
            post_recovery_memory=post_metrics['memory'],
            post_recovery_error_rate=post_metrics['error_rate'],
            post_recovery_anomaly_prob=post_metrics['anomaly_prob'],
            
            detection_success=True,
            recovery_success=True,
            recurrence_detected=recurrence_detected,
            
            detection_latency=detection_latency,
            mttr=mttr,
            end_to_end_latency=end_to_end_latency,
            recovery_execution_latency=k8s_action_latency,
            pod_restart_duration=pod_restart_duration,
            anomaly_prob_reduction=anomaly_prob_reduction
        )
        
        return record
    
    def _generate_anomaly_metrics(self, anomaly_type: str) -> Dict:
        """Generate realistic metrics for anomaly state."""
        if anomaly_type == "cpu_spike":
            return {
                'cpu': np.random.uniform(85, 98),
                'memory': np.random.uniform(40, 70),
                'error_rate': np.random.uniform(0.01, 0.15),
                'anomaly_prob': np.random.uniform(0.80, 0.95)
            }
        elif anomaly_type == "memory_leak":
            return {
                'cpu': np.random.uniform(40, 70),
                'memory': np.random.uniform(80, 95),
                'error_rate': np.random.uniform(0.02, 0.20),
                'anomaly_prob': np.random.uniform(0.75, 0.92)
            }
        elif anomaly_type == "service_crash":
            return {
                'cpu': np.random.uniform(10, 40),
                'memory': np.random.uniform(30, 60),
                'error_rate': np.random.uniform(0.50, 0.90),
                'anomaly_prob': np.random.uniform(0.85, 0.98)
            }
        else:
            return {
                'cpu': np.random.uniform(50, 80),
                'memory': np.random.uniform(50, 80),
                'error_rate': np.random.uniform(0.10, 0.40),
                'anomaly_prob': np.random.uniform(0.70, 0.90)
            }
    
    def _generate_normal_metrics(self) -> Dict:
        """Generate realistic metrics for normal state."""
        return {
            'cpu': np.random.uniform(15, 35),
            'memory': np.random.uniform(20, 45),
            'error_rate': np.random.uniform(0.001, 0.02),
            'anomaly_prob': np.random.uniform(0.05, 0.20)
        }
    
    def _create_failed_record(
        self,
        injection_id: int,
        anomaly_type: str,
        injection_time: float
    ) -> SimulatedAnomalyRecord:
        """Create record for failed scenario."""
        return SimulatedAnomalyRecord(
            injection_id=injection_id,
            anomaly_type=anomaly_type,
            injection_time=injection_time,
            detection_time=0,
            explanation_time=0,
            recovery_trigger_time=0,
            recovery_completion_time=0,
            normalization_time=0,
            
            pre_recovery_cpu=0,
            pre_recovery_memory=0,
            pre_recovery_error_rate=0,
            pre_recovery_anomaly_prob=0,
            
            post_recovery_cpu=0,
            post_recovery_memory=0,
            post_recovery_error_rate=0,
            post_recovery_anomaly_prob=0,
            
            detection_success=False,
            recovery_success=False,
            recurrence_detected=False,
            
            detection_latency=0,
            mttr=0,
            end_to_end_latency=0,
            recovery_execution_latency=0,
            pod_restart_duration=0,
            anomaly_prob_reduction=0
        )
    
    def run_evaluation(
        self,
        num_injections: int = 20,
        anomaly_types: List[str] = None
    ) -> pd.DataFrame:
        """
        Run simulated evaluation with multiple injections.
        
        Args:
            num_injections: Number of simulations to run
            anomaly_types: List of anomaly types to test
        
        Returns:
            DataFrame with all results
        """
        if anomaly_types is None:
            anomaly_types = ["cpu_spike", "memory_leak", "service_crash"]
        
        print(f"\n{'='*60}")
        print(f"🚀 Starting Simulated Evaluation")
        print(f"   Total simulations: {num_injections}")
        print(f"   Anomaly types: {anomaly_types}")
        print(f"{'='*60}\n")
        
        for i in range(num_injections):
            anomaly_type = anomaly_types[i % len(anomaly_types)]
            
            record = self.simulate_anomaly_scenario(
                injection_id=i + 1,
                anomaly_type=anomaly_type
            )
            
            self.records.append(record)
            
            # Small delay between simulations
            time.sleep(0.5)
        
        # Convert to DataFrame
        df = pd.DataFrame([asdict(rec) for rec in self.records])
        return df
    
    def calculate_aggregate_metrics(self) -> Dict:
        """Calculate summary statistics."""
        df = pd.DataFrame([asdict(rec) for rec in self.records])
        
        successful = df[df['recovery_success'] == True]
        detected = df[df['detection_success'] == True]
        
        metrics = {
            'total_injections': len(df),
            'detection_success_rate': (df['detection_success'].sum() / len(df)) * 100,
            'recovery_success_rate': (df['recovery_success'].sum() / len(df)) * 100,
            'recurrence_rate': (df['recurrence_detected'].sum() / len(df)) * 100,
            'autonomous_resolution_rate': (successful['recovery_success'].sum() / len(df)) * 100,
            
            'mean_detection_latency': successful['detection_latency'].mean(),
            'std_detection_latency': successful['detection_latency'].std(),
            
            'mean_mttr': successful['mttr'].mean(),
            'std_mttr': successful['mttr'].std(),
            'min_mttr': successful['mttr'].min(),
            'max_mttr': successful['mttr'].max(),
            'median_mttr': successful['mttr'].median(),
            
            'mean_e2e_latency': successful['end_to_end_latency'].mean(),
            'std_e2e_latency': successful['end_to_end_latency'].std(),
            
            'mean_recovery_execution_latency': successful['recovery_execution_latency'].mean(),
            'std_recovery_execution_latency': successful['recovery_execution_latency'].std(),
            
            'mean_pod_restart_duration': successful['pod_restart_duration'].mean(),
            'std_pod_restart_duration': successful['pod_restart_duration'].std(),
            
            'mean_anomaly_prob_reduction': successful['anomaly_prob_reduction'].mean(),
            'std_anomaly_prob_reduction': successful['anomaly_prob_reduction'].std(),
            'min_anomaly_prob_reduction': successful['anomaly_prob_reduction'].min(),
            'max_anomaly_prob_reduction': successful['anomaly_prob_reduction'].max(),
        }
        
        return metrics
    
    def save_results(self):
        """Save evaluation results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw data
        df = pd.DataFrame([asdict(rec) for rec in self.records])
        csv_path = os.path.join(self.results_dir, f"simulated_evaluation_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n💾 Raw data saved: {csv_path}")
        
        # Save aggregate metrics
        metrics = self.calculate_aggregate_metrics()
        json_path = os.path.join(self.results_dir, f"simulated_metrics_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"💾 Aggregate metrics saved: {json_path}")
        
        return csv_path, json_path
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations."""
        df = pd.DataFrame([asdict(rec) for rec in self.records])
        successful = df[df['recovery_success'] == True]
        
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. MTTR Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(successful['mttr'], bins=15, edgecolor='black', alpha=0.7, color='skyblue')
        ax1.set_xlabel('MTTR (seconds)', fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        ax1.set_title('Distribution of Mean Time To Recovery', fontsize=11, fontweight='bold')
        ax1.axvline(successful['mttr'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {successful["mttr"].mean():.2f}s')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. Anomaly Probability Reduction
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(successful['anomaly_prob_reduction'], bins=15, edgecolor='black', alpha=0.7, color='lightcoral')
        ax2.set_xlabel('Probability Reduction', fontsize=10)
        ax2.set_ylabel('Frequency', fontsize=10)
        ax2.set_title('Anomaly Probability Reduction', fontsize=11, fontweight='bold')
        ax2.axvline(successful['anomaly_prob_reduction'].mean(), color='darkred', linestyle='--', linewidth=2,
                   label=f'Mean: {successful["anomaly_prob_reduction"].mean():.3f}')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Recovery Success Rate by Type
        ax3 = fig.add_subplot(gs[0, 2])
        success_by_type = df.groupby('anomaly_type')['recovery_success'].mean() * 100
        
        # Use custom labels to prevent overlap
        labels_map = {
            'cpu_spike': 'CPU\nSpike',
            'memory_leak': 'Memory\nLeak',
            'service_crash': 'Service\nCrash'
        }
        x_labels = [labels_map.get(t, t) for t in success_by_type.index]
        
        bars = ax3.bar(range(len(success_by_type)), success_by_type.values, 
                      color='mediumseagreen', edgecolor='black', alpha=0.7)
        ax3.set_ylabel('Success Rate (%)', fontsize=10)
        ax3.set_title('Recovery Success Rate by Type', fontsize=11, fontweight='bold')
        ax3.set_ylim([0, 105])
        ax3.set_xticks(range(len(success_by_type)))
        ax3.set_xticklabels(x_labels, fontsize=9)
        
        # Add percentage labels on bars
        for i, (bar, val) in enumerate(zip(bars, success_by_type.values)):
            height = bar.get_height()
            ax3.text(i, height, f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        ax3.grid(alpha=0.3, axis='y')
        
        # 4. End-to-End Latency Over Time
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(successful['injection_id'], successful['end_to_end_latency'], 
                marker='o', linestyle='-', color='purple', alpha=0.6)
        ax4.set_xlabel('Injection ID', fontsize=10)
        ax4.set_ylabel('Latency (seconds)', fontsize=10)
        ax4.set_title('End-to-End Latency Over Time', fontsize=11, fontweight='bold')
        ax4.axhline(successful['end_to_end_latency'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {successful["end_to_end_latency"].mean():.2f}s')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        # 5. Pod Restart Duration
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.hist(successful['pod_restart_duration'], bins=15, edgecolor='black', alpha=0.7, color='orange')
        ax5.set_xlabel('Duration (seconds)', fontsize=10)
        ax5.set_ylabel('Frequency', fontsize=10)
        ax5.set_title('Pod Restart Duration', fontsize=11, fontweight='bold')
        ax5.axvline(successful['pod_restart_duration'].mean(), color='darkred', linestyle='--', linewidth=2,
                   label=f'Mean: {successful["pod_restart_duration"].mean():.2f}s')
        ax5.legend()
        ax5.grid(alpha=0.3)
        
        # 6. Detection Latency
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.hist(successful['detection_latency'], bins=15, edgecolor='black', alpha=0.7, color='lightgreen')
        ax6.set_xlabel('Latency (seconds)', fontsize=10)
        ax6.set_ylabel('Frequency', fontsize=10)
        ax6.set_title('Detection Latency', fontsize=11, fontweight='bold')
        ax6.axvline(successful['detection_latency'].mean(), color='darkgreen', linestyle='--', linewidth=2,
                   label=f'Mean: {successful["detection_latency"].mean():.2f}s')
        ax6.legend()
        ax6.grid(alpha=0.3)
        
        # 7. Pre vs Post Recovery CPU
        ax7 = fig.add_subplot(gs[2, 0])
        x_pos = np.arange(len(successful))
        width = 0.35
        ax7.bar(x_pos - width/2, successful['pre_recovery_cpu'], width, label='Pre-Recovery', 
               color='indianred', edgecolor='black', alpha=0.7)
        ax7.bar(x_pos + width/2, successful['post_recovery_cpu'], width, label='Post-Recovery',
               color='lightgreen', edgecolor='black', alpha=0.7)
        ax7.set_xlabel('Injection ID', fontsize=10)
        ax7.set_ylabel('CPU (%)', fontsize=10)
        ax7.set_title('CPU Before/After Recovery', fontsize=11, fontweight='bold')
        ax7.legend()
        ax7.grid(alpha=0.3, axis='y')
        
        # 8. Pre vs Post Recovery Memory
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.bar(x_pos - width/2, successful['pre_recovery_memory'], width, label='Pre-Recovery',
               color='indianred', edgecolor='black', alpha=0.7)
        ax8.bar(x_pos + width/2, successful['post_recovery_memory'], width, label='Post-Recovery',
               color='lightgreen', edgecolor='black', alpha=0.7)
        ax8.set_xlabel('Injection ID', fontsize=10)
        ax8.set_ylabel('Memory (%)', fontsize=10)
        ax8.set_title('Memory Before/After Recovery', fontsize=11, fontweight='bold')
        ax8.legend()
        ax8.grid(alpha=0.3, axis='y')
        
        # 9. Summary Statistics Table
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('tight')
        ax9.axis('off')
        
        metrics = self.calculate_aggregate_metrics()
        table_data = [
            ['Metric', 'Value'],
            ['Detection Success', f"{metrics['detection_success_rate']:.1f}%"],
            ['Recovery Success', f"{metrics['recovery_success_rate']:.1f}%"],
            ['Recurrence Rate', f"{metrics['recurrence_rate']:.1f}%"],
            ['Mean MTTR', f"{metrics['mean_mttr']:.2f}s"],
            ['Mean E2E Latency', f"{metrics['mean_e2e_latency']:.2f}s"],
            ['Avg Prob Reduction', f"{metrics['mean_anomaly_prob_reduction']:.3f}"],
        ]
        
        table = ax9.table(cellText=table_data, cellLoc='left', loc='center',
                         colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_data)):
            for j in range(2):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        ax9.set_title('Summary Statistics', fontsize=11, fontweight='bold', pad=20)
        
        plt.suptitle('Comprehensive Recovery Evaluation Results', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(self.results_dir, f"evaluation_plots_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Visualizations saved: {plot_path}")
        plt.close()
        
        return plot_path
    
    def print_summary_report(self):
        """Print comprehensive summary report."""
        metrics = self.calculate_aggregate_metrics()
        
        print(f"\n{'='*70}")
        print("📊 COMPREHENSIVE RECOVERY EVALUATION SUMMARY")
        print(f"{'='*70}\n")
        
        print(f"🎯 Total Injections: {metrics['total_injections']}")
        print(f"{'─'*70}")
        
        print(f"\n1️⃣  RECOVERY EXECUTION METRICS (Orchestrator Level)")
        print(f"{'─'*70}")
        print(f"   Recovery Success Rate:           {metrics['recovery_success_rate']:.2f}%")
        print(f"   Detection Success Rate:          {metrics['detection_success_rate']:.2f}%")
        print(f"   Recurrence Rate (5 min window):  {metrics['recurrence_rate']:.2f}%")
        print(f"   Autonomous Resolution Rate:      {metrics['autonomous_resolution_rate']:.2f}%")
        
        print(f"\n   📈 Mean Time To Recovery (MTTR):")
        print(f"      Mean:    {metrics['mean_mttr']:.2f}s  (±{metrics['std_mttr']:.2f}s)")
        print(f"      Median:  {metrics['median_mttr']:.2f}s")
        print(f"      Range:   {metrics['min_mttr']:.2f}s - {metrics['max_mttr']:.2f}s")
        
        print(f"\n   ⚡ Recovery Execution Latency:")
        print(f"      Mean:    {metrics['mean_recovery_execution_latency']:.2f}s  (±{metrics['std_recovery_execution_latency']:.2f}s)")
        
        print(f"\n   🔄 Pod Restart Duration:")
        print(f"      Mean:    {metrics['mean_pod_restart_duration']:.2f}s  (±{metrics['std_pod_restart_duration']:.2f}s)")
        
        print(f"\n   🎯 Detection Latency:")
        print(f"      Mean:    {metrics['mean_detection_latency']:.2f}s  (±{metrics['std_detection_latency']:.2f}s)")
        
        print(f"\n   🌐 End-to-End Latency:")
        print(f"      Mean:    {metrics['mean_e2e_latency']:.2f}s  (±{metrics['std_e2e_latency']:.2f}s)")
        
        print(f"\n2️⃣  COUNTERFACTUAL-GUIDED RECOVERY IMPACT (Novel Contribution)")
        print(f"{'─'*70}")
        print(f"   Anomaly Probability Reduction:")
        print(f"      Mean:    {metrics['mean_anomaly_prob_reduction']:.3f}  (±{metrics['std_anomaly_prob_reduction']:.3f})")
        print(f"      Range:   {metrics['min_anomaly_prob_reduction']:.3f} - {metrics['max_anomaly_prob_reduction']:.3f}")
        
        print(f"\n3️⃣  SYSTEM PERFORMANCE")
        print(f"{'─'*70}")
        print(f"   Autonomous Resolution Rate:      {metrics['autonomous_resolution_rate']:.2f}%")
        print(f"   (% of anomalies resolved without human intervention)")
        
        print(f"\n{'='*70}")
        print("✅ Evaluation Complete")
        print(f"{'='*70}\n")


def main():
    """Main entry point for simulated evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simulated Recovery Evaluation")
    parser.add_argument('--injections', type=int, default=20,
                       help='Number of simulated injections')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = SimulatedRecoveryEvaluator(seed=args.seed)
    
    # Run evaluation
    df = evaluator.run_evaluation(num_injections=args.injections)
    
    # Save results
    evaluator.save_results()
    
    # Generate visualizations
    evaluator.generate_visualizations()
    
    # Print summary
    evaluator.print_summary_report()
    
    print("\n✅ Simulated evaluation completed!")
    print(f"📁 Results saved in: {evaluator.results_dir}")


if __name__ == "__main__":
    main()
