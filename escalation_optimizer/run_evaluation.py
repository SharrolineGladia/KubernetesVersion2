"""
Main Runner for Bandwidth-Aware Escalation Optimization Evaluation

Orchestrates complete evaluation workflow:
1. Generate anomaly scenarios with real confidence scores
2. Evaluate thresholds θ ∈ {0.6, 0.7, 0.8, 0.9}
3. Generate visualizations
4. Create publication-ready report
"""

import os
import sys
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.dirname(__file__))

from data_generator import AnomalyDataGenerator
from threshold_evaluator import ThresholdEvaluator
from publication_formatter import PublicationFormatter


def main():
    """Execute complete evaluation workflow."""
    
    print("=" * 80)
    print("  BANDWIDTH-AWARE ESCALATION OPTIMIZATION")
    print("  Complete Evaluation Framework")
    print("  Edge-Cloud Anomaly Management System")
    print("=" * 80)
    print(f"\n⏰ Evaluation started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, 'ml_detector', 'models', 'anomaly_detector_scaleinvariant.pkl')
    dataset_path = os.path.join(base_dir, 'ml_detector', 'datasets', 'metrics_dataset_scaleinvariant.csv')
    results_dir = os.path.join(base_dir, 'results', 'escalation')
    scenarios_csv = os.path.join(results_dir, 'anomaly_scenarios.csv')
    
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # ========================================================================
    # STEP 1: Generate Anomaly Scenarios
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: GENERATE ANOMALY SCENARIOS")
    print("=" * 80)
    
    if not os.path.exists(scenarios_csv):
        print("\n🔄 Generating evaluation dataset...")
        
        generator = AnomalyDataGenerator(
            model_path=model_path,
            dataset_path=dataset_path
        )
        
        # Generate 1000 scenarios for robust evaluation
        scenarios = generator.generate_scenarios(
            n_samples=1000,
            distribution={
                'normal': 0.5,          # 50% normal operation
                'cpu_spike': 0.2,       # 20% CPU anomalies
                'memory_leak': 0.15,    # 15% memory anomalies
                'service_crash': 0.15   # 15% crash anomalies
            }
        )
        
        generator.save_scenarios(scenarios, scenarios_csv)
    else:
        print(f"\n✅ Using existing scenarios from {scenarios_csv}")
    
    # ========================================================================
    # STEP 2: Evaluate Thresholds
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: EVALUATE CONFIDENCE THRESHOLDS")
    print("=" * 80)
    
    evaluator = ThresholdEvaluator(
        scenarios_csv=scenarios_csv,
        link_bandwidth_mbps=10.0,  # Edge-cloud link: 10 Mbps
        daily_anomalies=1000       # Expected daily anomaly count
    )
    
    # Evaluate thresholds
    thresholds = [0.6, 0.7, 0.8, 0.9]
    results_df = evaluator.evaluate_all_thresholds(thresholds)
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv = os.path.join(results_dir, f'threshold_evaluation_{timestamp}.csv')
    results_df.to_csv(results_csv, index=False)
    print(f"\n💾 Saved detailed results to {results_csv}")
    
    # ========================================================================
    # STEP 3: Generate Visualizations
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: GENERATE VISUALIZATIONS")
    print("=" * 80)
    
    evaluator.generate_visualizations(results_df, results_dir)
    
    # ========================================================================
    # STEP 4: Generate Publication Report
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: GENERATE PUBLICATION-READY REPORT")
    print("=" * 80)
    
    formatter = PublicationFormatter(results_df)
    report_path = os.path.join(results_dir, f'PUBLICATION_REPORT_{timestamp}.md')
    formatter.generate_complete_report(report_path)
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE")
    print("=" * 80)
    
    print("\n📂 Output Files:")
    print(f"   • Scenarios:      {scenarios_csv}")
    print(f"   • Metrics (CSV):  {results_csv}")
    print(f"   • Plots (PNG):    {results_dir}/escalation_analysis_{timestamp}.png")
    print(f"   • Report (MD):    {report_path}")
    
    print("\n📊 Quick Summary:")
    print(f"   • Evaluated {len(results_df)} threshold configurations")
    print(f"   • Threshold range: {results_df['threshold'].min():.1f} - {results_df['threshold'].max():.1f}")
    print(f"   • Escalation rate range: {results_df['escalation_rate'].min():.1%} - {results_df['escalation_rate'].max():.1%}")
    print(f"   • Bandwidth range: {results_df['daily_bandwidth_mb'].min():.2f} - {results_df['daily_bandwidth_mb'].max():.2f} MB/day")
    print(f"   • Accuracy range: {results_df['overall_accuracy'].min():.2%} - {results_df['overall_accuracy'].max():.2%}")
    
    # Find recommended threshold
    # Normalize and score
    norm_acc = results_df['overall_accuracy'] / results_df['overall_accuracy'].max()
    norm_bw = 1 - (results_df['daily_bandwidth_mb'] / results_df['daily_bandwidth_mb'].max())
    norm_latency = 1 - (results_df['expected_latency_ms'] / results_df['expected_latency_ms'].max())
    score = 0.4 * norm_acc + 0.35 * norm_bw + 0.25 * norm_latency
    best_idx = score.idxmax()
    
    print("\n🎯 Recommended Configuration:")
    print(f"   • Optimal threshold: θ = {results_df.loc[best_idx, 'threshold']:.1f}")
    print(f"   • Escalation rate: {results_df.loc[best_idx, 'escalation_rate']:.1%}")
    print(f"   • Daily bandwidth: {results_df.loc[best_idx, 'daily_bandwidth_mb']:.2f} MB/day")
    print(f"   • Expected latency: {results_df.loc[best_idx, 'expected_latency_ms']:.2f} ms")
    print(f"   • Overall accuracy: {results_df.loc[best_idx, 'overall_accuracy']:.2%}")
    
    print("\n💡 Next Steps:")
    print("   1. Review publication report for paper-ready content")
    print("   2. Examine visualizations for trend analysis")
    print("   3. Use recommended threshold in production deployment")
    
    print(f"\n⏰ Evaluation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == '__main__':
    main()
