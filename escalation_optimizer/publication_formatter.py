"""
Results Formatter for Publication-Ready Output

Generates LaTeX tables, markdown summaries, and analysis suitable for
journal paper insertion.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict


class PublicationFormatter:
    """Format evaluation results for publication."""
    
    def __init__(self, results_df: pd.DataFrame):
        """
        Initialize formatter.
        
        Args:
            results_df: DataFrame with threshold evaluation results
        """
        self.results_df = results_df
    
    def generate_summary_table(self) -> str:
        """Generate publication-quality summary table in markdown."""
        
        table = []
        table.append("\n## Table 1: Threshold Evaluation Results\n")
        table.append("| Threshold θ | Escalation Rate | Bandwidth (MB/day) | Expected Latency (ms) | Accuracy | Edge/Cloud Split |")
        table.append("|-------------|-----------------|--------------------|-----------------------|----------|------------------|")
        
        for _, row in self.results_df.iterrows():
            table.append(
                f"| {row['threshold']:.1f} | "
                f"{row['escalation_rate']:.1%} | "
                f"{row['daily_bandwidth_mb']:.2f} | "
                f"{row['expected_latency_ms']:.2f} ± {row['std_edge_latency_ms']:.2f} | "
                f"{row['overall_accuracy']:.2%} | "
                f"{row['edge_workload_pct']:.1f}% / {row['cloud_workload_pct']:.1f}% |"
            )
        
        return "\n".join(table)
    
    def generate_latex_table(self) -> str:
        """Generate LaTeX table for journal submission."""
        
        latex = []
        latex.append("\\begin{table}[ht]")
        latex.append("\\centering")
        latex.append("\\caption{Bandwidth-Aware Escalation Optimization: Threshold Evaluation Results}")
        latex.append("\\label{tab:escalation_threshold}")
        latex.append("\\begin{tabular}{ccccc}")
        latex.append("\\hline")
        latex.append("\\textbf{Threshold} & \\textbf{Escalation} & \\textbf{Bandwidth} & \\textbf{Latency} & \\textbf{Accuracy} \\\\")
        latex.append("$\\theta$ & Rate (\\%) & (MB/day) & (ms) & (\\%) \\\\")
        latex.append("\\hline")
        
        for _, row in self.results_df.iterrows():
            latex.append(
                f"{row['threshold']:.1f} & "
                f"{row['escalation_rate']*100:.1f} & "
                f"{row['daily_bandwidth_mb']:.2f} & "
                f"{row['expected_latency_ms']:.2f} $\\pm$ {row['std_edge_latency_ms']:.2f} & "
                f"{row['overall_accuracy']*100:.2f} \\\\"
            )
        
        latex.append("\\hline")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        return "\n".join(latex)
    
    def generate_bandwidth_analysis(self) -> str:
        """Generate bandwidth consumption analysis."""
        
        analysis = []
        analysis.append("\n## Bandwidth Consumption Analysis\n")
        
        # Overall statistics
        mean_payload_raw = self.results_df['mean_payload_raw_kb'].mean()
        mean_payload_compressed = self.results_df['mean_payload_compressed_kb'].mean()
        compression_ratio = mean_payload_raw / mean_payload_compressed
        
        analysis.append(f"**Payload Characteristics:**")
        analysis.append(f"- Mean raw payload size: {mean_payload_raw:.2f} KB")
        analysis.append(f"- Mean compressed payload size: {mean_payload_compressed:.2f} KB")
        analysis.append(f"- Compression ratio: {compression_ratio:.2f}x")
        analysis.append(f"- Bandwidth savings: {(1 - 1/compression_ratio)*100:.1f}%")
        
        analysis.append(f"\n**Daily Bandwidth by Threshold:**")
        for _, row in self.results_df.iterrows():
            analysis.append(
                f"- θ = {row['threshold']:.1f}: "
                f"{row['daily_bandwidth_mb']:.2f} MB/day "
                f"({row['daily_bandwidth_gb']:.3f} GB/day) "
                f"[{row['escalation_rate']:.1%} escalation rate]"
            )
        
        # Bandwidth savings comparison
        max_bw = self.results_df['daily_bandwidth_mb'].max()
        min_bw = self.results_df['daily_bandwidth_mb'].min()
        savings = (max_bw - min_bw) / max_bw * 100
        
        analysis.append(f"\n**Optimization Impact:**")
        analysis.append(f"- Maximum bandwidth (θ = 0.6): {max_bw:.2f} MB/day")
        analysis.append(f"- Minimum bandwidth (θ = 0.9): {min_bw:.2f} MB/day")
        analysis.append(f"- Potential savings: {savings:.1f}% through threshold optimization")
        
        return "\n".join(analysis)
    
    def generate_latency_analysis(self) -> str:
        """Generate latency performance analysis."""
        
        analysis = []
        analysis.append("\n## Latency Performance Analysis\n")
        
        # Edge vs Cloud comparison
        mean_edge = self.results_df['mean_edge_latency_ms'].mean()
        mean_cloud = self.results_df['mean_cloud_latency_ms'].mean()
        overhead = mean_cloud - mean_edge
        overhead_pct = (overhead / mean_edge) * 100
        
        analysis.append(f"**Edge vs Cloud Latency:**")
        analysis.append(f"- Mean edge-only latency: {mean_edge:.2f} ms")
        analysis.append(f"- Mean escalated latency: {mean_cloud:.2f} ms")
        analysis.append(f"- Escalation overhead: {overhead:.2f} ms ({overhead_pct:.1f}% increase)")
        
        analysis.append(f"\n**Expected Latency by Threshold:**")
        for _, row in self.results_df.iterrows():
            analysis.append(
                f"- θ = {row['threshold']:.1f}: "
                f"{row['expected_latency_ms']:.2f} ms "
                f"[{row['edge_rate']:.1%} edge, {row['escalation_rate']:.1%} cloud]"
            )
        
        # Find optimal balance
        best_idx = self.results_df['expected_latency_ms'].idxmin()
        best_threshold = self.results_df.loc[best_idx, 'threshold']
        best_latency = self.results_df.loc[best_idx, 'expected_latency_ms']
        
        analysis.append(f"\n**Optimal Configuration:**")
        analysis.append(f"- Lowest expected latency at θ = {best_threshold:.1f}")
        analysis.append(f"- Expected latency: {best_latency:.2f} ms")
        
        return "\n".join(analysis)
    
    def generate_accuracy_tradeoff(self) -> str:
        """Generate accuracy trade-off analysis."""
        
        analysis = []
        analysis.append("\n## Accuracy Trade-off Analysis\n")
        
        # Edge vs Cloud accuracy
        mean_edge_acc = self.results_df['edge_accuracy'].mean()
        mean_cloud_acc = self.results_df['cloud_accuracy'].mean()
        
        analysis.append(f"**Edge vs Cloud Classification:**")
        analysis.append(f"- Mean edge accuracy: {mean_edge_acc:.2%}")
        analysis.append(f"- Mean cloud accuracy: {mean_cloud_acc:.2%}")
        analysis.append(f"- Cloud improvement: {(mean_cloud_acc - mean_edge_acc)*100:.2f} percentage points")
        
        analysis.append(f"\n**Overall Accuracy by Threshold:**")
        for _, row in self.results_df.iterrows():
            analysis.append(
                f"- θ = {row['threshold']:.1f}: "
                f"{row['overall_accuracy']:.2%} "
                f"(Edge: {row['edge_accuracy']:.2%}, Cloud: {row['cloud_accuracy']:.2%})"
            )
        
        # Bandwidth-Accuracy trade-off
        analysis.append(f"\n**Bandwidth-Accuracy Trade-off:**")
        for _, row in self.results_df.iterrows():
            bw_per_accuracy = row['daily_bandwidth_mb'] / row['overall_accuracy']
            analysis.append(
                f"- θ = {row['threshold']:.1f}: "
                f"{bw_per_accuracy:.2f} MB/day per unit accuracy"
            )
        
        # Find Pareto optimal
        # Maximize accuracy, minimize bandwidth
        normalized_acc = self.results_df['overall_accuracy'] / self.results_df['overall_accuracy'].max()
        normalized_bw = 1 - (self.results_df['daily_bandwidth_mb'] / self.results_df['daily_bandwidth_mb'].max())
        pareto_score = normalized_acc + normalized_bw
        best_pareto_idx = pareto_score.idxmax()
        best_pareto_threshold = self.results_df.loc[best_pareto_idx, 'threshold']
        
        analysis.append(f"\n**Pareto Optimal Configuration:**")
        analysis.append(f"- Best balance at θ = {best_pareto_threshold:.1f}")
        analysis.append(
            f"- Accuracy: {self.results_df.loc[best_pareto_idx, 'overall_accuracy']:.2%}, "
            f"Bandwidth: {self.results_df.loc[best_pareto_idx, 'daily_bandwidth_mb']:.2f} MB/day"
        )
        
        return "\n".join(analysis)
    
    def generate_key_observations(self) -> str:
        """Generate key observations for paper discussion."""
        
        obs = []
        obs.append("\n## Key Observations\n")
        
        # Observation 1: Escalation rate impact
        escalation_range = (
            self.results_df['escalation_rate'].max() - 
            self.results_df['escalation_rate'].min()
        )
        obs.append(
            f"1. **Threshold Impact on Escalation**: Varying θ from 0.6 to 0.9 modulates "
            f"escalation rate by {escalation_range:.1%}, enabling fine-grained control over "
            f"edge-cloud workload distribution."
        )
        
        # Observation 2: Bandwidth efficiency
        bw_reduction = (
            1 - self.results_df.loc[self.results_df['threshold'] == 0.9, 'daily_bandwidth_mb'].values[0] /
            self.results_df.loc[self.results_df['threshold'] == 0.6, 'daily_bandwidth_mb'].values[0]
        )
        obs.append(
            f"\n2. **Bandwidth Optimization**: Increasing confidence threshold from 0.6 to 0.9 "
            f"reduces daily bandwidth consumption by {bw_reduction:.1%}, demonstrating significant "
            f"network efficiency gains for edge deployments with limited connectivity."
        )
        
        # Observation 3: Latency vs accuracy trade-off
        min_latency_idx = self.results_df['expected_latency_ms'].idxmin()
        max_acc_idx = self.results_df['overall_accuracy'].idxmax()
        obs.append(
            f"\n3. **Latency-Accuracy Trade-off**: The system exhibits a Pareto frontier between "
            f"latency ({self.results_df['expected_latency_ms'].min():.2f}-"
            f"{self.results_df['expected_latency_ms'].max():.2f} ms) and accuracy "
            f"({self.results_df['overall_accuracy'].min():.2%}-"
            f"{self.results_df['overall_accuracy'].max():.2%}), enabling application-specific "
            f"optimization based on operational requirements."
        )
        
        # Observation 4: Compression effectiveness
        compression_ratio = self.results_df['compression_ratio'].mean()
        obs.append(
            f"\n4. **Compression Effectiveness**: Gzip compression achieves {compression_ratio:.2f}x "
            f"reduction in payload size, translating to {(1-1/compression_ratio)*100:.1f}% bandwidth "
            f"savings for escalated cases—critical for low-bandwidth edge environments."
        )
        
        # Observation 5: Recommended threshold
        # Find threshold that balances all metrics
        pareto_idx = self._find_pareto_optimal()
        recommended_threshold = self.results_df.loc[pareto_idx, 'threshold']
        obs.append(
            f"\n5. **Recommended Configuration**: Based on multi-objective optimization balancing "
            f"bandwidth, latency, and accuracy, θ = {recommended_threshold:.1f} emerges as optimal, "
            f"achieving {self.results_df.loc[pareto_idx, 'overall_accuracy']:.2%} accuracy with "
            f"{self.results_df.loc[pareto_idx, 'daily_bandwidth_mb']:.2f} MB/day bandwidth consumption "
            f"and {self.results_df.loc[pareto_idx, 'expected_latency_ms']:.2f} ms expected latency."
        )
        
        return "\n".join(obs)
    
    def _find_pareto_optimal(self) -> int:
        """Find Pareto optimal threshold balancing all objectives."""
        # Normalize metrics (higher is better)
        norm_acc = self.results_df['overall_accuracy'] / self.results_df['overall_accuracy'].max()
        norm_bw = 1 - (self.results_df['daily_bandwidth_mb'] / self.results_df['daily_bandwidth_mb'].max())
        norm_latency = 1 - (self.results_df['expected_latency_ms'] / self.results_df['expected_latency_ms'].max())
        
        # Weighted score (adjust weights based on priorities)
        # For edge: bandwidth and latency are critical
        score = 0.4 * norm_acc + 0.35 * norm_bw + 0.25 * norm_latency
        
        return score.idxmax()
    
    def generate_complete_report(self, output_path: str):
        """Generate complete publication-ready report."""
        
        report = []
        
        # Title
        report.append("# Bandwidth-Aware Escalation Optimization: Evaluation Results\n")
        report.append(f"**Generated:** {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}\n")
        report.append("---\n")
        
        # Summary table
        report.append(self.generate_summary_table())
        report.append("\n---\n")
        
        # Detailed analyses
        report.append(self.generate_bandwidth_analysis())
        report.append("\n---\n")
        
        report.append(self.generate_latency_analysis())
        report.append("\n---\n")
        
        report.append(self.generate_accuracy_tradeoff())
        report.append("\n---\n")
        
        # Key observations
        report.append(self.generate_key_observations())
        report.append("\n---\n")
        
        # LaTeX table
        report.append("\n## LaTeX Table for Paper\n")
        report.append("```latex")
        report.append(self.generate_latex_table())
        report.append("```\n")
        
        # Save report
        full_report = "\n".join(report)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_report)
        
        print(f"\n📄 Generated publication-ready report: {output_path}")
        
        return full_report


def main():
    """Test formatter with sample data."""
    
    # Load latest results
    base_dir = os.path.dirname(os.path.dirname(__file__))
    results_dir = os.path.join(base_dir, 'results', 'escalation')
    
    # Find latest CSV
    csv_files = [f for f in os.listdir(results_dir) if f.startswith('threshold_evaluation_') and f.endswith('.csv')]
    if not csv_files:
        print("❌ No evaluation results found. Run threshold_evaluator.py first.")
        return
    
    latest_csv = sorted(csv_files)[-1]
    csv_path = os.path.join(results_dir, latest_csv)
    
    print(f"📂 Loading results from {csv_path}")
    results_df = pd.read_csv(csv_path)
    
    # Generate report
    formatter = PublicationFormatter(results_df)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(results_dir, f'PUBLICATION_REPORT_{timestamp}.md')
    
    formatter.generate_complete_report(report_path)
    
    print("\n✅ Report generation complete!")


if __name__ == '__main__':
    main()
