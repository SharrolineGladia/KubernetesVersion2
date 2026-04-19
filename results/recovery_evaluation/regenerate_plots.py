"""
Quick script to regenerate just the visualization with fixed labels
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import os

# Find latest CSV file
csv_files = glob("simulated_evaluation_*.csv")
if not csv_files:
    print("❌ No evaluation CSV found. Run simulated_evaluator.py first.")
    exit(1)

latest_csv = sorted(csv_files)[-1]
print(f"📊 Loading data from: {latest_csv}")

# Load data
df = pd.read_csv(latest_csv)
successful = df[df['recovery_success'] == True]

print(f"✅ Loaded {len(df)} injections ({len(successful)} successful)")

# Create figure
fig = plt.figure(figsize=(16, 12))
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

# 3. Recovery Success Rate by Type - FIXED VERSION
ax3 = fig.add_subplot(gs[0, 2])
success_by_type = df.groupby('anomaly_type')['recovery_success'].mean() * 100

# Use custom labels to prevent overlap
labels = {
    'cpu_spike': 'CPU\nSpike',
    'memory_leak': 'Memory\nLeak',
    'service_crash': 'Service\nCrash'
}
x_labels = [labels.get(t, t) for t in success_by_type.index]

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

# Calculate metrics
metrics = {
    'detection_success_rate': (df['detection_success'].sum() / len(df)) * 100,
    'recovery_success_rate': (df['recovery_success'].sum() / len(df)) * 100,
    'recurrence_rate': (df['recurrence_detected'].sum() / len(df)) * 100,
    'mean_mttr': successful['mttr'].mean(),
    'mean_e2e_latency': successful['end_to_end_latency'].mean(),
    'mean_anomaly_prob_reduction': successful['anomaly_prob_reduction'].mean(),
}

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

# Save with timestamp
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plot_path = f"evaluation_plots_{timestamp}_fixed.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Fixed visualization saved: {plot_path}")
plt.close()
