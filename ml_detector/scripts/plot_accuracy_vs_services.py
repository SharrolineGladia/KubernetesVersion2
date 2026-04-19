"""
Plot Accuracy vs Number of Services for XGBoost Model
Visualizes how the scale-invariant XGBoost model performs across different service configurations
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from evaluation_results_scaleinvariant.txt
service_counts = [1, 2, 3, 5, 10]
accuracies = [80.78, 93.37, 99.46, 93.66, 76.89]

# Create the plot
plt.figure(figsize=(10, 6))

# Plot line with markers
plt.plot(service_counts, accuracies, marker='o', linewidth=2, markersize=10, 
         color='#2E86AB', label='XGBoost Model Accuracy')

# Add data labels
for i, (services, acc) in enumerate(zip(service_counts, accuracies)):
    plt.annotate(f'{acc:.2f}%', 
                xy=(services, acc), 
                xytext=(0, 10), 
                textcoords='offset points',
                ha='center',
                fontsize=10,
                fontweight='bold')

# Styling
plt.xlabel('Number of Services', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
plt.title('XGBoost Model Accuracy vs Number of Services\n(Scale-Invariant Feature Extraction)', 
          fontsize=14, fontweight='bold', pad=20)

# Add grid
plt.grid(True, alpha=0.3, linestyle='--')

# Set y-axis limits for better visualization
plt.ylim(70, 105)

# Add x-axis ticks
plt.xticks(service_counts)

# Add mean accuracy line
mean_accuracy = np.mean(accuracies)
plt.axhline(y=mean_accuracy, color='red', linestyle='--', linewidth=1.5, 
            alpha=0.7, label=f'Mean Accuracy: {mean_accuracy:.2f}%')

# Add legend
plt.legend(loc='lower left', fontsize=10)

# Add statistics box
stats_text = f'Mean: {mean_accuracy:.2f}%\n'
stats_text += f'Std Dev: {np.std(accuracies):.2f}%\n'
stats_text += f'Max: {max(accuracies):.2f}%\n'
stats_text += f'Min: {min(accuracies):.2f}%'
plt.text(0.98, 0.02, stats_text, 
         transform=plt.gca().transAxes,
         fontsize=9,
         verticalalignment='bottom',
         horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Save the plot
output_path = '../results/accuracy_vs_services.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Plot saved to: {output_path}")

# Show the plot
plt.show()

# Print summary statistics
print("\n" + "="*50)
print("XGBoost Model Performance Summary")
print("="*50)
for services, acc in zip(service_counts, accuracies):
    print(f"{services:2d} Service(s): {acc:6.2f}%")
print("-"*50)
print(f"Mean Accuracy:     {mean_accuracy:.2f}%")
print(f"Standard Deviation: {np.std(accuracies):.2f}%")
print(f"Maximum Accuracy:   {max(accuracies):.2f}%")
print(f"Minimum Accuracy:   {min(accuracies):.2f}%")
print(f"Range:             {max(accuracies) - min(accuracies):.2f}%")
print("="*50)
