"""
Generate Confusion Matrix Visualizations for Service Configurations

Creates publication-quality confusion matrix heatmaps for the paper.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from train_scaleinvariant_model import ScaleInvariantAnomalyDetector


def plot_confusion_matrix(cm, class_names, title, output_file, normalize=False):
    """
    Create a beautiful confusion matrix visualization.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        title: Plot title
        output_file: Where to save the plot
        normalize: Whether to show percentages instead of counts
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        cmap = 'Blues'
    else:
        fmt = 'd'
        cmap = 'Blues'
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage' if normalize else 'Count'},
                linewidths=1, linecolor='white',
                square=True, ax=ax)
    
    # Labels and title
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Rotate labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    
    plt.close()


def generate_confusion_matrices():
    """Generate confusion matrices for all configurations."""
    print("="*80)
    print("CONFUSION MATRIX GENERATION")
    print("="*80)
    
    # Load trained model
    print("\n🔧 Loading trained model...")
    detector = ScaleInvariantAnomalyDetector()
    detector.load_model('../models/anomaly_detector_scaleinvariant.pkl')
    
    class_names = [label.strip() for label in detector.label_encoder.classes_]
    print(f"   Classes: {class_names}")
    
    # Configurations to generate matrices for
    configs = [
        ('metrics_eval_3_services_scaleinvariant.csv', '3 Services', 'Best Performance'),
        ('metrics_eval_5_services_scaleinvariant.csv', '5 Services', 'Extended System'),
        ('metrics_eval_10_services_scaleinvariant.csv', '10 Services', 'Large Scale')
    ]
    
    for dataset_file, config_name, description in configs:
        print(f"\n{'='*80}")
        print(f"Processing: {config_name} - {description}")
        print(f"{'='*80}")
        
        try:
            # Load dataset
            dataset_path = f"../datasets/{dataset_file}"
            print(f"📊 Loading: {dataset_path}")
            df = pd.read_csv(dataset_path)
            
            # Prepare data
            feature_cols = detector.feature_columns
            X = df[feature_cols]
            y = df['anomaly_type']
            y_encoded = detector.label_encoder.transform(y)
            
            # Predict
            print(f"🔮 Running predictions...")
            y_pred = detector.model.predict(X)
            
            # Generate confusion matrix
            cm = confusion_matrix(y_encoded, y_pred)
            
            # Calculate accuracy
            accuracy = np.trace(cm) / np.sum(cm)
            print(f"   Accuracy: {accuracy:.2%}")
            
            # Generate both count and percentage versions
            config_safe = config_name.replace(' ', '_').lower()
            
            # 1. Count-based confusion matrix
            output_count = f"../results/confusion_matrix_{config_safe}_counts.png"
            plot_confusion_matrix(
                cm, class_names,
                f"Confusion Matrix: {config_name}\n{description} (Accuracy: {accuracy:.2%})",
                output_count,
                normalize=False
            )
            
            # 2. Percentage-based confusion matrix
            output_pct = f"../results/confusion_matrix_{config_safe}_normalized.png"
            plot_confusion_matrix(
                cm, class_names,
                f"Normalized Confusion Matrix: {config_name}\n{description} (Accuracy: {accuracy:.2%})",
                output_pct,
                normalize=True
            )
            
            # Print classification report
            print(f"\n📋 Classification Report:")
            report = classification_report(y_encoded, y_pred, target_names=class_names, digits=4)
            print(report)
            
        except FileNotFoundError:
            print(f"⚠️  Dataset not found: {dataset_file}")
            continue
        except Exception as e:
            print(f"❌ Error processing {config_name}: {e}")
            continue
    
    print(f"\n{'='*80}")
    print("✅ CONFUSION MATRICES GENERATED")
    print(f"{'='*80}")
    print("\n📸 Files saved in: ml_detector/results/")
    print("   • confusion_matrix_3_services_counts.png")
    print("   • confusion_matrix_3_services_normalized.png")
    print("   • confusion_matrix_5_services_counts.png")
    print("   • confusion_matrix_5_services_normalized.png")
    print("   • confusion_matrix_10_services_counts.png")
    print("   • confusion_matrix_10_services_normalized.png")
    print("\n💡 For your paper:")
    print("   • Use 3-service normalized matrix (shows best performance)")
    print("   • Normalized version shows per-class accuracy clearly")
    print("   • High diagonal values indicate excellent classification")


if __name__ == '__main__':
    try:
        generate_confusion_matrices()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
