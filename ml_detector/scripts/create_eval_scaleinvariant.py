"""
Create Evaluation Datasets with Scale-Invariant Features

Generates test sets simulating 1, 2, and 3 service configurations
using scale-invariant features that should work across all topologies.
"""

import pandas as pd
import numpy as np
from transform_dataset_scaleinvariant import ScaleInvariantTransformer


def create_evaluation_datasets(input_csv: str, output_dir: str = '.'):
    """
    Create evaluation datasets for 1, 2, and 3 service configurations.
    
    Args:
        input_csv: Path to original metrics_dataset_enhanced_rounded.csv
        output_dir: Directory to save evaluation datasets
    """
    print("="*80)
    print("SCALE-INVARIANT EVALUATION DATASET GENERATION")
    print("="*80)
    print()
    
    transformer = ScaleInvariantTransformer()
    
    # Load original dataset
    print(f"📊 Loading original dataset: {input_csv}")
    df = pd.read_csv(input_csv)
    df.columns = df.columns.str.strip()
    print(f"   Shape: {df.shape}")
    print(f"   Samples: {len(df)}")
    
    # Class distribution
    print(f"\n📈 Class distribution:")
    for class_name, count in df['anomaly_type'].value_counts().items():
        percentage = (count / len(df)) * 100
        print(f"   {class_name:20s}: {count:5d} ({percentage:5.1f}%)")
    
    # Create configurations
    configs = [
        ('1_service', ['notification'], 'Notification Only'),
        ('2_services', ['notification', 'web_api'], 'Notification + Web API'),
        ('3_services', ['notification', 'web_api', 'processor'], 'Full System')
    ]
    
    results = {}
    
    for config_id, active_services, config_name in configs:
        print(f"\n{'='*80}")
        print(f"Creating: {config_name}")
        print(f"{'='*80}")
        
        # Create modified dataset
        df_config = df.copy()
        
        # Zero out inactive services
        all_services = ['notification', 'web_api', 'processor']
        inactive_services = [s for s in all_services if s not in active_services]
        
        if inactive_services:
            print(f"   Zeroing out: {', '.join(inactive_services)}")
            for service in inactive_services:
                # Find columns that start with the service name
                service_cols = [col for col in df_config.columns 
                               if col.strip().startswith(service)]
                for col in service_cols:
                    df_config[col] = 0.0
                print(f"   ├─ {service}: {len(service_cols)} metrics zeroed")
        else:
            print(f"   Using all services")
        
        # Transform to scale-invariant features
        print(f"   Transforming to scale-invariant features...")
        transformed_rows = []
        
        for idx, row in df_config.iterrows():
            features = transformer.aggregate_row(row)
            features['timestamp'] = row.get('timestamp', '')
            features['anomaly_label'] = row.get('anomaly_label', 0)
            features['anomaly_type'] = row.get('anomaly_type', 'normal')
            transformed_rows.append(features)
            
            if (idx + 1) % 500 == 0:
                print(f"      Processed {idx + 1}/{len(df_config)} rows...")
        
        # Create dataframe
        df_transformed = pd.DataFrame(transformed_rows)
        feature_cols = transformer.feature_names
        label_cols = ['timestamp', 'anomaly_label', 'anomaly_type']
        df_transformed = df_transformed[feature_cols + label_cols]
        
        # Save
        output_path = f"{output_dir}/metrics_eval_{config_id}_scaleinvariant.csv"
        df_transformed.to_csv(output_path, index=False)
        
        print(f"   ✅ Saved: {output_path}")
        print(f"   Shape: {df_transformed.shape}")
        
        results[config_id] = df_transformed
    
    print(f"\n{'='*80}")
    print("✅ ALL EVALUATION DATASETS CREATED")
    print(f"{'='*80}")
    print(f"\n📊 Summary:")
    print(f"   {'Configuration':<25} {'Samples':<10} {'Features'}")
    print(f"   {'-'*55}")
    for config_id in ['1_service', '2_services', '3_services']:
        df_result = results[config_id]
        print(f"   {config_id:<25} {len(df_result):<10} {len(transformer.feature_names)}")
    
    print(f"\n🚀 Key Insight:")
    print(f"   All datasets use SAME scale-invariant features")
    print(f"   → Model should achieve similar accuracy across all configs")
    print(f"\nNext: Run python evaluate_scaleinvariant_model.py")
    
    return results


if __name__ == '__main__':
    create_evaluation_datasets('metrics_dataset_enhanced_rounded.csv')
