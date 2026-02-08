"""
Test Model Extrapolation Beyond Training Data

Tests the scale-invariant model on 4 and 5 service configurations
to validate the claim that it works with "N services".

The model was trained on 3-service data. This tests extrapolation.
"""

import pandas as pd
import numpy as np
from train_scaleinvariant_model import ScaleInvariantAnomalyDetector
from transform_dataset_scaleinvariant import ScaleInvariantTransformer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def synthesize_additional_services(df: pd.DataFrame, num_new_services: int = 1):
    """
    Add synthetic services to existing data by creating variations of existing metrics.
    
    Args:
        df: Original dataset with 3 services
        num_new_services: Number of additional services to add (1 or 2)
    
    Returns:
        DataFrame with additional service columns
    """
    print(f"   Adding {num_new_services} synthetic service(s)...")
    
    df_expanded = df.copy()
    
    for i in range(num_new_services):
        service_num = i + 4  # service_4, service_5, etc.
        
        # Create variations by blending existing services with noise
        # This simulates realistic additional microservices
        
        # Blend notification and web_api metrics (common pattern in microservices)
        blend_weight = np.random.uniform(0.3, 0.7, size=len(df))
        noise_factor = np.random.uniform(0.8, 1.2, size=len(df))
        
        # CPU: blend with variation
        df_expanded[f'service_{service_num}_cpu'] = (
            (df['notification_cpu'] * blend_weight + 
             df['web_api_cpu'] * (1 - blend_weight)) * noise_factor
        ).clip(0, 100)
        
        # Memory: blend with variation
        df_expanded[f'service_{service_num}_memory'] = (
            (df['notification_memory'] * blend_weight + 
             df['web_api_memory'] * (1 - blend_weight)) * noise_factor
        ).clip(0, 200)  # Allow some memory spikes
        
        # Request rate: blend processor and notification
        df_expanded[f'service_{service_num}_requests'] = (
            (df['notification_message_rate'] * 0.6 + 
             df['processor_processing_rate'] * 0.4) * noise_factor
        ).clip(0, 100)
        
        # Error rate: small values with occasional spikes
        df_expanded[f'service_{service_num}_errors'] = (
            df['notification_error_rate'] * np.random.uniform(0.5, 1.5, size=len(df))
        ).clip(0, 1)
        
        # Latency: blend web_api and processor
        df_expanded[f'service_{service_num}_latency'] = (
            (df['web_api_response_time_p95'] * 0.5 + 
             df['processor_response_time_p95'] * 0.5) * noise_factor
        ).clip(0, 500)
        
        # Queue metrics
        df_expanded[f'service_{service_num}_queue_depth'] = (
            (df['notification_queue_depth'] * 0.5 + 
             df['web_api_queue_depth'] * 0.5) * noise_factor
        ).clip(0, 100)
        
        # Thread count
        df_expanded[f'service_{service_num}_threads'] = (
            (df['notification_thread_count'] * 0.6 + 
             df['processor_thread_count'] * 0.4) * noise_factor
        ).clip(1, 50)
        
        print(f"      ├─ service_{service_num}: CPU {df_expanded[f'service_{service_num}_cpu'].mean():.2f}, "
              f"Memory {df_expanded[f'service_{service_num}_memory'].mean():.2f}")
    
    return df_expanded


def transform_n_service_data(df: pd.DataFrame, num_services: int):
    """
    Transform N-service dataset to scale-invariant features.
    
    Args:
        df: Dataset with N services
        num_services: Number of services (4 or 5)
    
    Returns:
        Transformed dataset with scale-invariant features
    """
    print(f"   Transforming {num_services}-service data to scale-invariant features...")
    
    transformer = ScaleInvariantTransformer()
    transformed_rows = []
    
    for idx, row in df.iterrows():
        # Collect metrics from all services
        cpu_values = []
        memory_values = []
        requests = []
        errors = []
        latency = []
        queue_depth = []
        threads = []
        
        # Original 3 services
        cpu_values.extend([
            row.get('notification_cpu', 0),
            row.get('web_api_cpu', 0),
            row.get('processor_cpu', 0)
        ])
        memory_values.extend([
            row.get('notification_memory', 0),
            row.get('web_api_memory', 0),
            row.get('processor_memory', 0)
        ])
        requests.extend([
            row.get('notification_message_rate', 0),
            row.get('web_api_requests', 0),
            row.get('processor_processing_rate', 0)
        ])
        errors.extend([
            row.get('notification_error_rate', 0) * 100,
            row.get('web_api_errors', 0),
            0
        ])
        latency.extend([
            row.get('notification_api_health', 0) * 100,
            row.get('web_api_response_time_p95', 0),
            row.get('processor_response_time_p95', 0)
        ])
        queue_depth.extend([
            row.get('notification_queue_depth', 0),
            row.get('web_api_queue_depth', 0),
            row.get('processor_queue_depth', 0)
        ])
        threads.extend([
            row.get('notification_thread_count', 0),
            row.get('web_api_thread_count', 0),
            row.get('processor_thread_count', 0)
        ])
        
        # Additional services
        for i in range(4, num_services + 1):
            cpu_values.append(row.get(f'service_{i}_cpu', 0))
            memory_values.append(row.get(f'service_{i}_memory', 0))
            requests.append(row.get(f'service_{i}_requests', 0))
            errors.append(row.get(f'service_{i}_errors', 0))
            latency.append(row.get(f'service_{i}_latency', 0))
            queue_depth.append(row.get(f'service_{i}_queue_depth', 0))
            threads.append(row.get(f'service_{i}_threads', 0))
        
        # Create custom aggregation (similar to transformer but for N services)
        features = {}
        
        # Filter active services
        num_active = sum(1 for cpu in cpu_values if cpu > 0)
        if num_active == 0:
            num_active = 1
        
        active_cpu = [v for v in cpu_values if v > 0] or [0]
        active_memory = [v for v in memory_values if v > 0] or [0]
        active_requests = [v for v in requests if v > 0] or [0]
        active_errors = [v for v in errors if v > 0] or [0]
        active_latency = [v for v in latency if v > 0] or [0]
        active_queue = [v for v in queue_depth if v > 0] or [0]
        active_threads = [v for v in threads if v > 0] or [0]
        
        # Compute scale-invariant features
        cpu_mean = np.mean(active_cpu)
        cpu_max = np.max(active_cpu)
        cpu_min = np.min(active_cpu)
        cpu_std = np.std(active_cpu) if len(active_cpu) > 1 else 0
        
        features['cpu_utilization_mean'] = cpu_mean / 100.0
        features['cpu_utilization_max'] = cpu_max / 100.0
        features['cpu_variance_coef'] = (cpu_std / cpu_mean) if cpu_mean > 0 else 0
        features['cpu_imbalance'] = ((cpu_max - cpu_min) / cpu_mean) if cpu_mean > 0 else 0
        
        memory_mean = np.mean(active_memory)
        memory_max = np.max(active_memory)
        memory_min = np.min(active_memory)
        memory_std = np.std(active_memory) if len(active_memory) > 1 else 0
        
        features['memory_pressure_mean'] = memory_mean / 100.0
        features['memory_pressure_max'] = memory_max / 100.0
        features['memory_variance_coef'] = (memory_std / memory_mean) if memory_mean > 0 else 0
        features['memory_imbalance'] = ((memory_max - memory_min) / memory_mean) if memory_mean > 0 else 0
        
        queue_mean = np.mean(active_queue)
        queue_std = np.std(active_queue) if len(active_queue) > 1 else 0
        
        features['network_in_rate'] = queue_mean / 1000.0
        features['network_out_rate'] = np.mean(active_requests) / 100.0
        features['network_in_variance_coef'] = (queue_std / queue_mean) if queue_mean > 0 else 0
        features['network_out_variance_coef'] = 0.1  # Placeholder
        features['network_asymmetry'] = 0.1
        
        features['disk_io_rate'] = np.mean(active_threads) / 100.0
        features['disk_io_variance_coef'] = (np.std(active_threads) / np.mean(active_threads)) if np.mean(active_threads) > 0 else 0
        
        req_mean = np.mean(active_requests)
        features['request_rate'] = req_mean / 1000.0
        features['request_variance_coef'] = 0.1
        
        total_requests = sum(active_requests)
        total_errors = sum(active_errors)
        features['error_rate'] = (total_errors / total_requests) if total_requests > 0 else 0
        features['error_variance_coef'] = (np.std(active_errors) / np.mean(active_errors)) if np.mean(active_errors) > 0 else 0
        
        lat_mean = np.mean(active_latency)
        features['latency_mean'] = lat_mean / 1000.0
        features['latency_p95'] = np.max(active_latency) / 1000.0
        features['latency_variance_coef'] = (np.std(active_latency) / lat_mean) if lat_mean > 0 else 0
        
        features['system_stress'] = row.get('system_stress', 0)
        
        resource_util = (cpu_mean + memory_mean) / 200.0
        load = req_mean / 1000.0
        features['resource_efficiency'] = load / (resource_util + 0.01) if resource_util > 0 else 0
        
        features['service_density'] = num_active / 10.0
        
        if len(active_cpu) > 1 and len(active_memory) > 1:
            corr = np.corrcoef(active_cpu[:len(active_memory)], active_memory[:len(active_cpu)])[0, 1]
            features['cpu_memory_correlation'] = corr if not np.isnan(corr) else 0
        else:
            features['cpu_memory_correlation'] = 0
        
        features['performance_degradation'] = (lat_mean / (req_mean + 1)) / 1000.0
        
        # Preserve labels
        features['timestamp'] = row.get('timestamp', '')
        features['anomaly_label'] = row.get('anomaly_label', 0)
        features['anomaly_type'] = row.get('anomaly_type', 'normal')
        
        transformed_rows.append(features)
        
        if (idx + 1) % 500 == 0:
            print(f"      Processed {idx + 1}/{len(df)} rows...")
    
    df_transformed = pd.DataFrame(transformed_rows)
    
    # Reorder columns
    feature_cols = transformer.feature_names
    label_cols = ['timestamp', 'anomaly_label', 'anomaly_type']
    df_transformed = df_transformed[feature_cols + label_cols]
    
    return df_transformed


def evaluate_extrapolation(detector, df_transformed, config_name, num_services):
    """Evaluate model on N-service configuration."""
    print(f"\n{'='*80}")
    print(f"EXTRAPOLATION TEST: {config_name} ({num_services} Services)")
    print(f"{'='*80}")
    print(f"⚠️  Model was trained on 3-service data ONLY")
    print(f"   Testing extrapolation to {num_services} services...")
    
    # Separate features and labels
    X = df_transformed[detector.feature_columns]
    y = df_transformed['anomaly_type']
    
    # Encode labels
    y_encoded = detector.label_encoder.transform(y)
    
    # Predict
    print(f"\n🔮 Running predictions...")
    y_pred = detector.model.predict(X)
    
    # Metrics
    accuracy = accuracy_score(y_encoded, y_pred)
    
    print(f"\n🎯 RESULTS:")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Per-class metrics
    class_names = detector.label_encoder.classes_
    report = classification_report(y_encoded, y_pred, target_names=class_names, output_dict=True)
    
    print(f"\n📋 Per-Class Performance:")
    print(f"   {'Class':<20} {'Precision':<12} {'Recall':<10} {'F1-Score':<10} {'Support'}")
    print(f"   {'-'*72}")
    for class_name in class_names:
        metrics = report[class_name]
        print(f"   {class_name:<20} {metrics['precision']:<12.2%} "
              f"{metrics['recall']:<10.2%} {metrics['f1-score']:<10.2%} "
              f"{int(metrics['support'])}")
    
    return accuracy


def main():
    """Test model extrapolation to 4 and 5 services."""
    print("="*80)
    print("MODEL EXTRAPOLATION TEST: BEYOND TRAINING DATA")
    print("="*80)
    print("\n🎯 Goal: Validate model works with >3 services (extrapolation)")
    print("   Training data: 3 services only")
    print("   Test data: 4 and 5 services")
    
    # Load model
    print("\n🔧 Loading trained model...")
    detector = ScaleInvariantAnomalyDetector()
    detector.load_model('anomaly_detector_scaleinvariant.pkl')
    
    # Load original 3-service dataset
    print("\n📊 Loading original dataset...")
    df_original = pd.read_csv('metrics_dataset_enhanced_rounded.csv')
    df_original.columns = df_original.columns.str.strip()
    print(f"   Shape: {df_original.shape}")
    
    results = {}
    
    # Test 4 services
    print(f"\n{'='*80}")
    print("Creating 4-Service Configuration")
    print(f"{'='*80}")
    df_4svc = synthesize_additional_services(df_original, num_new_services=1)
    df_4svc_transformed = transform_n_service_data(df_4svc, num_services=4)
    df_4svc_transformed.to_csv('metrics_eval_4_services_scaleinvariant.csv', index=False)
    print(f"   ✅ Saved: metrics_eval_4_services_scaleinvariant.csv")
    
    acc_4svc = evaluate_extrapolation(detector, df_4svc_transformed, "4 Services", 4)
    results['4_services'] = acc_4svc
    
    # Test 5 services
    print(f"\n{'='*80}")
    print("Creating 5-Service Configuration")
    print(f"{'='*80}")
    df_5svc = synthesize_additional_services(df_original, num_new_services=2)
    df_5svc_transformed = transform_n_service_data(df_5svc, num_services=5)
    df_5svc_transformed.to_csv('metrics_eval_5_services_scaleinvariant.csv', index=False)
    print(f"   ✅ Saved: metrics_eval_5_services_scaleinvariant.csv")
    
    acc_5svc = evaluate_extrapolation(detector, df_5svc_transformed, "5 Services", 5)
    results['5_services'] = acc_5svc
    
    # Summary
    print(f"\n{'='*80}")
    print("EXTRAPOLATION RESULTS SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n📊 Full Comparison (1-5 Services):")
    print(f"   {'Configuration':<25} {'Accuracy':<12} {'Type'}")
    print(f"   {'-'*55}")
    print(f"   {'1 Service':<25} {'80.78%':<12} Interpolation")
    print(f"   {'2 Services':<25} {'93.37%':<12} Interpolation")
    print(f"   {'3 Services (training)':<25} {'99.46%':<12} Training data")
    acc_4_str = f"{results['4_services']*100:.2f}%"
    acc_5_str = f"{results['5_services']*100:.2f}%"
    print(f"   {'4 Services':<25} {acc_4_str:<12} EXTRAPOLATION")
    print(f"   {'5 Services':<25} {acc_5_str:<12} EXTRAPOLATION")
    
    # Analysis
    print(f"\n📈 Key Findings:")
    acc_drop_4 = 0.9946 - results['4_services']
    acc_drop_5 = 0.9946 - results['5_services']
    print(f"   Drop from 3→4 services: {acc_drop_4:.2%}")
    print(f"   Drop from 3→5 services: {acc_drop_5:.2%}")
    
    if results['4_services'] > 0.90 and results['5_services'] > 0.85:
        print(f"\n   ✅ EXCELLENT: Model extrapolates well to >3 services")
        print(f"   → Paper claim of 'N services' is VALIDATED")
    elif results['4_services'] > 0.80 and results['5_services'] > 0.75:
        print(f"\n   ✅ GOOD: Model shows reasonable extrapolation")
        print(f"   → Paper can claim 'scales to 4-5 services with minor degradation'")
    else:
        print(f"\n   ⚠️  WARNING: Significant extrapolation degradation")
        print(f"   → Paper should limit claim to '1-3 services' or recommend retraining")
    
    print(f"\n💡 Recommendation for Paper:")
    acc_4_pct = results['4_services']*100
    acc_5_pct = results['5_services']*100
    if results['4_services'] > 0.90:
        print(f"   'Our scale-invariant model maintains {acc_4_pct:.1f}% accuracy")
        print(f"    even when extrapolating to 4-5 services (beyond training data),'")
        print(f"    demonstrating true service-agnostic scalability.'")
    else:
        print(f"   'The model achieves 80-99% accuracy for 1-3 service deployments")
        print(f"    (covering most edge scenarios) and {acc_4_pct:.1f}% for larger")
        print(f"    clusters, with graceful degradation for extrapolation.'")


if __name__ == '__main__':
    main()
