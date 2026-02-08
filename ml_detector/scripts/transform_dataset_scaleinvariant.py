"""
Scale-Invariant Feature Transformation for Service-Agnostic Anomaly Detection

This transformer creates features that are invariant to the number of services,
making the model work with 1, 2, 3, or N services without retraining.

Key principles:
- Use ratios and percentages instead of absolute values
- Use coefficients of variation (std/mean) instead of raw std
- Use normalized metrics that have physical meaning
- Focus on relative patterns, not absolute magnitudes
"""

import pandas as pd
import numpy as np
from typing import Dict, List


class ScaleInvariantTransformer:
    """
    Transform service-specific metrics to scale-invariant features.
    
    Features are designed to be meaningful regardless of service count,
    enabling the model to work across heterogeneous edge deployments.
    """
    
    def __init__(self):
        # Define scale-invariant feature names
        self.feature_names = [
            # CPU metrics (utilization-based)
            'cpu_utilization_mean',      # Average CPU usage ratio
            'cpu_utilization_max',       # Peak CPU usage
            'cpu_variance_coef',         # Relative CPU spread (std/mean)
            'cpu_imbalance',             # Resource distribution unevenness
            
            # Memory metrics (pressure-based)
            'memory_pressure_mean',      # Average memory usage
            'memory_pressure_max',       # Peak memory usage
            'memory_variance_coef',      # Relative memory spread
            'memory_imbalance',          # Memory distribution unevenness
            
            # Network metrics (traffic patterns)
            'network_in_rate',           # Average incoming traffic rate
            'network_out_rate',          # Average outgoing traffic rate
            'network_in_variance_coef',  # Network in variability
            'network_out_variance_coef', # Network out variability
            'network_asymmetry',         # In/out traffic balance
            
            # Disk I/O metrics
            'disk_io_rate',              # Average disk I/O rate
            'disk_io_variance_coef',     # Disk I/O variability
            
            # Request handling metrics
            'request_rate',              # Average request throughput
            'request_variance_coef',     # Request variability
            'error_rate',                # Error percentage
            'error_variance_coef',       # Error variability
            
            # Latency metrics (performance)
            'latency_mean',              # Average response time
            'latency_p95',               # 95th percentile latency
            'latency_variance_coef',     # Latency variability
            
            # System-wide metrics
            'system_stress',             # Overall system stress
            'resource_efficiency',       # How efficiently resources are used
            'service_density',           # Number of active services
            
            # Cross-metric correlations
            'cpu_memory_correlation',    # CPU-memory usage pattern
            'performance_degradation',   # Latency vs load ratio
        ]
    
    def aggregate_row(self, row: pd.Series) -> Dict[str, float]:
        """
        Transform a single row to scale-invariant features.
        
        Args:
            row: Original row with service-specific metrics
        
        Returns:
            Dictionary of scale-invariant features
        """
        features = {}
        
        # Extract service-specific metrics
        # Map actual column names from dataset
        cpu_values = [
            row.get('notification_cpu', 0),
            row.get('web_api_cpu', 0),
            row.get('processor_cpu', 0)
        ]
        memory_values = [
            row.get('notification_memory', 0),
            row.get('web_api_memory', 0),
            row.get('processor_memory', 0)
        ]
        # Use available metrics as proxies for network/disk/requests
        requests = [
            row.get('notification_message_rate', 0),
            row.get('web_api_requests', 0),
            row.get('processor_processing_rate', 0)
        ]
        errors = [
            row.get('notification_error_rate', 0) * 100,  # Convert to percentage
            row.get('web_api_errors', 0),
            0  # No processor error metric
        ]
        latency = [
            row.get('notification_api_health', 0) * 100,  # Invert health to latency proxy
            row.get('web_api_response_time_p95', 0),
            row.get('processor_response_time_p95', 0)
        ]
        # Network and disk - use queue metrics as proxies
        network_in = [
            row.get('notification_queue_depth', 0),
            row.get('web_api_queue_depth', 0),
            row.get('processor_queue_depth', 0)
        ]
        network_out = [
            row.get('notification_delivery_success', 0) * 100,
            row.get('web_api_requests_per_second', 0),
            0  # No processor network out
        ]
        disk_io = [
            row.get('notification_thread_count', 0),
            row.get('web_api_thread_count', 0),
            row.get('processor_thread_count', 0)
        ]
        
        # Count active services (non-zero activity)
        num_active = sum(1 for cpu in cpu_values if cpu > 0)
        if num_active == 0:
            num_active = 1  # Avoid division by zero
        
        # Filter to only active services
        active_cpu = [v for v in cpu_values if v > 0] or [0]
        active_memory = [v for v in memory_values if v > 0] or [0]
        active_network_in = [v for v in network_in if v > 0] or [0]
        active_network_out = [v for v in network_out if v > 0] or [0]
        active_disk_io = [v for v in disk_io if v > 0] or [0]
        active_requests = [v for v in requests if v > 0] or [0]
        active_errors = [v for v in errors if v > 0] or [0]
        active_latency = [v for v in latency if v > 0] or [0]
        
        # CPU features (normalized to 0-1 assuming max ~100%)
        cpu_mean = np.mean(active_cpu)
        cpu_max = np.max(active_cpu)
        cpu_min = np.min(active_cpu)
        cpu_std = np.std(active_cpu) if len(active_cpu) > 1 else 0
        
        features['cpu_utilization_mean'] = cpu_mean / 100.0  # Normalize to 0-1
        features['cpu_utilization_max'] = cpu_max / 100.0
        features['cpu_variance_coef'] = (cpu_std / cpu_mean) if cpu_mean > 0 else 0
        features['cpu_imbalance'] = ((cpu_max - cpu_min) / cpu_mean) if cpu_mean > 0 else 0
        
        # Memory features (normalized to 0-1 assuming max ~100%)
        memory_mean = np.mean(active_memory)
        memory_max = np.max(active_memory)
        memory_min = np.min(active_memory)
        memory_std = np.std(active_memory) if len(active_memory) > 1 else 0
        
        features['memory_pressure_mean'] = memory_mean / 100.0
        features['memory_pressure_max'] = memory_max / 100.0
        features['memory_variance_coef'] = (memory_std / memory_mean) if memory_mean > 0 else 0
        features['memory_imbalance'] = ((memory_max - memory_min) / memory_mean) if memory_mean > 0 else 0
        
        # Network features (rates, not absolute values)
        net_in_mean = np.mean(active_network_in)
        net_out_mean = np.mean(active_network_out)
        net_in_std = np.std(active_network_in) if len(active_network_in) > 1 else 0
        net_out_std = np.std(active_network_out) if len(active_network_out) > 1 else 0
        
        # Normalize by typical bandwidth (e.g., 1000 MB/s)
        features['network_in_rate'] = net_in_mean / 1000.0
        features['network_out_rate'] = net_out_mean / 1000.0
        features['network_in_variance_coef'] = (net_in_std / net_in_mean) if net_in_mean > 0 else 0
        features['network_out_variance_coef'] = (net_out_std / net_out_mean) if net_out_mean > 0 else 0
        features['network_asymmetry'] = abs(net_in_mean - net_out_mean) / (net_in_mean + net_out_mean + 1e-6)
        
        # Disk I/O features
        disk_mean = np.mean(active_disk_io)
        disk_std = np.std(active_disk_io) if len(active_disk_io) > 1 else 0
        
        features['disk_io_rate'] = disk_mean / 1000.0  # Normalize
        features['disk_io_variance_coef'] = (disk_std / disk_mean) if disk_mean > 0 else 0
        
        # Request handling features
        req_mean = np.mean(active_requests)
        req_std = np.std(active_requests) if len(active_requests) > 1 else 0
        err_mean = np.mean(active_errors)
        err_std = np.std(active_errors) if len(active_errors) > 1 else 0
        
        features['request_rate'] = req_mean / 1000.0  # Normalize to typical scale
        features['request_variance_coef'] = (req_std / req_mean) if req_mean > 0 else 0
        
        # Error rate as percentage of requests
        total_requests = sum(active_requests)
        total_errors = sum(active_errors)
        features['error_rate'] = (total_errors / total_requests) if total_requests > 0 else 0
        features['error_variance_coef'] = (err_std / err_mean) if err_mean > 0 else 0
        
        # Latency features
        lat_mean = np.mean(active_latency)
        lat_std = np.std(active_latency) if len(active_latency) > 1 else 0
        lat_max = np.max(active_latency)
        
        features['latency_mean'] = lat_mean / 1000.0  # Normalize to seconds
        features['latency_p95'] = lat_max / 1000.0  # Approximate P95 with max
        features['latency_variance_coef'] = (lat_std / lat_mean) if lat_mean > 0 else 0
        
        # System-wide metrics
        features['system_stress'] = row.get('system_stress', 0)
        
        # Resource efficiency: how well are resources utilized?
        # High CPU + High memory + High requests = efficient
        # High CPU + Low requests = inefficient
        resource_util = (cpu_mean + memory_mean) / 200.0  # 0-1
        load = req_mean / 1000.0
        features['resource_efficiency'] = load / (resource_util + 0.01) if resource_util > 0 else 0
        
        # Service density (normalized)
        features['service_density'] = num_active / 10.0  # Assuming max ~10 services
        
        # Cross-metric correlations
        # CPU-memory correlation: do they move together?
        if len(active_cpu) > 1 and len(active_memory) > 1:
            features['cpu_memory_correlation'] = np.corrcoef(
                active_cpu[:len(active_memory)], 
                active_memory[:len(active_cpu)]
            )[0, 1] if not np.isnan(np.corrcoef(active_cpu[:len(active_memory)], active_memory[:len(active_cpu)])[0, 1]) else 0
        else:
            features['cpu_memory_correlation'] = 0
        
        # Performance degradation: latency vs load
        # High latency with low load = degradation
        features['performance_degradation'] = (lat_mean / (req_mean + 1)) / 1000.0
        
        return features
    
    def transform_dataset(self, input_csv: str, output_csv: str):
        """
        Transform entire dataset to scale-invariant features.
        
        Args:
            input_csv: Path to original dataset
            output_csv: Path to save transformed dataset
        """
        print("="*80)
        print("SCALE-INVARIANT FEATURE TRANSFORMATION")
        print("="*80)
        print()
        
        # Load original dataset
        print(f"📊 Loading dataset: {input_csv}")
        df = pd.read_csv(input_csv)
        df.columns = df.columns.str.strip()
        print(f"   Original shape: {df.shape}")
        print(f"   Original features: {len([c for c in df.columns if c not in ['timestamp', 'anomaly_label', 'anomaly_type']])}")
        
        # Transform each row
        print(f"\n🔄 Transforming to scale-invariant features...")
        transformed_rows = []
        
        for idx, row in df.iterrows():
            # Transform features
            features = self.aggregate_row(row)
            
            # Preserve labels
            features['timestamp'] = row.get('timestamp', '')
            features['anomaly_label'] = row.get('anomaly_label', 0)
            features['anomaly_type'] = row.get('anomaly_type', 'normal')
            
            transformed_rows.append(features)
            
            if (idx + 1) % 500 == 0:
                print(f"   Processed {idx + 1}/{len(df)} rows...")
        
        # Create new dataframe
        df_transformed = pd.DataFrame(transformed_rows)
        
        # Reorder columns: features first, then labels
        feature_cols = self.feature_names
        label_cols = ['timestamp', 'anomaly_label', 'anomaly_type']
        df_transformed = df_transformed[feature_cols + label_cols]
        
        # Save
        df_transformed.to_csv(output_csv, index=False)
        
        print(f"\n✅ Transformation complete!")
        print(f"   New shape: {df_transformed.shape}")
        print(f"   New features: {len(self.feature_names)}")
        print(f"   Saved to: {output_csv}")
        
        # Show feature summary
        print(f"\n📊 Feature Summary:")
        for col in feature_cols[:10]:  # Show first 10
            print(f"   {col:30s} min={df_transformed[col].min():.4f}  "
                  f"max={df_transformed[col].max():.4f}  "
                  f"mean={df_transformed[col].mean():.4f}")
        print(f"   ... and {len(feature_cols) - 10} more features")
        
        return df_transformed


def main():
    """Transform dataset to scale-invariant features."""
    transformer = ScaleInvariantTransformer()
    
    input_file = 'metrics_dataset_enhanced_rounded.csv'
    output_file = 'metrics_dataset_scaleinvariant.csv'
    
    df_transformed = transformer.transform_dataset(input_file, output_file)
    
    print(f"\n📈 Class Distribution:")
    for class_name, count in df_transformed['anomaly_type'].value_counts().items():
        percentage = (count / len(df_transformed)) * 100
        print(f"   {class_name:20s}: {count:5d} ({percentage:5.1f}%)")


if __name__ == '__main__':
    main()
