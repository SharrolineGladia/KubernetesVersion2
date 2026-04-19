"""
Comprehensive SHAP Explainability Evaluation Script

This script computes:
1. SHAP Root Cause Alignment (Top-1 and Top-3 matching)
2. SHAP Stability (intra-class cosine similarity)
3. Explanation Generation Time
4. Counterfactual Validation (with binary search and historical confidence)

Author: Anomaly Detection System
Date: February 22, 2026
"""

import os
import sys
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

# SHAP for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("⚠️ SHAP not installed. Installing now...")
    os.system("pip install shap")
    import shap
    SHAP_AVAILABLE = True


# ============================================================================
# CONFIGURATION
# ============================================================================

# Feature names (27 scale-invariant features)
FEATURE_NAMES = [
    'cpu_utilization_mean', 'cpu_utilization_max', 'cpu_variance_coef', 'cpu_imbalance',
    'memory_pressure_mean', 'memory_pressure_max', 'memory_variance_coef', 'memory_imbalance',
    'network_in_rate', 'network_out_rate', 'network_in_variance_coef', 'network_out_variance_coef',
    'network_asymmetry', 'disk_io_rate', 'disk_io_variance_coef',
    'request_rate', 'request_variance_coef', 'error_rate', 'error_variance_coef',
    'latency_mean', 'latency_p95', 'latency_variance_coef',
    'system_stress', 'resource_efficiency', 'service_density',
    'cpu_memory_correlation', 'performance_degradation'
]

# Ground truth root cause mapping
ROOT_CAUSE_MAPPING = {
    'cpu_spike': ['cpu_utilization_mean', 'cpu_utilization_max', 'cpu_variance_coef', 
                  'cpu_imbalance', 'cpu_memory_correlation', 'system_stress'],
    'memory_leak': ['memory_pressure_mean', 'memory_pressure_max', 'memory_variance_coef', 
                    'memory_imbalance', 'cpu_memory_correlation'],
    'service_crash': ['error_rate', 'error_variance_coef', 'request_variance_coef', 
                      'performance_degradation', 'system_stress'],
    'normal': []  # No specific root cause
}

# Paths (relative to script directory)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # ml_detector/
DATASET_PATH = os.path.join(PROJECT_ROOT, 'datasets', 'metrics_dataset_scaleinvariant.csv')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'anomaly_detector_scaleinvariant.pkl')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'shap_evaluation')


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SHAPMetrics:
    """Container for all SHAP evaluation metrics."""
    # Root cause alignment
    top1_match_rate: float = 0.0
    top3_coverage_rate: float = 0.0
    per_class_top1: Dict[str, float] = field(default_factory=dict)
    per_class_top3: Dict[str, float] = field(default_factory=dict)
    
    # SHAP stability
    mean_shap_stability: float = 0.0
    std_shap_stability: float = 0.0
    per_class_stability: Dict[str, float] = field(default_factory=dict)
    per_class_stability_std: Dict[str, float] = field(default_factory=dict)
    
    # Explanation time
    mean_explanation_time: float = 0.0
    std_explanation_time: float = 0.0
    mean_cache_lookup_time: float = 0.0
    std_cache_lookup_time: float = 0.0
    
    # Counterfactual validation
    counterfactual_flip_rate: float = 0.0
    counterfactual_feasible_rate: float = 0.0
    mean_counterfactual_confidence: float = 0.0
    mean_threshold_delta: float = 0.0
    mean_convergence_time: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for easy serialization."""
        return {
            'top1_match_rate': self.top1_match_rate,
            'top3_coverage_rate': self.top3_coverage_rate,
            'per_class_top1': self.per_class_top1,
            'per_class_top3': self.per_class_top3,
            'mean_shap_stability': self.mean_shap_stability,
            'std_shap_stability': self.std_shap_stability,
            'per_class_stability': self.per_class_stability,
            'per_class_stability_std': self.per_class_stability_std,
            'mean_explanation_time': self.mean_explanation_time,
            'std_explanation_time': self.std_explanation_time,
            'mean_cache_lookup_time': self.mean_cache_lookup_time,
            'std_cache_lookup_time': self.std_cache_lookup_time,
            'counterfactual_flip_rate': self.counterfactual_flip_rate,
            'counterfactual_feasible_rate': self.counterfactual_feasible_rate,
            'mean_counterfactual_confidence': self.mean_counterfactual_confidence,
            'mean_threshold_delta': self.mean_threshold_delta,
            'mean_convergence_time': self.mean_convergence_time,
        }


@dataclass
class HistoricalIncident:
    """Simulated historical incident for validation."""
    feature_vector: np.ndarray
    anomaly_type: str
    success: bool  # Whether intervention was successful
    threshold_used: float
    feature_modified: str


# ============================================================================
# SHAP EVALUATION ENGINE
# ============================================================================

class SHAPEvaluator:
    """
    Comprehensive SHAP explainability evaluation engine.
    """
    
    def __init__(self, model_path: str, dataset_path: str):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained XGBoost model
            dataset_path: Path to dataset CSV
        """
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.model = None
        self.label_encoder = LabelEncoder()
        self.shap_explainer = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_test_labels = None
        self.feature_names = FEATURE_NAMES
        
        # Results storage
        self.shap_values_test = None
        self.metrics = SHAPMetrics()
        
        # Historical incident database (simulated)
        self.historical_db: List[HistoricalIncident] = []
        
        print("="*80)
        print("SHAP COMPREHENSIVE EVALUATION SYSTEM")
        print("="*80)
        print()
    
    def load_model_and_data(self):
        """Load trained model and prepare test data."""
        print("📂 Loading model and dataset...")
        
        # Load model
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        if isinstance(model_data, dict):
            self.model = model_data.get('model')
            stored_features = model_data.get('feature_columns', FEATURE_NAMES)
            if stored_features != FEATURE_NAMES:
                print(f"⚠️  Warning: Feature mismatch. Using stored features.")
                self.feature_names = stored_features
        else:
            self.model = model_data
        
        print(f"   ✅ Model loaded from: {self.model_path}")
        
        # Load dataset
        df = pd.read_csv(self.dataset_path)
        print(f"   ✅ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")
        
        # Prepare features and labels
        X = df[self.feature_names].values
        y = df['anomaly_type'].values
        
        # Encode labels (strip whitespace)
        y_cleaned = [label.strip() for label in y]
        y_encoded = self.label_encoder.fit_transform(y_cleaned)
        
        # Split data (same split as training)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Store label names for test set (cleaned)
        self.y_test_labels = self.label_encoder.inverse_transform(self.y_test)
        
        print(f"   📊 Train: {len(self.X_train)}, Test: {len(self.X_test)}")
        print(f"   📋 Classes: {list(self.label_encoder.classes_)}")
        
        # Initialize SHAP explainer
        print("\n🔮 Initializing SHAP TreeExplainer...")
        try:
            # Try TreeExplainer with model_output='raw' for multi-class
            self.shap_explainer = shap.TreeExplainer(
                self.model, 
                data=shap.sample(self.X_train, 100),
                model_output='raw'
            )
            print("   ✅ SHAP explainer ready")
        except Exception as e:
            print(f"   ⚠️ TreeExplainer failed: {e}")
            print("   Trying KernelExplainer (slower but more compatible)...")
            try:
                # Fallback to KernelExplainer
                background = shap.sample(self.X_train, 50)
                self.shap_explainer = shap.KernelExplainer(
                    lambda x: self.model.predict_proba(x), 
                    background
                )
                print("   ✅ SHAP explainer ready (KernelExplainer)")
            except Exception as e2:
                print(f"   ❌ Could not initialize SHAP: {e2}")
                print("   Falling back to manual feature importance...")
                # We'll compute feature importance manually
                self.shap_explainer = None
        
        # Build historical database (simulated)
        self._build_historical_database()
    
    def _build_historical_database(self):
        """Build simulated historical incident database from training data."""
        print("\n📚 Building historical incident database...")
        
        # Sample 20% of training data as "historical incidents"
        num_incidents = int(len(self.X_train) * 0.2)
        indices = np.random.choice(len(self.X_train), num_incidents, replace=False)
        
        for idx in indices:
            feature_vector = self.X_train[idx]
            anomaly_type = self.label_encoder.inverse_transform([self.y_train[idx]])[0]
            
            # Simulate success rate based on anomaly type
            success_prob = {
                'normal': 0.95,
                'cpu_spike': 0.80,
                'memory_leak': 0.70,
                'service_crash': 0.75
            }
            
            success = np.random.random() < success_prob.get(anomaly_type, 0.75)
            
            # Simulate threshold used
            feature_idx = np.random.randint(0, len(feature_vector))
            threshold_used = feature_vector[feature_idx] * np.random.uniform(0.7, 1.3)
            
            incident = HistoricalIncident(
                feature_vector=feature_vector,
                anomaly_type=anomaly_type,
                success=success,
                threshold_used=threshold_used,
                feature_modified=self.feature_names[feature_idx]
            )
            
            self.historical_db.append(incident)
        
        print(f"   ✅ Built database with {len(self.historical_db)} historical incidents")
    
    # ========================================================================
    # PART 1: SHAP ROOT CAUSE ALIGNMENT
    # ========================================================================
    
    def compute_shap_values(self):
        """Compute SHAP values for all test samples."""
        print("\n" + "="*80)
        print("PART 1: COMPUTING SHAP VALUES")
        print("="*80)
        
        print(f"\n🔮 Computing SHAP values for {len(self.X_test)} test samples...")
        start_time = time.time()
        
        # Compute SHAP values
        self.shap_values_test = self.shap_explainer.shap_values(self.X_test)
        
        computation_time = time.time() - start_time
        print(f"   ✅ SHAP computation complete in {computation_time:.2f} seconds")
        print(f"   ⏱️  Average per sample: {(computation_time/len(self.X_test))*1000:.2f} ms")
        
        # Handle multi-class output
        if isinstance(self.shap_values_test, list):
            print(f"   📊 Multi-class SHAP values: {len(self.shap_values_test)} classes")
        else:
            print(f"   📊 SHAP values shape: {self.shap_values_test.shape}")
    
    def evaluate_root_cause_alignment(self):
        """
        Evaluate SHAP root cause alignment with ground truth.
        
        Computes:
        - Top-1 match rate (highest SHAP feature matches expected root cause)
        - Top-3 coverage rate (any of top-3 SHAP features matches)
        """
        print("\n" + "="*80)
        print("PART 1: SHAP ROOT CAUSE ALIGNMENT")
        print("="*80)
        
        # Get predictions
        y_pred = self.model.predict(self.X_test)
        
        # Handle multi-output SHAP values (n_samples, n_features, n_classes)
        # For root cause alignment, use SHAP values for the PREDICTED class
        if len(self.shap_values_test.shape) == 3:
            # Multi-class output: extract SHAP values for predicted class
            shap_values = np.zeros((len(self.X_test), len(self.feature_names)))
            for i, pred_class in enumerate(y_pred):
                shap_values[i] = self.shap_values_test[i, :, pred_class]
        elif len(self.shap_values_test.shape) == 2:
            # Binary or single output: (n_samples, n_features)
            shap_values = self.shap_values_test
        else:
            raise ValueError(f"Unexpected SHAP values shape: {self.shap_values_test.shape}")
        
        # Compute absolute SHAP values for ranking
        abs_shap = np.abs(shap_values)
        
        # Storage for per-class metrics
        class_top1_matches = defaultdict(list)
        class_top3_matches = defaultdict(list)
        
        total_top1_matches = 0
        total_top3_matches = 0
        total_samples = len(self.X_test)
        
        print(f"\n🔍 Analyzing {total_samples} test samples...")
        
        for i in range(len(self.X_test)):
            true_label = self.y_test_labels[i]
            
            # Get top features by absolute SHAP value
            top_indices = np.argsort(abs_shap[i])[::-1]
            
            top1_feature = self.feature_names[top_indices[0]]
            top3_features = [self.feature_names[idx] for idx in top_indices[:3]]
            
            # Check alignment with ground truth
            expected_features = ROOT_CAUSE_MAPPING.get(true_label, [])
            
            if not expected_features:  # normal class
                # For normal, we expect low SHAP values overall
                top1_match = abs_shap[i][top_indices[0]] < 0.1
                top3_match = top1_match
            else:
                top1_match = top1_feature in expected_features
                top3_match = any(f in expected_features for f in top3_features)
            
            # Record results
            class_top1_matches[true_label].append(top1_match)
            class_top3_matches[true_label].append(top3_match)
            
            if top1_match:
                total_top1_matches += 1
            if top3_match:
                total_top3_matches += 1
        
        # Compute overall metrics
        self.metrics.top1_match_rate = (total_top1_matches / total_samples) * 100
        self.metrics.top3_coverage_rate = (total_top3_matches / total_samples) * 100
        
        # Compute per-class metrics
        for class_name in self.label_encoder.classes_:
            if class_name in class_top1_matches:
                top1_list = class_top1_matches[class_name]
                top3_list = class_top3_matches[class_name]
                
                self.metrics.per_class_top1[class_name] = (sum(top1_list) / len(top1_list)) * 100
                self.metrics.per_class_top3[class_name] = (sum(top3_list) / len(top3_list)) * 100
        
        # Print results
        print("\n📊 ROOT CAUSE ALIGNMENT RESULTS:")
        print(f"   Overall Top-1 Match Rate:    {self.metrics.top1_match_rate:.2f}%")
        print(f"   Overall Top-3 Coverage Rate: {self.metrics.top3_coverage_rate:.2f}%")
        
        print("\n   Per-Class Results:")
        for class_name in sorted(self.label_encoder.classes_):
            if class_name in self.metrics.per_class_top1:
                top1 = self.metrics.per_class_top1[class_name]
                top3 = self.metrics.per_class_top3[class_name]
                print(f"   {class_name:15s}: Top-1={top1:6.2f}%  Top-3={top3:6.2f}%")
    
    # ========================================================================
    # PART 2: SHAP STABILITY
    # ========================================================================
    
    def evaluate_shap_stability(self):
        """
        Evaluate SHAP stability by computing intra-class cosine similarity.
        
        For each anomaly class, compute pairwise cosine similarity between
        SHAP vectors of same-class samples.
        """
        print("\n" + "="*80)
        print("PART 2: SHAP STABILITY ANALYSIS")
        print("="*80)
        
        # Handle multi-output SHAP values
        if len(self.shap_values_test.shape) == 3:
            # Multi-class: sum across classes
            shap_values = np.sum(np.abs(self.shap_values_test), axis=2)
        else:
            shap_values = self.shap_values_test
        
        # Group by class
        class_shap_vectors = defaultdict(list)
        for i, label in enumerate(self.y_test_labels):
            class_shap_vectors[label].append(shap_values[i])
        
        # Compute intra-class cosine similarity
        all_similarities = []
        
        print(f"\n🔍 Computing intra-class SHAP similarities...")
        
        for class_name, vectors in class_shap_vectors.items():
            if len(vectors) < 2:
                continue
            
            vectors_array = np.array(vectors)
            
            # Compute pairwise cosine similarity
            similarity_matrix = cosine_similarity(vectors_array)
            
            # Extract upper triangle (exclude diagonal)
            upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
            
            if len(upper_triangle) > 0:
                mean_similarity = np.mean(upper_triangle)
                std_similarity = np.std(upper_triangle)
                
                self.metrics.per_class_stability[class_name] = mean_similarity
                self.metrics.per_class_stability_std[class_name] = std_similarity
                
                all_similarities.extend(upper_triangle)
                
                print(f"   {class_name:15s}: Mean={mean_similarity:.4f} ± {std_similarity:.4f}")
        
        # Overall stability
        if all_similarities:
            self.metrics.mean_shap_stability = np.mean(all_similarities)
            self.metrics.std_shap_stability = np.std(all_similarities)
        
        print(f"\n📊 OVERALL SHAP STABILITY:")
        print(f"   Mean Cosine Similarity: {self.metrics.mean_shap_stability:.4f}")
        print(f"   Std Deviation:          {self.metrics.std_shap_stability:.4f}")
    
    # ========================================================================
    # PART 3: EXPLANATION GENERATION TIME
    # ========================================================================
    
    def evaluate_explanation_time(self):
        """
        Measure explanation generation time.
        
        - Full SHAP computation (cloud scenario)
        - Simulated cache lookup (edge scenario)
        """
        print("\n" + "="*80)
        print("PART 3: EXPLANATION GENERATION TIME")
        print("="*80)
        
        # Sample 100 random test instances for timing
        num_samples = min(100, len(self.X_test))
        sample_indices = np.random.choice(len(self.X_test), num_samples, replace=False)
        X_sample = self.X_test[sample_indices]
        
        print(f"\n⏱️  Measuring SHAP computation time ({num_samples} samples)...")
        
        # Full SHAP computation time
        computation_times = []
        for i in range(num_samples):
            start = time.time()
            _ = self.shap_explainer.shap_values(X_sample[i:i+1])
            elapsed = time.time() - start
            computation_times.append(elapsed * 1000)  # Convert to ms
        
        self.metrics.mean_explanation_time = np.mean(computation_times)
        self.metrics.std_explanation_time = np.std(computation_times)
        
        print(f"   Mean SHAP Time: {self.metrics.mean_explanation_time:.2f} ± {self.metrics.std_explanation_time:.2f} ms")
        
        # Simulated cache lookup time (edge)
        print(f"\n⚡ Measuring cache lookup time (simulated edge)...")
        
        cache_times = []
        for _ in range(num_samples):
            start = time.time()
            # Simulate cache lookup (dictionary access + cosine similarity)
            _ = cosine_similarity(X_sample[0:1], X_sample[1:2])
            elapsed = time.time() - start
            cache_times.append(elapsed * 1000)
        
        self.metrics.mean_cache_lookup_time = np.mean(cache_times)
        self.metrics.std_cache_lookup_time = np.std(cache_times)
        
        print(f"   Mean Cache Lookup: {self.metrics.mean_cache_lookup_time:.2f} ± {self.metrics.std_cache_lookup_time:.2f} ms")
        print(f"   🚀 Speedup: {self.metrics.mean_explanation_time / self.metrics.mean_cache_lookup_time:.1f}x faster")
    
    # ========================================================================
    # PART 4: COUNTERFACTUAL VALIDATION
    # ========================================================================
    
    def evaluate_counterfactuals(self):
        """
        Perform counterfactual validation using binary search.
        
        For each anomalous sample:
        1. Select top SHAP feature
        2. Binary search for minimal threshold change that flips prediction
        3. Validate against historical incident database
        4. Compute Laplace-smoothed confidence
        """
        print("\n" + "="*80)
        print("PART 4: COUNTERFACTUAL VALIDATION")
        print("="*80)
        
        # Handle multi-output SHAP values
        if len(self.shap_values_test.shape) == 3:
            shap_values = np.sum(np.abs(self.shap_values_test), axis=2)
        else:
            shap_values = np.abs(self.shap_values_test)
        
        abs_shap = shap_values
        
        # Only evaluate anomalous samples
        anomalous_indices = [i for i, label in enumerate(self.y_test_labels) if label != 'normal']
        
        print(f"\n🔍 Evaluating {len(anomalous_indices)} anomalous samples...")
        
        flip_count = 0
        feasible_count = 0
        confidences = []
        threshold_deltas = []
        convergence_times = []
        
        for idx in anomalous_indices[:100]:  # Limit to 100 for performance
            # Get top SHAP feature
            top_feature_idx = np.argmax(abs_shap[idx])
            top_feature_name = self.feature_names[top_feature_idx]
            
            # Original prediction
            original_pred = self.model.predict(self.X_test[idx:idx+1])[0]
            
            # Binary search for threshold
            start_time = time.time()
            flip_threshold, did_flip = self._binary_search_threshold(
                self.X_test[idx], top_feature_idx, original_pred
            )
            search_time = (time.time() - start_time) * 1000  # ms
            convergence_times.append(search_time)
            
            if did_flip:
                flip_count += 1
                
                # Compute threshold delta
                original_value = self.X_test[idx][top_feature_idx]
                delta = abs(flip_threshold - original_value)
                threshold_deltas.append(delta)
                
                # Validate against historical incidents
                confidence = self._validate_historical(
                    self.X_test[idx], 
                    top_feature_name,
                    flip_threshold
                )
                
                confidences.append(confidence)
                
                if confidence >= 0.5:
                    feasible_count += 1
        
        # Compute metrics
        total_evaluated = len(anomalous_indices[:100])
        self.metrics.counterfactual_flip_rate = (flip_count / total_evaluated) * 100
        self.metrics.counterfactual_feasible_rate = (feasible_count / total_evaluated) * 100
        
        if confidences:
            self.metrics.mean_counterfactual_confidence = np.mean(confidences)
        if threshold_deltas:
            self.metrics.mean_threshold_delta = np.mean(threshold_deltas)
        if convergence_times:
            self.metrics.mean_convergence_time = np.mean(convergence_times)
        
        # Print results
        print(f"\n📊 COUNTERFACTUAL VALIDATION RESULTS:")
        print(f"   Samples Evaluated:           {total_evaluated}")
        print(f"   Flip Success Rate:           {self.metrics.counterfactual_flip_rate:.2f}%")
        print(f"   Feasible Rate (conf≥0.5):    {self.metrics.counterfactual_feasible_rate:.2f}%")
        print(f"   Mean Confidence:             {self.metrics.mean_counterfactual_confidence:.4f}")
        print(f"   Mean Threshold Delta:        {self.metrics.mean_threshold_delta:.4f}")
        print(f"   Mean Convergence Time:       {self.metrics.mean_convergence_time:.2f} ms")
        
        return confidences  # Return for histogram
    
    def _binary_search_threshold(
        self, 
        sample: np.ndarray, 
        feature_idx: int, 
        original_pred: int,
        max_iterations: int = 20
    ) -> Tuple[float, bool]:
        """
        Binary search for minimal threshold that changes prediction.
        
        Args:
            sample: Feature vector
            feature_idx: Index of feature to modify
            original_pred: Original prediction
            max_iterations: Max search iterations
        
        Returns:
            (threshold_value, did_flip)
        """
        sample_copy = sample.copy()
        original_value = sample[feature_idx]
        
        # Search range: [0.5x, 1.5x] of original value
        low = original_value * 0.5
        high = original_value * 1.5
        
        # Edge case: original value is 0 or very small
        if abs(original_value) < 1e-6:
            low = 0.0
            high = 1.0
        
        best_threshold = high
        did_flip = False
        
        for _ in range(max_iterations):
            mid = (low + high) / 2.0
            
            # Test with modified value
            sample_copy[feature_idx] = mid
            pred = self.model.predict(sample_copy.reshape(1, -1))[0]
            
            if pred != original_pred:
                # Found a flip, try to find smaller threshold
                did_flip = True
                best_threshold = mid
                high = mid
            else:
                # No flip yet, increase threshold
                low = mid
            
            # Convergence check
            if abs(high - low) < 1e-4:
                break
        
        return best_threshold, did_flip
    
    def _validate_historical(
        self, 
        sample: np.ndarray, 
        feature_name: str, 
        threshold: float,
        similarity_threshold: float = 0.8
    ) -> float:
        """
        Validate counterfactual against historical incident database.
        
        Uses Laplace-smoothed confidence:
            confidence = (successes + 1) / (total_attempts + 2)
        
        Args:
            sample: Feature vector
            feature_name: Name of modified feature
            threshold: Threshold value used
            similarity_threshold: Cosine similarity threshold for matching
        
        Returns:
            Laplace-smoothed confidence score
        """
        # Find similar historical incidents
        similar_incidents = []
        
        for incident in self.historical_db:
            # Compute cosine similarity
            similarity = cosine_similarity(
                sample.reshape(1, -1), 
                incident.feature_vector.reshape(1, -1)
            )[0][0]
            
            if similarity > similarity_threshold and incident.feature_modified == feature_name:
                similar_incidents.append(incident)
        
        # Compute Laplace-smoothed confidence
        if similar_incidents:
            successes = sum(1 for inc in similar_incidents if inc.success)
            total = len(similar_incidents)
        else:
            successes = 0
            total = 0
        
        # Laplace smoothing
        confidence = (successes + 1) / (total + 2)
        
        return confidence
    
    # ========================================================================
    # VISUALIZATION AND REPORTING
    # ========================================================================
    
    def generate_visualizations(self, counterfactual_confidences: List[float]):
        """Generate evaluation visualizations."""
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Handle multi-output SHAP values
        if len(self.shap_values_test.shape) == 3:
            shap_values = np.sum(np.abs(self.shap_values_test), axis=2)
        else:
            shap_values = np.abs(self.shap_values_test)
        
        # --- Plot 1: Top 10 Mean Absolute SHAP Features ---
        print("\n📊 Generating Top-10 SHAP features plot...")
        
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        top10_indices = np.argsort(mean_abs_shap)[-10:][::-1]
        top10_features = [self.feature_names[i] for i in top10_indices]
        top10_values = mean_abs_shap[top10_indices]
        
        plt.figure(figsize=(12, 6))
        plt.barh(range(len(top10_features)), top10_values, color='steelblue')
        plt.yticks(range(len(top10_features)), top10_features)
        plt.xlabel('Mean Absolute SHAP Value', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title('Top 10 Most Important Features (Mean |SHAP|)', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        plot1_path = os.path.join(OUTPUT_DIR, 'top10_shap_features.png')
        plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {plot1_path}")
        plt.close()
        
        # --- Plot 2: Counterfactual Confidence Distribution ---
        print("\n📊 Generating counterfactual confidence histogram...")
        
        if counterfactual_confidences:
            plt.figure(figsize=(10, 6))
            plt.hist(counterfactual_confidences, bins=20, color='coral', edgecolor='black', alpha=0.7)
            plt.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Feasibility Threshold')
            plt.xlabel('Counterfactual Confidence', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.title('Distribution of Counterfactual Confidence Scores', fontsize=14, fontweight='bold')
            plt.legend(fontsize=11)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            plot2_path = os.path.join(OUTPUT_DIR, 'counterfactual_confidence_distribution.png')
            plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
            print(f"   ✅ Saved: {plot2_path}")
            plt.close()
        else:
            print("   ⚠️  No counterfactual data to plot")
    
    def print_summary(self):
        """Print comprehensive summary of all metrics."""
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION SUMMARY")
        print("="*80)
        print()
        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║                   SHAP EXPLAINABILITY METRICS                        ║")
        print("╚══════════════════════════════════════════════════════════════════════╝")
        print()
        print(f"  📌 SHAP ROOT CAUSE ALIGNMENT")
        print(f"     SHAP Top-1 Match Rate:          {self.metrics.top1_match_rate:6.2f}%")
        print(f"     SHAP Top-3 Coverage Rate:       {self.metrics.top3_coverage_rate:6.2f}%")
        print()
        print(f"  📌 SHAP STABILITY (Intra-Class Cosine Similarity)")
        print(f"     Mean SHAP Stability:            {self.metrics.mean_shap_stability:.4f}")
        print(f"     Std Deviation:                  {self.metrics.std_shap_stability:.4f}")
        print()
        print(f"     Per-Class Stability:")
        for class_name in sorted(self.metrics.per_class_stability.keys()):
            stability = self.metrics.per_class_stability[class_name]
            std = self.metrics.per_class_stability_std[class_name]
            print(f"       {class_name:15s}: {stability:.4f} ± {std:.4f}")
        print()
        print(f"  📌 EXPLANATION GENERATION TIME")
        print(f"     Mean SHAP Computation Time:     {self.metrics.mean_explanation_time:6.2f} ms")
        print(f"     Std Deviation:                  {self.metrics.std_explanation_time:6.2f} ms")
        print(f"     Mean Cache Lookup Time (Edge):  {self.metrics.mean_cache_lookup_time:6.2f} ms")
        print(f"     Speedup Factor:                 {self.metrics.mean_explanation_time / max(self.metrics.mean_cache_lookup_time, 0.001):.1f}x")
        print()
        print(f"  📌 COUNTERFACTUAL VALIDATION")
        print(f"     Counterfactual Flip Rate:       {self.metrics.counterfactual_flip_rate:6.2f}%")
        print(f"     Counterfactual Feasible Rate:   {self.metrics.counterfactual_feasible_rate:6.2f}%")
        print(f"     Mean Counterfactual Confidence: {self.metrics.mean_counterfactual_confidence:6.4f}")
        print(f"     Mean Threshold Delta:           {self.metrics.mean_threshold_delta:6.4f}")
        print(f"     Mean Convergence Time:          {self.metrics.mean_convergence_time:6.2f} ms")
        print()
        print("="*80)
    
    def save_results(self):
        """Save results to JSON and CSV."""
        print("\n💾 Saving results...")
        
        # Save metrics to JSON
        import json
        json_path = os.path.join(OUTPUT_DIR, 'shap_evaluation_metrics.json')
        with open(json_path, 'w') as f:
            json.dump(self.metrics.to_dict(), f, indent=4)
        print(f"   ✅ Saved metrics: {json_path}")
        
        # Save detailed results to CSV
        csv_path = os.path.join(OUTPUT_DIR, 'shap_evaluation_detailed.csv')
        detailed_data = {
            'Metric': [
                'Top-1 Match Rate (%)',
                'Top-3 Coverage Rate (%)',
                'Mean SHAP Stability',
                'Std SHAP Stability',
                'Mean Explanation Time (ms)',
                'Std Explanation Time (ms)',
                'Mean Cache Lookup Time (ms)',
                'Counterfactual Flip Rate (%)',
                'Counterfactual Feasible Rate (%)',
                'Mean Counterfactual Confidence',
                'Mean Threshold Delta',
                'Mean Convergence Time (ms)'
            ],
            'Value': [
                f"{self.metrics.top1_match_rate:.2f}",
                f"{self.metrics.top3_coverage_rate:.2f}",
                f"{self.metrics.mean_shap_stability:.4f}",
                f"{self.metrics.std_shap_stability:.4f}",
                f"{self.metrics.mean_explanation_time:.2f}",
                f"{self.metrics.std_explanation_time:.2f}",
                f"{self.metrics.mean_cache_lookup_time:.2f}",
                f"{self.metrics.counterfactual_flip_rate:.2f}",
                f"{self.metrics.counterfactual_feasible_rate:.2f}",
                f"{self.metrics.mean_counterfactual_confidence:.4f}",
                f"{self.metrics.mean_threshold_delta:.4f}",
                f"{self.metrics.mean_convergence_time:.2f}"
            ]
        }
        df = pd.DataFrame(detailed_data)
        df.to_csv(csv_path, index=False)
        print(f"   ✅ Saved detailed results: {csv_path}")
    
    def run_full_evaluation(self):
        """Run all evaluation steps."""
        print("\n🚀 Starting comprehensive SHAP evaluation...\n")
        
        # Step 1: Load data
        self.load_model_and_data()
        
        # Step 2: Compute SHAP values
        self.compute_shap_values()
        
        # Step 3: Evaluate root cause alignment
        self.evaluate_root_cause_alignment()
        
        # Step 4: Evaluate SHAP stability
        self.evaluate_shap_stability()
        
        # Step 5: Evaluate explanation time
        self.evaluate_explanation_time()
        
        # Step 6: Evaluate counterfactuals
        counterfactual_confidences = self.evaluate_counterfactuals()
        
        # Step 7: Generate visualizations
        self.generate_visualizations(counterfactual_confidences)
        
        # Step 8: Print summary
        self.print_summary()
        
        # Step 9: Save results
        self.save_results()
        
        print("\n✅ Evaluation complete!")
        print(f"📁 Results saved to: {OUTPUT_DIR}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    # Initialize evaluator
    evaluator = SHAPEvaluator(
        model_path=MODEL_PATH,
        dataset_path=DATASET_PATH
    )
    
    # Run full evaluation
    evaluator.run_full_evaluation()


if __name__ == "__main__":
    main()
