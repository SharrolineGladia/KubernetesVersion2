"""
Comprehensive SHAP Explainability Evaluation Script - Version 2

Enhanced with:
1. Fixed SHAP Root Cause Alignment (feature groups, anomaly-only scoring)
2. Strengthened Counterfactual Search (top-2 features, combined perturbation)
3. NEW: Recovery Alignment Metric (historical validation)
4. Enhanced visualizations and reporting

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
from collections import defaultdict, Counter

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

# SHAP for explainability
try:
    import shap
    from tqdm import tqdm
    SHAP_AVAILABLE = True
except ImportError:
    print("⚠️ SHAP not installed. Installing now...")
    os.system("pip install shap tqdm")
    import shap
    from tqdm import tqdm
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

# =========================
# NEW: Feature Groups
# =========================
CPU_FEATURES = [f for f in FEATURE_NAMES if 'cpu' in f.lower()]
MEMORY_FEATURES = [f for f in FEATURE_NAMES if 'memory' in f.lower()]
CRASH_FEATURES = ['error_rate', 'error_variance_coef', 'restart_count'] 
# Note: restart_count not in dataset, so we use error_rate, error_variance_coef, request_variance_coef
CRASH_FEATURES = ['error_rate', 'error_variance_coef', 'request_variance_coef', 
                  'performance_degradation', 'system_stress']
NETWORK_FEATURES = [f for f in FEATURE_NAMES if 'network' in f.lower()]

# NEW ground-truth mapping
ROOT_CAUSE_MAPPING_V2 = {
    'cpu_spike': CPU_FEATURES,
    'memory_leak': MEMORY_FEATURES,
    'service_crash': CRASH_FEATURES,
    'normal': []  # Measured by sparsity instead
}

# Recovery action mapping (for recovery alignment metric)
RECOVERY_ACTIONS = {
    'cpu_spike': 'scale_up_cpu',
    'memory_leak': 'increase_memory',
    'service_crash': 'restart_pod',
    'normal': 'no_action'
}

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATASET_PATH = os.path.join(PROJECT_ROOT, 'datasets', 'metrics_dataset_scaleinvariant.csv')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'anomaly_detector_scaleinvariant.pkl')
OUTPUT_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), 'results', 'explanation engine')


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SHAPMetricsV2:
    """Enhanced container for SHAP evaluation metrics."""
    # Root cause alignment (anomaly-only)
    anomaly_only_top1_match_rate: float = 0.0
    anomaly_only_top3_coverage_rate: float = 0.0
    per_class_top1: Dict[str, float] = field(default_factory=dict)
    per_class_top3: Dict[str, float] = field(default_factory=dict)
    normal_sparsity_score: float = 0.0  # NEW: for "normal" class
    
    # SHAP stability
    mean_shap_stability: float = 0.0
    std_shap_stability: float = 0.0
    per_class_stability: Dict[str, float] = field(default_factory=dict)
    per_class_stability_std: Dict[str, float] = field(default_factory=dict)
    
    # Explanation time
    mean_explanation_time: float = 0.0
    std_explanation_time: float = 0.0
    mean_cache_lookup_time: float = 0.0
    
    # Counterfactual validation (ENHANCED)
    old_flip_rate: float = 0.0  # Original narrow search
    new_flip_rate: float = 0.0  # Enhanced bounded search
    old_feasible_rate: float = 0.0
    new_feasible_rate: float = 0.0
    mean_single_delta: float = 0.0
    mean_combined_delta: float = 0.0
    mean_convergence_time: float = 0.0
    
    # NEW: Recovery alignment
    recovery_alignment_rate: float = 0.0
    mean_recovery_confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'anomaly_only_top1_match_rate': self.anomaly_only_top1_match_rate,
            'anomaly_only_top3_coverage_rate': self.anomaly_only_top3_coverage_rate,
            'per_class_top1': self.per_class_top1,
            'per_class_top3': self.per_class_top3,
            'normal_sparsity_score': self.normal_sparsity_score,
            'mean_shap_stability': self.mean_shap_stability,
            'std_shap_stability': self.std_shap_stability,
            'per_class_stability': self.per_class_stability,
            'per_class_stability_std': self.per_class_stability_std,
            'mean_explanation_time': self.mean_explanation_time,
            'std_explanation_time': self.std_explanation_time,
            'mean_cache_lookup_time': self.mean_cache_lookup_time,
            'old_flip_rate': self.old_flip_rate,
            'new_flip_rate': self.new_flip_rate,
            'old_feasible_rate': self.old_feasible_rate,
            'new_feasible_rate': self.new_feasible_rate,
            'mean_single_delta': self.mean_single_delta,
            'mean_combined_delta': self.mean_combined_delta,
            'mean_convergence_time': self.mean_convergence_time,
            'recovery_alignment_rate': self.recovery_alignment_rate,
            'mean_recovery_confidence': self.mean_recovery_confidence,
        }


@dataclass
class HistoricalIncident:
    """Historical incident with recovery information."""
    feature_vector: np.ndarray
    anomaly_type: str
    recovery_action: str  # NEW
    success: bool
    threshold_used: float
    feature_modified: str


@dataclass
class CounterfactualResult:
    """Result from counterfactual search."""
    sample_idx: int
    original_pred: int
    top1_feature_idx: int
    top2_feature_idx: int
    single_flip: bool
    combined_flip: bool
    single_delta: float
    combined_delta: float
    convergence_time: float
    historical_confidence: float
    recovery_action: str
    recovery_aligned: bool


# ============================================================================
# ENHANCED SHAP EVALUATION ENGINE
# ============================================================================

class SHAPEvaluatorV2:
    """
    Enhanced SHAP explainability evaluation engine with:
    - Fixed root cause alignment
    - Strengthened counterfactual search
    - Recovery alignment metric
    """
    
    def __init__(self, model_path: str, dataset_path: str):
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
        self.metrics = SHAPMetricsV2()
        self.historical_db: List[HistoricalIncident] = []
        self.counterfactual_results: List[CounterfactualResult] = []
        
        print("="*80)
        print("SHAP COMPREHENSIVE EVALUATION SYSTEM V2")
        print("="*80)
        print("Enhanced with:")
        print("  ✓ Fixed Root Cause Alignment (feature groups, anomaly-only)")
        print("  ✓ Strengthened Counterfactual Search (top-2, combined)")
        print("  ✓ NEW Recovery Alignment Metric")
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
        
        # Encode labels
        y_cleaned = [label.strip() for label in y]
        y_encoded = self.label_encoder.fit_transform(y_cleaned)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        self.y_test_labels = self.label_encoder.inverse_transform(self.y_test)
        
        print(f"   📊 Train: {len(self.X_train)}, Test: {len(self.X_test)}")
        print(f"   📋 Classes: {list(self.label_encoder.classes_)}")
        
        # Print feature groups
        print(f"\n   🔍 Feature Groups:")
        print(f"      CPU Features:     {len(CPU_FEATURES)} features")
        print(f"      Memory Features:  {len(MEMORY_FEATURES)} features")
        print(f"      Crash Features:   {len(CRASH_FEATURES)} features")
        print(f"      Network Features: {len(NETWORK_FEATURES)} features")
        
        # Initialize SHAP explainer
        self._init_shap_explainer()
        
        # Build historical database
        self._build_historical_database()
    
    def _init_shap_explainer(self):
        """Initialize SHAP explainer."""
        print("\n🔮 Initializing SHAP explainer...")
        try:
            background = shap.sample(self.X_train, 50)
            self.shap_explainer = shap.KernelExplainer(
                lambda x: self.model.predict_proba(x), 
                background
            )
            print("   ✅ SHAP explainer ready (KernelExplainer)")
        except Exception as e:
            print(f"   ❌ Could not initialize SHAP: {e}")
            sys.exit(1)
    
    def _build_historical_database(self):
        """Build historical incident database with recovery actions."""
        print("\n📚 Building historical incident database...")
        
        num_incidents = int(len(self.X_train) * 0.2)
        indices = np.random.choice(len(self.X_train), num_incidents, replace=False)
        
        for idx in indices:
            feature_vector = self.X_train[idx]
            anomaly_type = self.label_encoder.inverse_transform([self.y_train[idx]])[0]
            
            # Assign recovery action
            recovery_action = RECOVERY_ACTIONS.get(anomaly_type, 'no_action')
            
            # Success probability
            success_prob = {
                'normal': 0.95,
                'cpu_spike': 0.80,
                'memory_leak': 0.70,
                'service_crash': 0.75
            }
            success = np.random.random() < success_prob.get(anomaly_type, 0.75)
            
            # Feature modified
            feature_idx = np.random.randint(0, len(feature_vector))
            threshold_used = feature_vector[feature_idx] * np.random.uniform(0.7, 1.3)
            
            incident = HistoricalIncident(
                feature_vector=feature_vector,
                anomaly_type=anomaly_type,
                recovery_action=recovery_action,
                success=success,
                threshold_used=threshold_used,
                feature_modified=self.feature_names[feature_idx]
            )
            
            self.historical_db.append(incident)
        
        print(f"   ✅ Built database with {len(self.historical_db)} historical incidents")
    
    # ========================================================================
    # PART 1: SHAP VALUES COMPUTATION
    # ========================================================================
    
    def compute_shap_values(self):
        """Compute SHAP values for test samples."""
        print("\n" + "="*80)
        print("PART 1: COMPUTING SHAP VALUES")
        print("="*80)
        
        print(f"\n🔮 Computing SHAP values for {len(self.X_test)} test samples...")
        
        # Use tqdm for progress bar
        self.shap_values_test = []
        start_time = time.time()
        
        for i in tqdm(range(len(self.X_test)), desc="Computing SHAP"):
            shap_val = self.shap_explainer.shap_values(self.X_test[i:i+1])
            self.shap_values_test.append(shap_val)
        
        # Convert to numpy array
        self.shap_values_test = np.array(self.shap_values_test).squeeze()
        
        computation_time = time.time() - start_time
        print(f"\n   ✅ SHAP computation complete in {computation_time:.2f} seconds")
        print(f"   ⏱️  Average per sample: {(computation_time/len(self.X_test))*1000:.2f} ms")
        
        # Store mean computation time
        self.metrics.mean_explanation_time = (computation_time/len(self.X_test))*1000
    
    # ========================================================================
    # PART 1: FIXED SHAP ROOT CAUSE ALIGNMENT
    # ========================================================================
    
    def evaluate_root_cause_alignment_v2(self):
        """
        Enhanced root cause alignment evaluation.
        
        Changes:
        - Uses feature groups (CPU, MEMORY, CRASH, NETWORK)
        - Computes anomaly-only metrics (excludes 'normal')
        - Measures sparsity for 'normal' class
        """
        print("\n" + "="*80)
        print("PART 1: SHAP ROOT CAUSE ALIGNMENT (ENHANCED)")
        print("="*80)
        
        y_pred = self.model.predict(self.X_test)
        
        # Handle multi-class SHAP values
        if len(self.shap_values_test.shape) == 3:
            # Extract SHAP for predicted class
            shap_values = np.zeros((len(self.X_test), len(self.feature_names)))
            for i, pred_class in enumerate(y_pred):
                shap_values[i] = self.shap_values_test[i, :, pred_class]
        elif len(self.shap_values_test.shape) == 2:
            shap_values = self.shap_values_test
        else:
            raise ValueError(f"Unexpected SHAP values shape: {self.shap_values_test.shape}")
        
        abs_shap = np.abs(shap_values)
        
        # Storage
        class_top1_matches = defaultdict(list)
        class_top3_matches = defaultdict(list)
        normal_shap_magnitudes = []
        
        anomaly_top1_matches = 0
        anomaly_top3_matches = 0
        anomaly_count = 0
        
        print(f"\n🔍 Analyzing {len(self.X_test)} test samples...")
        
        for i in range(len(self.X_test)):
            true_label = self.y_test_labels[i]
            
            # Get top features
            top_indices = np.argsort(abs_shap[i])[::-1]
            top1_feature = self.feature_names[top_indices[0]]
            top3_features = [self.feature_names[idx] for idx in top_indices[:3]]
            
            # Check alignment
            if true_label == 'normal':
                # Measure sparsity: mean of absolute SHAP values
                mean_shap = np.mean(abs_shap[i])
                normal_shap_magnitudes.append(mean_shap)
                # Don't include in anomaly metrics
            else:
                # Anomaly class
                expected_features = ROOT_CAUSE_MAPPING_V2.get(true_label, [])
                
                top1_match = top1_feature in expected_features
                top3_match = any(f in expected_features for f in top3_features)
                
                class_top1_matches[true_label].append(top1_match)
                class_top3_matches[true_label].append(top3_match)
                
                if top1_match:
                    anomaly_top1_matches += 1
                if top3_match:
                    anomaly_top3_matches += 1
                anomaly_count += 1
        
        # Compute anomaly-only metrics
        if anomaly_count > 0:
            self.metrics.anomaly_only_top1_match_rate = (anomaly_top1_matches / anomaly_count) * 100
            self.metrics.anomaly_only_top3_coverage_rate = (anomaly_top3_matches / anomaly_count) * 100
        
        # Normal sparsity score
        if normal_shap_magnitudes:
            self.metrics.normal_sparsity_score = np.mean(normal_shap_magnitudes)
        
        # Per-class metrics
        for class_name in self.label_encoder.classes_:
            if class_name != 'normal' and class_name in class_top1_matches:
                top1_list = class_top1_matches[class_name]
                top3_list = class_top3_matches[class_name]
                
                self.metrics.per_class_top1[class_name] = (sum(top1_list) / len(top1_list)) * 100
                self.metrics.per_class_top3[class_name] = (sum(top3_list) / len(top3_list)) * 100
        
        # Print results
        print("\n📊 ANOMALY-ONLY ROOT CAUSE ALIGNMENT:")
        print(f"   Overall Top-1 Match Rate:    {self.metrics.anomaly_only_top1_match_rate:.2f}%")
        print(f"   Overall Top-3 Coverage Rate: {self.metrics.anomaly_only_top3_coverage_rate:.2f}%")
        
        print("\n   Per-Class Results (Anomalies Only):")
        for class_name in sorted([c for c in self.label_encoder.classes_ if c != 'normal']):
            if class_name in self.metrics.per_class_top1:
                top1 = self.metrics.per_class_top1[class_name]
                top3 = self.metrics.per_class_top3[class_name]
                print(f"   {class_name:15s}: Top-1={top1:6.2f}%  Top-3={top3:6.2f}%")
        
        print(f"\n   Normal Class Sparsity Score: {self.metrics.normal_sparsity_score:.4f}")
        print(f"   (Lower = sparser explanations, as expected for normal)")
    
    # ========================================================================
    # PART 2: SHAP STABILITY (unchanged)
    # ========================================================================
    
    def evaluate_shap_stability(self):
        """Evaluate SHAP stability (same as before)."""
        print("\n" + "="*80)
        print("PART 2: SHAP STABILITY ANALYSIS")
        print("="*80)
        
        # Handle multi-output SHAP
        if len(self.shap_values_test.shape) == 3:
            shap_values = np.sum(np.abs(self.shap_values_test), axis=2)
        else:
            shap_values = self.shap_values_test
        
        # Group by class
        class_shap_vectors = defaultdict(list)
        for i, label in enumerate(self.y_test_labels):
            class_shap_vectors[label].append(shap_values[i])
        
        all_similarities = []
        
        print(f"\n🔍 Computing intra-class SHAP similarities...")
        
        for class_name, vectors in class_shap_vectors.items():
            if len(vectors) < 2:
                continue
            
            vectors_array = np.array(vectors)
            similarity_matrix = cosine_similarity(vectors_array)
            upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
            
            if len(upper_triangle) > 0:
                mean_similarity = np.mean(upper_triangle)
                std_similarity = np.std(upper_triangle)
                
                self.metrics.per_class_stability[class_name] = mean_similarity
                self.metrics.per_class_stability_std[class_name] = std_similarity
                
                all_similarities.extend(upper_triangle)
                
                print(f"   {class_name:15s}: Mean={mean_similarity:.4f} ± {std_similarity:.4f}")
        
        if all_similarities:
            self.metrics.mean_shap_stability = np.mean(all_similarities)
            self.metrics.std_shap_stability = np.std(all_similarities)
        
        print(f"\n📊 OVERALL SHAP STABILITY:")
        print(f"   Mean Cosine Similarity: {self.metrics.mean_shap_stability:.4f}")
        print(f"   Std Deviation:          {self.metrics.std_shap_stability:.4f}")
    
    # ========================================================================
    # PART 3: STRENGTHENED COUNTERFACTUAL SEARCH
    # ========================================================================
    
    def evaluate_counterfactuals_v2(self):
        """
        Enhanced counterfactual validation.
        
        Changes:
        - Test top-2 SHAP features independently
        - Test combined perturbation
        - Bounded search ±20% of feature range
        - Recovery alignment metric
        """
        print("\n" + "="*80)
        print("PART 3: COUNTERFACTUAL VALIDATION (ENHANCED)")
        print("="*80)
        
        # Handle multi-output SHAP
        if len(self.shap_values_test.shape) == 3:
            shap_values = np.sum(np.abs(self.shap_values_test), axis=2)
        else:
            shap_values = np.abs(self.shap_values_test)
        
        abs_shap = shap_values
        
        # Only evaluate anomalous samples
        anomalous_indices = [i for i, label in enumerate(self.y_test_labels) if label != 'normal']
        
        print(f"\n🔍 Evaluating {len(anomalous_indices)} anomalous samples...")
        print("   Testing: Top-1 feature | Top-2 feature | Combined perturbation")
        
        # OLD method counters (for comparison)
        old_flip_count = 0
        old_feasible_count = 0
        old_confidences = []
        
        # NEW method counters
        new_flip_count = 0
        new_feasible_count = 0
        single_deltas = []
        combined_deltas = []
        convergence_times = []
        recovery_confidences = []
        recovery_aligned_count = 0
        
        for idx in tqdm(anomalous_indices[:100], desc="Counterfactual Search"):
            # Get top 2 SHAP features
            top_indices = np.argsort(abs_shap[idx])[::-1]
            top1_idx = top_indices[0]
            top2_idx = top_indices[1]
            
            original_pred = self.model.predict(self.X_test[idx:idx+1])[0]
            true_label = self.y_test_labels[idx]
            
            # ===== OLD METHOD: Test top-1 with narrow search =====
            start_time = time.time()
            old_threshold, old_flip = self._binary_search_narrow(
                self.X_test[idx], top1_idx, original_pred
            )
            old_time = (time.time() - start_time) * 1000
            
            if old_flip:
                old_flip_count += 1
                old_conf = self._validate_historical_simple(
                    self.X_test[idx], 
                    self.feature_names[top1_idx],
                    old_threshold
                )
                old_confidences.append(old_conf)
                if old_conf >= 0.5:
                    old_feasible_count += 1
            
            # ===== NEW METHOD: Test top-1 with expanded search =====
            start_time = time.time()
            new_threshold1, new_flip1, delta1 = self._binary_search_expanded(
                self.X_test[idx], top1_idx, original_pred
            )
            
            # Test top-2 feature
            new_threshold2, new_flip2, delta2 = self._binary_search_expanded(
                self.X_test[idx], top2_idx, original_pred
            )
            
            # Test combined perturbation
            combined_flip, combined_delta = self._test_combined_perturbation(
                self.X_test[idx], 
                top1_idx, top2_idx,
                original_pred,
                delta1, delta2
            )
            
            search_time = (time.time() - start_time) * 1000
            convergence_times.append(search_time)
            
            # Check if any method succeeded
            any_flip = new_flip1 or new_flip2 or combined_flip
            
            if any_flip:
                new_flip_count += 1
                
                # Use the method with smallest delta
                if combined_flip:
                    best_delta = combined_delta
                    best_feature = f"{self.feature_names[top1_idx]}+{self.feature_names[top2_idx]}"
                elif new_flip1:
                    best_delta = delta1
                    best_feature = self.feature_names[top1_idx]
                else:
                    best_delta = delta2
                    best_feature = self.feature_names[top2_idx]
                
                single_deltas.append(min(delta1, delta2))
                if combined_flip:
                    combined_deltas.append(combined_delta)
                
                # Validate with historical database + recovery alignment
                recovery_conf, suggested_action, recovery_aligned = self._validate_recovery_alignment(
                    self.X_test[idx],
                    best_feature,
                    best_delta,
                    true_label
                )
                
                recovery_confidences.append(recovery_conf)
                if recovery_conf >= 0.5:
                    new_feasible_count += 1
                if recovery_aligned:
                    recovery_aligned_count += 1
                
                # Store result
                result = CounterfactualResult(
                    sample_idx=idx,
                    original_pred=original_pred,
                    top1_feature_idx=top1_idx,
                    top2_feature_idx=top2_idx,
                    single_flip=new_flip1 or new_flip2,
                    combined_flip=combined_flip,
                    single_delta=min(delta1, delta2),
                    combined_delta=combined_delta if combined_flip else 0.0,
                    convergence_time=search_time,
                    historical_confidence=recovery_conf,
                    recovery_action=suggested_action,
                    recovery_aligned=recovery_aligned
                )
                self.counterfactual_results.append(result)
        
        # Compute metrics
        total_evaluated = len(anomalous_indices[:100])
        
        # OLD metrics
        self.metrics.old_flip_rate = (old_flip_count / total_evaluated) * 100
        self.metrics.old_feasible_rate = (old_feasible_count / total_evaluated) * 100
        
        # NEW metrics
        self.metrics.new_flip_rate = (new_flip_count / total_evaluated) * 100
        self.metrics.new_feasible_rate = (new_feasible_count / total_evaluated) * 100
        
        if single_deltas:
            self.metrics.mean_single_delta = np.mean(single_deltas)
        if combined_deltas:
            self.metrics.mean_combined_delta = np.mean(combined_deltas)
        if convergence_times:
            self.metrics.mean_convergence_time = np.mean(convergence_times)
        
        # Recovery alignment
        if new_flip_count > 0:
            self.metrics.recovery_alignment_rate = (recovery_aligned_count / new_flip_count) * 100
        if recovery_confidences:
            self.metrics.mean_recovery_confidence = np.mean(recovery_confidences)
        
        # Print comparison
        print(f"\n📊 COUNTERFACTUAL VALIDATION COMPARISON:")
        print(f"\n   OLD METHOD (narrow search, top-1 only):")
        print(f"      Flip Rate:        {self.metrics.old_flip_rate:.2f}%")
        print(f"      Feasible Rate:    {self.metrics.old_feasible_rate:.2f}%")
        
        print(f"\n   NEW METHOD (expanded search, top-2, combined):")
        print(f"      Flip Rate:        {self.metrics.new_flip_rate:.2f}%")
        print(f"      Feasible Rate:    {self.metrics.new_feasible_rate:.2f}%")
        print(f"      Mean Single Δ:    {self.metrics.mean_single_delta:.4f}")
        print(f"      Mean Combined Δ:  {self.metrics.mean_combined_delta:.4f}")
        print(f"      Mean Conv. Time:  {self.metrics.mean_convergence_time:.2f} ms")
        
        print(f"\n   📌 RECOVERY ALIGNMENT (NEW):")
        print(f"      Alignment Rate:   {self.metrics.recovery_alignment_rate:.2f}%")
        print(f"      Mean Confidence:  {self.metrics.mean_recovery_confidence:.4f}")
        
        print(f"\n   🚀 IMPROVEMENT:")
        print(f"      Flip Rate:        +{self.metrics.new_flip_rate - self.metrics.old_flip_rate:.2f}%")
        print(f"      Feasible Rate:    +{self.metrics.new_feasible_rate - self.metrics.old_feasible_rate:.2f}%")
    
    def _binary_search_narrow(self, sample: np.ndarray, feature_idx: int, 
                              original_pred: int, max_iter: int = 20) -> Tuple[float, bool]:
        """Original narrow binary search (0.5x to 1.5x)."""
        sample_copy = sample.copy()
        original_value = sample[feature_idx]
        
        low = original_value * 0.5
        high = original_value * 1.5
        
        if abs(original_value) < 1e-6:
            low = 0.0
            high = 1.0
        
        best_threshold = high
        did_flip = False
        
        for _ in range(max_iter):
            mid = (low + high) / 2.0
            sample_copy[feature_idx] = mid
            pred = self.model.predict(sample_copy.reshape(1, -1))[0]
            
            if pred != original_pred:
                did_flip = True
                best_threshold = mid
                high = mid
            else:
                low = mid
            
            if abs(high - low) < 1e-4:
                break
        
        return best_threshold, did_flip
    
    def _binary_search_expanded(self, sample: np.ndarray, feature_idx: int, 
                                original_pred: int, max_iter: int = 25) -> Tuple[float, bool, float]:
        """Expanded binary search (±20% of feature range)."""
        sample_copy = sample.copy()
        original_value = sample[feature_idx]
        
        # Get feature range from training data
        feature_values = self.X_train[:, feature_idx]
        feature_min = np.min(feature_values)
        feature_max = np.max(feature_values)
        feature_range = feature_max - feature_min
        
        # Expand search to ±20% of range
        low = max(feature_min, original_value - 0.2 * feature_range)
        high = min(feature_max, original_value + 0.2 * feature_range)
        
        best_threshold = high
        did_flip = False
        best_delta = float('inf')
        
        for _ in range(max_iter):
            mid = (low + high) / 2.0
            sample_copy[feature_idx] = mid
            pred = self.model.predict(sample_copy.reshape(1, -1))[0]
            
            if pred != original_pred:
                did_flip = True
                best_threshold = mid
                best_delta = min(best_delta, abs(mid - original_value))
                high = mid
            else:
                low = mid
            
            if abs(high - low) < 1e-4:
                break
        
        if not did_flip:
            best_delta = 0.0
        
        return best_threshold, did_flip, best_delta
    
    def _test_combined_perturbation(self, sample: np.ndarray, 
                                    idx1: int, idx2: int, 
                                    original_pred: int,
                                    delta1: float, delta2: float) -> Tuple[bool, float]:
        """Test combined perturbation of both features."""
        if delta1 == 0.0 or delta2 == 0.0:
            return False, 0.0
        
        sample_copy = sample.copy()
        
        # Try half the individual deltas combined
        sample_copy[idx1] += delta1 * 0.5
        sample_copy[idx2] += delta2 * 0.5
        
        pred = self.model.predict(sample_copy.reshape(1, -1))[0]
        
        if pred != original_pred:
            combined_delta = np.sqrt((delta1 * 0.5)**2 + (delta2 * 0.5)**2)
            return True, combined_delta
        
        return False, 0.0
    
    def _validate_historical_simple(self, sample: np.ndarray, feature_name: str, 
                                    threshold: float, threshold_sim: float = 0.8) -> float:
        """Simple historical validation (for old method comparison)."""
        similar_incidents = []
        
        for incident in self.historical_db:
            similarity = cosine_similarity(
                sample.reshape(1, -1), 
                incident.feature_vector.reshape(1, -1)
            )[0][0]
            
            if similarity > threshold_sim and incident.feature_modified == feature_name:
                similar_incidents.append(incident)
        
        if similar_incidents:
            successes = sum(1 for inc in similar_incidents if inc.success)
            total = len(similar_incidents)
        else:
            successes = 0
            total = 0
        
        return (successes + 1) / (total + 2)
    
    def _validate_recovery_alignment(self, sample: np.ndarray, feature_name: str, 
                                     delta: float, true_label: str,
                                     threshold_sim: float = 0.8) -> Tuple[float, str, bool]:
        """
        NEW: Validate with recovery alignment.
        
        Returns:
            (confidence, suggested_recovery_action, is_aligned)
        """
        # Find similar historical incidents
        similar_incidents = []
        
        for incident in self.historical_db:
            similarity = cosine_similarity(
                sample.reshape(1, -1), 
                incident.feature_vector.reshape(1, -1)
            )[0][0]
            
            if similarity > threshold_sim:
                similar_incidents.append(incident)
        
        # Extract most common recovery action
        if similar_incidents:
            recovery_actions = [inc.recovery_action for inc in similar_incidents if inc.success]
            if recovery_actions:
                most_common_action = Counter(recovery_actions).most_common(1)[0][0]
            else:
                most_common_action = 'no_action'
            
            # Compute confidence
            successes = sum(1 for inc in similar_incidents if inc.success)
            total = len(similar_incidents)
            confidence = (successes + 1) / (total + 2)
        else:
            most_common_action = 'no_action'
            confidence = (0 + 1) / (0 + 2)
        
        # Check alignment
        expected_action = RECOVERY_ACTIONS.get(true_label, 'no_action')
        is_aligned = (most_common_action == expected_action)
        
        return confidence, most_common_action, is_aligned
    
    # ========================================================================
    # VISUALIZATION AND REPORTING
    # ========================================================================
    
    def generate_visualizations_v2(self):
        """Generate enhanced visualizations."""
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Handle multi-output SHAP
        if len(self.shap_values_test.shape) == 3:
            shap_values = np.sum(np.abs(self.shap_values_test), axis=2)
        else:
            shap_values = np.abs(self.shap_values_test)
        
        # --- Plot 1: Top 10 SHAP Features ---
        print("\n📊 Generating Top-10 SHAP features plot...")
        
        mean_abs_shap = np.mean(shap_values, axis=0)
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
        
        if self.counterfactual_results:
            confidences = [r.historical_confidence for r in self.counterfactual_results]
            
            plt.figure(figsize=(10, 6))
            plt.hist(confidences, bins=20, color='coral', edgecolor='black', alpha=0.7)
            plt.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Feasibility Threshold')
            plt.xlabel('Counterfactual Confidence', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.title('Distribution of Counterfactual Confidence Scores\n(Enhanced Method)', 
                     fontsize=14, fontweight='bold')
            plt.legend(fontsize=11)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            plot2_path = os.path.join(OUTPUT_DIR, 'counterfactual_confidence_distribution.png')
            plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
            print(f"   ✅ Saved: {plot2_path}")
            plt.close()
        
        # --- Plot 3: OLD vs NEW Comparison ---
        print("\n📊 Generating OLD vs NEW comparison plot...")
        
        metrics_labels = ['Flip Rate', 'Feasible Rate']
        old_values = [self.metrics.old_flip_rate, self.metrics.old_feasible_rate]
        new_values = [self.metrics.new_flip_rate, self.metrics.new_feasible_rate]
        
        x = np.arange(len(metrics_labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, old_values, width, label='OLD (narrow, top-1)', 
                       color='lightcoral', alpha=0.8)
        bars2 = ax.bar(x + width/2, new_values, width, label='NEW (expanded, top-2, combined)', 
                       color='mediumseagreen', alpha=0.8)
        
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_title('Counterfactual Method Comparison\n(OLD vs NEW)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_labels)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plot3_path = os.path.join(OUTPUT_DIR, 'counterfactual_comparison.png')
        plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {plot3_path}")
        plt.close()
    
    def print_summary_v2(self):
        """Print enhanced summary."""
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION SUMMARY V2")
        print("="*80)
        print()
        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║           SHAP EXPLAINABILITY METRICS (ENHANCED)                     ║")
        print("╚══════════════════════════════════════════════════════════════════════╝")
        print()
        print(f"  📌 SHAP ROOT CAUSE ALIGNMENT (Anomaly-Only)")
        print(f"     ANOMALY-ONLY Top-1 Match:       {self.metrics.anomaly_only_top1_match_rate:6.2f}%")
        print(f"     ANOMALY-ONLY Top-3 Coverage:    {self.metrics.anomaly_only_top3_coverage_rate:6.2f}%")
        print(f"     Normal Sparsity Score:          {self.metrics.normal_sparsity_score:6.4f}")
        print()
        print(f"  📌 SHAP STABILITY")
        print(f"     Mean SHAP Stability:            {self.metrics.mean_shap_stability:.4f}")
        print(f"     Std Deviation:                  {self.metrics.std_shap_stability:.4f}")
        print()
        print(f"  📌 EXPLANATION TIME")
        print(f"     Mean SHAP Computation:          {self.metrics.mean_explanation_time:6.2f} ms")
        print(f"     Mean Cache Lookup:              {self.metrics.mean_cache_lookup_time:6.2f} ms")
        print()
        print(f"  📌 COUNTERFACTUAL VALIDATION (Enhanced)")
        print(f"     OLD Flip Rate:                  {self.metrics.old_flip_rate:6.2f}%")
        print(f"     NEW Flip Rate:                  {self.metrics.new_flip_rate:6.2f}%  [+{self.metrics.new_flip_rate - self.metrics.old_flip_rate:.2f}%]")
        print(f"     OLD Feasible Rate:              {self.metrics.old_feasible_rate:6.2f}%")
        print(f"     NEW Feasible Rate:              {self.metrics.new_feasible_rate:6.2f}%  [+{self.metrics.new_feasible_rate - self.metrics.old_feasible_rate:.2f}%]")
        print(f"     Mean Single Delta:              {self.metrics.mean_single_delta:6.4f}")
        print(f"     Mean Combined Delta:            {self.metrics.mean_combined_delta:6.4f}")
        print(f"     Mean Convergence Time:          {self.metrics.mean_convergence_time:6.2f} ms")
        print()
        print(f"  📌 RECOVERY ALIGNMENT (NEW)")
        print(f"     Recovery Alignment Rate:        {self.metrics.recovery_alignment_rate:6.2f}%")
        print(f"     Mean Recovery Confidence:       {self.metrics.mean_recovery_confidence:6.4f}")
        print()
        print("="*80)
        print("\n✨ KEY IMPROVEMENTS:")
        print(f"   • Flip Rate improved by:     +{self.metrics.new_flip_rate - self.metrics.old_flip_rate:.2f}%")
        print(f"   • Feasible Rate improved by: +{self.metrics.new_feasible_rate - self.metrics.old_feasible_rate:.2f}%")
        print(f"   • Recovery alignment:        {self.metrics.recovery_alignment_rate:.2f}% match with historical actions")
        print("="*80)
    
    def save_results_v2(self):
        """Save enhanced results."""
        print("\n💾 Saving results...")
        
        # JSON
        import json
        json_path = os.path.join(OUTPUT_DIR, 'shap_evaluation_metrics_v2.json')
        with open(json_path, 'w') as f:
            json.dump(self.metrics.to_dict(), f, indent=4)
        print(f"   ✅ Saved metrics: {json_path}")
        
        # CSV
        csv_path = os.path.join(OUTPUT_DIR, 'shap_evaluation_detailed_v2.csv')
        detailed_data = {
            'Metric': [
                'Anomaly-Only Top-1 Match Rate (%)',
                'Anomaly-Only Top-3 Coverage Rate (%)',
                'Normal Sparsity Score',
                'Mean SHAP Stability',
                'Std SHAP Stability',
                'Mean Explanation Time (ms)',
                'Mean Cache Lookup Time (ms)',
                'OLD Flip Rate (%)',
                'NEW Flip Rate (%)',
                'OLD Feasible Rate (%)',
                'NEW Feasible Rate (%)',
                'Mean Single Delta',
                'Mean Combined Delta',
                'Mean Convergence Time (ms)',
                'Recovery Alignment Rate (%)',
                'Mean Recovery Confidence'
            ],
            'Value': [
                f"{self.metrics.anomaly_only_top1_match_rate:.2f}",
                f"{self.metrics.anomaly_only_top3_coverage_rate:.2f}",
                f"{self.metrics.normal_sparsity_score:.4f}",
                f"{self.metrics.mean_shap_stability:.4f}",
                f"{self.metrics.std_shap_stability:.4f}",
                f"{self.metrics.mean_explanation_time:.2f}",
                f"{self.metrics.mean_cache_lookup_time:.2f}",
                f"{self.metrics.old_flip_rate:.2f}",
                f"{self.metrics.new_flip_rate:.2f}",
                f"{self.metrics.old_feasible_rate:.2f}",
                f"{self.metrics.new_feasible_rate:.2f}",
                f"{self.metrics.mean_single_delta:.4f}",
                f"{self.metrics.mean_combined_delta:.4f}",
                f"{self.metrics.mean_convergence_time:.2f}",
                f"{self.metrics.recovery_alignment_rate:.2f}",
                f"{self.metrics.mean_recovery_confidence:.4f}"
            ]
        }
        df = pd.DataFrame(detailed_data)
        df.to_csv(csv_path, index=False)
        print(f"   ✅ Saved detailed results: {csv_path}")
        
        # Save counterfactual details
        if self.counterfactual_results:
            cf_csv_path = os.path.join(OUTPUT_DIR, 'counterfactual_details_v2.csv')
            cf_data = {
                'sample_idx': [r.sample_idx for r in self.counterfactual_results],
                'single_flip': [r.single_flip for r in self.counterfactual_results],
                'combined_flip': [r.combined_flip for r in self.counterfactual_results],
                'single_delta': [r.single_delta for r in self.counterfactual_results],
                'combined_delta': [r.combined_delta for r in self.counterfactual_results],
                'convergence_time_ms': [r.convergence_time for r in self.counterfactual_results],
                'confidence': [r.historical_confidence for r in self.counterfactual_results],
                'recovery_action': [r.recovery_action for r in self.counterfactual_results],
                'recovery_aligned': [r.recovery_aligned for r in self.counterfactual_results]
            }
            df_cf = pd.DataFrame(cf_data)
            df_cf.to_csv(cf_csv_path, index=False)
            print(f"   ✅ Saved counterfactual details: {cf_csv_path}")
    
    def run_full_evaluation(self):
        """Run all evaluation steps."""
        print("\n🚀 Starting comprehensive SHAP evaluation V2...\n")
        
        # Step 1: Load data
        self.load_model_and_data()
        
        # Step 2: Compute SHAP values
        self.compute_shap_values()
        
        # Step 3: Evaluate root cause alignment (enhanced)
        self.evaluate_root_cause_alignment_v2()
        
        # Step 4: Evaluate SHAP stability
        self.evaluate_shap_stability()
        
        # Step 5: Evaluate counterfactuals (enhanced)
        self.evaluate_counterfactuals_v2()
        
        # Step 6: Generate visualizations
        self.generate_visualizations_v2()
        
        # Step 7: Print summary
        self.print_summary_v2()
        
        # Step 8: Save results
        self.save_results_v2()
        
        print("\n✅ Evaluation V2 complete!")
        print(f"📁 Results saved to: {OUTPUT_DIR}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    evaluator = SHAPEvaluatorV2(
        model_path=MODEL_PATH,
        dataset_path=DATASET_PATH
    )
    
    evaluator.run_full_evaluation()


if __name__ == "__main__":
    main()
