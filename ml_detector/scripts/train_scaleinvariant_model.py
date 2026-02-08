"""
Train XGBoost Model on Scale-Invariant Features

This model works across 1, 2, 3, or N services without retraining
because features are topology-agnostic (ratios, percentages, coefficients).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import pickle
import time


class ScaleInvariantAnomalyDetector:
    """XGBoost detector trained on scale-invariant features."""
    
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_columns = [
            'cpu_utilization_mean', 'cpu_utilization_max', 'cpu_variance_coef', 'cpu_imbalance',
            'memory_pressure_mean', 'memory_pressure_max', 'memory_variance_coef', 'memory_imbalance',
            'network_in_rate', 'network_out_rate', 'network_in_variance_coef', 'network_out_variance_coef',
            'network_asymmetry', 'disk_io_rate', 'disk_io_variance_coef',
            'request_rate', 'request_variance_coef', 'error_rate', 'error_variance_coef',
            'latency_mean', 'latency_p95', 'latency_variance_coef',
            'system_stress', 'resource_efficiency', 'service_density',
            'cpu_memory_correlation', 'performance_degradation'
        ]
    
    def train(self, csv_path: str):
        """
        Train the model on scale-invariant features.
        
        Args:
            csv_path: Path to transformed dataset
        """
        print("="*80)
        print("TRAINING SCALE-INVARIANT ANOMALY DETECTOR")
        print("="*80)
        print()
        
        # Load data
        print(f"📊 Loading dataset: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"   Shape: {df.shape}")
        print(f"   Samples: {len(df)}")
        
        # Separate features and labels
        X = df[self.feature_columns]
        y = df['anomaly_type']
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"\n📈 Class distribution:")
        for class_name, count in y.value_counts().items():
            percentage = (count / len(y)) * 100
            print(f"   {class_name:20s}: {count:5d} ({percentage:5.1f}%)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"\n🔀 Train/Test split:")
        print(f"   Training: {len(X_train)} samples")
        print(f"   Testing:  {len(X_test)} samples")
        
        # Train model
        print(f"\n🤖 Training XGBoost classifier...")
        start_time = time.time()
        
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss'
        )
        
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f"   ✅ Training completed in {training_time:.2f} seconds")
        
        # Evaluate
        print(f"\n📊 Evaluation on test set:")
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Per-class metrics
        print(f"\n📋 Per-class performance:")
        report = classification_report(
            y_test, y_pred, 
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        print(f"   {'Class':<20} {'Precision':<12} {'Recall':<10} {'F1-Score':<10} {'Support'}")
        print(f"   {'-'*72}")
        for class_name in self.label_encoder.classes_:
            metrics = report[class_name]
            print(f"   {class_name:<20} {metrics['precision']:<12.2%} "
                  f"{metrics['recall']:<10.2%} {metrics['f1-score']:<10.2%} "
                  f"{int(metrics['support'])}")
        
        # Feature importance
        print(f"\n🔍 Top 10 Most Important Features:")
        feature_importance = sorted(
            zip(self.feature_columns, self.model.feature_importances_),
            key=lambda x: x[1],
            reverse=True
        )
        for i, (feature, importance) in enumerate(feature_importance[:10], 1):
            print(f"   {i:2d}. {feature:30s} {importance:.4f} ({importance*100:.2f}%)")
        
        return accuracy
    
    def save_model(self, filepath: str):
        """Save trained model to disk."""
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"\n💾 Model saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.feature_columns = model_data['feature_columns']
        
        print(f"📦 Model loaded from: {filepath}")
        print(f"   Classes: {self.label_encoder.classes_}")
        print(f"   Features: {len(self.feature_columns)}")
    
    def predict(self, features: dict):
        """
        Predict anomaly for given features.
        
        Args:
            features: Dictionary of scale-invariant features
        
        Returns:
            tuple: (anomaly_type, confidence)
        """
        # Convert to dataframe
        X = pd.DataFrame([features])[self.feature_columns]
        
        # Predict
        y_pred = self.model.predict(X)[0]
        y_proba = self.model.predict_proba(X)[0]
        
        # Decode
        anomaly_type = self.label_encoder.inverse_transform([y_pred])[0]
        confidence = y_proba[y_pred]
        
        return anomaly_type, confidence


def main():
    """Train and save the scale-invariant model."""
    detector = ScaleInvariantAnomalyDetector()
    
    # Train
    dataset_path = 'metrics_dataset_scaleinvariant.csv'
    accuracy = detector.train(dataset_path)
    
    # Save
    model_path = 'anomaly_detector_scaleinvariant.pkl'
    detector.save_model(model_path)
    
    print(f"\n{'='*80}")
    print("✅ TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"\nModel: {model_path}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Features: {len(detector.feature_columns)} scale-invariant features")
    print(f"\n🚀 This model works with ANY number of services!")
    print(f"   • Edge node with 1 service: ✅")
    print(f"   • Edge cluster with 2-3 services: ✅")
    print(f"   • Cloud datacenter with 10+ services: ✅")


if __name__ == '__main__':
    main()
