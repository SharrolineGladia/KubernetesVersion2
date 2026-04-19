"""Debug script to check model predictions."""
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load model
print("Loading model...")
with open(PROJECT_ROOT / 'ml_detector' / 'models' / 'anomaly_detector_scaleinvariant.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
label_encoder = model_data['label_encoder']
feature_columns = model_data['feature_columns']

print(f"Model type: {type(model)}")
print(f"Feature columns ({len(feature_columns)}): {feature_columns[:5]}...")
print(f"Label encoder classes: {label_encoder.classes_}")

# Load dataset
print("\nLoading dataset...")
df = pd.read_csv(PROJECT_ROOT / 'ml_detector' / 'datasets' / 'metrics_dataset_scaleinvariant.csv')
# DON'T strip - the model was trained with leading spaces!

print(f"Dataset shape: {df.shape}")
print(f"Dataset columns: {df.columns.tolist()}")
print(f"Unique anomaly types: {df['anomaly_type'].unique()}")

# Extract features using model's feature columns
print(f"\nExtracting features using model's feature_columns...")
X = df[feature_columns].values
y_true_labels = df['anomaly_type'].values

# Encode true labels
y_true_encoded = label_encoder.transform(y_true_labels)

# Predict
print("Making predictions...")
y_pred_encoded = model.predict(X)

# Decode predictions
y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)

# Calculate accuracy
accuracy = accuracy_score(y_true_labels, y_pred_labels)
print(f"\n✅ Model Accuracy: {accuracy:.2%}")

# Show classification report
print("\nClassification Report:")
print(classification_report(y_true_labels, y_pred_labels))

# Now test on a small sample to see what we're getting in data_generator
print("\n" + "="*80)
print("TESTING SAMPLE (like data_generator does)")
print("="*80)

sample_df = df[df['anomaly_type'] == ' normal'].sample(n=10, random_state=42)
X_sample = sample_df[feature_columns].values
y_true_sample = sample_df['anomaly_type'].values

y_pred_sample = model.predict(X_sample)
y_pred_labels_sample = label_encoder.inverse_transform(y_pred_sample)

probabilities = model.predict_proba(X_sample)

print("\nSample predictions:")
for i in range(5):
    print(f"  True: {y_true_sample[i]:15s} | Pred: {y_pred_labels_sample[i]:15s} | Conf: {probabilities[i].max():.3f}")

print("\n" + "="*80)
print("If this shows 99% accuracy, then the issue is in data_generator.py")
print("="*80)
