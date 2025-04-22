import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
from tensorflow.keras import backend as K

# --- Configuration ---
SAVE_DIR = 'sepsis_model_files'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'sepsis_model.keras')
IMPUTER_SAVE_PATH = os.path.join(SAVE_DIR, 'imputer.joblib')
SCALER_SAVE_PATH = os.path.join(SAVE_DIR, 'scaler.joblib')
FEATURES_SAVE_PATH = os.path.join(SAVE_DIR, 'feature_list.joblib')
THRESHOLD = 0.15  # Lowered further to improve recall

# --- Define Focal Loss for Model Loading ---
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        fl = -alpha * tf.pow(1.0 - pt, gamma) * tf.math.log(pt)
        fl = tf.where(tf.equal(y_true, 1), fl, (1 - alpha) * tf.pow(pt, gamma) * tf.math.log(1 - pt))
        return tf.reduce_mean(fl)
    return focal_loss_fn

# --- Load Prediction Assets ---
def load_prediction_assets(model_path, imputer_path, scaler_path, features_path):
    try:
        model = tf.keras.models.load_model(model_path, custom_objects={'focal_loss_fn': focal_loss(gamma=2.0, alpha=0.25)})
        imputer = joblib.load(imputer_path)
        scaler = joblib.load(scaler_path)
        features = joblib.load(features_path)
        print("Model, imputer, scaler, and feature list loaded successfully.")
        return model, imputer, scaler, features
    except Exception as e:
        print(f"Error loading assets: {e}")
        return None, None, None, None

model, imputer, scaler, feature_list = load_prediction_assets(
    MODEL_SAVE_PATH, IMPUTER_SAVE_PATH, SCALER_SAVE_PATH, FEATURES_SAVE_PATH
)

# --- Prediction Function ---
def predict_sepsis(new_data, model, imputer, scaler, feature_list):
    try:
        input_df = pd.DataFrame(new_data, columns=feature_list)
        for col in feature_list:
            if col not in input_df.columns:
                input_df[col] = np.nan
        input_df = input_df[feature_list]
        input_imputed = imputer.transform(input_df)
        input_scaled = scaler.transform(input_imputed)
        probabilities = model.predict(input_scaled, verbose=0).flatten()
        classes = (probabilities > THRESHOLD).astype(int)
        return probabilities, classes
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None

# --- Generate 10 Sample Cases ---
# Median values for unspecified features (approximated based on clinical data)
median_values = {
    'HR': 80, 'Temp': 36.8, 'Resp': 18, 'WBC': 8.0, 'Lactate': 1.5, 'BUN': 20,
    'MAP': 80, 'Creatinine': 1.0, 'Glucose': 120, 'Potassium': 4.0, 'Age': 60,
    'ICULOS': 10, 'Gender': 0, 'HospAdmTime': -0.05, 'Platelets': 200
}

# 5 Sepsis-like samples (more extreme features)
sepsis_samples = [
    {**median_values, 'HR': 135, 'Temp': 39.5, 'Resp': 34, 'WBC': 18.5, 'Lactate': 4.5, 'BUN': 50, 'MAP': 48, 'Creatinine': 2.5, 'ICULOS': 25},
    {**median_values, 'HR': 130, 'Temp': 39.2, 'Resp': 32, 'WBC': 17.0, 'Lactate': 4.0, 'BUN': 45, 'MAP': 50, 'Glucose': 200, 'Age': 75},
    {**median_values, 'HR': 125, 'Temp': 39.0, 'Resp': 30, 'WBC': 19.0, 'Lactate': 3.8, 'BUN': 40, 'MAP': 52, 'Potassium': 5.0, 'Creatinine': 2.2},
    {**median_values, 'HR': 140, 'Temp': 39.3, 'Resp': 35, 'WBC': 20.0, 'Lactate': 4.2, 'BUN': 48, 'MAP': 47, 'ICULOS': 30, 'Platelets': 150},
    {**median_values, 'HR': 128, 'Temp': 39.1, 'Resp': 33, 'WBC': 17.5, 'Lactate': 3.9, 'BUN': 42, 'MAP': 49, 'Age': 65, 'Glucose': 180}
]

# 5 Non-sepsis samples (normal ranges)
non_sepsis_samples = [
    {**median_values, 'HR': 68, 'Temp': 36.4, 'Resp': 14, 'WBC': 6.5, 'Lactate': 0.8, 'BUN': 10, 'MAP': 92, 'Creatinine': 0.7, 'ICULOS': 5},
    {**median_values, 'HR': 70, 'Temp': 36.5, 'Resp': 15, 'WBC': 7.0, 'Lactate': 1.0, 'BUN': 12, 'MAP': 90, 'Glucose': 100, 'Age': 50},
    {**median_values, 'HR': 72, 'Temp': 36.6, 'Resp': 16, 'WBC': 7.5, 'Lactate': 1.1, 'BUN': 14, 'MAP': 88, 'Potassium': 3.8, 'Creatinine': 0.8},
    {**median_values, 'HR': 75, 'Temp': 36.7, 'Resp': 17, 'WBC': 8.0, 'Lactate': 1.2, 'BUN': 15, 'MAP': 85, 'Platelets': 220, 'ICULOS': 7},
    {**median_values, 'HR': 78, 'Temp': 36.8, 'Resp': 18, 'WBC': 8.5, 'Lactate': 1.3, 'BUN': 16, 'MAP': 87, 'Age': 55, 'Glucose': 110}
]

# Combine samples
samples = sepsis_samples + non_sepsis_samples

# --- Make Predictions ---
if model and imputer and scaler and feature_list:
    print("\nMaking predictions on 10 samples (5 sepsis, 5 non-sepsis)...")
    probs, classes = predict_sepsis(samples, model, imputer, scaler, feature_list)
    
    if probs is not None:
        print("\nPrediction Results:")
        misclassified = []
        for i, (prob, cls) in enumerate(zip(probs, classes)):
            label = 'Sepsis' if cls == 1 else 'No Sepsis'
            sample_type = 'Sepsis-like' if i < 5 else 'Non-sepsis'
            print(f"Sample {i+1} ({sample_type}): Probability={prob:.4f}, Predicted={label}")
            # Check for misclassified sepsis-like samples
            if i < 5 and cls == 0:
                misclassified.append((i+1, samples[i]))
        
        # Print misclassified samples' feature values
        if misclassified:
            print("\nMisclassified Sepsis-like Samples:")
            for sample_num, sample_data in misclassified:
                print(f"\nSample {sample_num} Features:")
                for key, value in sample_data.items():
                    print(f"  {key}: {value}")
        else:
            print("\nMisclassified Samples: None")
else:
    print("Cannot make predictions due to missing assets.")

print("\n--- END OF SCRIPT ---")