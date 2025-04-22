import streamlit as st
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
THRESHOLD = 0.15  # Threshold for classification (aligned with training)

# --- Define Focal Loss for Model Loading ---
def focal_loss(gamma=2.5, alpha=0.5):
    def focal_loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        fl = -alpha * tf.pow(1.0 - pt, gamma) * tf.math.log(pt)
        fl = tf.where(tf.equal(y_true, 1), fl, (1 - alpha) * tf.pow(pt, gamma) * tf.math.log(1 - pt))
        return tf.reduce_mean(fl)
    return focal_loss_fn

# --- Load Prediction Assets ---
@st.cache_resource
def load_prediction_assets(model_path, imputer_path, scaler_path, features_path):
    try:
        model = tf.keras.models.load_model(model_path, custom_objects={'focal_loss_fn': focal_loss(gamma=2.5, alpha=0.5)})
        imputer = joblib.load(imputer_path)
        scaler = joblib.load(scaler_path)
        features = joblib.load(features_path)
        st.success("Model, imputer, scaler, and feature list loaded successfully.")
        return model, imputer, scaler, features
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None, None, None

# --- Prediction Function ---
def predict_sepsis(new_data, model, imputer, scaler, feature_list):
    try:
        input_df = pd.DataFrame([new_data], columns=feature_list)
        for col in feature_list:
            if col not in input_df.columns:
                input_df[col] = np.nan
        input_df = input_df[feature_list]
        input_imputed = imputer.transform(input_df)
        input_scaled = scaler.transform(input_imputed)
        probability = model.predict(input_scaled, verbose=0).flatten()[0]
        class_pred = 1 if probability > THRESHOLD else 0
        return probability, class_pred
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

# --- Streamlit App ---
st.title("Sepsis Prediction App")
st.write("Enter patient data to predict the likelihood of sepsis. All fields are optional; missing values will be imputed.")

# Load assets
model, imputer, scaler, feature_list = load_prediction_assets(
    MODEL_SAVE_PATH, IMPUTER_SAVE_PATH, SCALER_SAVE_PATH, FEATURES_SAVE_PATH
)

if model is None:
    st.stop()

# Default values (approximated medians from clinical data)
default_values = {
    'HR': 80.0, 'Temp': 36.8, 'Resp': 18.0, 'WBC': 8.0, 'Lactate': 1.5,
    'BUN': 20.0, 'MAP': 80.0, 'Creatinine': 1.0, 'Glucose': 120.0,
    'Potassium': 4.0, 'Age': 60.0, 'ICULOS': 10.0, 'Gender': 0.0,
    'HospAdmTime': -0.05, 'Platelets': 200.0
}

# Input form
st.subheader("Patient Data Input")
input_data = {}
with st.form(key="patient_form"):
    col1, col2 = st.columns(2)
    
    # Input fields for core features
    with col1:
        for feature in ['HR', 'Temp', 'Resp', 'WBC', 'Lactate', 'BUN', 'MAP']:
            if feature in feature_list:
                input_data[feature] = st.number_input(
                    feature, min_value=0.0, max_value=1000.0, value=default_values.get(feature, 0.0), step=0.1
                )
    
    with col2:
        for feature in ['Creatinine', 'Glucose', 'Potassium', 'Age', 'ICULOS']:
            if feature in feature_list:
                input_data[feature] = st.number_input(
                    feature, min_value=0.0, max_value=1000.0, value=default_values.get(feature, 0.0), step=0.1
                )
    
    # Input fields for categorical/binary features
    if 'Gender' in feature_list:
        input_data['Gender'] = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
    
    # Input fields for other features
    for feature in ['HospAdmTime', 'Platelets']:
        if feature in feature_list:
            input_data[feature] = st.number_input(
                feature, min_value=-1000.0, max_value=1000.0, value=default_values.get(feature, 0.0), step=0.01
            )
    
    # Interaction features (computed automatically)
    if 'Lactate_WBC' in feature_list and 'Lactate' in input_data and 'WBC' in input_data:
        input_data['Lactate_WBC'] = input_data['Lactate'] / (input_data['WBC'] + 1e-6)
    if 'HR_MAP' in feature_list and 'HR' in input_data and 'MAP' in input_data:
        input_data['HR_MAP'] = input_data['HR'] / (input_data['MAP'] + 1e-6)
    
    submit_button = st.form_submit_button(label="Predict Sepsis")

# Prediction
if submit_button:
    st.subheader("Prediction Results")
    probability, class_pred = predict_sepsis(input_data, model, imputer, scaler, feature_list)
    
    if probability is not None:
        label = "Sepsis" if class_pred == 1 else "No Sepsis"
        st.write(f"**Predicted Probability of Sepsis**: {probability:.4f}")
        st.write(f"**Prediction**: {label}")
        
        if class_pred == 1:
            st.warning("Warning: This patient is predicted to have sepsis. Immediate clinical evaluation is recommended.")
        else:
            st.success("This patient is not predicted to have sepsis.")
        
        # Display input values for reference
        with st.expander("View Input Values"):
            st.write("Input Features Used:")
            for key, value in input_data.items():
                st.write(f"{key}: {value}")
    else:
        st.error("Unable to make a prediction due to an error.")

# Instructions
st.subheader("Instructions")
st.write("""
1. Enter patient data in the fields above. All fields are optional; missing values will be imputed automatically.
2. Click 'Predict Sepsis' to get the prediction.
3. The model uses a threshold of 0.15 to classify sepsis (probability > 0.15 = Sepsis).
4. Ensure the model and preprocessing files are in the `sepsis_model_files` directory.
""")