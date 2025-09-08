
  # app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.metrics import MeanAbsoluteError

# Load models
xgb_model = joblib.load("/content/bodyfat_model_xgb.pkl")
rf_model = joblib.load("/content/bodyfat_model_rf.pkl")
tf_model = tf.keras.models.load_model("/content/bodyfat_model_tf.h5", custom_objects={'mae': MeanAbsoluteError()})


st.title("🏋️ Body Fat Estimator App")
st.write("Enter your body measurements below and select a model to estimate your body fat percentage.")

# Input fields (adjust based on your dataset)
age = st.number_input("Age", min_value=10, max_value=100, value=25)
weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
height = st.number_input("Height (cm)", min_value=120.0, max_value=220.0, value=170.0)
neck = st.number_input("Neck (cm)", min_value=20.0, max_value=60.0, value=35.0)
chest = st.number_input("Chest (cm)", min_value=60.0, max_value=150.0, value=95.0)
abdomen = st.number_input("Abdomen (cm)", min_value=60.0, max_value=150.0, value=85.0)
hip = st.number_input("Hip (cm)", min_value=70.0, max_value=160.0, value=95.0)
thigh = st.number_input("Thigh (cm)", min_value=30.0, max_value=90.0, value=55.0)
knee = st.number_input("Knee (cm)", min_value=20.0, max_value=60.0, value=38.0)
ankle = st.number_input("Ankle (cm)", min_value=15.0, max_value=40.0, value=22.0)
bicep = st.number_input("Bicep (cm)", min_value=20.0, max_value=60.0, value=32.0)
forearm = st.number_input("Forearm (cm)", min_value=15.0, max_value=50.0, value=28.0)
wrist = st.number_input("Wrist (cm)", min_value=10.0, max_value=30.0, value=18.0)

# Prepare input
features = np.array([[age, weight, height, neck, chest, abdomen, hip, thigh, knee, ankle, bicep, forearm, wrist]])

# Model selection
model_choice = st.selectbox("Choose a model", ["XGBoost", "Random Forest", "TensorFlow (Neural Net)"])

# Load the appropriate scaler based on the model choice
scaler_file = ""
if model_choice == "XGBoost":
    scaler_file = "scaler_xgb.pkl"
elif model_choice == "Random Forest":
    scaler_file = "scaler_rf.pkl"
else:  # TensorFlow
    scaler_file = "scaler_tf.pkl" # Load the scaler saved in the TF training cell


try:
    scaler = joblib.load(scaler_file)
except FileNotFoundError:
    st.error(f"Scaler file not found: {scaler_file}. Please ensure it exists and the corresponding training cell has been run.")
    st.stop()


features_scaled = scaler.transform(features)

if st.button("Predict Body Fat %"):
    if model_choice == "XGBoost":
        prediction = xgb_model.predict(features_scaled)[0]
    elif model_choice == "Random Forest":
        prediction = rf_model.predict(features_scaled)[0]
    else:  # TensorFlow
        prediction = tf_model.predict(features_scaled)[0][0]

    st.success(f"Estimated Body Fat: **{prediction:.2f}%**")
  