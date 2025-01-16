# -*- coding: utf-8 -*-
"""
Biomass Pyrolysis Yield Forecast
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# Page setup
st.set_page_config(
    page_title='Biomass Pyrolysis Yield Forecast',
    page_icon='📊',
    layout='wide'
)

# Custom styles for title
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        color: white;
    }
    .header-background {
        background-color: #1e1e1e;
        padding: 10px;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Dashboard title
st.markdown("<div class='header-background'><h1 class='main-title'>Biomass Pyrolysis Yield Forecast</h1></div>", unsafe_allow_html=True)

# Load models and scaler
MODEL_PATHS = {
    "GBDT-Char": "GBDT-Char-1.15.joblib",
    "GBDT-Oil": "GBDT-Oil-1.15.joblib",
    "GBDT-Gas": "GBDT-Gas-1.15.joblib"
}
SCALER_PATH = "scaler.joblib"

# Function to load the selected model
def load_model(model_name):
    return joblib.load(MODEL_PATHS[model_name])

# Function to load scaler
def load_scaler():
    return joblib.load(SCALER_PATH)

# Sidebar for model selection
st.sidebar.header("Select a Model")
model_name = st.sidebar.selectbox(
    "Available Models", list(MODEL_PATHS.keys())
)
st.sidebar.write(f"Current selected model: **{model_name}**")

# Feature categories
feature_categories = {
    "Proximate Analysis": ["M(wt%)", "Ash(wt%)", "VM(wt%)", "FC(wt%)"],
    "Ultimate Analysis": ["C(wt%)", "H(wt%)", "N(wt%)", "O(wt%)"],
    "Pyrolysis Conditions": ["PS(mm)", "SM(g)", "FT(℃)", "HR(℃/min)", "FR(mL/min)", "RT(min)"]
}

# Input features
features = {}
st.sidebar.header("Input Features")

for category, feature_list in feature_categories.items():
    st.sidebar.subheader(category)
    for feature in feature_list:
        features[feature] = st.sidebar.slider(feature, min_value=0.0, max_value=100.0, value=50.0)

# Convert to DataFrame
input_data = pd.DataFrame([features])

# Predict and evaluate
if st.button("Predict"):
    try:
        # Load the selected model and scaler
        model = load_model(model_name)
        scaler = load_scaler()

        # Verify feature names match
        if not all(f in model.feature_names_in_ for f in input_data.columns):
            st.error("Feature mismatch: Ensure input features match model training features.")
            st.stop()

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Perform prediction
        y_pred = model.predict(input_data_scaled)[0]

        # Display prediction result
        st.subheader("Prediction Results")
        st.write(f"Predicted Yield: **{y_pred:.2f}**")

        # Validate prediction range with training data
        st.markdown(
            f"<div style='color: green; font-size: 20px;'>Prediction complete. Verify prediction falls within expected training range.</div>",
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
