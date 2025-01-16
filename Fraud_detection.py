# -*- coding: utf-8 -*-
"""
Biomass Pyrolysis Yield Forecast
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_squared_error
import shap

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
SCALER_PATHS = {
    "GBDT-Char": "scaler-Char-1.15.joblib",
    "GBDT-Oil": "scaler-Oil-1.15.joblib",
    "GBDT-Gas": "scaler-Gas-1.15.joblib"
}

# Function to load the selected model
def load_model(model_name):
    return joblib.load(MODEL_PATHS[model_name])

# Function to load scaler
def load_scaler(model_name):
    return joblib.load(SCALER_PATHS[model_name])

# Sidebar for SHAP summary plot
st.sidebar.header("SHAP Analysis")
if st.sidebar.button("Generate SHAP Summary"):
    try:
        model_name = "GBDT-Char"  # Example for SHAP
        model = load_model(model_name)
        scaler = load_scaler(model_name)
        
        # Generate synthetic data (replace with real data if available)
        synthetic_data = pd.DataFrame(np.random.rand(100, len(model.feature_importances_)), columns=[f'Feature_{i}' for i in range(len(model.feature_importances_))])
        scaled_data = scaler.transform(synthetic_data)

        explainer = shap.Explainer(model, scaled_data)
        shap_values = explainer(scaled_data)

        # Display SHAP summary plot
        st.sidebar.subheader("SHAP Summary Plot")
        shap.summary_plot(shap_values, synthetic_data, show=False)
        st.pyplot(bbox_inches='tight')
    except Exception as e:
        st.sidebar.error(f"SHAP analysis error: {e}")

# Layout for input and prediction
st.header("Input Features")
cols = st.columns([1, 1, 1])  # Adjust column width as needed

feature_categories = {
    "Proximate Analysis": ["M(wt%)", "Ash(wt%)", "VM(wt%)", "FC(wt%)"],
    "Ultimate Analysis": ["C(wt%)", "H(wt%)", "N(wt%)", "O(wt%)"],
    "Pyrolysis Conditions": ["PS(mm)", "SM(g)", "FT(℃)", "HR(℃/min)", "FR(mL/min)", "RT(min)"]
}

features = {}

for i, (category, feature_list) in enumerate(feature_categories.items()):
    with cols[i % len(cols)]:  # Distribute features across columns
        st.subheader(category)
        for feature in feature_list:
            features[feature] = st.slider(feature, min_value=0.0, max_value=100.0, value=50.0)

# Convert to DataFrame
input_data = pd.DataFrame([features])

# Predict and evaluate
if st.button("Predict"):
    try:
        # Load the selected model and scaler
        model_name = st.selectbox("Select Model", list(MODEL_PATHS.keys()))
        model = load_model(model_name)
        scaler = load_scaler(model_name)

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Perform prediction
        y_pred = model.predict(input_data_scaled)[0]

        # Display prediction result
        st.subheader("Prediction Results")
        st.markdown(f"**Predicted Yield:** {y_pred:.2f}")

    except Exception as e:
        st.error(f"Prediction error: {e}")
