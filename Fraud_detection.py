# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 11:02:43 2022

@author: Kevin Boss
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_squared_error
import shap
import matplotlib.pyplot as plt

# Fix for np.bool deprecation
np.bool = bool

st.set_page_config(
    page_title='Biomass Pyrolysis Yield Forecast',
    page_icon='📊',
    layout='wide'
)

# Custom styles for title and layout
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
        margin-bottom: 20px;
    }
    .group-title {
        font-size: 18px;
        font-weight: bold;
        color: #E74C3C;
        margin-bottom: 10px;
    }
    .container {
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Dashboard title
st.markdown("<div class='header-background'><h1 class='main-title'>Biomass Pyrolysis Yield Forecast</h1></div>", unsafe_allow_html=True)

# Load models
MODEL_PATHS = {
    "GBDT-Char": "GBDT-Char-1.15.joblib",
    "GBDT-Oil": "GBDT-Oil-1.15.joblib",
    "GBDT-Gas": "GBDT-Gas-1.15.joblib"
}

# Function to load the selected model
def load_model(model_name):
    return joblib.load(MODEL_PATHS[model_name])

# Sidebar for model selection
st.sidebar.header("Select a model")
model_name = st.sidebar.selectbox(
    "Available Models", list(MODEL_PATHS.keys())
)
st.sidebar.write(f"Currently selected model: **{model_name}**")

# Input features grouped by categories
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<div class='group-title'>Proximate Compositions</div>", unsafe_allow_html=True)
    M = st.slider("M (wt%)", 0.0, 100.0, 10.0)
    Ash = st.slider("Ash (wt%)", 0.0, 100.0, 5.0)
    VM = st.slider("VM (wt%)", 0.0, 100.0, 60.0)
    FC = st.slider("FC (wt%)", 0.0, 100.0, 25.0)

with col2:
    st.markdown("<div class='group-title'>Ultimate Compositions</div>", unsafe_allow_html=True)
    C = st.slider("C (wt%)", 0.0, 100.0, 50.0)
    H = st.slider("H (wt%)", 0.0, 100.0, 6.0)
    N = st.slider("N (wt%)", 0.0, 100.0, 2.0)
    O = st.slider("O (wt%)", 0.0, 100.0, 40.0)

with col3:
    st.markdown("<div class='group-title'>Pyrolysis Conditions</div>", unsafe_allow_html=True)
    PS = st.slider("PS (mm)", 0.0, 5.0, 1.0)
    SM = st.slider("SM (g)", 0.0, 20.0, 10.0)
    FT = st.slider("FT (°C)", 300, 800, 500)
    HR = st.slider("HR (°C/min)", 1.0, 50.0, 10.0)
    FR = st.slider("FR (mL/min)", 10.0, 500.0, 100.0)
    RT = st.slider("RT (min)", 1, 60, 30)

# Combine all features into a single input
input_data = pd.DataFrame([{
    "M(wt%)": M, "Ash(wt%)": Ash, "VM(wt%)": VM, "FC(wt%)": FC,
    "C(wt%)": C, "H(wt%)": H, "N(wt%)": N, "O(wt%)": O,
    "PS(mm)": PS, "SM(g)": SM, "FT(°C)": FT, "HR(°C/min)": HR,
    "FR(mL/min)": FR, "RT(min)": RT
}])

# Predict and evaluate
if st.button("Predict"):
    try:
        # Load the selected model
        model = load_model(model_name)

        # Perform prediction
        y_pred = model.predict(input_data)[0]

        # Generate synthetic test data for evaluation
        test_data = np.random.rand(100, 14)  # Replace with actual test data
        test_target = np.random.rand(100)  # Replace with actual test target
        y_test_pred = model.predict(test_data)

        # Evaluate model
        r2 = r2_score(test_target, y_test_pred)
        rmse = np.sqrt(mean_squared_error(test_target, y_test_pred))

        # Display results
        st.subheader("Prediction Results")
        st.write(f"Predicted Yield: **{y_pred:.2f}**")
        st.write(f"R² (Coefficient of Determination): **{r2:.4f}**")
        st.write(f"RMSE (Root Mean Squared Error): **{rmse:.4f}**")

        # SHAP Analysis
        st.subheader("SHAP Analysis")
        explainer = shap.Explainer(model, test_data)
        shap_values = explainer(test_data)
        shap.summary_plot(shap_values, test_data, plot_type="bar", show=False)
        st.pyplot(bbox_inches='tight')

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
