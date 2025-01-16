# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 11:02:43 2022

@author: Kevin Boss
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(
    page_title='Model Selection for Regression Analysis',
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
st.markdown("<div class='header-background'><h1 class='main-title'>回归模型选择与预测</h1></div>", unsafe_allow_html=True)

# Load models
MODEL_PATHS = {
    "GBDT-Char": "GBDT-Char-1.15.pkl",
    "GBDT-Oil": "GBDT-Oil-1.15.pkl",
    "GBDT-Gas": "GBDT-Gas-1.15.pkl"
}

# Function to load the selected model
def load_model(model_name):
    with open(MODEL_PATHS[model_name], "rb") as file:
        return pickle.load(file)

# Sidebar for model selection
st.sidebar.header("选择一个模型")
model_name = st.sidebar.selectbox(
    "可用模型", list(MODEL_PATHS.keys())
)
st.sidebar.write(f"当前选择的模型: **{model_name}**")

# Input features
st.sidebar.header("输入特征值")
features = {}
feature_names = [
    "M (wt%)", "Ash (wt%)", "VM (wt%)", "FC (wt%)", "C (wt%)", 
    "H (wt%)", "N (wt%)", "O (wt%)", "PS (mm)", "SM (g)", 
    "FT (°C)", "HR (°C/min)", "FR (mL/min)", "RT (min)"
]

for feature in feature_names:
    features[feature] = st.sidebar.number_input(f"{feature}", value=0.0)

# Convert to DataFrame
input_data = pd.DataFrame([features])

# Predict and evaluate
if st.button("预测"):
    try:
        # Load the selected model
        model = load_model(model_name)
        
        # Perform prediction
        y_pred = model.predict(input_data)[0]
        
        # Generate synthetic test data for evaluation
        # (Replace with actual data if available)
        test_data = np.random.rand(100, 14)
        test_target = np.random.rand(100)
        y_test_pred = model.predict(test_data)
        
        # Evaluate model
        r2 = r2_score(test_target, y_test_pred)
        rmse = np.sqrt(mean_squared_error(test_target, y_test_pred))
        
        # Display results
        st.subheader("预测结果")
        st.write(f"预测值 (Y): {y_pred:.2f}")
        st.write(f"R² (判定系数): {r2:.4f}")
        st.write(f"RMSE (均方根误差): {rmse:.4f}")
    except Exception as e:
        st.error(f"预测过程中出现错误: {e}")
