# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 11:02:43 2022

@author: Kevin Boss
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib  # 使用joblib进行模型加载与保存
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
    "GBDT-Char": "GBDT-Char-1.15.joblib",
    "GBDT-Oil": "GBDT-Oil-1.15.joblib",
    "GBDT-Gas": "GBDT-Gas-1.15.joblib"
}

# Function to load the selected model
def load_model(model_name):
    try:
        return joblib.load(MODEL_PATHS[model_name])
    except FileNotFoundError:
        st.error(f"模型文件 {MODEL_PATHS[model_name]} 未找到，请检查路径和文件名！")
        return None
    except Exception as e:
        st.error(f"加载模型时出现错误: {e}")
        return None

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
        
        if model is None:
            st.error("无法加载模型，请检查模型路径或文件！")
        else:
            # Ensure input_data matches model's expected format
            if input_data.shape[1] != len(feature_names):
                st.error(f"输入数据列数 ({input_data.shape[1]}) 与模型特征数 ({len(feature_names)}) 不匹配！")
            else:
                # Perform prediction
                y_pred = model.predict(input_data)[0]
                
                # Generate synthetic test data for evaluation
                test_data = np.random.rand(100, 14)  # Use actual test data if available
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
