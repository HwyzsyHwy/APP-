# -*- coding: utf-8 -*-
"""
Biomass Pyrolysis Yield Forecast
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_squared_error

# 页面设置
st.set_page_config(
    page_title='Biomass Pyrolysis Yield Forecast',
    page_icon='📊',
    layout='wide'
)

# 自定义样式
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
    .red-text {
        color: #ff5c5c;
        font-size: 20px;
        font-weight: bold;
    }
    .input-section {
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 主标题
st.markdown("<div class='header-background'><h1 class='main-title'>Biomass Pyrolysis Yield Forecast</h1></div>", unsafe_allow_html=True)

# 模型选择放置在顶部
st.header("Select a Model")
model_name = st.selectbox(
    "Available Models", ["GBDT-Char", "GBDT-Oil", "GBDT-Gas"]
)
st.write(f"Current selected model: **{model_name}**")

# 加载模型和Scaler
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

# 加载函数
def load_model(model_name):
    return joblib.load(MODEL_PATHS[model_name])

def load_scaler(model_name):
    return joblib.load(SCALER_PATHS[model_name])

# 特征分类
feature_categories = {
    "Proximate Analysis": ["M(wt%)", "Ash(wt%)", "VM(wt%)", "FC(wt%)"],
    "Ultimate Analysis": ["C(wt%)", "H(wt%)", "N(wt%)", "O(wt%)"],
    "Pyrolysis Conditions": ["PS(mm)", "SM(g)", "FT(℃)", "HR(℃/min)", "FR(mL/min)", "RT(min)"]
}

# 输入特征部分
st.markdown("<h3 style='color: orange;'>Input Features</h3>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

# 左列：Proximate Analysis
with col1:
    st.subheader("Proximate Analysis")
    features = {}
    for feature in feature_categories["Proximate Analysis"]:
        features[feature] = st.slider(feature, min_value=0.0, max_value=100.0, value=50.0)

# 中列：Ultimate Analysis
with col2:
    st.subheader("Ultimate Analysis")
    for feature in feature_categories["Ultimate Analysis"]:
        features[feature] = st.slider(feature, min_value=0.0, max_value=100.0, value=50.0)

# 右列：Pyrolysis Conditions
with col3:
    st.subheader("Pyrolysis Conditions")
    for feature in feature_categories["Pyrolysis Conditions"]:
        features[feature] = st.slider(feature, min_value=0.0, max_value=600.0 if "PS" not in feature else 100.0, value=50.0)

# 转换为DataFrame
input_data = pd.DataFrame([features])

# 预测按钮和结果
st.markdown("<h3 style='color: orange;'>Prediction Results</h3>", unsafe_allow_html=True)
if st.button("Predict"):
    try:
        # 加载所选模型和Scaler
        model = load_model(model_name)
        scaler = load_scaler(model_name)

        # 数据标准化
        input_data_scaled = scaler.transform(input_data)

        # 预测
        y_pred = model.predict(input_data_scaled)[0]

        # 显示预测结果
        st.markdown(f"<div class='red-text'>Predicted Yield: {y_pred:.2f}</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
