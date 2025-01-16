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
    </style>
    """,
    unsafe_allow_html=True
)

# 主标题
st.markdown("<div class='header-background'><h1 class='main-title'>Biomass Pyrolysis Yield Forecast</h1></div>", unsafe_allow_html=True)

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
    "Biomass Compositions": ["C_biomass(wt%)", "H_biomass(wt%)", "N_biomass(wt%)", "O_biomass(wt%)", "Ash_biomass(wt%)"],
    "Pyrolysis Conditions": ["T_py(°C)", "Rt_py(min)"],
    "Adsorption Conditions": ["T_ad(°C)", "pH_ad", "C0(mmol/L)"],
    "Heavy Metal Properties": ["X", "r", "Ncharge"]
}

# 布局设置
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

# 输入特征值
input_features = {}
for category, features in feature_categories.items():
    with st.container():
        if category in ["Biomass Compositions"]:
            with col1:
                st.subheader(category)
                for feature in features:
                    input_features[feature] = st.slider(feature, min_value=0.0, max_value=100.0, value=50.0)
        elif category in ["Pyrolysis Conditions"]:
            with col2:
                st.subheader(category)
                for feature in features:
                    input_features[feature] = st.slider(feature, min_value=1.0, max_value=850.0 if "T_py" in feature else 600.0, value=325.0)
        elif category in ["Adsorption Conditions"]:
            with col3:
                st.subheader(category)
                for feature in features:
                    input_features[feature] = st.slider(feature, min_value=0.0, max_value=100.0 if "C0" in feature else 9.0, value=6.0)
        elif category in ["Heavy Metal Properties"]:
            with col4:
                st.subheader(category)
                for feature in features:
                    input_features[feature] = st.slider(feature, min_value=0.0, max_value=120.0 if "r" in feature else 3.0, value=1.0)

# 转换为DataFrame
input_data = pd.DataFrame([input_features])

# 预测按钮
if st.button("Predict"):
    try:
        # 加载模型和Scaler
        model = load_model("GBDT-Char")  # 这里可以根据需求切换模型
        scaler = load_scaler("GBDT-Char")

        # 数据标准化
        input_data_scaled = scaler.transform(input_data)

        # 预测
        y_pred = model.predict(input_data_scaled)[0]

        # 显示预测结果
        st.markdown("<div class='red-text'>Heavy metal adsorption capacity of biochar (mmol/g): {:.2f}</div>".format(y_pred), unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
