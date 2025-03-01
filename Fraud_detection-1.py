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
    body {
        background-color: #0e1117;
        color: white;
    }
    .main-title {
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        color: white;
        margin-bottom: 20px;
        padding: 10px;
        background-color: #1e1e1e;
        border-radius: 5px;
    }
    .ultimate-section {
        background-color: #DAA520;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        color: black;
    }
    .proximate-section {
        background-color: #32CD32;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        color: black;
    }
    .pyrolysis-section {
        background-color: #FF7F50;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        color: black;
    }
    .section-title {
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    .stSlider {
        padding-top: 5px;
        padding-bottom: 5px;
    }
    .yield-result {
        background-color: #1E1E1E;
        color: white;
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        padding: 15px;
        border-radius: 8px;
        margin-top: 20px;
    }
    .button-container {
        display: flex;
        justify-content: space-between;
        margin-top: 20px;
    }
    .predict-button {
        background-color: #ff4b4b;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        cursor: pointer;
    }
    .clear-button {
        background-color: #ff4b4b;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        cursor: pointer;
    }
    div.stSlider > div > div > div {
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 主标题
st.markdown("<h1 class='main-title'>GUI for Bio-Char Yield Prediction based on ELT-PSO Model</h1>", unsafe_allow_html=True)

# 隐藏模型选择，让它不那么突出
with st.expander("Model Selection", expanded=False):
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

# 创建三列布局
col1, col2, col3 = st.columns(3)

# Proximate Analysis (绿色区域) - 在第一列
with col1:
    st.markdown("<div class='proximate-section'><div class='section-title'>Proximate Analysis</div>", unsafe_allow_html=True)
    features = {}
    for feature in feature_categories["Proximate Analysis"]:
        if feature == "M(wt%)":
            features[feature] = st.slider(feature, min_value=0.0, max_value=20.0, value=5.0, key=f"proximate_{feature}")
        elif feature == "Ash(wt%)":
            features[feature] = st.slider(feature, min_value=0.0, max_value=25.0, value=8.0, key=f"proximate_{feature}")
        elif feature == "VM(wt%)":
            features[feature] = st.slider(feature, min_value=0.0, max_value=110.0, value=75.0, key=f"proximate_{feature}")
        elif feature == "FC(wt%)":
            features[feature] = st.slider(feature, min_value=0.0, max_value=120.0, value=15.0, key=f"proximate_{feature}")
    st.markdown("</div>", unsafe_allow_html=True)

# Ultimate Analysis (黄色区域) - 在第二列
with col2:
    st.markdown("<div class='ultimate-section'><div class='section-title'>Ultimate Analysis</div>", unsafe_allow_html=True)
    for feature in feature_categories["Ultimate Analysis"]:
        if feature == "C(wt%)":
            features[feature] = st.slider(feature, min_value=30.0, max_value=110.0, value=60.0, key=f"ultimate_{feature}")
        elif feature == "H(wt%)":
            features[feature] = st.slider(feature, min_value=0.0, max_value=15.0, value=5.0, key=f"ultimate_{feature}")
        elif feature == "N(wt%)":
            features[feature] = st.slider(feature, min_value=0.0, max_value=5.0, value=1.0, key=f"ultimate_{feature}")
        elif feature == "O(wt%)":
            features[feature] = st.slider(feature, min_value=30.0, max_value=60.0, value=38.0, key=f"ultimate_{feature}")
    st.markdown("</div>", unsafe_allow_html=True)

# Pyrolysis Conditions (橙色区域) - 在第三列
with col3:
    st.markdown("<div class='pyrolysis-section'><div class='section-title'>Pyrolysis Conditions</div>", unsafe_allow_html=True)
    for feature in feature_categories["Pyrolysis Conditions"]:
        if feature == "PS(mm)":
            features[feature] = st.slider(feature, min_value=0.0, max_value=20.0, value=6.0, key=f"pyrolysis_{feature}")
        elif feature == "SM(g)":
            features[feature] = st.slider(feature, min_value=0.0, max_value=200.0, value=75.0, key=f"pyrolysis_{feature}")
        elif feature == "FT(℃)":
            features[feature] = st.slider(feature, min_value=250.0, max_value=1100.0, value=600.0, key=f"pyrolysis_{feature}")
        elif feature == "HR(℃/min)":
            features[feature] = st.slider(feature, min_value=0.0, max_value=200.0, value=50.0, key=f"pyrolysis_{feature}")
        elif feature == "FR(mL/min)":
            features[feature] = st.slider(feature, min_value=0.0, max_value=120.0, value=50.0, key=f"pyrolysis_{feature}")
        elif feature == "RT(min)":
            features[feature] = st.slider(feature, min_value=5.0, max_value=100.0, value=30.0, key=f"pyrolysis_{feature}")
    st.markdown("</div>", unsafe_allow_html=True)

# 转换为DataFrame
input_data = pd.DataFrame([features])

# 预测结果显示区域和按钮
result_col, button_col = st.columns([3, 1])

with result_col:
    prediction_placeholder = st.empty()
    
with button_col:
    predict_button = st.button("PUSH", key="predict")
    clear_button = st.button("CLEAR", key="clear")

# 处理预测逻辑
if predict_button:
    try:
        # 加载所选模型和Scaler
        model = load_model(model_name)
        scaler = load_scaler(model_name)

        # 数据标准化
        input_data_scaled = scaler.transform(input_data)

        # 预测
        y_pred = model.predict(input_data_scaled)[0]

        # 显示预测结果
        prediction_placeholder.markdown(
            f"<div class='yield-result'>Biochar Yield (%) <br> {y_pred:.2f}</div>",
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"预测过程中出现错误: {e}")

# 清除按钮逻辑
if clear_button:
    # 不需要实际清除，因为Streamlit会在页面刷新时重置
    st.experimental_rerun()