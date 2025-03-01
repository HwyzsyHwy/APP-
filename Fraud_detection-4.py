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
    page_icon='🔥',
    layout='wide',
    initial_sidebar_state='collapsed'
)

# 自定义样式
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: white;
        text-shadow: 2px 2px 4px #000000;
        margin-bottom: 20px;
    }
    .header-background {
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .header-subtitle {
        text-align: center;
        color: #E0E7FF;
        font-size: 18px;
        margin-top: -15px;
    }
    .section-header {
        color: #3B82F6;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 15px;
        padding-bottom: 8px;
        border-bottom: 2px solid #3B82F6;
    }
    .card {
        background-color: #F8FAFC;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    .prediction-card {
        background: linear-gradient(135deg, #1E3A8A, #2563EB);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        margin-top: 30px;
    }
    .prediction-value {
        color: white;
        font-size: 48px;
        font-weight: bold;
        margin: 20px 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
    .prediction-label {
        color: #E0E7FF;
        font-size: 22px;
        font-weight: bold;
        letter-spacing: 1px;
    }
    .feature-label {
        font-weight: bold;
        color: #334155;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        font-weight: bold;
        border: none;
        padding: 12px 30px;
        border-radius: 8px;
        cursor: pointer;
        width: 100%;
        font-size: 18px;
        margin-top: 20px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #1E40AF;
        transform: translateY(-2px);
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.15);
    }
    .category-title {
        color: #1E3A8A;
        font-size: 20px;
        font-weight: bold;
        padding: 8px 15px;
        background-color: #EFF6FF;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .model-selector {
        padding: 15px;
        background-color: #F1F5F9;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 2px solid #CBD5E1;
    }
    /* 美化滑块样式 */
    .stSlider {
        padding-top: 10px;
        padding-bottom: 20px;
    }
    /* 分割线 */
    hr {
        margin: 30px 0;
        border: 0;
        height: 1px;
        background: #E2E8F0;
    }
    /* 美化选择框 */
    div[data-baseweb="select"] {
        margin-top: 5px;
    }
    /* 提高滑块辨识度 */
    .stSlider div[data-baseweb="slider"] div[role="slider"] {
        background-color: #3B82F6 !important;
    }
    
    /* 修改第一列数字输入控件的整个背景色为绿色 */
    .proximate-inputs [data-testid="stNumberInput"] {
        background-color: #32CD32 !important;
        border-radius: 5px !important;
    }
    
    /* 确保输入控件内部所有元素（加减按钮等）也使用绿色背景 */
    .proximate-inputs [data-testid="stNumberInput"] > div,
    .proximate-inputs [data-testid="stNumberInput"] div[data-baseweb="input"],
    .proximate-inputs [data-testid="stNumberInput"] div[data-baseweb="base-input"] {
        background-color: #32CD32 !important;
    }
    
    /* 确保输入框内的文本颜色可见 */
    .proximate-inputs [data-testid="stNumberInput"] input {
        color: black !important;
        background-color: #32CD32 !important;
    }
    
    /* 确保加减按钮的样式一致 */
    .proximate-inputs [data-testid="stNumberInput"] button,
    .proximate-inputs [data-testid="stNumberInput"] button:hover {
        background-color: #32CD32 !important;
        color: black !important;
    }
    
    /* 强制覆盖任何其他的样式 */
    .proximate-inputs [data-testid="stNumberInput"] * {
        background-color: #32CD32 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 主标题和副标题
st.markdown("<div class='header-background'><h1 class='main-title'>Biomass Pyrolysis Yield Forecast</h1><p class='header-subtitle'>Advanced prediction model for biomass pyrolysis products</p></div>", unsafe_allow_html=True)

# 模型选择卡片
st.markdown("<h2 class='section-header'>Model Selection</h2>", unsafe_allow_html=True)
with st.container():
    st.markdown("<div class='model-selector'>", unsafe_allow_html=True)
    model_name = st.selectbox(
        "Select prediction model",
        ["GBDT-Char", "GBDT-Oil", "GBDT-Gas"],
        format_func=lambda x: {
            "GBDT-Char": "🔷 GBDT-Char (Biochar yield prediction)",
            "GBDT-Oil": "🔶 GBDT-Oil (Bio-oil yield prediction)",
            "GBDT-Gas": "🔴 GBDT-Gas (Syngas yield prediction)"
        }[x]
    )
    st.write(f"You've selected: **{model_name}** model")
    st.markdown("</div>", unsafe_allow_html=True)

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
st.markdown("<h2 class='section-header'>Input Parameters</h2>", unsafe_allow_html=True)
st.markdown("<p>Adjust the sliders below to set the biomass characteristics and pyrolysis conditions.</p>", unsafe_allow_html=True)

# 创建三列布局
col1, col2, col3 = st.columns(3)

# 左列：Proximate Analysis
with col1:
    st.markdown("<div class='proximate-inputs'>", unsafe_allow_html=True)
    st.markdown("<div style='background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold;'>Proximate Analysis</div>", unsafe_allow_html=True)
    features = {}
    for feature in feature_categories["Proximate Analysis"]:
        if feature == "M(wt%)":
            features[feature] = st.number_input(feature, min_value=0.0, max_value=20.0, value=5.0, step=0.1, format="%.2f")
        elif feature == "Ash(wt%)":
            features[feature] = st.number_input(feature, min_value=0.0, max_value=25.0, value=8.0, step=0.1, format="%.2f")
        elif feature == "VM(wt%)":
            features[feature] = st.number_input(feature, min_value=0.0, max_value=110.0, value=75.0, step=0.1, format="%.2f")
        elif feature == "FC(wt%)":
            features[feature] = st.number_input(feature, min_value=0.0, max_value=120.0, value=15.0, step=0.1, format="%.2f")
    st.markdown("</div>", unsafe_allow_html=True)

# 中列：Ultimate Analysis
with col2:
    st.markdown("<div style='background-color: #E0C34A; color: white; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold;'>Ultimate Analysis</div>", unsafe_allow_html=True)
    for feature in feature_categories["Ultimate Analysis"]:
        if feature == "C(wt%)":
            features[feature] = st.number_input(feature, min_value=30.0, max_value=110.0, value=60.0, step=0.1, format="%.2f")
        elif feature == "H(wt%)":
            features[feature] = st.number_input(feature, min_value=0.0, max_value=15.0, value=5.0, step=0.1, format="%.2f")
        elif feature == "N(wt%)":
            features[feature] = st.number_input(feature, min_value=0.0, max_value=5.0, value=1.0, step=0.1, format="%.2f")
        elif feature == "O(wt%)":
            features[feature] = st.number_input(feature, min_value=30.0, max_value=60.0, value=38.0, step=0.1, format="%.2f")

# 右列：Pyrolysis Conditions
with col3:
    st.markdown("<div style='background-color: #F27854; color: white; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold;'>Pyrolysis Conditions</div>", unsafe_allow_html=True)
    for feature in feature_categories["Pyrolysis Conditions"]:
        if feature == "PS(mm)":
            features[feature] = st.number_input(feature, min_value=0.0, max_value=20.0, value=6.0, step=0.1, format="%.2f")
        elif feature == "SM(g)":
            features[feature] = st.number_input(feature, min_value=0.0, max_value=200.0, value=75.0, step=0.1, format="%.2f")
        elif feature == "FT(℃)":
            features[feature] = st.number_input(feature, min_value=250.0, max_value=1100.0, value=600.0, step=0.1, format="%.2f")
        elif feature == "HR(℃/min)":
            features[feature] = st.number_input(feature, min_value=0.0, max_value=200.0, value=50.0, step=0.1, format="%.2f")
        elif feature == "FR(mL/min)":
            features[feature] = st.number_input(feature, min_value=0.0, max_value=120.0, value=50.0, step=0.1, format="%.2f")
        elif feature == "RT(min)":
            features[feature] = st.number_input(feature, min_value=5.0, max_value=100.0, value=30.0, step=0.1, format="%.2f")

# 转换为DataFrame
input_data = pd.DataFrame([features])

# 预测按钮和结果
st.markdown("<h2 class='section-header'>Prediction</h2>", unsafe_allow_html=True)
predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])

with predict_col2:
    prediction_button = st.button("Run Prediction", help="Click to predict the pyrolysis yield based on your input parameters")
    
    # 预测结果显示区域
    if prediction_button:
        try:
            # 显示加载状态
            with st.spinner("Calculating prediction..."):
                # 加载所选模型和Scaler
                model = load_model(model_name)
                scaler = load_scaler(model_name)

                # 数据标准化
                input_data_scaled = scaler.transform(input_data)

                # 预测
                y_pred = model.predict(input_data_scaled)[0]

            # 显示预测结果
            yield_labels = {
                "GBDT-Char": "Biochar Yield",
                "GBDT-Oil": "Bio-oil Yield",
                "GBDT-Gas": "Syngas Yield"
            }
            
            st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='prediction-label'>{yield_labels[model_name]} Prediction</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='prediction-value'>{y_pred:.2f}%</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # 添加一些预测解释
            st.success("Prediction successfully completed!")
            st.info(f"Based on the input parameters, the model predicts a {yield_labels[model_name].lower()} of {y_pred:.2f}%.")

        except Exception as e:
            st.error(f"预测过程中出现错误: {e}")
            st.markdown("请确保已经正确安装所有依赖并且模型文件位于正确的路径。")

# 页脚
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94A3B8;'>© 2023 Biomass Pyrolysis Research Team. All rights reserved.</p>", unsafe_allow_html=True) 