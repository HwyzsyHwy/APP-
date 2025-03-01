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
    .data-block {
        display: flex;
        justify-content: space-between;
        margin-bottom: 10px;
        overflow: hidden;
    }
    .label-block {
        background-color: #DAA520;
        color: white;
        font-weight: bold;
        padding: 8px 15px;
        border-radius: 4px;
        width: 30%;
        text-align: center;
    }
    .value-block {
        background-color: #DAA520;
        color: white;
        font-weight: bold;
        padding: 8px 15px;
        border-radius: 4px;
        width: 67%;
        text-align: center;
    }
    .ultimate-label {
        background-color: #DAA520;
    }
    .ultimate-value {
        background-color: #DAA520;
    }
    .proximate-label {
        background-color: #32CD32;
    }
    .proximate-value {
        background-color: #32CD32;
    }
    .pyrolysis-label {
        background-color: #FF7F50;
    }
    .pyrolysis-value {
        background-color: #FF7F50;
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
    /* 移除Streamlit的默认样式 */
    div.block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 主标题
st.markdown("<h1 class='main-title'>GUI for Bio-Char Yield Prediction based on ELT-PSO Model</h1>", unsafe_allow_html=True)

# 初始化会话状态，用于保存是否需要清除数据
if 'clear_pressed' not in st.session_state:
    st.session_state.clear_pressed = False

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

# 定义默认值
default_values = {
    "M(wt%)": 5.0,
    "Ash(wt%)": 8.6,
    "VM(wt%)": 73.5,
    "FC(wt%)": 13.2,
    "C(wt%)": 52.05,
    "H(wt%)": 5.37,
    "N(wt%)": 0.49,
    "O(wt%)": 42.1,
    "PS(mm)": 1.5,
    "SM(g)": 75.0,
    "FT(℃)": 500.0,
    "HR(℃/min)": 10.0,
    "FR(mL/min)": 2.0,
    "RT(min)": 30.0
}

# 初始化特征值，如果从未设置则使用默认值
for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

# 特征分类
feature_categories = {
    "Proximate Analysis": ["VM(wt%)", "FC(wt%)", "MC(wt%)", "Ash(wt%)"],
    "Ultimate Analysis": ["C(wt%)", "H(wt%)", "N(wt%)", "O(wt%)"],
    "Pyrolysis Conditions": ["PS(mm)", "SM(g)", "FT(℃)", "HR(℃/min)", "FR(mL/min)", "RT(min)"]
}

# 创建三列布局
col1, col2, col3 = st.columns(3)

# 定义一个函数来处理数值输入和显示数据块
def display_data_block(column, category, feature, min_val, max_val, css_class):
    with column:
        # 添加数据块标题
        if feature == list(feature_categories[category])[0]:
            st.markdown(f"<div class='section-title'>{category}</div>", unsafe_allow_html=True)
        
        # 处理输入框的回调
        def update_value():
            try:
                new_val = float(st.session_state[f"input_{feature}"])
                # 确保值在范围内
                new_val = max(min_val, min(max_val, new_val))
                st.session_state[feature] = new_val
            except ValueError:
                pass

        # 创建数据块显示
        st.markdown(f"""
        <div class='data-block'>
            <div class='label-block {css_class}-label'>{feature}</div>
            <div class='value-block {css_class}-value'>{st.session_state[feature]}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # 添加隐藏输入框用于更改值
        st.number_input(
            f"Change {feature}",
            min_value=min_val,
            max_value=max_val,
            value=st.session_state[feature],
            key=f"input_{feature}",
            on_change=update_value,
            label_visibility="hidden"
        )

# Ultimate Analysis (黄色区域) - 在第一列
with col1:
    st.markdown("<div class='ultimate-section'>", unsafe_allow_html=True)
    for feature in feature_categories["Ultimate Analysis"]:
        min_val = 30.0 if feature in ["C(wt%)", "O(wt%)"] else 0.0
        max_val = 110.0 if feature == "C(wt%)" else (15.0 if feature == "H(wt%)" else (5.0 if feature == "N(wt%)" else 60.0))
        display_data_block(col1, "Ultimate Analysis", feature, min_val, max_val, "ultimate")
    st.markdown("</div>", unsafe_allow_html=True)

# Proximate Analysis (绿色区域) - 在第二列
with col2:
    st.markdown("<div class='proximate-section'>", unsafe_allow_html=True)
    for feature in feature_categories["Proximate Analysis"]:
        if feature in ["MC(wt%)", "M(wt%)"]:
            min_val, max_val = 0.0, 20.0
        elif feature == "Ash(wt%)":
            min_val, max_val = 0.0, 25.0
        elif feature == "VM(wt%)":
            min_val, max_val = 0.0, 110.0
        elif feature == "FC(wt%)":
            min_val, max_val = 0.0, 120.0
        
        display_data_block(col2, "Proximate Analysis", feature, min_val, max_val, "proximate")
    st.markdown("</div>", unsafe_allow_html=True)

# Pyrolysis Conditions (橙色区域) - 在第三列
with col3:
    st.markdown("<div class='pyrolysis-section'>", unsafe_allow_html=True)
    for feature in feature_categories["Pyrolysis Conditions"]:
        min_val = 250.0 if feature == "FT(℃)" else (5.0 if feature == "RT(min)" else 0.0)
        max_val = 1100.0 if feature == "FT(℃)" else (200.0 if feature in ["SM(g)", "HR(℃/min)"] else (120.0 if feature == "FR(mL/min)" else (100.0 if feature == "RT(min)" else 20.0)))
        
        display_data_block(col3, "Pyrolysis Conditions", feature, min_val, max_val, "pyrolysis")
    st.markdown("</div>", unsafe_allow_html=True)

# 构建特征字典用于预测
features = {}
for category in feature_categories:
    for feature in feature_categories[category]:
        features[feature] = st.session_state[feature]

# 转换为DataFrame
input_data = pd.DataFrame([features])

# 预测结果显示区域和按钮
result_col, button_col = st.columns([3, 1])

with result_col:
    prediction_placeholder = st.empty()
    
with button_col:
    predict_button = st.button("PUSH", key="predict")
    
    # 定义Clear按钮的回调函数
    def clear_values():
        for key, value in default_values.items():
            st.session_state[key] = value
        # 清除预测结果
        if 'prediction_result' in st.session_state:
            st.session_state.prediction_result = None
    
    clear_button = st.button("CLEAR", key="clear", on_click=clear_values)

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
        
        # 保存预测结果到session_state
        st.session_state.prediction_result = y_pred

        # 显示预测结果
        prediction_placeholder.markdown(
            f"<div class='yield-result'>Biochar Yield (%) <br> {y_pred:.2f}</div>",
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"预测过程中出现错误: {e}")

# 如果有保存的预测结果，显示它
if 'prediction_result' in st.session_state and st.session_state.prediction_result is not None:
    prediction_placeholder.markdown(
        f"<div class='yield-result'>Biochar Yield (%) <br> {st.session_state.prediction_result:.2f}</div>",
        unsafe_allow_html=True
    )