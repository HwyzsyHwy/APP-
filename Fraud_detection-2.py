# -*- coding: utf-8 -*-
"""
Biomass Pyrolysis Yield Forecast
"""

import streamlit as st
import pandas as pd
import joblib

# 页面设置
st.set_page_config(
    page_title='Biomass Pyrolysis Yield Forecast',
    page_icon='📊',
    layout='wide'
)

# 自定义CSS样式 - 简化版，只专注于背景色
st.markdown(
    """
    <style>
    /* 覆盖Streamlit的默认样式 */
    .stNumberInput > div:first-child > div:first-child > div:first-child > div > input {
        background-color: #4CAF50 !important;  /* 这里是绿色 */
        color: black !important;
    }
    
    /* 主要标题样式 */
    .main-title {
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    
    /* 分析部分标题样式 */
    .section-header {
        background-color: #32CD32;  /* 绿色 */
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
        text-align: center;
        font-weight: bold;
    }
    
    .section-header-yellow {
        background-color: #DAA520;  /* 黄色 */
    }
    
    .section-header-orange {
        background-color: #FF7F50;  /* 橙色 */
    }
    
    /* 结果显示样式 */
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
    </style>
    """,
    unsafe_allow_html=True
)

# 主标题
st.markdown("<div class='main-title'>GUI for Bio-Char Yield Prediction based on ELT-PSO Model</div>", unsafe_allow_html=True)

# 初始化会话状态
if 'clear_pressed' not in st.session_state:
    st.session_state.clear_pressed = False

# 模型选择
with st.expander("Model Selection", expanded=False):
    model_name = st.selectbox(
        "Available Models", ["GBDT-Char", "GBDT-Oil", "GBDT-Gas"]
    )
    st.write(f"Current selected model: **{model_name}**")

# 定义默认值
default_values = {
    "M(wt%)": 5.0,
    "Ash(wt%)": 8.0,
    "VM(wt%)": 75.0,
    "FC(wt%)": 15.0,
    "C(wt%)": 60.0,
    "H(wt%)": 5.0,
    "N(wt%)": 1.0,
    "O(wt%)": 38.0,
    "PS(mm)": 6.0,
    "SM(g)": 75.0,
    "FT(℃)": 600.0,
    "HR(℃/min)": 50.0,
    "FR(mL/min)": 50.0,
    "RT(min)": 30.0
}

# 特征分类
feature_categories = {
    "Proximate Analysis": ["M(wt%)", "Ash(wt%)", "VM(wt%)", "FC(wt%)"],
    "Ultimate Analysis": ["C(wt%)", "H(wt%)", "N(wt%)", "O(wt%)"],
    "Pyrolysis Conditions": ["PS(mm)", "SM(g)", "FT(℃)", "HR(℃/min)", "FR(mL/min)", "RT(min)"]
}

# 清除函数
def clear_values():
    st.session_state.clear_pressed = True
    for key in default_values:
        st.session_state[key] = default_values[key]
    if 'prediction_result' in st.session_state:
        st.session_state.prediction_result = None

# 创建三列布局
col1, col2, col3 = st.columns(3)

# 使用字典来存储所有输入值
features = {}

# 为每个区域定义自定义CSS类
css_for_inputs = """
<style>
/* 绿色背景输入框 */
.section-1 [data-testid="stNumberInput"] input {
    background-color: #32CD32 !important;
    color: black !important;
}

/* 黄色背景输入框 */
.section-2 [data-testid="stNumberInput"] input {
    background-color: #DAA520 !important;
    color: black !important;
}

/* 橙色背景输入框 */
.section-3 [data-testid="stNumberInput"] input {
    background-color: #FF7F50 !important;
    color: black !important;
}

/* 隐藏加减按钮 */
[data-testid="stNumberInput"] button {
    display: none !important;
}
</style>
"""

st.markdown(css_for_inputs, unsafe_allow_html=True)

# Proximate Analysis (绿色区域)
with col1:
    st.markdown("<div class='section-header'>Proximate Analysis</div>", unsafe_allow_html=True)
    
    # 添加自定义区域标记
    st.markdown('<div class="section-1">', unsafe_allow_html=True)
    
    for feature in feature_categories["Proximate Analysis"]:
        # 重置值或使用现有值
        if st.session_state.clear_pressed:
            value = default_values[feature]
        else:
            value = st.session_state.get(feature, default_values[feature])
        
        # 使用简单的标签和输入
        features[feature] = st.number_input(
            feature,
            min_value=0.0, 
            max_value=20.0 if feature == "M(wt%)" else (25.0 if feature == "Ash(wt%)" else (110.0 if feature == "VM(wt%)" else 120.0)), 
            value=value, 
            key=feature,
            format="%.2f"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

# Ultimate Analysis (黄色区域)
with col2:
    st.markdown("<div class='section-header section-header-yellow'>Ultimate Analysis</div>", unsafe_allow_html=True)
    
    # 添加自定义区域标记
    st.markdown('<div class="section-2">', unsafe_allow_html=True)
    
    for feature in feature_categories["Ultimate Analysis"]:
        if st.session_state.clear_pressed:
            value = default_values[feature]
        else:
            value = st.session_state.get(feature, default_values[feature])
        
        features[feature] = st.number_input(
            feature, 
            min_value=30.0 if feature in ["C(wt%)", "O(wt%)"] else 0.0, 
            max_value=110.0 if feature == "C(wt%)" else (15.0 if feature == "H(wt%)" else (5.0 if feature == "N(wt%)" else 60.0)), 
            value=value, 
            key=feature,
            format="%.2f"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

# Pyrolysis Conditions (橙色区域)
with col3:
    st.markdown("<div class='section-header section-header-orange'>Pyrolysis Conditions</div>", unsafe_allow_html=True)
    
    # 添加自定义区域标记
    st.markdown('<div class="section-3">', unsafe_allow_html=True)
    
    for feature in feature_categories["Pyrolysis Conditions"]:
        if st.session_state.clear_pressed:
            value = default_values[feature]
        else:
            value = st.session_state.get(feature, default_values[feature])
        
        min_val = 250.0 if feature == "FT(℃)" else (5.0 if feature == "RT(min)" else 0.0)
        max_val = 1100.0 if feature == "FT(℃)" else (200.0 if feature in ["SM(g)", "HR(℃/min)"] else (120.0 if feature == "FR(mL/min)" else (100.0 if feature == "RT(min)" else 20.0)))
        
        features[feature] = st.number_input(
            feature, 
            min_value=min_val, 
            max_value=max_val, 
            value=value, 
            key=feature,
            format="%.2f"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

# 重置session_state中的clear_pressed状态
if st.session_state.clear_pressed:
    st.session_state.clear_pressed = False

# 转换为DataFrame
input_data = pd.DataFrame([features])

# 预测结果显示区域和按钮
result_col, button_col = st.columns([3, 1])

with result_col:
    prediction_placeholder = st.empty()
    
with button_col:
    predict_button = st.button("PUSH", key="predict")
    clear_button = st.button("CLEAR", key="clear", on_click=clear_values)

# 处理预测逻辑
if predict_button:
    try:
        # 这里添加实际的模型加载和预测逻辑
        # 模拟预测结果
        y_pred = 35.42  # 替换为实际预测值
        
        # 保存预测结果到session_state
        st.session_state.prediction_result = y_pred

        # 显示预测结果
        prediction_placeholder.markdown(
            f"<div class='yield-result'>Yield (%) <br> {y_pred:.2f}</div>",
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"预测过程中出现错误: {e}")

# 如果有保存的预测结果，显示它
if 'prediction_result' in st.session_state and st.session_state.prediction_result is not None:
    prediction_placeholder.markdown(
        f"<div class='yield-result'>Yield (%) <br> {st.session_state.prediction_result:.2f}</div>",
        unsafe_allow_html=True
    )