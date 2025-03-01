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

# 主标题
st.markdown("<h1 style='text-align:center;font-size:32px;font-weight:bold;margin-bottom:20px;color:white;'>GUI for Bio-Char Yield Prediction based on ELT-PSO Model</h1>", unsafe_allow_html=True)

# 初始化会话状态
if 'clear_pressed' not in st.session_state:
    st.session_state.clear_pressed = False

# 模型选择
with st.expander("Model Selection", expanded=False):
    model_name = st.selectbox(
        "Available Models", ["GBDT-Char", "GBDT-Oil", "GBDT-Gas"]
    )
    st.write(f"Current selected model: **{model_name}**")

# 模型路径
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

# 创建三列布局
col1, col2, col3 = st.columns(3)

# 使用字典来存储所有输入值
features = {}

# 创建自定义样式
custom_css = """
<style>
    .section-header {
        color: white;
        font-weight: bold;
        font-size: 22px;
        text-align: center;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    
    .input-label {
        padding: 5px;
        border-radius: 5px;
        margin-bottom: 5px;
        font-size: 18px;
        color: white;
    }
    
    .yield-result {
        background-color: #1E1E1E;
        color: white;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        padding: 15px;
        border-radius: 8px;
        margin-top: 20px;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# Proximate Analysis (绿色区域)
with col1:
    # 使用Streamlit的容器功能
    prox_container = st.container()
    prox_container.markdown("<div class='section-header' style='background-color: #32CD32;'>Proximate Analysis</div>", unsafe_allow_html=True)
    
    for feature in feature_categories["Proximate Analysis"]:
        # 重置值或使用现有值
        if st.session_state.clear_pressed:
            value = default_values[feature]
        else:
            value = st.session_state.get(f"proximate_{feature}", default_values[feature])
        
        # 创建两列布局
        label_col, input_col = st.columns([1, 0.5])
        
        with label_col:
            st.markdown(f"<div class='input-label' style='background-color: #32CD32;'>{feature}</div>", unsafe_allow_html=True)
        
        with input_col:
            features[feature] = st.number_input(
                "", 
                min_value=0.0, 
                max_value=20.0 if feature == "M(wt%)" else (25.0 if feature == "Ash(wt%)" else (110.0 if feature == "VM(wt%)" else 120.0)), 
                value=value, 
                key=f"proximate_{feature}", 
                format="%.2f",
                label_visibility="collapsed"
            )

# Ultimate Analysis (黄色区域)
with col2:
    ult_container = st.container()
    ult_container.markdown("<div class='section-header' style='background-color: #DAA520;'>Ultimate Analysis</div>", unsafe_allow_html=True)
    
    for feature in feature_categories["Ultimate Analysis"]:
        if st.session_state.clear_pressed:
            value = default_values[feature]
        else:
            value = st.session_state.get(f"ultimate_{feature}", default_values[feature])
        
        # 创建两列布局
        label_col, input_col = st.columns([1, 0.5])
        
        with label_col:
            st.markdown(f"<div class='input-label' style='background-color: #DAA520;'>{feature}</div>", unsafe_allow_html=True)
        
        with input_col:
            features[feature] = st.number_input(
                "", 
                min_value=30.0 if feature in ["C(wt%)", "O(wt%)"] else 0.0, 
                max_value=110.0 if feature == "C(wt%)" else (15.0 if feature == "H(wt%)" else (5.0 if feature == "N(wt%)" else 60.0)), 
                value=value, 
                key=f"ultimate_{feature}", 
                format="%.2f",
                label_visibility="collapsed"
            )

# Pyrolysis Conditions (橙色区域)
with col3:
    pyro_container = st.container()
    pyro_container.markdown("<div class='section-header' style='background-color: #FF7F50;'>Pyrolysis Conditions</div>", unsafe_allow_html=True)
    
    for feature in feature_categories["Pyrolysis Conditions"]:
        if st.session_state.clear_pressed:
            value = default_values[feature]
        else:
            value = st.session_state.get(f"pyrolysis_{feature}", default_values[feature])
        
        min_val = 250.0 if feature == "FT(℃)" else (5.0 if feature == "RT(min)" else 0.0)
        max_val = 1100.0 if feature == "FT(℃)" else (200.0 if feature in ["SM(g)", "HR(℃/min)"] else (120.0 if feature == "FR(mL/min)" else (100.0 if feature == "RT(min)" else 20.0)))
        
        # 创建两列布局
        label_col, input_col = st.columns([1, 0.5])
        
        with label_col:
            st.markdown(f"<div class='input-label' style='background-color: #FF7F50;'>{feature}</div>", unsafe_allow_html=True)
        
        with input_col:
            features[feature] = st.number_input(
                "", 
                min_value=min_val, 
                max_value=max_val, 
                value=value, 
                key=f"pyrolysis_{feature}", 
                format="%.2f",
                label_visibility="collapsed"
            )

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
    
    # 定义Clear按钮的回调函数
    def clear_values():
        st.session_state.clear_pressed = True
        # 清除显示
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
            f"<div class='yield-result'>Yield (wt%) <br> {y_pred:.2f}</div>",
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"预测过程中出现错误: {e}")

# 如果有保存的预测结果，显示它
if 'prediction_result' in st.session_state and st.session_state.prediction_result is not None:
    prediction_placeholder.markdown(
        f"<div class='yield-result'>Yield (wt%) <br> {st.session_state.prediction_result:.2f}</div>",
        unsafe_allow_html=True
    )