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
    .input-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 10px;
    }
    .input-label {
        flex: 1;
        margin-right: 10px;
    }
    .input-field {
        flex: 0.3; /* 控制输入框的宽度 */
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

# Proximate Analysis (绿色区域) - 在第一列
with col1:
    st.markdown("<div class='proximate-section'><div class='section-title'>Proximate Analysis</div>", unsafe_allow_html=True)
    features = {}
    for feature in feature_categories["Proximate Analysis"]:
        if st.session_state.clear_pressed:
            value = default_values[feature]
        else:
            value = st.session_state.get(f"proximate_{feature}", default_values[feature])
        
        # 创建输入行
        st.markdown("<div class='input-row'>", unsafe_allow_html=True)
        st.markdown(f"<div class='input-label'>{feature}</div>", unsafe_allow_html=True)
        features[feature] = st.number_input("", min_value=0.0, max_value=20.0 if feature == "M(wt%)" else (25.0 if feature == "Ash(wt%)" else (110.0 if feature == "VM(wt%)" else 120.0)), value=value, key=f"proximate_{feature}", format="%.2f", help="Enter value", width=100)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Ultimate Analysis (黄色区域) - 在第二列
with col2:
    st.markdown("<div class='ultimate-section'><div class='section-title'>Ultimate Analysis</div>", unsafe_allow_html=True)
    for feature in feature_categories["Ultimate Analysis"]:
        if st.session_state.clear_pressed:
            value = default_values[feature]
        else:
            value = st.session_state.get(f"ultimate_{feature}", default_values[feature])
        
        # 创建输入行
        st.markdown("<div class='input-row'>", unsafe_allow_html=True)
        st.markdown(f"<div class='input-label'>{feature}</div>", unsafe_allow_html=True)
        features[feature] = st.number_input("", min_value=30.0 if feature in ["C(wt%)", "O(wt%)"] else 0.0, max_value=110.0 if feature == "C(wt%)" else (15.0 if feature == "H(wt%)" else (5.0 if feature == "N(wt%)" else 60.0)), value=value, key=f"ultimate_{feature}", format="%.2f", help="Enter value", width=100)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Pyrolysis Conditions (橙色区域) - 在第三列
with col3:
    st.markdown("<div class='pyrolysis-section'><div class='section-title'>Pyrolysis Conditions</div>", unsafe_allow_html=True)
    for feature in feature_categories["Pyrolysis Conditions"]:
        if st.session_state.clear_pressed:
            value = default_values[feature]
        else:
            value = st.session_state.get(f"pyrolysis_{feature}", default_values[feature])
        
        min_val = 250.0 if feature == "FT(℃)" else (5.0 if feature == "RT(min)" else 0.0)
        max_val = 1100.0 if feature == "FT(℃)" else (200.0 if feature in ["SM(g)", "HR(℃/min)"] else (120.0 if feature == "FR(mL/min)" else (100.0 if feature == "RT(min)" else 20.0)))
        
        # 创建输入行
        st.markdown("<div class='input-row'>", unsafe_allow_html=True)
        st.markdown(f"<div class='input-label'>{feature}</div>", unsafe_allow_html=True)
        features[feature] = st.number_input("", min_value=min_val, max_value=max_val, value=value, key=f"pyrolysis_{feature}", format="%.2f", help="Enter value", width=100)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

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
        # 尝试更新预测结果区域，清除显示
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