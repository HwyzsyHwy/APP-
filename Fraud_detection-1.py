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
    }
    .section-header {
        font-weight: bold;
        text-align: center;
        padding: 10px;
        border-radius: 4px;
        margin-bottom: 10px;
        color: white;
    }
    .ultimate-header {
        background-color: #DAA520;
    }
    .proximate-header {
        background-color: #32CD32;
    }
    .pyrolysis-header {
        background-color: #FF7F50;
    }
    .data-row {
        display: flex;
        margin-bottom: 10px;
    }
    .data-label {
        width: 40%;
        color: white;
        font-weight: bold;
        padding: 8px;
        border-radius: 4px;
        text-align: center;
    }
    .data-value {
        width: 58%;
        color: white;
        font-weight: bold;
        padding: 8px;
        border-radius: 4px;
        text-align: center;
        margin-left: 2%;
    }
    .ultimate-label, .ultimate-value {
        background-color: #DAA520;
    }
    .proximate-label, .proximate-value {
        background-color: #32CD32;
    }
    .pyrolysis-label, .pyrolysis-value {
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
    .stButton button {
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    /* 隐藏Streamlit默认元素 */
    div[data-testid="stNumberInput"] label {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 主标题
st.markdown("<h1 class='main-title'>GUI for Bio-Char Yield Prediction based on ELT-PSO Model</h1>", unsafe_allow_html=True)

# 模型选择放置在顶部但不太突出
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

# 初始化会话状态，用于保存是否需要清除数据
if 'clear_pressed' not in st.session_state:
    st.session_state.clear_pressed = False

# 创建三列布局
col1, col2, col3 = st.columns(3)

# Ultimate Analysis (黄色区域) - 第一列
with col1:
    st.markdown("<div class='section-header ultimate-header'>Ultimate Analysis</div>", unsafe_allow_html=True)
    
    # C(wt%)
    st.markdown("<div class='data-row'><div class='data-label ultimate-label'>C (%)</div><div class='data-value ultimate-value'>52.05</div></div>", unsafe_allow_html=True)
    c_value = st.number_input("C(wt%)", min_value=30.0, max_value=110.0, value=52.05, step=0.01, key="c_input")
    
    # H(wt%)
    st.markdown("<div class='data-row'><div class='data-label ultimate-label'>H (%)</div><div class='data-value ultimate-value'>5.37</div></div>", unsafe_allow_html=True)
    h_value = st.number_input("H(wt%)", min_value=0.0, max_value=15.0, value=5.37, step=0.01, key="h_input")
    
    # N(wt%)
    st.markdown("<div class='data-row'><div class='data-label ultimate-label'>N (%)</div><div class='data-value ultimate-value'>0.49</div></div>", unsafe_allow_html=True)
    n_value = st.number_input("N(wt%)", min_value=0.0, max_value=5.0, value=0.49, step=0.01, key="n_input")
    
    # O(wt%)
    st.markdown("<div class='data-row'><div class='data-label ultimate-label'>O (%)</div><div class='data-value ultimate-value'>42.1</div></div>", unsafe_allow_html=True)
    o_value = st.number_input("O(wt%)", min_value=30.0, max_value=60.0, value=42.1, step=0.01, key="o_input")

# Proximate Analysis (绿色区域) - 第二列
with col2:
    st.markdown("<div class='section-header proximate-header'>Proximate Analysis</div>", unsafe_allow_html=True)
    
    # FC(wt%)
    st.markdown("<div class='data-row'><div class='data-label proximate-label'>FC (%)</div><div class='data-value proximate-value'>13.2</div></div>", unsafe_allow_html=True)
    fc_value = st.number_input("FC(wt%)", min_value=0.0, max_value=120.0, value=13.2, step=0.01, key="fc_input")
    
    # VM(wt%)
    st.markdown("<div class='data-row'><div class='data-label proximate-label'>VM (%)</div><div class='data-value proximate-value'>73.5</div></div>", unsafe_allow_html=True)
    vm_value = st.number_input("VM(wt%)", min_value=0.0, max_value=110.0, value=73.5, step=0.01, key="vm_input")
    
    # M(wt%)
    st.markdown("<div class='data-row'><div class='data-label proximate-label'>MC (%)</div><div class='data-value proximate-value'>4.7</div></div>", unsafe_allow_html=True)
    m_value = st.number_input("M(wt%)", min_value=0.0, max_value=20.0, value=4.7, step=0.01, key="m_input")
    
    # Ash(wt%)
    st.markdown("<div class='data-row'><div class='data-label proximate-label'>Ash (%)</div><div class='data-value proximate-value'>8.6</div></div>", unsafe_allow_html=True)
    ash_value = st.number_input("Ash(wt%)", min_value=0.0, max_value=25.0, value=8.6, step=0.01, key="ash_input")

# Pyrolysis Conditions (橙色区域) - 第三列
with col3:
    st.markdown("<div class='section-header pyrolysis-header'>Pyrolysis Condition</div>", unsafe_allow_html=True)
    
    # Temperature (℃)
    st.markdown("<div class='data-row'><div class='data-label pyrolysis-label'>Temperature (C)</div><div class='data-value pyrolysis-value'>500</div></div>", unsafe_allow_html=True)
    temp_value = st.number_input("FT(℃)", min_value=250.0, max_value=1100.0, value=500.0, step=10.0, key="temp_input")
    
    # Heating Rate (℃/min)
    st.markdown("<div class='data-row'><div class='data-label pyrolysis-label'>Heating Rate (C/min)</div><div class='data-value pyrolysis-value'>10</div></div>", unsafe_allow_html=True)
    hr_value = st.number_input("HR(℃/min)", min_value=0.0, max_value=200.0, value=10.0, step=1.0, key="hr_input")
    
    # Particle Size (mm)
    st.markdown("<div class='data-row'><div class='data-label pyrolysis-label'>Particle Size (mm)</div><div class='data-value pyrolysis-value'>1.5</div></div>", unsafe_allow_html=True)
    ps_value = st.number_input("PS(mm)", min_value=0.0, max_value=20.0, value=1.5, step=0.1, key="ps_input")
    
    # N2 Flow (L/min)
    st.markdown("<div class='data-row'><div class='data-label pyrolysis-label'>N2 Flow (L/min)</div><div class='data-value pyrolysis-value'>2</div></div>", unsafe_allow_html=True)
    fr_value = st.number_input("FR(mL/min)", min_value=0.0, max_value=120.0, value=2.0, step=0.1, key="fr_input")

# 为其他必要的特征提供默认值
sm_value = 75.0  # Sample mass
rt_value = 30.0  # Residence time

# 转换为DataFrame
features = {
    'C(wt%)': c_value,
    'H(wt%)': h_value,
    'N(wt%)': n_value,
    'O(wt%)': o_value,
    'VM(wt%)': vm_value,
    'FC(wt%)': fc_value,
    'M(wt%)': m_value,
    'Ash(wt%)': ash_value,
    'PS(mm)': ps_value,
    'FT(℃)': temp_value,
    'HR(℃/min)': hr_value,
    'FR(mL/min)': fr_value,
    'SM(g)': sm_value,
    'RT(min)': rt_value
}

input_data = pd.DataFrame([features])

# 预测结果显示区域和按钮
result_col, button_col1, button_col2 = st.columns([3, 1, 1])

with result_col:
    prediction_placeholder = st.empty()

# 显示按钮
with button_col1:
    predict_button = st.button("PUSH", key="predict")
    
with button_col2:
    # 定义Clear按钮的回调函数
    def clear_values():
        st.session_state.clear_pressed = True
        # 清除预测结果
        if 'prediction_result' in st.session_state:
            del st.session_state['prediction_result']
    
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
        st.session_state['prediction_result'] = y_pred

        # 显示预测结果
        prediction_placeholder.markdown(
            f"<div class='yield-result'>Biochar Yield (%) <br> {y_pred:.2f}</div>",
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"预测过程中出现错误: {e}")

# 如果有保存的预测结果，显示它
if 'prediction_result' in st.session_state:
    prediction_placeholder.markdown(
        f"<div class='yield-result'>Biochar Yield (%) <br> {st.session_state['prediction_result']:.2f}</div>",
        unsafe_allow_html=True
    )

# 重置session_state中的clear_pressed状态
if st.session_state.clear_pressed:
    st.session_state.clear_pressed = False
    st.rerun()  # 使用st.rerun()替代experimental_rerun