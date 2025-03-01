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
    .section-header {
        font-weight: bold;
        text-align: center;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 15px;
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
    .data-block {
        margin-bottom: 20px;
    }
    .data-label {
        background-color: #DAA520;
        color: white;
        font-weight: bold;
        padding: 8px 15px;
        border-radius: 4px;
        margin-bottom: 5px;
        text-align: center;
    }
    .data-value {
        background-color: #DAA520;
        color: white;
        font-weight: bold;
        padding: 8px 15px;
        border-radius: 4px;
        margin-bottom: 15px;
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
    .stButton button {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    /* 隐藏number input */
    div[data-testid="stNumberInput"] {
        position: relative;
    }
    div[data-testid="stNumberInput"] label {
        display: none;
    }
    div[data-testid="stNumberInput"] div[data-testid="stVerticalBlock"] {
        position: absolute;
        bottom: -10px;
        right: 0;
        width: 80px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 主标题
st.markdown("<h1 class='main-title'>GUI for Bio-Char Yield Prediction based on ELT-PSO Model</h1>", unsafe_allow_html=True)

# 模型选择
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

# 初始化默认值
if 'initialized' not in st.session_state:
    st.session_state['initialized'] = True
    
    # Ultimate Analysis
    st.session_state['C(wt%)'] = 52.05
    st.session_state['H(wt%)'] = 5.37
    st.session_state['N(wt%)'] = 0.49
    st.session_state['O(wt%)'] = 42.1
    
    # Proximate Analysis
    st.session_state['VM(wt%)'] = 73.5
    st.session_state['FC(wt%)'] = 13.2
    st.session_state['M(wt%)'] = 5.0
    st.session_state['Ash(wt%)'] = 8.6
    
    # Pyrolysis Conditions
    st.session_state['PS(mm)'] = 1.5
    st.session_state['FT(℃)'] = 500.0
    st.session_state['HR(℃/min)'] = 10.0
    st.session_state['FR(mL/min)'] = 2.0
    st.session_state['SM(g)'] = 75.0
    st.session_state['RT(min)'] = 30.0

# 创建三列布局
col1, col2, col3 = st.columns(3)

# Ultimate Analysis (黄色区域) - 第一列
with col1:
    st.markdown("<div class='section-header ultimate-header'>Ultimate Analysis</div>", unsafe_allow_html=True)
    
    # C(wt%)
    st.markdown("<div class='data-label ultimate-label'>C(wt%)</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='data-value ultimate-value'>{st.session_state['C(wt%)']}</div>", unsafe_allow_html=True)
    c_value = st.number_input("C input", value=st.session_state['C(wt%)'], min_value=30.0, max_value=110.0, step=0.01, key="c_input")
    st.session_state['C(wt%)'] = c_value
    
    # H(wt%)
    st.markdown("<div class='data-label ultimate-label'>H(wt%)</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='data-value ultimate-value'>{st.session_state['H(wt%)']}</div>", unsafe_allow_html=True)
    h_value = st.number_input("H input", value=st.session_state['H(wt%)'], min_value=0.0, max_value=15.0, step=0.01, key="h_input")
    st.session_state['H(wt%)'] = h_value
    
    # N(wt%)
    st.markdown("<div class='data-label ultimate-label'>N(wt%)</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='data-value ultimate-value'>{st.session_state['N(wt%)']}</div>", unsafe_allow_html=True)
    n_value = st.number_input("N input", value=st.session_state['N(wt%)'], min_value=0.0, max_value=5.0, step=0.01, key="n_input")
    st.session_state['N(wt%)'] = n_value
    
    # O(wt%)
    st.markdown("<div class='data-label ultimate-label'>O(wt%)</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='data-value ultimate-value'>{st.session_state['O(wt%)']}</div>", unsafe_allow_html=True)
    o_value = st.number_input("O input", value=st.session_state['O(wt%)'], min_value=30.0, max_value=60.0, step=0.01, key="o_input")
    st.session_state['O(wt%)'] = o_value

# Proximate Analysis (绿色区域) - 第二列
with col2:
    st.markdown("<div class='section-header proximate-header'>Proximate Analysis</div>", unsafe_allow_html=True)
    
    # VM(wt%)
    st.markdown("<div class='data-label proximate-label'>VM(wt%)</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='data-value proximate-value'>{st.session_state['VM(wt%)']}</div>", unsafe_allow_html=True)
    vm_value = st.number_input("VM input", value=st.session_state['VM(wt%)'], min_value=0.0, max_value=110.0, step=0.01, key="vm_input")
    st.session_state['VM(wt%)'] = vm_value
    
    # FC(wt%)
    st.markdown("<div class='data-label proximate-label'>FC(wt%)</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='data-value proximate-value'>{st.session_state['FC(wt%)']}</div>", unsafe_allow_html=True)
    fc_value = st.number_input("FC input", value=st.session_state['FC(wt%)'], min_value=0.0, max_value=120.0, step=0.01, key="fc_input")
    st.session_state['FC(wt%)'] = fc_value
    
    # Ash(wt%)
    st.markdown("<div class='data-label proximate-label'>Ash(wt%)</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='data-value proximate-value'>{st.session_state['Ash(wt%)']}</div>", unsafe_allow_html=True)
    ash_value = st.number_input("Ash input", value=st.session_state['Ash(wt%)'], min_value=0.0, max_value=25.0, step=0.01, key="ash_input")
    st.session_state['Ash(wt%)'] = ash_value
    
    # M(wt%)
    st.markdown("<div class='data-label proximate-label'>M(wt%)</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='data-value proximate-value'>{st.session_state['M(wt%)']}</div>", unsafe_allow_html=True)
    m_value = st.number_input("M input", value=st.session_state['M(wt%)'], min_value=0.0, max_value=20.0, step=0.01, key="m_input")
    st.session_state['M(wt%)'] = m_value

# Pyrolysis Conditions (橙色区域) - 第三列
with col3:
    st.markdown("<div class='section-header pyrolysis-header'>Pyrolysis Condition</div>", unsafe_allow_html=True)
    
    # Temperature (℃)
    st.markdown("<div class='data-label pyrolysis-label'>Temperature (℃)</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='data-value pyrolysis-value'>{st.session_state['FT(℃)']}</div>", unsafe_allow_html=True)
    temp_value = st.number_input("Temp input", value=st.session_state['FT(℃)'], min_value=250.0, max_value=1100.0, step=10.0, key="temp_input")
    st.session_state['FT(℃)'] = temp_value
    
    # Heating Rate (℃/min)
    st.markdown("<div class='data-label pyrolysis-label'>Heating Rate (℃/min)</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='data-value pyrolysis-value'>{st.session_state['HR(℃/min)']}</div>", unsafe_allow_html=True)
    hr_value = st.number_input("HR input", value=st.session_state['HR(℃/min)'], min_value=0.0, max_value=200.0, step=1.0, key="hr_input")
    st.session_state['HR(℃/min)'] = hr_value
    
    # Particle Size (mm)
    st.markdown("<div class='data-label pyrolysis-label'>Particle Size (mm)</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='data-value pyrolysis-value'>{st.session_state['PS(mm)']}</div>", unsafe_allow_html=True)
    ps_value = st.number_input("PS input", value=st.session_state['PS(mm)'], min_value=0.0, max_value=20.0, step=0.1, key="ps_input")
    st.session_state['PS(mm)'] = ps_value
    
    # N2 Flow (L/min)
    st.markdown("<div class='data-label pyrolysis-label'>N2 Flow (L/min)</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='data-value pyrolysis-value'>{st.session_state['FR(mL/min)']}</div>", unsafe_allow_html=True)
    fr_value = st.number_input("FR input", value=st.session_state['FR(mL/min)'], min_value=0.0, max_value=120.0, step=0.1, key="fr_input")
    st.session_state['FR(mL/min)'] = fr_value

# 预测结果显示区域和按钮
result_col, button_col1, button_col2 = st.columns([3, 1, 1])

with result_col:
    prediction_placeholder = st.empty()
    
with button_col1:
    predict_button = st.button("PUSH")
    
with button_col2:
    # 定义Clear按钮的回调函数
    def clear_values():
        # 重置为默认值
        st.session_state['C(wt%)'] = 52.05
        st.session_state['H(wt%)'] = 5.37
        st.session_state['N(wt%)'] = 0.49
        st.session_state['O(wt%)'] = 42.1
        st.session_state['VM(wt%)'] = 73.5
        st.session_state['FC(wt%)'] = 13.2
        st.session_state['M(wt%)'] = 5.0
        st.session_state['Ash(wt%)'] = 8.6
        st.session_state['PS(mm)'] = 1.5
        st.session_state['FT(℃)'] = 500.0
        st.session_state['HR(℃/min)'] = 10.0
        st.session_state['FR(mL/min)'] = 2.0
        st.session_state['SM(g)'] = 75.0
        st.session_state['RT(min)'] = 30.0
        
        # 清除预测结果
        if 'prediction_result' in st.session_state:
            del st.session_state['prediction_result']
    
    clear_button = st.button("CLEAR", on_click=clear_values)

# 处理预测逻辑
if predict_button:
    try:
        # 构建特征字典
        features = {
            'C(wt%)': st.session_state['C(wt%)'],
            'H(wt%)': st.session_state['H(wt%)'],
            'N(wt%)': st.session_state['N(wt%)'],
            'O(wt%)': st.session_state['O(wt%)'],
            'VM(wt%)': st.session_state['VM(wt%)'],
            'FC(wt%)': st.session_state['FC(wt%)'],
            'M(wt%)': st.session_state['M(wt%)'],
            'Ash(wt%)': st.session_state['Ash(wt%)'],
            'PS(mm)': st.session_state['PS(mm)'],
            'FT(℃)': st.session_state['FT(℃)'],
            'HR(℃/min)': st.session_state['HR(℃/min)'],
            'FR(mL/min)': st.session_state['FR(mL/min)'],
            'SM(g)': st.session_state['SM(g)'],
            'RT(min)': st.session_state['RT(min)']
        }
        
        # 转换为DataFrame
        input_data = pd.DataFrame([features])
        
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