import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os

# 设置页面配置
st.set_page_config(page_title="生物质热解产率预测", layout="wide")

# 自定义CSS样式
st.markdown("""
<style>
/* 整体背景颜色 */
.stApp {
    background-color: #1E1E1E;
    color: white;
}

/* 隐藏加减按钮 */
input[type=number]::-webkit-outer-spin-button,
input[type=number]::-webkit-inner-spin-button {
    -webkit-appearance: none !important;
    margin: 0 !important;
}

input[type=number] {
    -moz-appearance: textfield !important;
}

/* 标签和输入框的共享容器 */
.input-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    width: 100%;
    padding: 8px 12px;
    border-radius: 4px;
    margin-bottom: 15px;
    box-sizing: border-box;
}

/* 标签样式 */
.label-text {
    flex: 1;
    font-weight: normal;
}

/* 输入框样式 */
.input-value {
    text-align: right;
    width: auto;
    background: transparent;
    border: none;
    color: inherit;
}

/* 颜色设置 */
.proximate-row {
    background-color: #4CAF50;
    color: white;
}

.ultimate-row {
    background-color: #FFEB3B;
    color: black;
}

.pyrolysis-row {
    background-color: #FF9800;
    color: black;
}

/* 标题样式 */
.title-container {
    background-color: #333;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 20px;
    text-align: center;
}

/* 分析类型标题 */
.section-title {
    width: 100%;
    padding: 10px;
    border-radius: 4px;
    margin-bottom: 15px;
    font-weight: bold;
    font-size: 18px;
}

.proximate-title {
    background-color: rgba(76, 175, 80, 0.2);
    color: #4CAF50;
}

.ultimate-title {
    background-color: rgba(255, 235, 59, 0.2);
    color: #FFEB3B;
}

.pyrolysis-title {
    background-color: rgba(255, 152, 0, 0.2);
    color: #FF9800;
}

/* 按钮样式 */
.stButton > button {
    background-color: #4CAF50;
    color: white;
    border: none;
    width: 100%;
    padding: 12px;
    font-weight: bold;
    border-radius: 4px;
}

/* 隐藏Streamlit元素 */
div.stNumberInput > div {
    display: none;
}

/* 自定义输入容器 */
.custom-input-container {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

.custom-input-container input {
    background: transparent !important;
    border: none !important;
    color: inherit !important;
    text-align: right;
    padding: 0 !important;
    width: 70px !important;
}

/* 预测结果 */
.result-container {
    background-color: #333;
    padding: 20px;
    border-radius: 5px;
    margin-top: 20px;
    text-align: center;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# 标题
st.markdown("<div class='title-container'><h1>生物质热解产率预测</h1></div>", unsafe_allow_html=True)

# 创建三列布局
col1, col2, col3 = st.columns(3)

# 初始化会话状态（如果尚未初始化）
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# 默认值字典
default_values = {
    'M': 5.0, 'Ash': 8.0, 'VM': 75.0, 'FC': 15.0,
    'C': 45.0, 'H': 5.5, 'O': 45.0, 'N': 0.5, 'S': 0.1,
    'Temperature': 500.0, 'Heating_Rate': 10.0, 'Holding_Time': 30.0
}

# 清除函数
def clear_inputs():
    for key in default_values:
        st.session_state[key] = default_values[key]
    st.session_state.prediction_result = None

# 第一列: 近似分析
with col1:
    st.markdown("<div class='section-title proximate-title'>Proximate Analysis</div>", unsafe_allow_html=True)
    
    # 创建行布局，标签和输入框在同一行
    for label, key in [("M(wt%)", "M"), ("Ash(wt%)", "Ash"), ("VM(wt%)", "VM"), ("FC(wt%)", "FC")]:
        # 开始输入行
        st.markdown(f"<div class='input-row proximate-row'><div class='label-text'>{label}</div><div class='input-value'></div></div>", unsafe_allow_html=True)
        
        # 添加输入框（隐藏）
        st.number_input("", 
                      min_value=0.0, 
                      max_value=100.0, 
                      value=default_values[key], 
                      step=0.1, 
                      key=key,
                      label_visibility="collapsed")

# 第二列: 元素分析
with col2:
    st.markdown("<div class='section-title ultimate-title'>Ultimate Analysis</div>", unsafe_allow_html=True)
    
    for label, key in [("C(wt%)", "C"), ("H(wt%)", "H"), ("O(wt%)", "O"), ("N(wt%)", "N"), ("S(wt%)", "S")]:
        # 开始输入行
        st.markdown(f"<div class='input-row ultimate-row'><div class='label-text'>{label}</div><div class='input-value'></div></div>", unsafe_allow_html=True)
        
        # 添加输入框（隐藏）
        st.number_input("", 
                      min_value=0.0, 
                      max_value=100.0, 
                      value=default_values[key], 
                      step=0.1, 
                      key=key,
                      label_visibility="collapsed")

# 第三列: 热解条件
with col3:
    st.markdown("<div class='section-title pyrolysis-title'>Pyrolysis Conditions</div>", unsafe_allow_html=True)
    
    for label, key, max_val in [
        ("Temperature(°C)", "Temperature", 1000.0), 
        ("Heating Rate(°C/min)", "Heating_Rate", 100.0), 
        ("Holding Time(min)", "Holding_Time", 120.0)
    ]:
        # 开始输入行
        st.markdown(f"<div class='input-row pyrolysis-row'><div class='label-text'>{label}</div><div class='input-value'></div></div>", unsafe_allow_html=True)
        
        # 添加输入框（隐藏）
        st.number_input("", 
                      min_value=0.0, 
                      max_value=max_val, 
                      value=default_values[key], 
                      step=0.1, 
                      key=key,
                      label_visibility="collapsed")

# 按钮行
col1, col2 = st.columns(2)

# 预测按钮
with col1:
    if st.button("PUSH", key="push_btn", use_container_width=True):
        try:
            # 收集所有输入并创建一个特征向量
            features = [
                st.session_state.M, st.session_state.Ash, st.session_state.VM, st.session_state.FC,
                st.session_state.C, st.session_state.H, st.session_state.O, st.session_state.N, st.session_state.S,
                st.session_state.Temperature, st.session_state.Heating_Rate, st.session_state.Holding_Time
            ]
            
            # 这里应该有模型加载和预测的逻辑
            # 例如: model = joblib.load('model.pkl')
            # 这里我们假设模型预测结果为35.5
            st.session_state.prediction_result = 35.5
        except Exception as e:
            st.error(f"预测过程中出错: {str(e)}")

# 清除按钮
with col2:
    if st.button("CLEAR", key="clear_btn", on_click=clear_inputs, use_container_width=True):
        pass

# 显示预测结果
if st.session_state.prediction_result is not None:
    st.markdown(
        f"<div class='result-container'>"
        f"<h3>Yield (%): {st.session_state.prediction_result:.2f}</h3>"
        f"</div>",
        unsafe_allow_html=True
    )

# 注入自定义JavaScript以移动输入框到正确位置
st.markdown("""
<script>
// 在页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 等待Streamlit完全加载
    setTimeout(function() {
        // 获取所有输入框
        const inputs = document.querySelectorAll('input[type="number"]');
        // 获取所有准备放置输入框的容器
        const containers = document.querySelectorAll('.input-value');
        
        // 确保数量匹配
        if(inputs.length === containers.length) {
            for(let i = 0; i < inputs.length; i++) {
                // 移动输入框到目标容器
                containers[i].appendChild(inputs[i]);
                // 设置样式
                inputs[i].style.background = 'transparent';
                inputs[i].style.border = 'none';
                inputs[i].style.color = 'inherit';
                inputs[i].style.textAlign = 'right';
                inputs[i].style.width = '70px';
                inputs[i].style.padding = '0';
            }
        }
    }, 1000); // 延迟1秒，确保Streamlit元素已加载
}, false);
</script>
""", unsafe_allow_html=True)