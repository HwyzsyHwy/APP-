import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os

# 设置页面配置
st.set_page_config(page_title="生物质热解产率预测", layout="wide")

# 自定义CSS样式 - 完全重写
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

/* 直接复制截图中的布局样式 */
.label-box {
    width: 100%;
    background-color: #4CAF50;
    color: white;
    padding: 8px 12px;
    border-radius: 4px;
    margin-bottom: 5px;
}

.input-box {
    width: 100%;
    background-color: #202123;
    color: white;
    padding: 8px 12px;
    border-radius: 4px;
    margin-bottom: 15px;
    border: none;
}

.ultimate-label {
    background-color: #FFEB3B !important;
    color: black !important;
}

.pyrolysis-label {
    background-color: #FF9800 !important;
    color: black !important;
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
.push-button {
    background-color: #4CAF50;
    color: white;
    border: none;
    width: 100%;
    padding: 12px;
    text-align: center;
    font-weight: bold;
    border-radius: 4px;
    cursor: pointer;
}

.clear-button {
    background-color: #4CAF50;
    color: white;
    border: none;
    width: 100%;
    padding: 12px;
    text-align: center;
    font-weight: bold;
    border-radius: 4px;
    cursor: pointer;
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

/* 隐藏Streamlit默认元素 */
.block-container {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}

/* 移除输入框的边框和轮廓 */
input[type=number] {
    border: none !important;
    outline: none !important;
    width: 100% !important;
}

/* 确保输入框能够继承背景色 */
.stNumberInput {
    background-color: transparent !important;
}

.stNumberInput div {
    background-color: transparent !important;
}

/* 隐藏所有+-按钮 */
.stNumberInput button {
    display: none !important;
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
    
    # 直接使用HTML创建标签
    for label, key in [("M(wt%)", "M"), ("Ash(wt%)", "Ash"), ("VM(wt%)", "VM"), ("FC(wt%)", "FC")]:
        st.markdown(f"<div class='label-box'>{label}</div>", unsafe_allow_html=True)
        
        # 隐藏标签的输入框
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
        st.markdown(f"<div class='label-box ultimate-label'>{label}</div>", unsafe_allow_html=True)
        
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
        st.markdown(f"<div class='label-box pyrolysis-label'>{label}</div>", unsafe_allow_html=True)
        
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

# 注入自定义JavaScript以隐藏加减按钮和调整输入样式
st.markdown("""
<script>
// 等待页面加载完成
document.addEventListener('DOMContentLoaded', function() {
    // 获取所有数字输入框
    const inputs = document.querySelectorAll('input[type="number"]');
    
    // 修改每个输入框
    inputs.forEach(input => {
        // 隐藏spinners
        input.style.appearance = 'textfield';
        input.style.MozAppearance = 'textfield';
        input.style.webkitAppearance = 'none';
        
        // 设置背景颜色
        let parentDiv = input.closest('.stNumberInput');
        if (parentDiv) {
            parentDiv.style.backgroundColor = '#202123';
            parentDiv.style.borderRadius = '4px';
            parentDiv.style.padding = '8px';
        }
    });
}, false);
</script>
""", unsafe_allow_html=True)