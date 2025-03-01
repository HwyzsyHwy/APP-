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

/* 设置输入框的背景颜色 */
input[type="number"] {
    background-color: green !important;
    color: white !important;
}

/* 设置第一列(Proximate Analysis)的输入框和标签颜色 */
.proximate-input input[type="number"], .proximate-label {
    background-color: #4CAF50 !important; /* 绿色 */
    color: white !important;
    padding: 8px !important;
    border-radius: 4px !important;
}

/* 设置第二列(Ultimate Analysis)的输入框和标签颜色 */
.ultimate-input input[type="number"], .ultimate-label {
    background-color: #FFEB3B !important; /* 黄色 */
    color: black !important;
    padding: 8px !important;
    border-radius: 4px !important;
}

/* 设置第三列(Pyrolysis Conditions)的输入框和标签颜色 */
.pyrolysis-input input[type="number"], .pyrolysis-label {
    background-color: #FF9800 !important; /* 橙色 */
    color: black !important;
    padding: 8px !important;
    border-radius: 4px !important;
}

/* 设置标题样式 */
.title-container {
    background-color: #333;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 20px;
}

/* 设置分析类型容器样式 */
.analysis-container {
    border: 1px solid #444;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 15px;
}

/* 设置按钮样式 */
.stButton button {
    background-color: #4CAF50;
    color: white;
    border: none;
    padding: 10px 24px;
    border-radius: 4px;
    cursor: pointer;
}

.stButton button:hover {
    background-color: #45a049;
}

/* 清除按钮样式 */
.clear-button button {
    background-color: #f44336;
    color: white;
}

.clear-button button:hover {
    background-color: #d32f2f;
}

/* 数据行样式 */
.data-row {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
}

.data-label {
    flex: 1;
    padding: 8px;
    border-radius: 4px;
}

.data-input {
    flex: 1;
}

</style>
""", unsafe_allow_html=True)

# 标题
st.markdown("<div class='title-container'><h1 style='text-align: center; color: white;'>生物质热解产率预测</h1></div>", unsafe_allow_html=True)

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
    st.markdown("<div class='analysis-container' style='background-color: rgba(76, 175, 80, 0.2);'><h3 style='color: #4CAF50;'>Proximate Analysis</h3>", unsafe_allow_html=True)
    
    # 为每个输入字段创建行
    for label, key in [("M(wt%)", "M"), ("Ash(wt%)", "Ash"), ("VM(wt%)", "VM"), ("FC(wt%)", "FC")]:
        st.markdown(f"""
        <div class='data-row'>
            <div class='data-label proximate-label'>{label}</div>
            <div class='data-input'>
                <div class='proximate-input'>
                    <!-- 这里将通过Streamlit注入输入框 -->
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 使用相同的key插入输入框，但使其隐藏标签
        st.number_input("", 
                        min_value=0.0, 
                        max_value=100.0, 
                        value=default_values[key], 
                        step=0.1, 
                        key=key,
                        label_visibility="collapsed")
    
    st.markdown("</div>", unsafe_allow_html=True)

# 第二列: 元素分析
with col2:
    st.markdown("<div class='analysis-container' style='background-color: rgba(255, 235, 59, 0.2);'><h3 style='color: #FFEB3B;'>Ultimate Analysis</h3>", unsafe_allow_html=True)
    
    for label, key in [("C(wt%)", "C"), ("H(wt%)", "H"), ("O(wt%)", "O"), ("N(wt%)", "N"), ("S(wt%)", "S")]:
        st.markdown(f"""
        <div class='data-row'>
            <div class='data-label ultimate-label'>{label}</div>
            <div class='data-input'>
                <div class='ultimate-input'>
                    <!-- 这里将通过Streamlit注入输入框 -->
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 使用相同的key插入输入框，但使其隐藏标签
        st.number_input("", 
                        min_value=0.0, 
                        max_value=100.0, 
                        value=default_values[key], 
                        step=0.1, 
                        key=key,
                        label_visibility="collapsed")
    
    st.markdown("</div>", unsafe_allow_html=True)

# 第三列: 热解条件
with col3:
    st.markdown("<div class='analysis-container' style='background-color: rgba(255, 152, 0, 0.2);'><h3 style='color: #FF9800;'>Pyrolysis Conditions</h3>", unsafe_allow_html=True)
    
    for label, key, max_val in [
        ("Temperature(°C)", "Temperature", 1000.0), 
        ("Heating Rate(°C/min)", "Heating_Rate", 100.0), 
        ("Holding Time(min)", "Holding_Time", 120.0)
    ]:
        st.markdown(f"""
        <div class='data-row'>
            <div class='data-label pyrolysis-label'>{label}</div>
            <div class='data-input'>
                <div class='pyrolysis-input'>
                    <!-- 这里将通过Streamlit注入输入框 -->
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 使用相同的key插入输入框，但使其隐藏标签
        st.number_input("", 
                        min_value=0.0, 
                        max_value=max_val, 
                        value=default_values[key], 
                        step=0.1, 
                        key=key,
                        label_visibility="collapsed")
    
    st.markdown("</div>", unsafe_allow_html=True)

# 按钮行
col1, col2 = st.columns(2)

# 预测按钮
with col1:
    push_button = st.button("PUSH", use_container_width=True)

# 清除按钮
with col2:
    st.markdown("<div class='clear-button'>", unsafe_allow_html=True)
    clear_button = st.button("CLEAR", on_click=clear_inputs, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# 预测逻辑
if push_button:
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
        prediction = 35.5
        
        # 保存预测结果到会话状态
        st.session_state.prediction_result = prediction
    except Exception as e:
        st.error(f"预测过程中出错: {str(e)}")

# 显示预测结果
if st.session_state.prediction_result is not None:
    st.markdown(
        f"<div style='background-color: #333; padding: 20px; border-radius: 5px; margin-top: 20px;'>"
        f"<h3 style='text-align: center; color: white;'>Yield (%): {st.session_state.prediction_result:.2f}</h3>"
        f"</div>",
        unsafe_allow_html=True
    )