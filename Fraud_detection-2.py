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

/* 隐藏Streamlit原生输入框样式 */
div.stNumberInput {
    position: absolute;
    left: -9999px;
}

/* 标签行样式 */
.input-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    width: 100%;
    padding: 10px 15px;
    border-radius: 4px;
    margin-bottom: 15px;
    box-sizing: border-box;
}

/* 输入框样式 */
.custom-input {
    width: 60px;
    padding: 2px 4px;
    background-color: rgba(0, 0, 0, 0.2);
    border: none;
    border-radius: 3px;
    color: inherit;
    text-align: center;
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

# 初始化session_state
if 'input_values' not in st.session_state:
    st.session_state.input_values = {
        'M': 5.0, 'Ash': 8.0, 'VM': 75.0, 'FC': 15.0,
        'C': 45.0, 'H': 5.5, 'O': 45.0, 'N': 0.5, 'S': 0.1,
        'Temperature': 500.0, 'Heating_Rate': 10.0, 'Holding_Time': 30.0
    }

if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# 默认值字典（用于重置）
default_values = {
    'M': 5.0, 'Ash': 8.0, 'VM': 75.0, 'FC': 15.0,
    'C': 45.0, 'H': 5.5, 'O': 45.0, 'N': 0.5, 'S': 0.1,
    'Temperature': 500.0, 'Heating_Rate': 10.0, 'Holding_Time': 30.0
}

# 更新输入值的函数
def update_value(key, value):
    try:
        # 尝试将输入转换为浮点数
        st.session_state.input_values[key] = float(value)
    except:
        # 如果转换失败，保持原值不变
        pass

# 清除函数
def clear_inputs():
    st.session_state.input_values = default_values.copy()
    st.session_state.prediction_result = None

# 第一列: 近似分析
with col1:
    st.markdown("<div class='section-title proximate-title'>Proximate Analysis</div>", unsafe_allow_html=True)
    
    # 使用自定义HTML和JavaScript创建带输入框的彩色标签
    for label, key in [("M(wt%)", "M"), ("Ash(wt%)", "Ash"), ("VM(wt%)", "VM"), ("FC(wt%)", "FC")]:
        st.markdown(f"""
        <div class='input-row proximate-row'>
            <span>{label}</span>
            <input type='number' class='custom-input' id='{key}_input' value='{st.session_state.input_values[key]}' 
                   onchange="updateStreamlit(this.id, this.value)" step="0.1" min="0" max="100">
        </div>
        """, unsafe_allow_html=True)
        
        # 隐藏的Streamlit输入用于存储值（但不在UI中显示）
        st.number_input("", 
                      min_value=0.0, 
                      max_value=100.0, 
                      value=st.session_state.input_values[key], 
                      step=0.1, 
                      key=key,
                      label_visibility="collapsed")

# 第二列: 元素分析
with col2:
    st.markdown("<div class='section-title ultimate-title'>Ultimate Analysis</div>", unsafe_allow_html=True)
    
    for label, key in [("C(wt%)", "C"), ("H(wt%)", "H"), ("O(wt%)", "O"), ("N(wt%)", "N"), ("S(wt%)", "S")]:
        st.markdown(f"""
        <div class='input-row ultimate-row'>
            <span>{label}</span>
            <input type='number' class='custom-input' id='{key}_input' value='{st.session_state.input_values[key]}' 
                   onchange="updateStreamlit(this.id, this.value)" step="0.1" min="0" max="100">
        </div>
        """, unsafe_allow_html=True)
        
        # 隐藏的Streamlit输入用于存储值
        st.number_input("", 
                      min_value=0.0, 
                      max_value=100.0, 
                      value=st.session_state.input_values[key], 
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
        st.markdown(f"""
        <div class='input-row pyrolysis-row'>
            <span>{label}</span>
            <input type='number' class='custom-input' id='{key}_input' value='{st.session_state.input_values[key]}' 
                   onchange="updateStreamlit(this.id, this.value)" step="0.1" min="0" max="{max_val}">
        </div>
        """, unsafe_allow_html=True)
        
        # 隐藏的Streamlit输入用于存储值
        st.number_input("", 
                      min_value=0.0, 
                      max_value=max_val, 
                      value=st.session_state.input_values[key], 
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
                st.session_state.input_values['M'], 
                st.session_state.input_values['Ash'], 
                st.session_state.input_values['VM'], 
                st.session_state.input_values['FC'],
                st.session_state.input_values['C'], 
                st.session_state.input_values['H'], 
                st.session_state.input_values['O'], 
                st.session_state.input_values['N'], 
                st.session_state.input_values['S'],
                st.session_state.input_values['Temperature'], 
                st.session_state.input_values['Heating_Rate'], 
                st.session_state.input_values['Holding_Time']
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

# 添加JavaScript以实现自定义输入框与Streamlit的交互
st.markdown("""
<script>
// 函数：当自定义输入框值改变时更新Streamlit
function updateStreamlit(id, value) {
    // 从输入框ID中提取键名
    const key = id.split('_')[0];
    
    // 找到对应的隐藏Streamlit输入框
    const streamlitInputs = document.querySelectorAll('.stNumberInput input');
    for (let input of streamlitInputs) {
        const inputKey = input.getAttribute('aria-label') || '';
        if (inputKey === key || inputKey.includes(key)) {
            // 更新Streamlit输入框的值
            input.value = value;
            
            // 触发change事件，通知Streamlit值已更改
            const event = new Event('change', { bubbles: true });
            input.dispatchEvent(event);
            
            // 模拟按下Enter键，提交更改
            const keyEvent = new KeyboardEvent('keydown', {
                key: 'Enter',
                code: 'Enter',
                keyCode: 13,
                which: 13,
                bubbles: true
            });
            input.dispatchEvent(keyEvent);
            
            break;
        }
    }
}

// 监听Streamlit重新渲染
const observer = new MutationObserver(function(mutations) {
    // 当页面内容变化时，重新绑定自定义输入框的值
    const customInputs = document.querySelectorAll('.custom-input');
    customInputs.forEach(input => {
        const key = input.id.split('_')[0];
        // 在这里可以添加代码来同步Streamlit的值到自定义输入框
    });
});

// 开始观察页面变化
observer.observe(document.body, { childList: true, subtree: true });
</script>
""", unsafe_allow_html=True)