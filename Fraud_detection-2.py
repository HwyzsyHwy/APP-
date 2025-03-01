import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os
import json

# 设置页面配置
st.set_page_config(page_title="生物质热解产率预测", layout="wide")

# 初始化会话状态
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# 默认值字典
default_values = {
    'M': 5.0, 'Ash': 8.0, 'VM': 75.0, 'FC': 15.0,
    'C': 45.0, 'H': 5.5, 'O': 45.0, 'N': 0.5, 'S': 0.1,
    'Temperature': 500.0, 'Heating_Rate': 10.0, 'Holding_Time': 30.0
}

# 初始化状态
for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

# 自定义CSS和JavaScript
st.markdown("""
<style>
/* 整体背景颜色 */
.stApp {
    background-color: #1E1E1E;
    color: white;
}

/* 容器样式 */
.main-container {
    padding: 20px;
}

.analysis-section {
    margin-bottom: 20px;
    border-radius: 5px;
    padding: 10px;
}

.section-header {
    padding: 8px;
    border-radius: 5px;
    margin-bottom: 15px;
    text-align: center;
    font-weight: bold;
    font-size: 18px;
}

/* 输入行样式 */
.input-row {
    display: flex;
    margin-bottom: 8px;
    width: 100%;
    height: 38px;
}

.label {
    display: flex;
    align-items: center;
    padding: 0 10px;
    width: 50%;
    height: 100%;
    border-top-left-radius: 4px;
    border-bottom-left-radius: 4px;
}

.input-field {
    width: 50%;
    height: 100%;
    border: none;
    padding: 0 10px;
    border-top-right-radius: 4px;
    border-bottom-right-radius: 4px;
    font-size: 16px;
    text-align: left;
}

/* 按钮样式 */
.button-container {
    display: flex;
    justify-content: space-between;
    margin-top: 20px;
}

.action-button {
    width: 48%;
    height: 38px;
    border: none;
    border-radius: 4px;
    color: white;
    font-weight: bold;
    cursor: pointer;
}

.push-button {
    background-color: #4CAF50;
}

.clear-button {
    background-color: #f44336;
}

/* 颜色设置 */
.proximate-header {
    background-color: rgba(76, 175, 80, 0.2);
    color: #4CAF50;
}

.proximate-label {
    background-color: #4CAF50;
    color: white;
}

.proximate-input {
    background-color: #4CAF50;
    color: white;
}

.ultimate-header {
    background-color: rgba(255, 235, 59, 0.2);
    color: #FFEB3B;
}

.ultimate-label {
    background-color: #FFEB3B;
    color: black;
}

.ultimate-input {
    background-color: #FFEB3B;
    color: black;
}

.pyrolysis-header {
    background-color: rgba(255, 152, 0, 0.2);
    color: #FF9800;
}

.pyrolysis-label {
    background-color: #FF9800;
    color: black;
}

.pyrolysis-input {
    background-color: #FF9800;
    color: black;
}

/* 结果样式 */
.result-container {
    background-color: #333;
    padding: 20px;
    border-radius: 5px;
    margin-top: 20px;
    text-align: center;
    font-size: 20px;
    color: white;
}

/* 标题样式 */
.app-title {
    background-color: #333;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 20px;
    text-align: center;
    font-size: 24px;
    color: white;
}
</style>

<script>
// 更新Streamlit会话状态的函数
function updateState(key, value) {
    const data = {
        key: key,
        value: parseFloat(value)
    };
    
    // 发送数据到Streamlit
    window.parent.postMessage({
        type: "streamlit:setComponentValue",
        data: data
    }, "*");
}

// 初始化输入字段的函数
function initInputs() {
    // 获取所有自定义输入字段
    const inputs = document.querySelectorAll('.input-field');
    
    // 为每个输入字段添加事件监听器
    inputs.forEach(input => {
        input.addEventListener('change', function() {
            updateState(this.id, this.value);
        });
    });
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', initInputs);
</script>
""", unsafe_allow_html=True)

# 创建自定义HTML组件
def custom_input_html():
    # 获取当前的值
    values = {key: st.session_state[key] for key in default_values.keys()}
    
    html = f"""
    <div class="main-container">
        <div class="app-title">生物质热解产率预测</div>
        
        <div class="row">
            <div class="col-md-4">
                <!-- Proximate Analysis -->
                <div class="analysis-section">
                    <div class="section-header proximate-header">Proximate Analysis</div>
                    
                    <div class="input-row">
                        <div class="label proximate-label">M(wt%)</div>
                        <input type="number" id="M" class="input-field proximate-input" value="{values['M']}" step="0.1" min="0" max="100" />
                    </div>
                    
                    <div class="input-row">
                        <div class="label proximate-label">Ash(wt%)</div>
                        <input type="number" id="Ash" class="input-field proximate-input" value="{values['Ash']}" step="0.1" min="0" max="100" />
                    </div>
                    
                    <div class="input-row">
                        <div class="label proximate-label">VM(wt%)</div>
                        <input type="number" id="VM" class="input-field proximate-input" value="{values['VM']}" step="0.1" min="0" max="100" />
                    </div>
                    
                    <div class="input-row">
                        <div class="label proximate-label">FC(wt%)</div>
                        <input type="number" id="FC" class="input-field proximate-input" value="{values['FC']}" step="0.1" min="0" max="100" />
                    </div>
                </div>
            </div>
            
            <div class="col-md-4">
                <!-- Ultimate Analysis -->
                <div class="analysis-section">
                    <div class="section-header ultimate-header">Ultimate Analysis</div>
                    
                    <div class="input-row">
                        <div class="label ultimate-label">C(wt%)</div>
                        <input type="number" id="C" class="input-field ultimate-input" value="{values['C']}" step="0.1" min="0" max="100" />
                    </div>
                    
                    <div class="input-row">
                        <div class="label ultimate-label">H(wt%)</div>
                        <input type="number" id="H" class="input-field ultimate-input" value="{values['H']}" step="0.1" min="0" max="100" />
                    </div>
                    
                    <div class="input-row">
                        <div class="label ultimate-label">O(wt%)</div>
                        <input type="number" id="O" class="input-field ultimate-input" value="{values['O']}" step="0.1" min="0" max="100" />
                    </div>
                    
                    <div class="input-row">
                        <div class="label ultimate-label">N(wt%)</div>
                        <input type="number" id="N" class="input-field ultimate-input" value="{values['N']}" step="0.1" min="0" max="100" />
                    </div>
                    
                    <div class="input-row">
                        <div class="label ultimate-label">S(wt%)</div>
                        <input type="number" id="S" class="input-field ultimate-input" value="{values['S']}" step="0.1" min="0" max="100" />
                    </div>
                </div>
            </div>
            
            <div class="col-md-4">
                <!-- Pyrolysis Conditions -->
                <div class="analysis-section">
                    <div class="section-header pyrolysis-header">Pyrolysis Conditions</div>
                    
                    <div class="input-row">
                        <div class="label pyrolysis-label">Temperature(°C)</div>
                        <input type="number" id="Temperature" class="input-field pyrolysis-input" value="{values['Temperature']}" step="0.1" min="0" max="1000" />
                    </div>
                    
                    <div class="input-row">
                        <div class="label pyrolysis-label">Heating Rate(°C/min)</div>
                        <input type="number" id="Heating_Rate" class="input-field pyrolysis-input" value="{values['Heating_Rate']}" step="0.1" min="0" max="100" />
                    </div>
                    
                    <div class="input-row">
                        <div class="label pyrolysis-label">Holding Time(min)</div>
                        <input type="number" id="Holding_Time" class="input-field pyrolysis-input" value="{values['Holding_Time']}" step="0.1" min="0" max="120" />
                    </div>
                </div>
            </div>
        </div>
        
        <div class="button-container">
            <button id="push-btn" class="action-button push-button" onclick="pushData()">PUSH</button>
            <button id="clear-btn" class="action-button clear-button" onclick="clearData()">CLEAR</button>
        </div>
        
        {f'<div class="result-container">Yield (%): {st.session_state.prediction_result:.2f}</div>' if st.session_state.prediction_result is not None else ''}
    </div>
    
    <script>
    // 提交数据
    function pushData() {{
        const data = {{
            action: 'push',
            values: {{}}
        }};
        
        // 收集所有输入值
        document.querySelectorAll('.input-field').forEach(input => {{
            data.values[input.id] = parseFloat(input.value);
        }});
        
        // 发送到Streamlit
        window.parent.postMessage({{
            type: "streamlit:setComponentValue",
            data: data
        }}, "*");
    }}
    
    // 清除数据
    function clearData() {{
        window.parent.postMessage({{
            type: "streamlit:setComponentValue",
            data: {{ action: 'clear' }}
        }}, "*");
    }}
    </script>
    """
    
    return html

# 自定义组件处理
def handle_custom_component():
    component_value = st.components.v1.html(
        custom_input_html(),
        height=800,
        scrolling=False
    )
    
    # 接收组件消息
    if component_value:
        if isinstance(component_value, dict):
            if 'key' in component_value and 'value' in component_value:
                # 更新单个值
                st.session_state[component_value['key']] = component_value['value']
            elif 'action' in component_value:
                if component_value['action'] == 'push':
                    # 执行预测
                    try:
                        # 更新所有值
                        for key, value in component_value.get('values', {}).items():
                            st.session_state[key] = value
                        
                        # 收集特征
                        features = [
                            st.session_state.M, st.session_state.Ash, st.session_state.VM, st.session_state.FC,
                            st.session_state.C, st.session_state.H, st.session_state.O, st.session_state.N, st.session_state.S,
                            st.session_state.Temperature, st.session_state.Heating_Rate, st.session_state.Holding_Time
                        ]
                        
                        # 简单的假预测结果
                        st.session_state.prediction_result = 35.5
                        
                        # 强制重新渲染
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"预测过程中出错: {str(e)}")
                elif component_value['action'] == 'clear':
                    # 重置所有值
                    for key, value in default_values.items():
                        st.session_state[key] = value
                    st.session_state.prediction_result = None
                    
                    # 强制重新渲染
                    st.experimental_rerun()

# 主程序
handle_custom_component()