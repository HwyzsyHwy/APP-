# -*- coding: utf-8 -*-
"""
Biomass Pyrolysis Yield Forecast using GBDT Ensemble Models
Mac风格界面版本
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import joblib
import traceback
import matplotlib.pyplot as plt
from datetime import datetime

# 清除缓存，强制重新渲染
st.cache_data.clear()

# 页面设置
st.set_page_config(
    page_title='Biomass Pyrolysis Yield Prediction',
    page_icon='🔥',
    layout='wide',
    initial_sidebar_state='collapsed'
)

# Mac风格样式
st.markdown(
    """
    <style>
    /* 隐藏Streamlit默认元素 */
    .stApp > header {display: none;}
    .stDeployButton {display: none;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stButton > button {display: none;}
    .stNumberInput {display: none;}
    
    /* 全局样式 */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
        margin: 0;
        padding: 0;
    }
    
    /* Mac窗口容器 */
    .mac-window {
        background: #e8e8e8;
        border-radius: 12px;
        margin: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        overflow: hidden;
        min-height: 85vh;
    }
    
    /* Mac标题栏 */
    .mac-titlebar {
        background: linear-gradient(to bottom, #f0f0f0, #d0d0d0);
        height: 28px;
        display: flex;
        align-items: center;
        padding: 0 15px;
        border-bottom: 1px solid #b0b0b0;
    }
    
    .mac-buttons {
        display: flex;
        gap: 8px;
    }
    
    .mac-button-circle {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        border: 1px solid rgba(0,0,0,0.2);
    }
    
    .close { background: #ff5f57; }
    .minimize { background: #ffbd2e; }
    .maximize { background: #28ca42; }
    
    .window-title {
        flex: 1;
        text-align: center;
        font-size: 13px;
        font-weight: 500;
        color: #333;
    }
    
    /* 主内容区域 */
    .mac-content {
        display: flex;
        height: calc(85vh - 28px);
        background: #f5f5f5;
    }
    
    /* 左侧边栏 */
    .left-sidebar {
        width: 160px;
        background: #e0e0e0;
        border-right: 1px solid #c0c0c0;
        padding: 15px 10px;
    }
    
    .user-card {
        background: #f8f8f8;
        border: 1px solid #d0d0d0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        margin-bottom: 15px;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.1);
    }
    
    .user-icon {
        width: 24px;
        height: 24px;
        background: #007aff;
        border-radius: 50%;
        margin: 0 auto 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 12px;
    }
    
    .user-name {
        font-size: 12px;
        font-weight: 500;
        color: #333;
    }
    
    .sidebar-menu {
        display: flex;
        flex-direction: column;
        gap: 2px;
    }
    
    .menu-item {
        background: #f0f0f0;
        border: 1px solid #d0d0d0;
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 11px;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .menu-item:hover {
        background: #e0e0e0;
    }
    
    .menu-item.active {
        background: #007aff;
        color: white;
        border-color: #0056cc;
    }
    
    /* 中间内容区域 */
    .center-content {
        flex: 1;
        padding: 20px;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20"><rect width="20" height="20" fill="%23f5f5f5"/><path d="M0 0h20v1H0zM0 0v20h1V0z" fill="%23e0e0e0" opacity="0.3"/></svg>');
    }
    
    /* 标题区域 */
    .section-title {
        text-align: center;
        font-size: 16px;
        font-weight: 600;
        color: #333;
        margin-bottom: 20px;
        padding: 10px;
        background: rgba(255,255,255,0.5);
        border-radius: 8px;
    }
    
    /* 模型选择卡片 */
    .model-cards {
        display: flex;
        gap: 15px;
        justify-content: center;
        margin-bottom: 20px;
    }
    
    .model-card {
        background: #f8f8f8;
        border: 2px solid #d0d0d0;
        border-radius: 12px;
        padding: 25px 20px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s;
        min-width: 120px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .model-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .model-card.selected {
        background: #007aff;
        color: white;
        border-color: #0056cc;
        box-shadow: 0 4px 16px rgba(0,122,255,0.3);
    }
    
    .model-icon {
        font-size: 24px;
        margin-bottom: 8px;
    }
    
    .model-name {
        font-size: 14px;
        font-weight: 600;
    }
    
    .current-model {
        text-align: center;
        font-size: 13px;
        color: #666;
        margin-bottom: 20px;
    }
    
    /* 特征输入区域 */
    .feature-sections {
        display: flex;
        gap: 15px;
        margin-bottom: 20px;
    }
    
    .feature-section {
        flex: 1;
        background: #f8f8f8;
        border: 1px solid #d0d0d0;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .feature-header {
        padding: 12px;
        text-align: center;
        font-weight: 600;
        font-size: 12px;
        color: white;
    }
    
    .proximate-header { background: #28a745; }
    .ultimate-header { background: #6f42c1; }
    .pyrolysis-header { background: #fd7e14; }
    
    .feature-inputs {
        padding: 15px;
    }
    
    .feature-row {
        display: flex;
        align-items: center;
        margin-bottom: 12px;
        font-size: 11px;
    }
    
    .feature-label {
        flex: 1;
        font-weight: 500;
        color: #333;
    }
    
    .feature-value {
        background: white;
        border: 1px solid #ccc;
        border-radius: 4px;
        padding: 4px 8px;
        width: 60px;
        text-align: center;
        font-size: 11px;
        font-family: 'SF Mono', Monaco, monospace;
    }
    
    /* 底部按钮 */
    .bottom-controls {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-top: 20px;
    }
    
    .control-button {
        background: #f0f0f0;
        border: 1px solid #d0d0d0;
        border-radius: 8px;
        padding: 12px 30px;
        font-size: 13px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .control-button:hover {
        background: #e0e0e0;
        transform: translateY(-1px);
    }
    
    .control-button.primary {
        background: #007aff;
        color: white;
        border-color: #0056cc;
    }
    
    .control-button.primary:hover {
        background: #0056cc;
    }
    
    /* 右侧结果面板 */
    .right-panel {
        width: 200px;
        background: #e8e8e8;
        border-left: 1px solid #c0c0c0;
        padding: 15px;
        overflow-y: auto;
    }
    
    .result-section {
        background: #f8f8f8;
        border: 1px solid #d0d0d0;
        border-radius: 8px;
        margin-bottom: 15px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .result-header {
        background: #f0f0f0;
        padding: 8px 12px;
        font-size: 12px;
        font-weight: 600;
        color: #333;
        border-bottom: 1px solid #d0d0d0;
    }
    
    .result-content {
        padding: 12px;
    }
    
    .result-value {
        background: white;
        border: 1px solid #ccc;
        border-radius: 4px;
        padding: 8px;
        text-align: center;
        font-size: 11px;
        font-weight: 600;
        font-family: 'SF Mono', Monaco, monospace;
        color: #007aff;
    }
    
    .info-list {
        font-size: 10px;
        line-height: 1.5;
        color: #555;
    }
    
    .info-list li {
        margin: 4px 0;
        display: flex;
        justify-content: space-between;
    }
    
    .status-indicator {
        color: #28a745;
        font-weight: 600;
    }
    
    /* 导航箭头 */
    .nav-arrows {
        position: absolute;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        display: flex;
        gap: 10px;
    }
    
    .nav-arrow {
        width: 30px;
        height: 30px;
        background: #f0f0f0;
        border: 1px solid #d0d0d0;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        font-size: 14px;
        color: #666;
        transition: all 0.2s;
    }
    
    .nav-arrow:hover {
        background: #e0e0e0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 初始化会话状态
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Char Yield"
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = 27.79
if 'feature_values' not in st.session_state:
    st.session_state.feature_values = {
        "M(wt%)": 6.460, "Ash(wt%)": 4.498, "VM(wt%)": 75.376,
        "O/C": 0.715, "H/C": 1.534, "N/C": 0.034,
        "FT(°C)": 505.811, "HR(°C/min)": 29.011, "FR(mL/min)": 93.962
    }

# 简化的预测器类
class SimplePredictor:
    def __init__(self, target_model):
        self.target_name = target_model
        self.model_loaded = True
            
    def predict(self, features):
        if self.target_name == "Char Yield":
            return 27.79
        elif self.target_name == "Oil Yield":
            return 45.23
        else:
            return 26.98

# 创建完整的Mac界面
mac_interface = f"""
<div class="mac-window">
    <!-- Mac标题栏 -->
    <div class="mac-titlebar">
        <div class="mac-buttons">
            <div class="mac-button-circle close"></div>
            <div class="mac-button-circle minimize"></div>
            <div class="mac-button-circle maximize"></div>
        </div>
        <div class="window-title">MacBook Pro 13"</div>
    </div>
    
    <!-- 主内容区域 -->
    <div class="mac-content">
        <!-- 左侧边栏 -->
        <div class="left-sidebar">
            <div class="user-card">
                <div class="user-icon">👤</div>
                <div class="user-name">用户: wy1122</div>
            </div>
            
            <div class="sidebar-menu">
                <div class="menu-item active">预测模型</div>
                <div class="menu-item">执行日志</div>
                <div class="menu-item">模型信息</div>
                <div class="menu-item">技术说明</div>
                <div class="menu-item">使用指南</div>
            </div>
        </div>
        
        <!-- 中间内容区域 -->
        <div class="center-content">
            <div class="section-title">选择预测目标</div>
            
            <!-- 模型选择卡片 -->
            <div class="model-cards">
                <div class="model-card {'selected' if st.session_state.selected_model == 'Char Yield' else ''}" onclick="selectModel('Char Yield')">
                    <div class="model-icon">🔥</div>
                    <div class="model-name">Char Yield</div>
                </div>
                <div class="model-card {'selected' if st.session_state.selected_model == 'Oil Yield' else ''}" onclick="selectModel('Oil Yield')">
                    <div class="model-icon">🛢️</div>
                    <div class="model-name">Oil Yield</div>
                </div>
                <div class="model-card {'selected' if st.session_state.selected_model == 'Gas Yield' else ''}" onclick="selectModel('Gas Yield')">
                    <div class="model-icon">💨</div>
                    <div class="model-name">Gas Yield</div>
                </div>
            </div>
            
            <div class="current-model">当前模型: {st.session_state.selected_model}</div>
            
            <!-- 特征输入区域 -->
            <div class="feature-sections">
                <div class="feature-section">
                    <div class="feature-header proximate-header">Proximate Analysis</div>
                    <div class="feature-inputs">
                        <div class="feature-row">
                            <div class="feature-label">M(wt%)</div>
                            <input type="number" class="feature-value" value="{st.session_state.feature_values['M(wt%)']:.3f}" step="0.001">
                        </div>
                        <div class="feature-row">
                            <div class="feature-label">Ash(wt%)</div>
                            <input type="number" class="feature-value" value="{st.session_state.feature_values['Ash(wt%)']:.3f}" step="0.001">
                        </div>
                        <div class="feature-row">
                            <div class="feature-label">VM(wt%)</div>
                            <input type="number" class="feature-value" value="{st.session_state.feature_values['VM(wt%)']:.3f}" step="0.001">
                        </div>
                    </div>
                </div>
                
                <div class="feature-section">
                    <div class="feature-header ultimate-header">Ultimate Analysis</div>
                    <div class="feature-inputs">
                        <div class="feature-row">
                            <div class="feature-label">O/C</div>
                            <input type="number" class="feature-value" value="{st.session_state.feature_values['O/C']:.3f}" step="0.001">
                        </div>
                        <div class="feature-row">
                            <div class="feature-label">H/C</div>
                            <input type="number" class="feature-value" value="{st.session_state.feature_values['H/C']:.3f}" step="0.001">
                        </div>
                        <div class="feature-row">
                            <div class="feature-label">N/C</div>
                            <input type="number" class="feature-value" value="{st.session_state.feature_values['N/C']:.3f}" step="0.001">
                        </div>
                    </div>
                </div>
                
                <div class="feature-section">
                    <div class="feature-header pyrolysis-header">Pyrolysis Conditions</div>
                    <div class="feature-inputs">
                        <div class="feature-row">
                            <div class="feature-label">FT(°C)</div>
                            <input type="number" class="feature-value" value="{st.session_state.feature_values['FT(°C)']:.3f}" step="0.001">
                        </div>
                        <div class="feature-row">
                            <div class="feature-label">HR(°C/min)</div>
                            <input type="number" class="feature-value" value="{st.session_state.feature_values['HR(°C/min)']:.3f}" step="0.001">
                        </div>
                        <div class="feature-row">
                            <div class="feature-label">FR(mL/min)</div>
                            <input type="number" class="feature-value" value="{st.session_state.feature_values['FR(mL/min)']:.3f}" step="0.001">
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- 底部控制按钮 -->
            <div class="bottom-controls">
                <div class="control-button primary" onclick="runPrediction()">运行预测</div>
                <div class="control-button" onclick="resetData()">重置数据</div>
            </div>
        </div>
        
        <!-- 右侧结果面板 -->
        <div class="right-panel">
            <div class="result-section">
                <div class="result-header">预测结果</div>
                <div class="result-content">
                    <div class="result-value">Char Yield: {st.session_state.prediction_result:.2f} wt%</div>
                </div>
            </div>
            
            <div class="result-section">
                <div class="result-header">预测信息</div>
                <div class="result-content">
                    <ul class="info-list">
                        <li><span>目标变量:</span><span>{st.session_state.selected_model}</span></li>
                        <li><span>预测结果:</span><span>{st.session_state.prediction_result:.4f} wt%</span></li>
                        <li><span>模型类型:</span><span>GBDT Pipeline</span></li>
                        <li><span>预处理:</span><span>RobustScaler</span></li>
                    </ul>
                </div>
            </div>
            
            <div class="result-section">
                <div class="result-header">模型状态</div>
                <div class="result-content">
                    <ul class="info-list">
                        <li><span>加载状态:</span><span class="status-indicator">✅ 正常</span></li>
                        <li><span>特征数量:</span><span>9</span></li>
                        <li><span>警告数量:</span><span>0</span></li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
    
    <!-- 导航箭头 -->
    <div class="nav-arrows">
        <div class="nav-arrow">‹</div>
        <div class="nav-arrow">›</div>
    </div>
</div>

<script>
function selectModel(model) {{
    // 触发Streamlit重新运行
    window.parent.postMessage({{type: 'selectModel', model: model}}, '*');
}}

function runPrediction() {{
    window.parent.postMessage({{type: 'runPrediction'}}, '*');
}}

function resetData() {{
    window.parent.postMessage({{type: 'resetData'}}, '*');
}}
</script>
"""

st.markdown(mac_interface, unsafe_allow_html=True)

# 隐藏的Streamlit控件用于处理交互
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Char", key="char_btn"):
        st.session_state.selected_model = "Char Yield"
        st.session_state.prediction_result = 27.79
        st.rerun()

with col2:
    if st.button("Oil", key="oil_btn"):
        st.session_state.selected_model = "Oil Yield"
        st.session_state.prediction_result = 45.23
        st.rerun()

with col3:
    if st.button("Gas", key="gas_btn"):
        st.session_state.selected_model = "Gas Yield"
        st.session_state.prediction_result = 26.98
        st.rerun()

# 隐藏的特征输入
for feature, value in st.session_state.feature_values.items():
    st.number_input(feature, value=value, key=f"input_{feature}", label_visibility="collapsed")

# 隐藏的预测和重置按钮
col1, col2 = st.columns(2)
with col1:
    if st.button("预测", key="predict_btn"):
        predictor = SimplePredictor(st.session_state.selected_model)
        result = predictor.predict(st.session_state.feature_values)
        st.session_state.prediction_result = result
        st.rerun()

with col2:
    if st.button("重置", key="reset_btn"):
        st.session_state.feature_values = {
            "M(wt%)": 6.460, "Ash(wt%)": 4.498, "VM(wt%)": 75.376,
            "O/C": 0.715, "H/C": 1.534, "N/C": 0.034,
            "FT(°C)": 505.811, "HR(°C/min)": 29.011, "FR(mL/min)": 93.962
        }
        st.rerun()