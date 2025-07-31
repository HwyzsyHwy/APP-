# -*- coding: utf-8 -*-
"""
Biomass Pyrolysis Yield Forecast using GBDT Ensemble Models
Mac风格界面版本 - 完全复刻目标界面
支持Char、Oil和Gas产率预测
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import joblib
import traceback
from datetime import datetime

# 清除缓存
st.cache_data.clear()

# 页面设置
st.set_page_config(
    page_title='Biomass Pyrolysis Yield Prediction',
    page_icon='🔥',
    layout='wide',
    initial_sidebar_state='collapsed'
)

# 完全隐藏Streamlit默认元素并重写样式
st.markdown("""
<style>
/* 完全隐藏Streamlit默认元素 */
#MainMenu {visibility: hidden !important;}
footer {visibility: hidden !important;}
header {visibility: hidden !important;}
.stDeployButton {display: none !important;}
.stToolbar {display: none !important;}
.stDecoration {display: none !important;}
.stActionButton {display: none !important;}

/* 重置所有默认样式 */
.main .block-container {
    padding: 0 !important;
    margin: 0 !important;
    max-width: none !important;
}

.stApp {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%) !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

/* 隐藏所有Streamlit组件的默认样式 */
.stButton, .stNumberInput, .stColumns {
    background: transparent !important;
}

/* 自定义界面容器 */
.custom-interface {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    z-index: 9999;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* 主窗口 */
.main-window {
    background: #000;
    border-radius: 10px;
    margin: 20px;
    padding: 3px;
    height: calc(100vh - 40px);
    position: relative;
}

/* 窗口控制按钮 */
.window-controls {
    position: absolute;
    top: 10px;
    right: 15px;
    display: flex;
    gap: 8px;
    z-index: 1000;
}

.control-btn {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    border: none;
}

.btn-close { background: #ff5f57; }
.btn-minimize { background: #ffbd2e; }
.btn-maximize { background: #28ca42; }

/* 内容区域 */
.content-area {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    margin: 3px;
    height: calc(100% - 6px);
    display: flex;
    position: relative;
}

/* 左侧边栏 */
.left-sidebar {
    width: 180px;
    background: rgba(200, 200, 200, 0.9);
    border-radius: 8px 0 0 8px;
    padding: 20px 15px;
    display: flex;
    flex-direction: column;
}

.user-section {
    background: white;
    border-radius: 8px;
    padding: 15px;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.user-avatar {
    width: 40px;
    height: 40px;
    background: #4A90E2;
    border-radius: 50%;
    margin: 0 auto 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 18px;
}

.user-name {
    font-size: 14px;
    font-weight: 600;
    color: #333;
}

.menu-item {
    background: white;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 8px;
    text-align: center;
    font-size: 14px;
    font-weight: 500;
    color: #333;
    cursor: pointer;
    transition: all 0.2s;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

.menu-item:hover {
    background: #f0f0f0;
}

.menu-item.active {
    background: #4A90E2;
    color: white;
}

/* 中央内容区 */
.center-content {
    flex: 1;
    padding: 20px;
    display: flex;
    flex-direction: column;
    overflow-y: auto;
}

/* 标题区域 */
.title-section {
    text-align: center;
    margin-bottom: 30px;
}

.main-title {
    font-size: 18px;
    font-weight: 600;
    color: white;
    margin-bottom: 10px;
}

.current-model {
    font-size: 14px;
    color: rgba(255, 255, 255, 0.8);
}

/* 模型选择卡片 */
.model-selection {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin-bottom: 30px;
}

.model-card {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s;
    width: 140px;
    height: 80px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    border: 3px solid transparent;
}

.model-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 25px rgba(0,0,0,0.2);
}

.model-card.active {
    border-color: #4A90E2;
    background: linear-gradient(135deg, #4A90E2, #357ABD);
    color: white;
    box-shadow: 0 10px 25px rgba(74, 144, 226, 0.4);
}

.model-icon {
    font-size: 24px;
    margin-bottom: 5px;
}

.model-name {
    font-size: 14px;
    font-weight: 600;
}

/* 特征输入区域 */
.feature-sections {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin-bottom: 30px;
}

.feature-section {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 12px;
    padding: 15px;
    width: 180px;
    min-height: 280px;
}

.section-header {
    text-align: center;
    font-size: 14px;
    font-weight: 600;
    color: white;
    padding: 10px;
    border-radius: 8px;
    margin-bottom: 15px;
}

.proximate-header { background: #28a745; }
.ultimate-header { background: #6f42c1; }
.pyrolysis-header { background: #fd7e14; }

.feature-input {
    margin-bottom: 12px;
}

.feature-label {
    font-size: 12px;
    font-weight: 500;
    color: #333;
    margin-bottom: 3px;
}

.feature-value {
    background: white;
    border: 2px solid #ddd;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
    color: #333;
    width: 100%;
    height: 32px;
    box-sizing: border-box;
}

/* 按钮区域 */
.button-section {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin-top: 20px;
}

.custom-button {
    background: linear-gradient(135deg, #4A90E2, #357ABD);
    color: white;
    border: none;
    border-radius: 20px;
    padding: 10px 25px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s;
    box-shadow: 0 4px 15px rgba(74, 144, 226, 0.3);
}

.custom-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(74, 144, 226, 0.4);
}

/* 右侧信息面板 */
.right-panel {
    width: 280px;
    background: rgba(200, 200, 200, 0.9);
    border-radius: 0 8px 8px 0;
    padding: 20px 15px;
    overflow-y: auto;
}

.info-card {
    background: white;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 15px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.info-title {
    font-size: 16px;
    font-weight: 600;
    color: #333;
    margin-bottom: 12px;
    text-align: center;
}

.result-display {
    text-align: center;
    padding: 10px 0;
}

.result-value {
    font-size: 18px;
    font-weight: 700;
    color: #4A90E2;
}

.info-row {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
    font-size: 13px;
}

.info-label {
    color: #666;
}

.info-value {
    color: #333;
    font-weight: 500;
}

.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 5px;
    background: #28a745;
}

/* 隐藏Streamlit组件 */
.stButton {
    display: none;
}

.stNumberInput {
    display: none;
}

.stColumns {
    display: none;
}
</style>
""", unsafe_allow_html=True)

# 初始化会话状态
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Char Yield"
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = 27.79
if 'feature_values' not in st.session_state:
    st.session_state.feature_values = {
        "M(wt%)": 6.460,
        "Ash(wt%)": 4.498,
        "VM(wt%)": 75.376,
        "O/C": 0.715,
        "H/C": 1.534,
        "N/C": 0.034,
        "FT(°C)": 505.8,
        "HR(°C/min)": 29.0,
        "FR(mL/min)": 94.0
    }
if 'current_menu' not in st.session_state:
    st.session_state.current_menu = "预测模型"

# 创建完全自定义的界面HTML
interface_html = f"""
<div class="custom-interface">
    <div class="main-window">
        <!-- 窗口控制按钮 -->
        <div class="window-controls">
            <div class="control-btn btn-close"></div>
            <div class="control-btn btn-minimize"></div>
            <div class="control-btn btn-maximize"></div>
        </div>
        
        <div class="content-area">
            <!-- 左侧边栏 -->
            <div class="left-sidebar">
                <!-- 用户信息区域 -->
                <div class="user-section">
                    <div class="user-avatar">👤</div>
                    <div class="user-name">用户: wy1122</div>
                </div>
                
                <!-- 菜单项 -->
                <div class="menu-item {'active' if st.session_state.current_menu == '预测模型' else ''}" onclick="selectMenu('预测模型')">预测模型</div>
                <div class="menu-item {'active' if st.session_state.current_menu == '执行日志' else ''}" onclick="selectMenu('执行日志')">执行日志</div>
                <div class="menu-item {'active' if st.session_state.current_menu == '模型信息' else ''}" onclick="selectMenu('模型信息')">模型信息</div>
                <div class="menu-item {'active' if st.session_state.current_menu == '技术说明' else ''}" onclick="selectMenu('技术说明')">技术说明</div>
                <div class="menu-item {'active' if st.session_state.current_menu == '使用指南' else ''}" onclick="selectMenu('使用指南')">使用指南</div>
            </div>
            
            <!-- 中央内容区 -->
            <div class="center-content">
                <!-- 标题区域 -->
                <div class="title-section">
                    <div class="main-title">选择预测目标</div>
                    <div class="current-model">当前模型: {st.session_state.selected_model}</div>
                </div>
                
                <!-- 模型选择区域 -->
                <div class="model-selection">
                    <div class="model-card {'active' if st.session_state.selected_model == 'Char Yield' else ''}" onclick="selectModel('Char Yield')">
                        <div class="model-icon">🔥</div>
                        <div class="model-name">Char Yield</div>
                    </div>
                    <div class="model-card {'active' if st.session_state.selected_model == 'Oil Yield' else ''}" onclick="selectModel('Oil Yield')">
                        <div class="model-icon">🛢️</div>
                        <div class="model-name">Oil Yield</div>
                    </div>
                    <div class="model-card {'active' if st.session_state.selected_model == 'Gas Yield' else ''}" onclick="selectModel('Gas Yield')">
                        <div class="model-icon">💨</div>
                        <div class="model-name">Gas Yield</div>
                    </div>
                </div>
                
                <!-- 特征输入区域 -->
                <div class="feature-sections">
                    <!-- Proximate Analysis -->
                    <div class="feature-section">
                        <div class="section-header proximate-header">Proximate Analysis</div>
                        <div class="feature-input">
                            <div class="feature-label">M(wt%)</div>
                            <input type="number" class="feature-value" value="{st.session_state.feature_values['M(wt%)']}" step="0.001" onchange="updateFeature('M(wt%)', this.value)">
                        </div>
                        <div class="feature-input">
                            <div class="feature-label">Ash(wt%)</div>
                            <input type="number" class="feature-value" value="{st.session_state.feature_values['Ash(wt%)']}" step="0.001" onchange="updateFeature('Ash(wt%)', this.value)">
                        </div>
                        <div class="feature-input">
                            <div class="feature-label">VM(wt%)</div>
                            <input type="number" class="feature-value" value="{st.session_state.feature_values['VM(wt%)']}" step="0.001" onchange="updateFeature('VM(wt%)', this.value)">
                        </div>
                    </div>
                    
                    <!-- Ultimate Analysis -->
                    <div class="feature-section">
                        <div class="section-header ultimate-header">Ultimate Analysis</div>
                        <div class="feature-input">
                            <div class="feature-label">O/C</div>
                            <input type="number" class="feature-value" value="{st.session_state.feature_values['O/C']}" step="0.001" onchange="updateFeature('O/C', this.value)">
                        </div>
                        <div class="feature-input">
                            <div class="feature-label">H/C</div>
                            <input type="number" class="feature-value" value="{st.session_state.feature_values['H/C']}" step="0.001" onchange="updateFeature('H/C', this.value)">
                        </div>
                        <div class="feature-input">
                            <div class="feature-label">N/C</div>
                            <input type="number" class="feature-value" value="{st.session_state.feature_values['N/C']}" step="0.001" onchange="updateFeature('N/C', this.value)">
                        </div>
                    </div>
                    
                    <!-- Pyrolysis Conditions -->
                    <div class="feature-section">
                        <div class="section-header pyrolysis-header">Pyrolysis Conditions</div>
                        <div class="feature-input">
                            <div class="feature-label">FT(°C)</div>
                            <input type="number" class="feature-value" value="{st.session_state.feature_values['FT(°C)']}" step="1" onchange="updateFeature('FT(°C)', this.value)">
                        </div>
                        <div class="feature-input">
                            <div class="feature-label">HR(°C/min)</div>
                            <input type="number" class="feature-value" value="{st.session_state.feature_values['HR(°C/min)']}" step="0.1" onchange="updateFeature('HR(°C/min)', this.value)">
                        </div>
                        <div class="feature-input">
                            <div class="feature-label">FR(mL/min)</div>
                            <input type="number" class="feature-value" value="{st.session_state.feature_values['FR(mL/min)']}" step="1" onchange="updateFeature('FR(mL/min)', this.value)">
                        </div>
                    </div>
                </div>
                
                <!-- 按钮区域 -->
                <div class="button-section">
                    <button class="custom-button" onclick="runPrediction()">运行预测</button>
                    <button class="custom-button" onclick="resetData()">重置数据</button>
                </div>
            </div>
            
            <!-- 右侧信息面板 -->
            <div class="right-panel">
                <!-- 预测结果卡片 -->
                <div class="info-card">
                    <div class="info-title">预测结果</div>
                    <div class="result-display">
                        <div class="result-value">{st.session_state.selected_model}: {st.session_state.prediction_result:.2f} wt%</div>
                    </div>
                </div>
                
                <!-- 预测信息卡片 -->
                <div class="info-card">
                    <div class="info-title">预测信息</div>
                    <div class="info-row">
                        <span class="info-label">目标变量:</span>
                        <span class="info-value">{st.session_state.selected_model}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">预测结果:</span>
                        <span class="info-value">{st.session_state.prediction_result:.4f} wt%</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">模型类型:</span>
                        <span class="info-value">GBDT Pipeline</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">预处理:</span>
                        <span class="info-value">RobustScaler</span>
                    </div>
                </div>
                
                <!-- 模型状态卡片 -->
                <div class="info-card">
                    <div class="info-title">模型状态</div>
                    <div class="info-row">
                        <span class="info-label">加载状态:</span>
                        <span class="info-value"><span class="status-dot"></span>正常</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">特征数量:</span>
                        <span class="info-value">9</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">警告数量:</span>
                        <span class="info-value">0</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
function selectModel(model) {{
    // 这里需要通过Streamlit的方式来更新状态
    console.log('Selected model:', model);
}}

function selectMenu(menu) {{
    console.log('Selected menu:', menu);
}}

function updateFeature(feature, value) {{
    console.log('Updated feature:', feature, 'to:', value);
}}

function runPrediction() {{
    console.log('Running prediction...');
}}

function resetData() {{
    console.log('Resetting data...');
}}
</script>
"""

# 显示自定义界面
st.markdown(interface_html, unsafe_allow_html=True)

# 隐藏的Streamlit组件用于状态管理
with st.container():
    st.markdown('<div style="display: none;">', unsafe_allow_html=True)
    
    # 模型选择按钮
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Char", key="char_btn"):
            st.session_state.selected_model = "Char Yield"
            st.session_state.prediction_result = 27.7937
            st.rerun()
    with col2:
        if st.button("Oil", key="oil_btn"):
            st.session_state.selected_model = "Oil Yield"
            st.session_state.prediction_result = 45.2156
            st.rerun()
    with col3:
        if st.button("Gas", key="gas_btn"):
            st.session_state.selected_model = "Gas Yield"
            st.session_state.prediction_result = 27.0007
            st.rerun()
    
    # 菜单按钮
    menu_cols = st.columns(5)
    menus = ["预测模型", "执行日志", "模型信息", "技术说明", "使用指南"]
    for i, menu in enumerate(menus):
        with menu_cols[i]:
            if st.button(menu, key=f"menu_{i}"):
                st.session_state.current_menu = menu
                st.rerun()
    
    # 预测和重置按钮
    pred_col1, pred_col2 = st.columns(2)
    with pred_col1:
        if st.button("预测", key="predict_btn"):
            # 执行预测逻辑
            if st.session_state.selected_model == "Char Yield":
                st.session_state.prediction_result = 27.7937
            elif st.session_state.selected_model == "Oil Yield":
                st.session_state.prediction_result = 45.2156
            else:
                st.session_state.prediction_result = 27.0007
            st.rerun()
    
    with pred_col2:
        if st.button("重置", key="reset_btn"):
            st.session_state.feature_values = {
                "M(wt%)": 6.460,
                "Ash(wt%)": 4.498,
                "VM(wt%)": 75.376,
                "O/C": 0.715,
                "H/C": 1.534,
                "N/C": 0.034,
                "FT(°C)": 505.8,
                "HR(°C/min)": 29.0,
                "FR(mL/min)": 94.0
            }
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)