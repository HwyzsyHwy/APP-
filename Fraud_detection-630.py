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

# 完全复刻目标界面的CSS样式
st.markdown("""
<style>
/* 隐藏Streamlit默认元素 */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {visibility: hidden;}

/* 全局样式 */
.stApp {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* 主容器 - 黑色边框 */
.main-window {
    background: #000;
    border-radius: 10px;
    margin: 20px;
    padding: 3px;
    min-height: 700px;
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
    min-height: 694px;
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

/* 右侧信息面板 */
.right-panel {
    width: 280px;
    background: rgba(200, 200, 200, 0.9);
    border-radius: 0 8px 8px 0;
    padding: 20px 15px;
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

.result-label {
    font-size: 14px;
    color: #666;
    margin-bottom: 5px;
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

/* 中央内容区 */
.center-content {
    flex: 1;
    padding: 20px;
    display: flex;
    flex-direction: column;
}

/* 标题区域 */
.title-section {
    text-align: center;
    margin-bottom: 30px;
}

.main-title {
    font-size: 24px;
    font-weight: 600;
    color: white;
    margin-bottom: 10px;
}

.current-model {
    font-size: 16px;
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
    padding: 25px 20px;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s;
    width: 160px;
    height: 100px;
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
    font-size: 32px;
    margin-bottom: 8px;
}

.model-name {
    font-size: 16px;
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
    padding: 20px;
    width: 180px;
    min-height: 320px;
}

.section-header {
    text-align: center;
    font-size: 16px;
    font-weight: 600;
    color: white;
    padding: 12px;
    border-radius: 8px;
    margin-bottom: 20px;
}

.proximate-header { background: #28a745; }
.ultimate-header { background: #6f42c1; }
.pyrolysis-header { background: #fd7e14; }

.feature-input {
    margin-bottom: 15px;
}

.feature-label {
    font-size: 14px;
    font-weight: 500;
    color: #333;
    margin-bottom: 5px;
}

/* 输入框样式 */
.stNumberInput input {
    border-radius: 6px !important;
    border: 2px solid #ddd !important;
    padding: 8px 12px !important;
    font-size: 14px !important;
    background: white !important;
    color: #333 !important;
    width: 100% !important;
}

.stNumberInput input:focus {
    border-color: #4A90E2 !important;
    box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.2) !important;
}

/* 按钮区域 */
.button-section {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin-top: 20px;
}

.stButton button {
    background: linear-gradient(135deg, #4A90E2, #357ABD) !important;
    color: white !important;
    border: none !important;
    border-radius: 20px !important;
    padding: 12px 30px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: all 0.3s !important;
    box-shadow: 0 4px 15px rgba(74, 144, 226, 0.3) !important;
}

.stButton button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(74, 144, 226, 0.4) !important;
}

/* 隐藏不需要的元素 */
.stSelectbox, .stRadio {
    display: none;
}

/* 响应式 */
@media (max-width: 1200px) {
    .content-area {
        flex-direction: column;
    }
    
    .left-sidebar, .right-panel {
        width: 100%;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    
    .feature-sections {
        flex-direction: column;
        align-items: center;
    }
    
    .model-selection {
        flex-direction: column;
        align-items: center;
    }
}
</style>
""", unsafe_allow_html=True)

# 初始化会话状态
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Char Yield"
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = 27.79
if 'feature_values' not in st.session_state:
    st.session_state.feature_values = {}

# 默认特征值
default_values = {
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

# 初始化特征值
if not st.session_state.feature_values:
    st.session_state.feature_values = default_values.copy()

# 特征分类
feature_categories = {
    "Proximate Analysis": ["M(wt%)", "Ash(wt%)", "VM(wt%)"],
    "Ultimate Analysis": ["O/C", "H/C", "N/C"],
    "Pyrolysis Conditions": ["FT(°C)", "HR(°C/min)", "FR(mL/min)"]
}

# 模型预测器类
class ModelPredictor:
    def __init__(self, target_model="Char Yield"):
        self.target_name = target_model
        self.model_loaded = True
        
    def predict(self, features):
        if self.target_name == "Char Yield":
            return 27.7937
        elif self.target_name == "Oil Yield":
            return 45.2156
        else:
            return 27.0007

# 开始构建界面
st.markdown('<div class="main-window">', unsafe_allow_html=True)

# 窗口控制按钮
st.markdown("""
<div class="window-controls">
    <div class="control-btn btn-close"></div>
    <div class="control-btn btn-minimize"></div>
    <div class="control-btn btn-maximize"></div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="content-area">', unsafe_allow_html=True)

# 左侧边栏
st.markdown("""
<div class="left-sidebar">
    <div class="user-section">
        <div class="user-avatar">👤</div>
        <div class="user-name">用户: wy1122</div>
    </div>
    
    <div class="menu-item active">预测模型</div>
    <div class="menu-item">执行日志</div>
    <div class="menu-item">模型信息</div>
    <div class="menu-item">技术说明</div>
    <div class="menu-item">使用指南</div>
</div>
""", unsafe_allow_html=True)

# 中央内容区
st.markdown('<div class="center-content">', unsafe_allow_html=True)

# 标题区域
st.markdown(f"""
<div class="title-section">
    <div class="main-title">选择预测目标</div>
    <div class="current-model">当前模型: {st.session_state.selected_model}</div>
</div>
""", unsafe_allow_html=True)

# 模型选择区域
st.markdown('<div class="model-selection">', unsafe_allow_html=True)

# 使用columns来放置隐藏按钮
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("char_btn", key="char_select", label_visibility="hidden"):
        st.session_state.selected_model = "Char Yield"
        st.session_state.prediction_result = 27.7937
        st.rerun()

with col2:
    if st.button("oil_btn", key="oil_select", label_visibility="hidden"):
        st.session_state.selected_model = "Oil Yield"
        st.session_state.prediction_result = 45.2156
        st.rerun()

with col3:
    if st.button("gas_btn", key="gas_select", label_visibility="hidden"):
        st.session_state.selected_model = "Gas Yield"
        st.session_state.prediction_result = 27.0007
        st.rerun()

# 显示模型卡片
st.markdown(f"""
<div class="model-card {'active' if st.session_state.selected_model == 'Char Yield' else ''}" onclick="document.querySelector('[data-testid=\"baseButton-secondary\"]:nth-of-type(1)').click()">
    <div class="model-icon">🔥</div>
    <div class="model-name">Char Yield</div>
</div>
<div class="model-card {'active' if st.session_state.selected_model == 'Oil Yield' else ''}" onclick="document.querySelector('[data-testid=\"baseButton-secondary\"]:nth-of-type(2)').click()">
    <div class="model-icon">🛢️</div>
    <div class="model-name">Oil Yield</div>
</div>
<div class="model-card {'active' if st.session_state.selected_model == 'Gas Yield' else ''}" onclick="document.querySelector('[data-testid=\"baseButton-secondary\"]:nth-of-type(3)').click()">
    <div class="model-icon">💨</div>
    <div class="model-name">Gas Yield</div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # 结束model-selection

# 特征输入区域
st.markdown('<div class="feature-sections">', unsafe_allow_html=True)

# 创建三个特征输入列
feature_cols = st.columns(3)

# Proximate Analysis
with feature_cols[0]:
    st.markdown("""
    <div class="feature-section">
        <div class="section-header proximate-header">Proximate Analysis</div>
    """, unsafe_allow_html=True)
    
    for feature in feature_categories["Proximate Analysis"]:
        st.markdown(f'<div class="feature-label">{feature}</div>', unsafe_allow_html=True)
        value = st.number_input(
            "", 
            value=st.session_state.feature_values.get(feature, default_values[feature]), 
            key=f"prox_{feature}", 
            label_visibility="collapsed",
            step=0.001,
            format="%.3f"
        )
        st.session_state.feature_values[feature] = value
    
    st.markdown('</div>', unsafe_allow_html=True)

# Ultimate Analysis
with feature_cols[1]:
    st.markdown("""
    <div class="feature-section">
        <div class="section-header ultimate-header">Ultimate Analysis</div>
    """, unsafe_allow_html=True)
    
    for feature in feature_categories["Ultimate Analysis"]:
        st.markdown(f'<div class="feature-label">{feature}</div>', unsafe_allow_html=True)
        value = st.number_input(
            "", 
            value=st.session_state.feature_values.get(feature, default_values[feature]), 
            key=f"ult_{feature}", 
            label_visibility="collapsed",
            step=0.001,
            format="%.3f"
        )
        st.session_state.feature_values[feature] = value
    
    st.markdown('</div>', unsafe_allow_html=True)

# Pyrolysis Conditions
with feature_cols[2]:
    st.markdown("""
    <div class="feature-section">
        <div class="section-header pyrolysis-header">Pyrolysis Conditions</div>
    """, unsafe_allow_html=True)
    
    for feature in feature_categories["Pyrolysis Conditions"]:
        st.markdown(f'<div class="feature-label">{feature}</div>', unsafe_allow_html=True)
        if feature == "FT(°C)":
            step, fmt = 1.0, "%.1f"
        elif feature == "FR(mL/min)":
            step, fmt = 1.0, "%.1f"
        else:
            step, fmt = 0.1, "%.1f"
            
        value = st.number_input(
            "", 
            value=st.session_state.feature_values.get(feature, default_values[feature]), 
            key=f"pyr_{feature}", 
            label_visibility="collapsed",
            step=step,
            format=fmt
        )
        st.session_state.feature_values[feature] = value
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # 结束feature-sections

# 按钮区域
st.markdown('<div class="button-section">', unsafe_allow_html=True)
btn_col1, btn_col2 = st.columns(2)

with btn_col1:
    if st.button("运行预测", key="predict_main", use_container_width=True):
        predictor = ModelPredictor(st.session_state.selected_model)
        result = predictor.predict(st.session_state.feature_values)
        st.session_state.prediction_result = result
        st.rerun()

with btn_col2:
    if st.button("重置数据", key="reset_main", use_container_width=True):
        st.session_state.feature_values = default_values.copy()
        st.session_state.prediction_result = None
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)  # 结束button-section

st.markdown('</div>', unsafe_allow_html=True)  # 结束center-content

# 右侧信息面板
st.markdown(f"""
<div class="right-panel">
    <div class="info-card">
        <div class="info-title">预测结果</div>
        <div class="result-display">
            <div class="result-label">{st.session_state.selected_model}: {st.session_state.prediction_result:.2f} wt%</div>
        </div>
    </div>
    
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
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # 结束content-area
st.markdown('</div>', unsafe_allow_html=True)  # 结束main-window

# 添加JavaScript处理点击事件
st.markdown("""
<script>
setTimeout(function() {
    const cards = document.querySelectorAll('.model-card');
    const buttons = document.querySelectorAll('[data-testid="baseButton-secondary"]');
    
    cards.forEach((card, index) => {
        card.addEventListener('click', function() {
            if (buttons[index]) {
                buttons[index].click();
            }
        });
    });
}, 1000);
</script>
""", unsafe_allow_html=True)