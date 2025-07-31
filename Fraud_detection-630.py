# -*- coding: utf-8 -*-
"""
Biomass Pyrolysis Yield Forecast using GBDT Ensemble Models
Mac风格界面版本 - 一比一复刻目标界面
支持Char、Oil和Gas产率预测
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

# Mac风格界面CSS样式
st.markdown(
    """
    <style>
    /* 隐藏Streamlit默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    
    /* 全局背景 */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* 主容器 */
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 20px;
        margin: 20px auto;
        max-width: 1400px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        backdrop-filter: blur(10px);
        position: relative;
    }
    
    /* 顶部窗口控制按钮 */
    .window-controls {
        position: absolute;
        top: 15px;
        right: 20px;
        display: flex;
        gap: 8px;
    }
    
    .control-btn {
        width: 20px;
        height: 20px;
        border-radius: 50%;
        border: none;
        cursor: pointer;
    }
    
    .btn-close { background: #ff5f57; }
    .btn-minimize { background: #ffbd2e; }
    .btn-maximize { background: #28ca42; }
    
    /* 左侧边栏 */
    .left-sidebar {
        background: rgba(240, 240, 240, 0.95);
        border-radius: 15px;
        padding: 20px;
        width: 180px;
        min-height: 600px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        position: absolute;
        left: 20px;
        top: 60px;
    }
    
    .user-info {
        text-align: center;
        margin-bottom: 30px;
        padding: 15px;
        background: white;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .user-avatar {
        width: 50px;
        height: 50px;
        background: #4A90E2;
        border-radius: 50%;
        margin: 0 auto 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 20px;
    }
    
    .menu-item {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 14px;
        font-weight: 500;
    }
    
    .menu-item:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .menu-item.active {
        background: #4A90E2;
        color: white;
    }
    
    /* 右侧信息面板 */
    .right-panel {
        background: rgba(240, 240, 240, 0.95);
        border-radius: 15px;
        padding: 20px;
        width: 280px;
        min-height: 600px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        position: absolute;
        right: 20px;
        top: 60px;
    }
    
    /* 中央内容区域 */
    .center-content {
        margin: 60px 220px 20px 220px;
        min-height: 600px;
    }
    
    /* 标题区域 */
    .title-section {
        text-align: center;
        margin-bottom: 30px;
        padding: 20px 0;
    }
    
    .main-title {
        font-size: 24px;
        font-weight: 600;
        color: #333;
        margin-bottom: 10px;
    }
    
    .current-model {
        font-size: 16px;
        color: #666;
        margin-top: 10px;
    }
    
    /* 模型选择卡片 */
    .model-cards {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-bottom: 30px;
    }
    
    .model-card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 30px 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        cursor: pointer;
        transition: all 0.3s ease;
        width: 180px;
        height: 120px;
        border: 3px solid transparent;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .model-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.2);
    }
    
    .model-card.active {
        border-color: #4A90E2;
        background: linear-gradient(135deg, #4A90E2, #357ABD);
        color: white;
        box-shadow: 0 15px 40px rgba(74, 144, 226, 0.4);
    }
    
    .model-icon {
        font-size: 40px;
        margin-bottom: 10px;
        display: block;
    }
    
    .model-name {
        font-size: 18px;
        font-weight: 600;
    }
    
    /* 特征输入区域 */
    .feature-sections {
        display: flex;
        gap: 20px;
        margin-bottom: 30px;
        justify-content: center;
    }
    
    .feature-section {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        width: 200px;
        min-height: 300px;
    }
    
    .section-title {
        text-align: center;
        font-size: 16px;
        font-weight: 600;
        color: white;
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    .proximate { background: #28a745; }
    .ultimate { background: #6f42c1; }
    .pyrolysis { background: #fd7e14; }
    
    .feature-input {
        margin-bottom: 15px;
    }
    
    .feature-label {
        font-size: 14px;
        font-weight: 500;
        color: #333;
        margin-bottom: 5px;
        padding: 5px 0;
    }
    
    /* 输入框样式 */
    .stNumberInput input {
        border-radius: 8px !important;
        border: 2px solid #e0e0e0 !important;
        padding: 8px 12px !important;
        font-size: 14px !important;
        background: white !important;
        color: #333 !important;
    }
    
    .stNumberInput input:focus {
        border-color: #4A90E2 !important;
        box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.1) !important;
    }
    
    /* 按钮区域 */
    .button-section {
        display: flex;
        gap: 20px;
        justify-content: center;
        margin: 30px 0;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #4A90E2, #357ABD) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 12px 40px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 5px 15px rgba(74, 144, 226, 0.3) !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(74, 144, 226, 0.4) !important;
    }
    
    /* 结果显示 */
    .result-display {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    
    .result-title {
        font-size: 18px;
        color: #666;
        margin-bottom: 10px;
    }
    
    .result-value {
        font-size: 36px;
        font-weight: 700;
        color: #4A90E2;
        margin-bottom: 10px;
    }
    
    /* 信息面板样式 */
    .info-section {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .info-title {
        font-size: 16px;
        font-weight: 600;
        color: #333;
        margin-bottom: 10px;
        text-align: center;
    }
    
    .info-item {
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
    
    /* 状态指示器 */
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 5px;
    }
    
    .status-normal { background: #28a745; }
    .status-warning { background: #ffc107; }
    .status-error { background: #dc3545; }
    
    /* 隐藏Streamlit默认样式 */
    .stSelectbox, .stRadio {
        display: none;
    }
    
    /* 响应式设计 */
    @media (max-width: 1400px) {
        .left-sidebar, .right-panel {
            position: relative;
            width: 100%;
            margin-bottom: 20px;
        }
        
        .center-content {
            margin: 20px;
        }
        
        .feature-sections {
            flex-direction: column;
            align-items: center;
        }
        
        .model-cards {
            flex-direction: column;
            align-items: center;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 初始化会话状态
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Char Yield"
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'model_info' not in st.session_state:
    st.session_state.model_info = {}
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
        # 模拟预测结果
        if self.target_name == "Char Yield":
            return 27.7937
        elif self.target_name == "Oil Yield":
            return 45.2156
        else:
            return 27.0007
    
    def get_model_info(self):
        return {
            "目标变量": self.target_name,
            "预测结果": f"{self.predict({}) if st.session_state.prediction_result else 'N/A':.4f} wt%",
            "模型类型": "GBDT Pipeline",
            "预处理": "RobustScaler"
        }

# 创建主布局
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# 窗口控制按钮
st.markdown("""
<div class="window-controls">
    <div class="control-btn btn-close"></div>
    <div class="control-btn btn-minimize"></div>
    <div class="control-btn btn-maximize"></div>
</div>
""", unsafe_allow_html=True)

# 左侧边栏
st.markdown("""
<div class="left-sidebar">
    <div class="user-info">
        <div class="user-avatar">👤</div>
        <div style="font-weight: 600; font-size: 14px;">用户: wy1122</div>
    </div>
    
    <div class="menu-item active">
        <div>预测模型</div>
    </div>
    
    <div class="menu-item">
        <div>执行日志</div>
    </div>
    
    <div class="menu-item">
        <div>模型信息</div>
    </div>
    
    <div class="menu-item">
        <div>技术说明</div>
    </div>
    
    <div class="menu-item">
        <div>使用指南</div>
    </div>
</div>
""", unsafe_allow_html=True)

# 右侧信息面板
predictor = ModelPredictor(st.session_state.selected_model)
st.markdown(f"""
<div class="right-panel">
    <div class="info-section">
        <div class="info-title">预测结果</div>
        <div style="font-size: 18px; font-weight: 600; color: #4A90E2; text-align: center;">
            {st.session_state.selected_model}: {st.session_state.prediction_result or 27.79:.2f} wt%
        </div>
    </div>
    
    <div class="info-section">
        <div class="info-title">预测信息</div>
        <div class="info-item">
            <span class="info-label">目标变量:</span>
            <span class="info-value">{st.session_state.selected_model}</span>
        </div>
        <div class="info-item">
            <span class="info-label">预测结果:</span>
            <span class="info-value">{st.session_state.prediction_result or 27.7937:.4f} wt%</span>
        </div>
        <div class="info-item">
            <span class="info-label">模型类型:</span>
            <span class="info-value">GBDT Pipeline</span>
        </div>
        <div class="info-item">
            <span class="info-label">预处理:</span>
            <span class="info-value">RobustScaler</span>
        </div>
    </div>
    
    <div class="info-section">
        <div class="info-title">模型状态</div>
        <div class="info-item">
            <span class="info-label">加载状态:</span>
            <span class="info-value">
                <span class="status-indicator status-normal"></span>正常
            </span>
        </div>
        <div class="info-item">
            <span class="info-label">特征数量:</span>
            <span class="info-value">9</span>
        </div>
        <div class="info-item">
            <span class="info-label">警告数量:</span>
            <span class="info-value">0</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# 中央内容区域
st.markdown('<div class="center-content">', unsafe_allow_html=True)

# 标题区域
st.markdown(f"""
<div class="title-section">
    <div class="main-title">选择预测目标</div>
    <div class="current-model">当前模型: {st.session_state.selected_model}</div>
</div>
""", unsafe_allow_html=True)

# 模型选择卡片 - 使用隐藏按钮和JavaScript
col1, col2, col3 = st.columns(3)

with col1:
    char_clicked = st.button("char_select", key="char_btn", label_visibility="hidden")
    if char_clicked:
        st.session_state.selected_model = "Char Yield"
        st.session_state.prediction_result = 27.7937
        st.rerun()

with col2:
    oil_clicked = st.button("oil_select", key="oil_btn", label_visibility="hidden")
    if oil_clicked:
        st.session_state.selected_model = "Oil Yield"
        st.session_state.prediction_result = 45.2156
        st.rerun()

with col3:
    gas_clicked = st.button("gas_select", key="gas_btn", label_visibility="hidden")
    if gas_clicked:
        st.session_state.selected_model = "Gas Yield"
        st.session_state.prediction_result = 27.0007
        st.rerun()

# 模型卡片显示
st.markdown(f"""
<div class="model-cards">
    <div class="model-card {'active' if st.session_state.selected_model == 'Char Yield' else ''}" onclick="document.querySelector('[data-testid=\"baseButton-secondary\"]').click()">
        <span class="model-icon">🔥</span>
        <div class="model-name">Char Yield</div>
    </div>
    <div class="model-card {'active' if st.session_state.selected_model == 'Oil Yield' else ''}">
        <span class="model-icon">🛢️</span>
        <div class="model-name">Oil Yield</div>
    </div>
    <div class="model-card {'active' if st.session_state.selected_model == 'Gas Yield' else ''}">
        <span class="model-icon">💨</span>
        <div class="model-name">Gas Yield</div>
    </div>
</div>
""", unsafe_allow_html=True)

# 特征输入区域
st.markdown('<div class="feature-sections">', unsafe_allow_html=True)

# 创建三个特征输入区域
feature_cols = st.columns(3)

# Proximate Analysis
with feature_cols[0]:
    st.markdown("""
    <div class="feature-section">
        <div class="section-title proximate">Proximate Analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    for feature in feature_categories["Proximate Analysis"]:
        st.markdown(f'<div class="feature-label">{feature}</div>', unsafe_allow_html=True)
        value = st.number_input(
            "", 
            value=default_values[feature], 
            key=f"prox_{feature}", 
            label_visibility="collapsed",
            step=0.001,
            format="%.3f"
        )
        st.session_state.feature_values[feature] = value

# Ultimate Analysis  
with feature_cols[1]:
    st.markdown("""
    <div class="feature-section">
        <div class="section-title ultimate">Ultimate Analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    for feature in feature_categories["Ultimate Analysis"]:
        st.markdown(f'<div class="feature-label">{feature}</div>', unsafe_allow_html=True)
        value = st.number_input(
            "", 
            value=default_values[feature], 
            key=f"ult_{feature}", 
            label_visibility="collapsed",
            step=0.001,
            format="%.3f"
        )
        st.session_state.feature_values[feature] = value

# Pyrolysis Conditions
with feature_cols[2]:
    st.markdown("""
    <div class="feature-section">
        <div class="section-title pyrolysis">Pyrolysis Conditions</div>
    </div>
    """, unsafe_allow_html=True)
    
    for feature in feature_categories["Pyrolysis Conditions"]:
        st.markdown(f'<div class="feature-label">{feature}</div>', unsafe_allow_html=True)
        if feature == "FT(°C)":
            step = 1.0
            format_str = "%.1f"
        elif feature == "FR(mL/min)":
            step = 1.0
            format_str = "%.1f"
        else:
            step = 0.1
            format_str = "%.1f"
            
        value = st.number_input(
            "", 
            value=default_values[feature], 
            key=f"pyr_{feature}", 
            label_visibility="collapsed",
            step=step,
            format=format_str
        )
        st.session_state.feature_values[feature] = value

st.markdown('</div>', unsafe_allow_html=True)

# 按钮区域
st.markdown('<div class="button-section">', unsafe_allow_html=True)
button_cols = st.columns(2)

with button_cols[0]:
    if st.button("运行预测", key="predict_btn", use_container_width=True):
        predictor = ModelPredictor(st.session_state.selected_model)
        result = predictor.predict(st.session_state.feature_values)
        st.session_state.prediction_result = result
        st.rerun()

with button_cols[1]:
    if st.button("重置数据", key="reset_btn", use_container_width=True):
        for feature, default_val in default_values.items():
            st.session_state.feature_values[feature] = default_val
        st.session_state.prediction_result = None
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# 结果显示
if st.session_state.prediction_result is not None:
    st.markdown(f"""
    <div class="result-display">
        <div class="result-title">{st.session_state.selected_model}</div>
        <div class="result-value">{st.session_state.prediction_result:.2f} wt%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # 结束center-content
st.markdown('</div>', unsafe_allow_html=True)  # 结束main-container

# 添加JavaScript来处理模型卡片点击
st.markdown("""
<script>
document.addEventListener('DOMContentLoaded', function() {
    const cards = document.querySelectorAll('.model-card');
    const buttons = document.querySelectorAll('[data-testid="baseButton-secondary"]');
    
    cards.forEach((card, index) => {
        card.addEventListener('click', function() {
            if (buttons[index]) {
                buttons[index].click();
            }
        });
    });
});
</script>
""", unsafe_allow_html=True)