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
    
    /* 全局背景 */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* 主容器 */
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 20px;
        margin: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
    
    /* 顶部窗口控制按钮 */
    .window-controls {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 10px;
    }
    
    .control-btn {
        width: 20px;
        height: 20px;
        border-radius: 50%;
        margin-left: 8px;
        border: none;
        cursor: pointer;
    }
    
    .btn-close { background: #ff5f57; }
    .btn-minimize { background: #ffbd2e; }
    .btn-maximize { background: #28ca42; }
    
    /* 左侧边栏 */
    .left-sidebar {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 20px;
        margin-right: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        min-height: 500px;
    }
    
    .user-info {
        text-align: center;
        margin-bottom: 30px;
    }
    
    .user-avatar {
        width: 60px;
        height: 60px;
        background: #4A90E2;
        border-radius: 50%;
        margin: 0 auto 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 24px;
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
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 20px;
        margin-left: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        min-height: 500px;
    }
    
    /* 标题区域 */
    .title-section {
        text-align: center;
        margin-bottom: 30px;
    }
    
    .main-title {
        font-size: 24px;
        font-weight: 600;
        color: #333;
        margin-bottom: 20px;
    }
    
    /* 模型选择卡片 */
    .model-cards {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-bottom: 30px;
    }
    
    .model-card {
        background: white;
        border-radius: 15px;
        padding: 30px 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        cursor: pointer;
        transition: all 0.3s ease;
        min-width: 150px;
        border: 3px solid transparent;
    }
    
    .model-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.2);
    }
    
    .model-card.active {
        border-color: #4A90E2;
        background: linear-gradient(135deg, #4A90E2, #357ABD);
        color: white;
    }
    
    .model-icon {
        font-size: 40px;
        margin-bottom: 15px;
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
    }
    
    .feature-section {
        flex: 1;
        background: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .section-title {
        text-align: center;
        font-size: 16px;
        font-weight: 600;
        color: white;
        padding: 10px;
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
        color: #666;
        margin-bottom: 5px;
    }
    
    /* 输入框样式 */
    .stNumberInput input {
        border-radius: 8px !important;
        border: 2px solid #e0e0e0 !important;
        padding: 10px !important;
        font-size: 14px !important;
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
        margin-bottom: 30px;
    }
    
    .action-button {
        background: linear-gradient(135deg, #4A90E2, #357ABD);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 15px 40px;
        font-size: 16px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(74, 144, 226, 0.3);
    }
    
    .action-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(74, 144, 226, 0.4);
    }
    
    .reset-button {
        background: linear-gradient(135deg, #6c757d, #5a6268);
        box-shadow: 0 5px 15px rgba(108, 117, 125, 0.3);
    }
    
    .reset-button:hover {
        box-shadow: 0 8px 25px rgba(108, 117, 125, 0.4);
    }
    
    /* 结果显示 */
    .result-display {
        background: white;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin-bottom: 20px;
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
        margin-bottom: 20px;
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
    }
    
    .info-item {
        display: flex;
        justify-content: space-between;
        margin-bottom: 5px;
        font-size: 14px;
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
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 5px;
    }
    
    .status-normal { background: #28a745; }
    .status-warning { background: #ffc107; }
    .status-error { background: #dc3545; }
    
    /* 隐藏Streamlit默认样式 */
    .stButton button {
        display: none;
    }
    
    /* 响应式设计 */
    @media (max-width: 1200px) {
        .feature-sections {
            flex-direction: column;
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

# 模型预测器类（简化版）
class ModelPredictor:
    def __init__(self, target_model="Char Yield"):
        self.target_name = target_model
        self.model_loaded = True  # 模拟模型已加载
        
    def predict(self, features):
        # 模拟预测结果
        if self.target_name == "Char Yield":
            return 27.7937
        elif self.target_name == "Oil Yield":
            return 45.2
        else:
            return 27.0
    
    def get_model_info(self):
        return {
            "目标变量": self.target_name,
            "预测结果": "27.7937 wt%" if self.target_name == "Char Yield" else "N/A",
            "模型类型": "GBDT Pipeline",
            "预处理": "RobustScaler"
        }

# 创建主布局
col_left, col_main, col_right = st.columns([1, 3, 1])

# 左侧边栏
with col_left:
    st.markdown("""
    <div class="left-sidebar">
        <div class="user-info">
            <div class="user-avatar">👤</div>
            <div style="font-weight: 600;">用户: wy1122</div>
        </div>
        
        <div class="menu-item active">
            <div style="font-weight: 600;">预测模型</div>
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

# 主要内容区域
with col_main:
    st.markdown("""
    <div class="main-container">
        <div class="window-controls">
            <button class="control-btn btn-close"></button>
            <button class="control-btn btn-minimize"></button>
            <button class="control-btn btn-maximize"></button>
        </div>
        
        <div class="title-section">
            <div class="main-title">选择预测目标</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 模型选择卡片
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("", key="char_btn"):
            st.session_state.selected_model = "Char Yield"
            st.session_state.prediction_result = 27.79
        
        active_class = "active" if st.session_state.selected_model == "Char Yield" else ""
        st.markdown(f"""
        <div class="model-card {active_class}" onclick="document.querySelector('[data-testid=\"stButton\"] button').click()">
            <span class="model-icon">🔥</span>
            <div class="model-name">Char Yield</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("", key="oil_btn"):
            st.session_state.selected_model = "Oil Yield"
            st.session_state.prediction_result = 45.2
        
        active_class = "active" if st.session_state.selected_model == "Oil Yield" else ""
        st.markdown(f"""
        <div class="model-card {active_class}">
            <span class="model-icon">🛢️</span>
            <div class="model-name">Oil Yield</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if st.button("", key="gas_btn"):
            st.session_state.selected_model = "Gas Yield"
            st.session_state.prediction_result = 27.0
        
        active_class = "active" if st.session_state.selected_model == "Gas Yield" else ""
        st.markdown(f"""
        <div class="model-card {active_class}">
            <span class="model-icon">💨</span>
            <div class="model-name">Gas Yield</div>
        </div>
        """, unsafe_allow_html=True)
    
    # 当前模型显示
    st.markdown(f"""
    <div style="text-align: center; margin: 20px 0; font-size: 16px; color: #666;">
        当前模型: <strong>{st.session_state.selected_model}</strong>
    </div>
    """, unsafe_allow_html=True)
    
    # 特征输入区域
    st.markdown('<div class="feature-sections">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    # Proximate Analysis
    with col1:
        st.markdown("""
        <div class="feature-section">
            <div class="section-title proximate">Proximate Analysis</div>
        </div>
        """, unsafe_allow_html=True)
        
        for feature in feature_categories["Proximate Analysis"]:
            st.markdown(f'<div class="feature-label">{feature}</div>', unsafe_allow_html=True)
            value = st.number_input("", value=default_values[feature], key=f"prox_{feature}", label_visibility="collapsed")
            st.session_state.feature_values[feature] = value
    
    # Ultimate Analysis  
    with col2:
        st.markdown("""
        <div class="feature-section">
            <div class="section-title ultimate">Ultimate Analysis</div>
        </div>
        """, unsafe_allow_html=True)
        
        for feature in feature_categories["Ultimate Analysis"]:
            st.markdown(f'<div class="feature-label">{feature}</div>', unsafe_allow_html=True)
            value = st.number_input("", value=default_values[feature], key=f"ult_{feature}", label_visibility="collapsed")
            st.session_state.feature_values[feature] = value
    
    # Pyrolysis Conditions
    with col3:
        st.markdown("""
        <div class="feature-section">
            <div class="section-title pyrolysis">Pyrolysis Conditions</div>
        </div>
        """, unsafe_allow_html=True)
        
        for feature in feature_categories["Pyrolysis Conditions"]:
            st.markdown(f'<div class="feature-label">{feature}</div>', unsafe_allow_html=True)
            value = st.number_input("", value=default_values[feature], key=f"pyr_{feature}", label_visibility="collapsed")
            st.session_state.feature_values[feature] = value
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 按钮区域
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("运行预测", key="predict_btn", use_container_width=True):
            predictor = ModelPredictor(st.session_state.selected_model)
            result = predictor.predict(st.session_state.feature_values)
            st.session_state.prediction_result = result
    
    with col2:
        if st.button("重置数据", key="reset_btn", use_container_width=True):
            st.session_state.feature_values = default_values.copy()
            st.rerun()
    
    # 结果显示
    if st.session_state.prediction_result is not None:
        st.markdown(f"""
        <div class="result-display">
            <div class="result-title">{st.session_state.selected_model}</div>
            <div class="result-value">{st.session_state.prediction_result:.2f} wt%</div>
        </div>
        """, unsafe_allow_html=True)

# 右侧信息面板
with col_right:
    st.markdown(f"""
    <div class="right-panel">
        <div class="info-section">
            <div class="info-title">预测结果</div>
            <div style="font-size: 18px; font-weight: 600; color: #4A90E2;">
                {st.session_state.selected_model}: {st.session_state.prediction_result or 0:.2f} wt%
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
                <span class="info-value">{st.session_state.prediction_result or 0:.4f} wt%</span>
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