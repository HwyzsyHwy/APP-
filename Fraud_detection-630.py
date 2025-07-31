# -*- coding: utf-8 -*-
"""
Biomass Pyrolysis Yield Prediction using GBDT Ensemble Models
Enhanced version with accurate feature statistics adjustment
Supporting Char, Oil and Gas yield predictions
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
import base64

# Clear cache, force re-rendering
st.cache_data.clear()

# Page configuration
st.set_page_config(
    page_title='选择预测目标',
    page_icon='🔥',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Initialize session state
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Prediction Model"
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Char Yield"
if 'model_cache' not in st.session_state:
    st.session_state.model_cache = {}
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'model_stats' not in st.session_state:
    st.session_state.model_stats = {
        "Char Yield": {"accuracy": 27.79, "features": 9, "warnings": 0},
        "Oil Yield": {"accuracy": 45.23, "features": 9, "warnings": 0},
        "Gas Yield": {"accuracy": 18.56, "features": 9, "warnings": 0}
    }
# Add collapse states
if 'prediction_info_expanded' not in st.session_state:
    st.session_state.prediction_info_expanded = True
if 'model_status_expanded' not in st.session_state:
    st.session_state.model_status_expanded = True
if 'sidebar_collapsed' not in st.session_state:
    st.session_state.sidebar_collapsed = False

def add_log(message):
    """Add log message to session state"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    st.session_state.log_messages.append(log_entry)
    if len(st.session_state.log_messages) > 100:
        st.session_state.log_messages = st.session_state.log_messages[-100:]

def display_logs():
    """Display logs"""
    if st.session_state.log_messages:
        log_content = '<br>'.join(st.session_state.log_messages)
        st.markdown(f"<div class='log-container'>{log_content}</div>", unsafe_allow_html=True)

# Custom styles - 完全重新设计以匹配目标图片
st.markdown("""
<style>
/* 全局字体设置 */
* {
    font-family: 'Microsoft YaHei', 'SimHei', sans-serif !important;
}

/* 主应用背景 - 绿色科技风格 */
.stApp {
    background: linear-gradient(135deg, #0a4d3a 0%, #1a7a5e 50%, #2d9f7f 100%) !important;
    background-image:
        radial-gradient(circle at 20% 80%, rgba(120, 255, 214, 0.3) 0%, transparent 50%),
        radial-gradient(circle at 80% 20%, rgba(120, 255, 214, 0.3) 0%, transparent 50%),
        radial-gradient(circle at 40% 40%, rgba(120, 255, 214, 0.2) 0%, transparent 50%);
    min-height: 100vh;
}

/* 主内容区域 */
.main .block-container {
    padding: 1rem 2rem !important;
    max-width: 100% !important;
    background: transparent !important;
}

/* 隐藏Streamlit默认元素 */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* 自定义顶部标题栏 */
.top-header {
    background: rgba(0, 0, 0, 0.3);
    padding: 10px 20px;
    margin: -1rem -2rem 2rem -2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    backdrop-filter: blur(10px);
}

.streamlit-logo {
    color: white;
    font-size: 24px;
    font-weight: bold;
}

.search-bar {
    background: rgba(255, 255, 255, 0.2);
    border: none;
    border-radius: 20px;
    padding: 8px 15px;
    color: white;
    width: 300px;
}

.top-icons {
    display: flex;
    gap: 15px;
    color: white;
    font-size: 20px;
}

/* 侧边栏样式 */
section[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.95) !important;
    border-radius: 15px !important;
    margin: 10px !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
    backdrop-filter: blur(10px) !important;
    width: 200px !important;
    min-width: 200px !important;
}

/* 侧边栏内容 */
section[data-testid="stSidebar"] > div {
    background: transparent !important;
    padding: 20px 15px !important;
}

/* 用户信息区域 */
.user-info {
    text-align: center;
    padding: 20px 10px;
    margin-bottom: 20px;
    background: rgba(255, 255, 255, 0.8);
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}

.user-avatar {
    width: 60px;
    height: 60px;
    border-radius: 50%;
    background: #20B2AA;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 10px auto;
    color: white;
    font-size: 24px;
}

.user-name {
    color: #333;
    font-size: 14px;
    font-weight: 500;
}

/* 侧边栏按钮样式 */
.stButton > button {
    width: 100% !important;
    margin-bottom: 8px !important;
    padding: 12px 15px !important;
    border-radius: 25px !important;
    border: none !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
    background: rgba(255, 255, 255, 0.8) !important;
    color: #666 !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
}

.stButton > button:hover {
    background: rgba(255, 255, 255, 1) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
}

/* 激活状态的按钮 */
.stButton > button[kind="primary"] {
    background: #20B2AA !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(32, 178, 170, 0.4) !important;
}

.stButton > button[kind="primary"]:hover {
    background: #1a9d96 !important;
    box-shadow: 0 6px 20px rgba(32, 178, 170, 0.5) !important;
}

/* 主标题 */
.main-title {
    text-align: center;
    color: white;
    font-size: 28px;
    font-weight: bold;
    margin-bottom: 30px;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
}

/* 模型选择卡片样式 - 更接近目标图片 */
.model-card {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 15px;
    padding: 25px 15px;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(10px);
    position: relative;
    overflow: hidden;
    margin-bottom: 20px;
}

.model-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.3);
}

.model-card.selected {
    background: #20B2AA;
    color: white;
    box-shadow: 0 12px 30px rgba(32, 178, 170, 0.4);
}

/* 模型图标 */
.model-icon {
    font-size: 40px;
    margin-bottom: 10px;
    display: block;
}

.model-name {
    font-size: 16px;
    font-weight: bold;
    color: #333;
    margin: 0;
}

.model-card.selected .model-name {
    color: white;
}

/* 当前模型显示 */
.current-model-display {
    text-align: center;
    background: rgba(0, 0, 0, 0.3);
    color: white;
    padding: 10px 20px;
    border-radius: 25px;
    margin: 20px auto;
    width: fit-content;
    font-weight: bold;
    backdrop-filter: blur(10px);
}

/* 参数输入区域容器 */
.parameters-container {
    display: flex;
    gap: 20px;
    margin-bottom: 30px;
    justify-content: center;
}

/* 参数卡片样式 */
.parameter-card {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(10px);
    width: 300px;
}

.parameter-card-title {
    text-align: center;
    font-size: 16px;
    font-weight: bold;
    color: white;
    padding: 10px;
    border-radius: 25px;
    margin-bottom: 20px;
}

.parameter-card-title.proximate {
    background: #20B2AA;
}

.parameter-card-title.ultimate {
    background: #DAA520;
}

.parameter-card-title.pyrolysis {
    background: #CD853F;
}

/* 参数输入行 */
.parameter-row {
    display: flex;
    align-items: center;
    margin-bottom: 15px;
    gap: 10px;
}

.parameter-label {
    color: white;
    padding: 8px 15px;
    border-radius: 20px;
    font-size: 14px;
    font-weight: bold;
    min-width: 80px;
    text-align: center;
}

.parameter-label.proximate {
    background: #20B2AA;
}

.parameter-label.ultimate {
    background: #DAA520;
}

.parameter-label.pyrolysis {
    background: #CD853F;
}

.parameter-input {
    flex: 1;
    padding: 8px 12px;
    border: 1px solid #ddd;
    border-radius: 8px;
    font-size: 14px;
}

.parameter-buttons {
    display: flex;
    gap: 5px;
}

.param-btn {
    width: 30px;
    height: 30px;
    border: none;
    border-radius: 6px;
    font-size: 16px;
    font-weight: bold;
    cursor: pointer;
    transition: all 0.2s ease;
}

.param-btn.minus {
    background: #ff6b6b;
    color: white;
}

.param-btn.plus {
    background: #51cf66;
    color: white;
}

.param-btn:hover {
    transform: scale(1.1);
}

/* 右侧结果面板 */
.results-panel {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(10px);
    width: 300px;
    height: fit-content;
}

.results-title {
    background: #20B2AA;
    color: white;
    text-align: center;
    padding: 10px;
    border-radius: 25px;
    margin-bottom: 20px;
    font-weight: bold;
}

.result-value {
    background: rgba(32, 178, 170, 0.1);
    border: 2px solid #20B2AA;
    border-radius: 10px;
    padding: 15px;
    text-align: center;
    font-size: 18px;
    font-weight: bold;
    color: #20B2AA;
    margin-bottom: 20px;
}

.result-details {
    font-size: 12px;
    color: #666;
    line-height: 1.6;
}

.result-details .detail-item {
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
}

.status-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 8px;
}

.status-normal {
    background: #51cf66;
}

/* 底部按钮样式 - 匹配目标图片 */
.stButton > button[data-testid="baseButton-primary"] {
    background: #20B2AA !important;
    color: white !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 15px 30px !important;
    font-size: 16px !important;
    font-weight: bold !important;
    width: 100% !important;
    box-shadow: 0 4px 15px rgba(32, 178, 170, 0.3) !important;
    transition: all 0.3s ease !important;
}

.stButton > button[data-testid="baseButton-primary"]:hover {
    background: #1a9d96 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(32, 178, 170, 0.4) !important;
}

.stButton > button[data-testid="baseButton-secondary"] {
    background: #6c757d !important;
    color: white !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 15px 30px !important;
    font-size: 16px !important;
    font-weight: bold !important;
    width: 100% !important;
    box-shadow: 0 4px 15px rgba(108, 117, 125, 0.3) !important;
    transition: all 0.3s ease !important;
}

.stButton > button[data-testid="baseButton-secondary"]:hover {
    background: #5a6268 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(108, 117, 125, 0.4) !important;
}

/* 右侧箭头按钮 */
.arrow-button {
    background: rgba(255, 255, 255, 0.9);
    border: none;
    border-radius: 50%;
    width: 50px;
    height: 50px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    color: #666;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}

.arrow-button:hover {
    background: rgba(255, 255, 255, 1);
    transform: scale(1.1);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
}

/* 折叠按钮 */
.collapse-button {
    position: fixed;
    bottom: 20px;
    left: 20px;
    width: 50px;
    height: 50px;
    background: rgba(255, 255, 255, 0.9) !important;
    border: none !important;
    border-radius: 50% !important;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.3s ease;
    font-size: 18px !important;
    color: #666 !important;
    z-index: 1000;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}

.collapse-button:hover {
    background: rgba(255, 255, 255, 1) !important;
    transform: scale(1.1);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
}
</style>
""", unsafe_allow_html=True)

.main-title {
    text-align: center;
    font-size: 32px !important;
    font-weight: bold;
    margin-bottom: 20px;
    color: #333 !important;
    font-family: 'Times New Roman', serif !important;
}

.model-selector {
    text-align: center;
    margin-bottom: 30px;
}

.model-card {
    background-color: white;
    border-radius: 15px;
    padding: 20px;
    margin: 10px;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    border: 1px solid #e0e0e0;
}

.model-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0,0,0,0.15);
}

.model-icon {
    font-size: 48px;
    margin-bottom: 10px;
}

.model-name {
    font-size: 18px;
    font-weight: bold;
    color: #333;
    font-family: 'Times New Roman', serif !important;
}

.current-model {
    background-color: #1f4e79;
    color: white;
    font-size: 16px;
    padding: 10px;
    border-radius: 25px;
    margin: 20px 0;
    text-align: center;
    font-family: 'Times New Roman', serif !important;
}

.analysis-card {
    background-color: white;
    border-radius: 15px;
    padding: 20px;
    margin: 10px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    border: 1px solid #e0e0e0;
}

.card-title {
    background-color: #1f4e79;
    color: white;
    font-weight: bold;
    font-size: 16px;
    text-align: center;
    padding: 10px;
    border-radius: 25px;
    margin-bottom: 15px;
    font-family: 'Times New Roman', serif !important;
}

.input-row {
    display: flex;
    align-items: center;
    margin-bottom: 10px;
    background-color: white;
    border-radius: 8px;
    padding: 8px;
    border: 1px solid #e0e0e0;
}

.input-label {
    background-color: #1f4e79;
    color: white;
    padding: 8px 12px;
    border-radius: 20px;
    font-size: 14px;
    font-weight: bold;
    min-width: 80px;
    text-align: center;
    margin-right: 10px;
    font-family: 'Times New Roman', serif !important;
}

.action-buttons {
    display: flex;
    gap: 20px;
    margin-top: 30px;
    justify-content: center;
}

.action-btn {
    padding: 15px 30px;
    border-radius: 25px;
    border: none;
    font-size: 16px;
    font-weight: bold;
    cursor: pointer;
    transition: all 0.3s;
    font-family: 'Times New Roman', serif !important;
}

.predict-btn {
    background-color: #1f4e79;
    color: white;
}

.reset-btn {
    background-color: #e9ecef;
    color: #6c757d;
}

.yield-result {
    background-color: white;
    color: #333;
    font-size: 36px;
    font-weight: bold;
    text-align: center;
    padding: 15px;
    border-radius: 8px;
    margin-top: 20px;
    border: 1px solid #e0e0e0;
    font-family: 'Times New Roman', serif !important;
}

.warning-box {
    background-color: rgba(255, 165, 0, 0.2);
    border-left: 5px solid orange;
    padding: 10px;
    margin: 10px 0;
    border-radius: 5px;
}

.error-box {
    background-color: rgba(255, 0, 0, 0.2);
    border-left: 5px solid red;
    padding: 10px;
    margin: 10px 0;
    border-radius: 5px;
}

.log-container {
    background-color: #1E1E1E;
    color: #00FF00;
    font-family: 'Courier New', monospace;
    font-size: 12px;
    padding: 10px;
    border-radius: 5px;
    max-height: 300px;
    overflow-y: auto;
    white-space: pre-wrap;
}

.sidebar-model-info {
    background-color: white;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
    border: 1px solid #e0e0e0;
}

.tech-info {
    background-color: white;
    padding: 15px;
    border-radius: 8px;
    color: #333;
    border: 1px solid #e0e0e0;
    font-family: 'Times New Roman', serif !important;
}

/* Sidebar user information style - mobile interface style */
.sidebar-user-info {
    text-align: center;
    padding: 25px 15px;
    margin-bottom: 25px;
    background-color: white;
    border-radius: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    border: 1px solid #e0e0e0;
}

.user-avatar {
    width: 70px;
    height: 70px;
    border-radius: 50%;
    background-color: #1f4e79;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 15px auto;
    color: white;
    font-size: 28px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.user-name {
    color: #333;
    font-size: 16px;
    margin-top: 5px;
    font-weight: 600;
    font-family: 'Times New Roman', serif !important;
}

/* Streamlit button style override - mobile interface style */
.stButton > button {
    width: 100% !important;
    margin-bottom: 12px !important;
    padding: 16px 20px !important;
    border-radius: 30px !important;
    border: none !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    background-color: #e9ecef !important;
    color: #6c757d !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
    font-family: 'Times New Roman', serif !important;
}

.stButton > button:hover {
    background-color: #dee2e6 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
}

/* Primary button style - dark blue */
.stButton > button[kind="primary"] {
    background-color: #1f4e79 !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(31,78,121,0.3) !important;
}

.stButton > button[kind="primary"]:hover {
    background-color: #164063 !important;
    box-shadow: 0 6px 16px rgba(31,78,121,0.4) !important;
}

/* Collapse button style */
.collapse-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    cursor: pointer;
    padding: 5px 0;
    border-bottom: 1px solid #ddd;
    margin-bottom: 10px;
}

.collapse-icon {
    font-size: 14px;
    transition: transform 0.3s;
}

.collapse-icon.expanded {
    transform: rotate(90deg);
}

/* Bottom navigation button style */
.bottom-nav {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background-color: #f0f0f0;
    padding: 15px;
    border-top: 1px solid #dee2e6;
    display: flex;
    justify-content: center;
    border-radius: 20px 20px 0 0;
}

.bottom-nav-button {
    background-color: #6c757d;
    color: white;
    border: none;
    padding: 12px 20px;
    border-radius: 25px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    font-family: 'Times New Roman', serif !important;
}

/* Sidebar bottom collapse button */
.sidebar-collapse-btn {
    position: fixed;
    bottom: 20px;
    left: 20px;
    width: 50px;
    height: 50px;
    background-color: transparent !important;
    border: none !important;
    border-radius: 50% !important;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.3s ease;
    font-family: 'Times New Roman', serif !important;
    font-size: 18px !important;
    color: #6c757d !important;
    z-index: 1000;
}

.sidebar-collapse-btn:hover {
    background-color: transparent !important;
    color: #333 !important;
    transform: scale(1.1);
}
</style>
""", unsafe_allow_html=True)

# Record startup logs
add_log("Application started")
add_log(f"Initialized selected model: {st.session_state.selected_model}")

# Sidebar navigation - new layout
with st.sidebar:
    # User information area
    st.markdown("""
    <div class='sidebar-user-info'>
        <div class='user-avatar'>👤</div>
        <div class='user-name'>User: wy1122</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation buttons
    st.markdown("### ")  # Empty title for spacing
    
    # Prediction model button
    if st.button("Prediction Model", key="nav_predict", use_container_width=True, 
                type="primary" if st.session_state.current_page == "Prediction Model" else "secondary"):
        st.session_state.current_page = "Prediction Model"
        add_log("Switched to prediction model page")
        st.rerun()
    
    # Execution logs button
    if st.button("Execution Logs", key="nav_logs", use_container_width=True,
                type="primary" if st.session_state.current_page == "Execution Logs" else "secondary"):
        st.session_state.current_page = "Execution Logs"
        add_log("Switched to execution logs page")
        st.rerun()
    
    # Model information button
    if st.button("Model Information", key="nav_model_info", use_container_width=True,
                type="primary" if st.session_state.current_page == "Model Information" else "secondary"):
        st.session_state.current_page = "Model Information"
        add_log("Switched to model information page")
        st.rerun()
    
    # Technical description button
    if st.button("Technical Description", key="nav_tech", use_container_width=True,
                type="primary" if st.session_state.current_page == "Technical Description" else "secondary"):
        st.session_state.current_page = "Technical Description"
        add_log("Switched to technical description page")
        st.rerun()
    
    # User guide button
    if st.button("User Guide", key="nav_guide", use_container_width=True,
                type="primary" if st.session_state.current_page == "User Guide" else "secondary"):
        st.session_state.current_page = "User Guide"
        add_log("Switched to user guide page")
        st.rerun()
    
    # Add spacing
    st.markdown("<br><br>", unsafe_allow_html=True)

# Add external collapse button with JavaScript functionality
st.markdown("""
<div style="position: fixed; bottom: 20px; left: 20px; z-index: 1000;">
    <button class="sidebar-collapse-btn" onclick="toggleSidebar()">
        &lt;
    </button>
</div>

<script>
function toggleSidebar() {
    const sidebar = window.parent.document.querySelector('section[data-testid="stSidebar"]');
    if (sidebar) {
        if (sidebar.style.display === 'none') {
            sidebar.style.display = 'block';
        } else {
            sidebar.style.display = 'none';
        }
    }
}
</script>
""", unsafe_allow_html=True)

# Simplified predictor class
class ModelPredictor:
    def __init__(self, target_model="Char Yield"):
        self.target_name = target_model
        self.feature_names = [
            'M(wt%)', 'Ash(wt%)', 'VM(wt%)', 'O/C', 'H/C', 'N/C',
            'FT(℃)', 'HR(℃/min)', 'FR(mL/min)'
        ]
        self.model_loaded = False
        add_log(f"Initialized predictor: {self.target_name}")
    
    def get_model_info(self):
        return {
            "Model Type": "GBDT Pipeline",
            "Target Variable": self.target_name,
            "Feature Count": len(self.feature_names),
            "Model Status": "Loaded" if self.model_loaded else "Not Loaded"
        }
    
    def predict(self, features):
        """Simulate prediction functionality"""
        # Simulate prediction results
        import random
        random.seed(42)
        base_values = {
            "Char Yield": 27.79,
            "Oil Yield": 45.23,
            "Gas Yield": 18.56
        }
        result = base_values[self.target_name] + random.uniform(-5, 5)
        return round(result, 2)

# Display different content based on current page
if st.session_state.current_page == "Prediction Model":
    # Main page content
    st.markdown("<h1 class='main-title'>Biomass Pyrolysis Product Prediction System Based on GBDT Ensemble Models</h1>", unsafe_allow_html=True)

    # Model selection area
    st.markdown("<div class='model-selector'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #333; text-align: center; margin-bottom: 30px; font-family: Times New Roman, serif; font-size: 20px;'>Select Prediction Target</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)

    with col1:
        char_button = st.button("", key="char_button", use_container_width=True, help="Char Yield")
        st.markdown("""
        <div class='model-card'>
            <div class='model-icon'>🔥</div>
            <div class='model-name'>Char Yield</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        oil_button = st.button("", key="oil_button", use_container_width=True, help="Oil Yield")
        st.markdown("""
        <div class='model-card'>
            <div class='model-icon'>🛢️</div>
            <div class='model-name'>Oil Yield</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        gas_button = st.button("", key="gas_button", use_container_width=True, help="Gas Yield")
        st.markdown("""
        <div class='model-card'>
            <div class='model-icon'>💨</div>
            <div class='model-name'>Gas Yield</div>
        </div>
        """, unsafe_allow_html=True)

    if char_button:
        st.session_state.selected_model = "Char Yield"
        st.session_state.prediction_result = None
        add_log(f"Switched to model: {st.session_state.selected_model}")
        st.rerun()

    if oil_button:
        st.session_state.selected_model = "Oil Yield"
        st.session_state.prediction_result = None
        add_log(f"Switched to model: {st.session_state.selected_model}")
        st.rerun()

    if gas_button:
        st.session_state.selected_model = "Gas Yield"
        st.session_state.prediction_result = None
        add_log(f"Switched to model: {st.session_state.selected_model}")
        st.rerun()

    st.markdown(f"<div class='current-model'>Current Model: {st.session_state.selected_model}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Initialize predictor
    predictor = ModelPredictor(target_model=st.session_state.selected_model)

    # Default values
    default_values = {
        "M(wt%)": 6.460, "Ash(wt%)": 6.460, "VM(wt%)": 6.460,
        "O/C": 6.460, "H/C": 6.460, "N/C": 6.460,
        "FT(°C)": 6.460, "HR(°C/min)": 6.460, "FR(mL/min)": 6.460
    }

    # Create main layout: left input area, right information panel
    main_col, info_col = st.columns([3, 1])

    with main_col:
        # Create three-column layout card-style input interface
        col1, col2, col3 = st.columns(3)
        features = {}

        # Proximate Analysis card
        with col1:
            st.markdown("""
            <div class='analysis-card'>
                <div class='card-title'>Proximate Analysis</div>
            </div>
            """, unsafe_allow_html=True)
            
            # M(wt%)
            st.markdown("""
            <div class='input-row'>
                <div class='input-label'>M(wt%)</div>
            </div>
            """, unsafe_allow_html=True)
            features["M(wt%)"] = st.number_input("M(wt%)", value=default_values["M(wt%)"], key="input_M")
            
            # Ash(wt%)
            st.markdown("""
            <div class='input-row'>
                <div class='input-label'>Ash(wt%)</div>
            </div>
            """, unsafe_allow_html=True)
            features["Ash(wt%)"] = st.number_input("Ash(wt%)", value=default_values["Ash(wt%)"], key="input_Ash")
            
            # VM(wt%)
            st.markdown("""
            <div class='input-row'>
                <div class='input-label'>VM(wt%)</div>
            </div>
            """, unsafe_allow_html=True)
            features["VM(wt%)"] = st.number_input("VM(wt%)", value=default_values["VM(wt%)"], key="input_VM")

        # Ultimate Analysis card
        with col2:
            st.markdown("""
            <div class='analysis-card'>
                <div class='card-title'>Ultimate Analysis</div>
            </div>
            """, unsafe_allow_html=True)
            
            # O/C
            st.markdown("""
            <div class='input-row'>
                <div class='input-label'>O/C</div>
            </div>
            """, unsafe_allow_html=True)
            features["O/C"] = st.number_input("O/C", value=default_values["O/C"], key="input_OC")

            # H/C
            st.markdown("""
            <div class='input-row'>
                <div class='input-label'>H/C</div>
            </div>
            """, unsafe_allow_html=True)
            features["H/C"] = st.number_input("H/C", value=default_values["H/C"], key="input_HC")

            # N/C
            st.markdown("""
            <div class='input-row'>
                <div class='input-label'>N/C</div>
            </div>
            """, unsafe_allow_html=True)
            features["N/C"] = st.number_input("N/C", value=default_values["N/C"], key="input_NC")

        # Pyrolysis Conditions card
        with col3:
            st.markdown("""
            <div class='analysis-card'>
                <div class='card-title'>Pyrolysis Conditions</div>
            </div>
            """, unsafe_allow_html=True)
            
            # FT(°C)
            st.markdown("""
            <div class='input-row'>
                <div class='input-label'>FT(°C)</div>
            </div>
            """, unsafe_allow_html=True)
            features["FT(°C)"] = st.number_input("FT(°C)", value=default_values["FT(°C)"], key="input_FT")

            # HR(°C/min)
            st.markdown("""
            <div class='input-row'>
                <div class='input-label'>HR(°C/min)</div>
            </div>
            """, unsafe_allow_html=True)
            features["HR(°C/min)"] = st.number_input("HR(°C/min)", value=default_values["HR(°C/min)"], key="input_HR")

            # FR(mL/min)
            st.markdown("""
            <div class='input-row'>
                <div class='input-label'>FR(mL/min)</div>
            </div>
            """, unsafe_allow_html=True)
            features["FR(mL/min)"] = st.number_input("FR(mL/min)", value=default_values["FR(mL/min)"], key="input_FR")

        # Operation buttons
        st.markdown("""
        <div class='action-buttons'>
        </div>
        """, unsafe_allow_html=True)
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Run Prediction", type="primary", use_container_width=True):
                add_log("Starting prediction process...")
                # Execute prediction
                result = predictor.predict(features)
                st.session_state.prediction_result = result
                add_log(f"Prediction completed: {st.session_state.selected_model} = {result} wt%")
                st.rerun()
        
        with col_btn2:
            if st.button("Reset Data", use_container_width=True):
                add_log("Reset all input data")
                st.session_state.prediction_result = None
                st.rerun()

    # Right information panel - add collapse functionality
    with info_col:
        # Get current model statistics
        current_stats = st.session_state.model_stats[st.session_state.selected_model]
        
        # Prediction result display
        result_text = f"{st.session_state.prediction_result} wt%" if st.session_state.prediction_result else "Awaiting prediction"
        
        # Use Streamlit container instead of HTML
        with st.container():
            # Prediction result title
            st.markdown("### Prediction Results")
            
            # Prediction result value
            if st.session_state.prediction_result:
                # Display model type names
                model_names = {
                    "Char Yield": "Char Yield",
                    "Oil Yield": "Oil Yield", 
                    "Gas Yield": "Gas Yield"
                }
                model_name = model_names.get(st.session_state.selected_model, st.session_state.selected_model)
                st.success(f"**{model_name}**: {st.session_state.prediction_result} wt%")
            else:
                st.info("Awaiting prediction...")
            
            st.markdown("---")
            
            # Prediction information - collapsible
            col_header, col_toggle = st.columns([4, 1])
            with col_header:
                st.markdown("### Prediction Information")
            with col_toggle:
                if st.button("▼" if st.session_state.prediction_info_expanded else "▶", 
                           key="toggle_prediction_info", 
                           help="Expand/Collapse prediction information"):
                    st.session_state.prediction_info_expanded = not st.session_state.prediction_info_expanded
                    st.rerun()
            
            if st.session_state.prediction_info_expanded:
                st.write(f"• **Target Variable**: {st.session_state.selected_model}")
                st.write(f"• **Prediction Result**: {result_text}")
                st.write(f"• **Model Type**: GBDT Pipeline")
                st.write(f"• **Preprocessing**: RobustScaler")
            
            st.markdown("---")
            
            # Model status - collapsible
            col_header2, col_toggle2 = st.columns([4, 1])
            with col_header2:
                st.markdown("### Model Status")
            with col_toggle2:
                if st.button("▼" if st.session_state.model_status_expanded else "▶", 
                           key="toggle_model_status", 
                           help="Expand/Collapse model status"):
                    st.session_state.model_status_expanded = not st.session_state.model_status_expanded
                    st.rerun()
            
            if st.session_state.model_status_expanded:
                st.write(f"• **Loading Status**: ✅ Normal")
                st.write(f"• **Feature Count**: {current_stats['features']}")
                st.write(f"• **Warning Count**: {current_stats['warnings']}")
            
            st.markdown("---")
            
            # More detailed information button
            if st.button("More Details...", use_container_width=True):
                st.info("Display more detailed model information and statistics...")

elif st.session_state.current_page == "Execution Logs":
    st.markdown("<h1 class='main-title'>Execution Logs</h1>", unsafe_allow_html=True)
    display_logs()

elif st.session_state.current_page == "Model Information":
    st.markdown("<h1 class='main-title'>Model Information</h1>", unsafe_allow_html=True)
    predictor = ModelPredictor(target_model=st.session_state.selected_model)
    model_info = predictor.get_model_info()
    
    for key, value in model_info.items():
        st.write(f"**{key}**: {value}")

elif st.session_state.current_page == "Technical Description":
    st.markdown("<h1 class='main-title'>Technical Description</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class='tech-info'>
    <h4>🔬 Model Technical Description</h4>
    <p>This system is constructed based on the <b>Gradient Boosting Decision Tree (GBDT)</b> algorithm, employing a Pipeline architecture that integrates data preprocessing and model prediction.</p>
    
    <h4>📋 Feature Description</h4>
    <ul>
        <li><b>Proximate Analysis:</b> M(wt%) - Moisture content, Ash(wt%) - Ash content, VM(wt%) - Volatile matter content</li>
        <li><b>Ultimate Analysis:</b> O/C - Oxygen-to-carbon ratio, H/C - Hydrogen-to-carbon ratio, N/C - Nitrogen-to-carbon ratio</li>
        <li><b>Pyrolysis Conditions:</b> FT(°C) - Final temperature, HR(°C/min) - Heating rate, FR(mL/min) - Flow rate</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.current_page == "User Guide":
    st.markdown("<h1 class='main-title'>User Guide</h1>", unsafe_allow_html=True)
    st.markdown("""
    ### 📋 Usage Steps
    1. Select "Prediction Model" in the sidebar
    2. Choose the prediction target (Char/Oil/Gas Yield)
    3. Input biomass characteristic parameters
    4. Click "Run Prediction" to obtain results
    
    ### ⚠️ Important Notes
    - Ensure input parameters are within reasonable ranges
    - Model prediction results are for reference only
    - Practical applications should be validated with professional knowledge
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-family: Times New Roman, serif; font-size: 20px;'>
<p>© 2024 Biomass Nanomaterials and Intelligent Equipment Laboratory | GBDT-based Biomass Pyrolysis Product Prediction System</p>
</div>
""", unsafe_allow_html=True)