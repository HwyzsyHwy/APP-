# -*- coding: utf-8 -*-
"""
Biomass Pyrolysis Yield Forecast using GBDT Ensemble Models
修复版本 - 根据实际特征统计信息正确调整
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
    initial_sidebar_state='expanded'
)

# 自定义样式（添加背景图片）
st.markdown(
    """
    <style>
    /* 全局字体设置和背景图片 */
    html, body, [class*="css"] {
        font-size: 16px !important;
    }

    /* 主应用背景 */
    .stApp {
        background-image: url('https://raw.githubusercontent.com/HwyzsyHwy/APP-/main/背景.png');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    /* 左侧边栏背景 */
    .css-1d391kg, .css-1lcbmhc, .css-1outpf7, section[data-testid="stSidebar"] {
        background-color: #f8f9fa !important;
    }

    /* 左侧边栏内容文字颜色 */
    section[data-testid="stSidebar"] * {
        color: #333333 !important;
    }

    /* 左侧边栏标题颜色 */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #333333 !important;
    }

    /* 调整主内容区域布局 - 三栏布局 */
    .main .block-container {
        max-width: 100% !important;
        margin-right: 0px !important;
        padding-left: 10px !important;
        padding-right: 10px !important;
    }

    /* 右侧参数显示区域样式 */
    .param-display-item {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 8px;
        padding: 8px 12px;
        margin: 5px 0;
        border-left: 4px solid #20b2aa;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .param-name {
        font-weight: bold;
        color: #333;
        font-size: 13px;
    }

    .param-value {
        color: #20b2aa;
        font-weight: bold;
        font-size: 14px;
    }

    /* 预测结果显示样式 */
    .prediction-display {
        background: linear-gradient(135deg, #20b2aa, #17a2b8);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(32, 178, 170, 0.3);
        margin-top: 15px;
    }

    .prediction-model {
        font-size: 12px;
        margin-bottom: 5px;
        opacity: 0.9;
    }

    .prediction-result {
        font-size: 18px;
        font-weight: bold;
    }

    /* 用户信息区域 */
    .user-info {
        text-align: center;
        padding: 20px 10px;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 20px;
    }

    .user-avatar {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        margin: 0 auto 10px auto;
        display: block;
        background-color: #20b2aa;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 24px;
    }

    .user-name {
        font-size: 16px;
        color: #333;
        margin: 0;
    }

    /* 导航按钮样式 */
    .nav-button {
        width: 100%;
        padding: 12px 20px;
        margin: 8px 0;
        border: none;
        border-radius: 25px;
        font-size: 16px;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: center;
        display: block;
        text-decoration: none;
    }

    .nav-button.active {
        background-color: #20b2aa !important;
        color: white !important;
    }

    .nav-button.inactive {
        background-color: #e0e0e0 !important;
        color: #666 !important;
    }

    .nav-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    /* 隐藏默认的streamlit按钮样式 */
    section[data-testid="stSidebar"] .stButton button {
        width: 100%;
        padding: 12px 20px;
        margin: 8px 0;
        border: none;
        border-radius: 25px;
        font-size: 16px;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: center;
    }

    /* 创建统一的整体白色半透明背景 */
    .main .block-container {
        background-color: rgba(255, 255, 255, 0.85) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 20px !important;
        padding: 30px !important;
        margin: 20px auto !important;
        max-width: 1200px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        min-height: 80vh !important;
    }

    /* 移除所有子元素的单独背景，让它们显示在统一背景上 */
    .main .block-container .stMarkdown,
    .main .block-container .stText,
    .main .block-container .stExpander,
    .main .block-container .stSelectbox,
    .main .block-container .stButton,
    .main .block-container .stDataFrame,
    .main .block-container .stMetric,
    .main .block-container .streamlit-expanderHeader,
    .main .block-container .streamlit-expanderContent,
    .main .block-container p,
    .main .block-container h1,
    .main .block-container h2,
    .main .block-container h3,
    .main .block-container h4,
    .main .block-container h5,
    .main .block-container h6,
    .main .block-container ul,
    .main .block-container li,
    .main .block-container div {
        background-color: transparent !important;
        backdrop-filter: none !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        border: none !important;
    }

    /* 确保文本颜色在白色背景上清晰可见 */
    .main .block-container * {
        color: #333 !important;
    }

    /* 标题样式 - 在统一背景上显示 */
    .main-title {
        text-align: center;
        font-size: 32px !important;
        font-weight: bold;
        margin-bottom: 20px;
        color: #333 !important;
        text-shadow: none !important;
        background-color: transparent !important;
        padding: 15px !important;
    }

    /* 区域标题样式 - 在统一背景上显示 */
    .section-header {
        color: #333 !important;
        font-weight: bold;
        font-size: 22px;
        text-align: center;
        padding: 10px;
        margin-bottom: 15px;
        background-color: transparent !important;
    }

    /* 输入标签样式 - 在统一背景上显示 */
    .input-label {
        padding: 5px;
        margin-bottom: 5px;
        font-size: 18px;
        color: #333 !important;
        background-color: transparent !important;
        font-weight: 500 !important;
    }
    
    /* 结果显示样式 */
    .yield-result {
        background-color: rgba(255, 255, 255, 0.8) !important;
        color: white;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        padding: 15px;
        border-radius: 8px;
        margin-top: 20px;
        backdrop-filter: blur(5px) !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
    }
    
    /* 强制应用白色背景到输入框 */
    [data-testid="stNumberInput"] input {
        background-color: white !important;
        color: black !important;
    }
    
    /* 增大按钮的字体 */
    .stButton button {
        font-size: 18px !important;
    }
    
    /* 警告样式 */
    .warning-box {
        background-color: rgba(255, 255, 255, 0.8);
        border-left: 5px solid orange;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
        backdrop-filter: blur(3px);
        color: #333;
    }

    /* 错误样式 */
    .error-box {
        background-color: rgba(255, 255, 255, 0.8);
        border-left: 5px solid red;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
        backdrop-filter: blur(3px);
        color: #333;
    }

    /* 成功样式 */
    .success-box {
        background-color: rgba(255, 255, 255, 0.8);
        border-left: 5px solid green;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
        backdrop-filter: blur(3px);
        color: #333;
    }
    
    /* 日志样式 - 保留背景框 */
    .log-container {
        height: 300px;
        overflow-y: auto;
        background-color: rgba(255, 255, 255, 0.8);
        color: #00FF00;
        font-family: 'Courier New', monospace;
        padding: 10px;
        border-radius: 5px;
        font-size: 14px !important;
        backdrop-filter: blur(5px);
        text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
    }

    /* 页面内容样式 - 与日志容器相同的白色半透明背景 */
    .page-content {
        background-color: rgba(255, 255, 255, 0.8) !important;
        color: #333 !important;
        padding: 20px !important;
        border-radius: 15px !important;
        backdrop-filter: blur(5px) !important;
        margin: 10px 0 !important;
        min-height: 400px !important;
    }

    /* 确保页面内容内的所有元素都没有单独背景 */
    .page-content * {
        background-color: transparent !important;
        backdrop-filter: none !important;
    }
    
    /* 模型选择器样式 */
    .model-selector {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
        backdrop-filter: blur(5px);
        color: white;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
    }

    /* 侧边栏导航按钮基础样式 - 灰色背景，矩形样式 */
    .stSidebar [data-testid="stButton"] > button {
        background-color: rgba(128, 128, 128, 0.7) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 20px !important;
        margin: 8px 0 !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        font-size: 16px !important;
    }

    /* 选中状态的侧边栏导航按钮 - 绿色高亮 */
    .stSidebar [data-testid="stButton"] > button[kind="primary"] {
        background-color: #20b2aa !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 8px rgba(32, 178, 170, 0.3) !important;
    }

    /* 侧边栏导航按钮悬停效果 */
    .stSidebar [data-testid="stButton"] > button:hover {
        background-color: rgba(100, 100, 100, 0.8) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
    }

    /* 选中的侧边栏按钮悬停效果 */
    .stSidebar [data-testid="stButton"] > button[kind="primary"]:hover {
        background-color: #1a9a92 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(32, 178, 170, 0.4) !important;
    }

    /* 自定义导航按钮样式 */
    .nav-button {
        background-color: rgba(128, 128, 128, 0.7);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 15px;
        margin: 5px 0;
        font-weight: 500;
        transition: all 0.3s ease;
        width: 100%;
        text-align: center;
        cursor: pointer;
        font-size: 14px;
    }

    /* 选中状态的导航按钮 */
    .nav-button-active {
        background-color: rgba(0, 150, 136, 0.9) !important;
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 10px 15px;
        margin: 5px 0;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
        text-align: center;
        cursor: pointer;
        font-size: 14px;
        box-shadow: 0 2px 4px rgba(0, 150, 136, 0.4);
    }

    /* 导航按钮悬停效果 */
    .nav-button:hover {
        background-color: rgba(100, 100, 100, 0.8);
        transform: translateY(-1px);
    }

    .nav-button-active:hover {
        background-color: rgba(0, 121, 107, 1.0) !important;
        transform: translateY(-1px);
    }

    /* 模型切换按钮组样式 */
    div[data-testid="stHorizontalBlock"] [data-testid="stButton"] {
        margin: 0 5px;
    }
    
    /* 填满屏幕 */
    .stApp {
        width: 100%;
        min-width: 100%;
        margin: 0 auto;
    }
    
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    
    /* 侧边栏模型信息样式 */
    .sidebar-model-info {
        background-color: #2E2E2E;
        padding: 10px;
        border-radius: 5px;
        margin-top: 20px;
    }
    
    /* 性能指标样式 */
    .performance-metrics {
        background-color: #2E2E2E;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }
    
    /* 技术说明样式 */
    .tech-info {
        background-color: #2E2E2E;
        padding: 15px;
        border-radius: 8px;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 初始化会话状态 - 添加页面导航
if 'current_page' not in st.session_state:
    st.session_state.current_page = "预测模型"

# 创建左侧边栏导航
with st.sidebar:
    # 用户信息区域
    st.markdown("""
    <div class="user-info">
        <img src="https://raw.githubusercontent.com/HwyzsyHwy/APP-/main/用户.png" class="user-avatar" alt="用户头像">
        <p class="user-name">用户：wy1122</p>
    </div>
    """, unsafe_allow_html=True)

    # 导航按钮
    col1, = st.columns([1])

    with col1:
        # 创建导航按钮 - 使用更直接的方法处理状态
        current_page = st.session_state.current_page

        # 预测模型按钮
        if st.button("预测模型", key="nav_predict", use_container_width=True, type="primary" if current_page == "预测模型" else "secondary"):
            st.session_state.current_page = "预测模型"
            st.rerun()

        # 执行日志按钮
        if st.button("执行日志", key="nav_log", use_container_width=True, type="primary" if current_page == "执行日志" else "secondary"):
            st.session_state.current_page = "执行日志"
            st.rerun()

        # 模型信息按钮
        if st.button("模型信息", key="nav_info", use_container_width=True, type="primary" if current_page == "模型信息" else "secondary"):
            st.session_state.current_page = "模型信息"
            st.rerun()

        # 技术说明按钮
        if st.button("技术说明", key="nav_tech", use_container_width=True, type="primary" if current_page == "技术说明" else "secondary"):
            st.session_state.current_page = "技术说明"
            st.rerun()

        # 使用指南按钮
        if st.button("使用指南", key="nav_guide", use_container_width=True, type="primary" if current_page == "使用指南" else "secondary"):
            st.session_state.current_page = "使用指南"
            st.rerun()

# 创建日志区域（仅在执行日志页面显示）
if st.session_state.current_page == "执行日志":
    log_container = st.sidebar.container()
    log_text = st.sidebar.empty()
else:
    log_container = None
    log_text = None

# 初始化日志字符串
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []

def log(message):
    """记录日志到侧边栏和会话状态"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    st.session_state.log_messages.append(log_entry)
    # 只保留最近的100条日志
    if len(st.session_state.log_messages) > 100:
        st.session_state.log_messages = st.session_state.log_messages[-100:]

    # 只在执行日志页面时更新日志显示
    if st.session_state.current_page == "执行日志" and log_text is not None:
        log_text.markdown(
            f"<div class='log-container'>{'<br>'.join(st.session_state.log_messages)}</div>",
            unsafe_allow_html=True
        )

# 记录启动日志
log("应用启动 - 根据图片特征统计信息正确修复版本")
log("特征顺序：M, Ash, VM, O/C, H/C, N/C, FT, HR, FR")

# 初始化会话状态 - 添加模型选择功能
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Char Yield"  # 默认选择Char产率模型
    log(f"初始化选定模型: {st.session_state.selected_model}")

# 添加模型缓存 - 避免重复加载相同模型
if 'model_cache' not in st.session_state:
    st.session_state.model_cache = {}
    
# 只在预测模型页面显示标题和模型选择器
if st.session_state.current_page == "预测模型":
    # 简洁的Streamlit样式标题
    st.markdown("""
    <div style="margin-bottom: 30px;">
        <h1 style="color: white; font-size: 2.5rem; font-weight: bold; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">
            Streamlit
        </h1>
        <div style="height: 3px; background: white; margin-top: 5px; border-radius: 2px;"></div>
    </div>
    """, unsafe_allow_html=True)

    # 添加模型选择区域 - 修改为可点击卡片样式
    st.markdown("<h3 style='color: white; text-align: center; margin-bottom: 30px;'>选择预测目标</h3>", unsafe_allow_html=True)

    # 添加模型选择卡片的自定义样式
    st.markdown("""
    <style>
    /* 模型选择卡片容器 */
    .model-card-container {
        display: flex;
        gap: 15px;
        margin: 20px 0;
        justify-content: space-between;
    }

    /* 模型选择卡片样式 */
    .model-card {
        flex: 1;
        height: 120px;
        border-radius: 15px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
        position: relative;
        padding: 20px;
        box-sizing: border-box;
    }

    /* 未选中状态的卡片 - 白色轻微透明背景 */
    .model-card.unselected {
        background: rgba(255,255,255,0.8);
        color: #333;
        border: 2px solid rgba(255,255,255,0.3);
    }

    /* 选中状态的卡片 - 白色背景 */
    .model-card.selected {
        background: rgba(255,255,255,0.95);
        color: #333;
        border: 2px solid rgba(255,255,255,0.5);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    /* 悬停效果 */
    .model-card:hover {
        transform: translateY(-1px);
    }

    /* 选中卡片的悬停效果 */
    .model-card.selected:hover {
        background-color: rgba(0, 121, 107, 1.0);
    }

    /* 图标样式 */
    .model-card-icon {
        width: 40px;
        height: 40px;
        margin-bottom: 10px;
    }

    /* 文字样式 */
    .model-card-text {
        font-size: 16px;
        font-weight: bold;
        text-align: center;
    }

    /* 让整个卡片可点击 */
    .model-card {
        cursor: pointer !important;
        position: relative !important;
    }

    /* 模型卡片按钮样式 - secondary按钮（未选中） */
    div[data-testid="stHorizontalBlock"] .stButton > button[kind="secondary"],
    div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
        background: rgba(255,255,255,0.1) !important;
        border: 2px solid rgba(255,255,255,0.3) !important;
        border-radius: 15px !important;
        padding: 20px !important;
        height: auto !important;
        min-height: 120px !important;
        color: white !important;
        font-weight: bold !important;
        font-size: 16px !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1) !important;
        transition: all 0.3s ease !important;
    }

    /* 模型卡片按钮样式 - primary按钮（选中） */
    div[data-testid="stHorizontalBlock"] .stButton > button[kind="primary"],
    div[data-testid="stHorizontalBlock"] button[kind="primary"] {
        background: linear-gradient(135deg, #20b2aa, #17a2b8) !important;
        border: 3px solid #20b2aa !important;
        border-radius: 15px !important;
        padding: 20px !important;
        height: auto !important;
        min-height: 120px !important;
        color: white !important;
        font-weight: bold !important;
        font-size: 16px !important;
        box-shadow: 0 12px 40px rgba(32, 178, 170, 0.3) !important;
        transform: translateY(-2px) !important;
        transition: all 0.3s ease !important;
    }

    /* 悬停效果 */
    div[data-testid="stHorizontalBlock"] .stButton > button[kind="secondary"]:hover,
    div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {
        background: rgba(255,255,255,0.2) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 40px rgba(255,255,255,0.1) !important;
    }

    div[data-testid="stHorizontalBlock"] .stButton > button[kind="primary"]:hover,
    div[data-testid="stHorizontalBlock"] button[kind="primary"]:hover {
        background: linear-gradient(135deg, #17a2b8, #20b2aa) !important;
        transform: translateY(-4px) !important;
        box-shadow: 0 16px 50px rgba(32, 178, 170, 0.4) !important;
    }

    </style>
    """, unsafe_allow_html=True)

    # 模型选择卡片
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔥\n\nChar Yield", key="char_card", use_container_width=True,
                    type="primary" if st.session_state.selected_model == "Char Yield" else "secondary"):
            if st.session_state.selected_model != "Char Yield":
                st.session_state.selected_model = "Char Yield"
                st.session_state.prediction_result = None
                st.session_state.warnings = []
                log(f"切换到模型: {st.session_state.selected_model}")
                st.rerun()

    with col2:
        if st.button("⚡️\n\nOil Yield", key="oil_card", use_container_width=True,
                    type="primary" if st.session_state.selected_model == "Oil Yield" else "secondary"):
            if st.session_state.selected_model != "Oil Yield":
                st.session_state.selected_model = "Oil Yield"
                st.session_state.prediction_result = None
                st.session_state.warnings = []
                log(f"切换到模型: {st.session_state.selected_model}")
                st.rerun()

    with col3:
        if st.button("💨\n\nGas Yield", key="gas_card", use_container_width=True,
                    type="primary" if st.session_state.selected_model == "Gas Yield" else "secondary"):
            if st.session_state.selected_model != "Gas Yield":
                st.session_state.selected_model = "Gas Yield"
                st.session_state.prediction_result = None
                st.session_state.warnings = []
                log(f"切换到模型: {st.session_state.selected_model}")
                st.rerun()

    # 添加CSS和JavaScript来强制改变按钮颜色
    selected_model = st.session_state.selected_model
    st.markdown(f"""
    <style>
    /* 强制覆盖所有按钮样式 */
    button[kind="secondary"],
    .stButton > button[kind="secondary"],
    [data-testid="stButton"] > button[kind="secondary"] {{
        background: rgba(255,255,255,0.1) !important;
        border: 2px solid rgba(255,255,255,0.3) !important;
        color: white !important;
        transition: all 0.3s ease !important;
    }}

    /* 强制设置primary按钮为青绿色 - 使用最高优先级选择器 */
    html body div.stApp button[kind="primary"],
    html body div.stApp .stButton > button[kind="primary"],
    html body div.stApp [data-testid="stButton"] > button[kind="primary"],
    button[kind="primary"],
    .stButton > button[kind="primary"],
    [data-testid="stButton"] > button[kind="primary"] {{
        background: #20b2aa !important;
        background-color: #20b2aa !important;
        background-image: none !important;
        border: 3px solid #20b2aa !important;
        border-color: #20b2aa !important;
        color: white !important;
        box-shadow: 0 8px 25px rgba(32, 178, 170, 0.4) !important;
        transform: translateY(-2px) !important;
        font-weight: 600 !important;
    }}

    /* 数字输入框按钮的强制样式 - 使用最强的选择器 */
    html body div.stApp button[aria-label="Increment"],
    html body div.stApp button[aria-label="Decrement"],
    html body div.stApp button[title="Increment"],
    html body div.stApp button[title="Decrement"],
    html body div.stApp [data-testid="stNumberInput"] button,
    html body div.stApp .stNumberInput button,
    html body div.stApp div[data-baseweb="input"] button,
    html body div.stApp input[type="number"] + button,
    html body div.stApp input[type="number"] ~ button,
    html body div.stApp button:has(svg),
    html body div.stApp button[kind="secondary"],
    button[aria-label="Increment"],
    button[aria-label="Decrement"],
    button[title="Increment"],
    button[title="Decrement"],
    [data-testid="stNumberInput"] button,
    .stNumberInput button,
    div[data-baseweb="input"] button,
    input[type="number"] + button,
    input[type="number"] ~ button,
    button:has(svg),
    button[kind="secondary"] {{
        color: white !important;
        border: none !important;
        font-weight: bold !important;
        border-radius: 4px !important;
        min-width: 24px !important;
        min-height: 24px !important;
        transition: all 0.2s ease !important;
        background-color: #20b2aa !important; /* 强制青绿色 */
        background: #20b2aa !important;
        background-image: none !important;
    }}

    /* 第一列按钮 - 青绿色 (Proximate Analysis) - 超高优先级 */
    div[data-testid="column"]:nth-child(1) button[aria-label="Increment"],
    div[data-testid="column"]:nth-child(1) button[aria-label="Decrement"],
    div[data-testid="column"]:nth-child(1) button[title="Increment"],
    div[data-testid="column"]:nth-child(1) button[title="Decrement"],
    div[data-testid="column"]:nth-child(1) [data-testid="stNumberInput"] button,
    div[data-testid="column"]:nth-child(1) button:has(svg),
    div[data-testid="column"]:nth-child(1) .stNumberInput button,
    div[data-testid="column"]:nth-child(1) div[data-baseweb="input"] button {{
        background-color: #20b2aa !important;
        background: #20b2aa !important;
        color: white !important;
        border: 2px solid #20b2aa !important;
    }}

    /* 第二列按钮 - 金黄色 (Ultimate Analysis) */
    [data-testid="column"]:nth-child(2) button[aria-label="Increment"],
    [data-testid="column"]:nth-child(2) button[aria-label="Decrement"],
    [data-testid="column"]:nth-child(2) button[title="Increment"],
    [data-testid="column"]:nth-child(2) button[title="Decrement"],
    [data-testid="column"]:nth-child(2) [data-testid="stNumberInput"] button,
    [data-testid="column"]:nth-child(2) button:has(svg) {{
        background-color: #daa520 !important;
    }}

    /* 第三列按钮 - 橙红色 (Pyrolysis Conditions) */
    [data-testid="column"]:nth-child(3) button[aria-label="Increment"],
    [data-testid="column"]:nth-child(3) button[aria-label="Decrement"],
    [data-testid="column"]:nth-child(3) button[title="Increment"],
    [data-testid="column"]:nth-child(3) button[title="Decrement"],
    [data-testid="column"]:nth-child(3) [data-testid="stNumberInput"] button,
    [data-testid="column"]:nth-child(3) button:has(svg) {{
        background-color: #cd5c5c !important;
    }}

    /* 备用方案：通过输入框的顺序 */
    [data-testid="stNumberInput"]:nth-of-type(1) button,
    [data-testid="stNumberInput"]:nth-of-type(2) button,
    [data-testid="stNumberInput"]:nth-of-type(3) button {{
        background-color: #20b2aa !important; /* 青绿色 */
    }}

    [data-testid="stNumberInput"]:nth-of-type(4) button,
    [data-testid="stNumberInput"]:nth-of-type(5) button,
    [data-testid="stNumberInput"]:nth-of-type(6) button {{
        background-color: #daa520 !important; /* 金黄色 */
    }}

    [data-testid="stNumberInput"]:nth-of-type(7) button,
    [data-testid="stNumberInput"]:nth-of-type(8) button,
    [data-testid="stNumberInput"]:nth-of-type(9) button {{
        background-color: #cd5c5c !important; /* 橙红色 */
    }}

    /* 最强力的备用方案 - 直接针对所有可能的按钮 */
    button:not([kind="primary"]):not([kind="primaryFormSubmit"]) {{
        background-color: #20b2aa !important;
    }}

    /* 超强力选择器 - 覆盖所有可能的Streamlit内部样式 */
    div[data-testid="column"]:nth-child(1) * button,
    div[data-testid="column"]:nth-child(1) button,
    div[data-testid="column"]:nth-child(1) [data-testid="stNumberInput"] * button,
    div[data-testid="column"]:nth-child(1) [data-testid="stNumberInput"] button,
    div[data-testid="column"]:nth-child(1) div[data-baseweb="input"] button,
    div[data-testid="column"]:nth-child(1) button[aria-label*="crement"],
    div[data-testid="column"]:nth-child(1) button[title*="crement"] {{
        background-color: #20b2aa !important;
        background: #20b2aa !important;
    }}

    div[data-testid="column"]:nth-child(2) * button,
    div[data-testid="column"]:nth-child(2) button,
    div[data-testid="column"]:nth-child(2) [data-testid="stNumberInput"] * button,
    div[data-testid="column"]:nth-child(2) [data-testid="stNumberInput"] button,
    div[data-testid="column"]:nth-child(2) div[data-baseweb="input"] button,
    div[data-testid="column"]:nth-child(2) button[aria-label*="crement"],
    div[data-testid="column"]:nth-child(2) button[title*="crement"] {{
        background-color: #daa520 !important;
        background: #daa520 !important;
    }}

    div[data-testid="column"]:nth-child(3) * button,
    div[data-testid="column"]:nth-child(3) button,
    div[data-testid="column"]:nth-child(3) [data-testid="stNumberInput"] * button,
    div[data-testid="column"]:nth-child(3) [data-testid="stNumberInput"] button,
    div[data-testid="column"]:nth-child(3) div[data-baseweb="input"] button,
    div[data-testid="column"]:nth-child(3) button[aria-label*="crement"],
    div[data-testid="column"]:nth-child(3) button[title*="crement"] {{
        background-color: #cd5c5c !important;
        background: #cd5c5c !important;
    }}

    /* 终极解决方案 - 使用CSS变量和更高优先级 */
    :root {{
        --col1-color: #20b2aa;
        --col2-color: #daa520;
        --col3-color: #cd5c5c;
    }}

    /* 使用属性选择器和通配符 */
    [data-testid="column"]:nth-child(1) [role="spinbutton"] ~ button,
    [data-testid="column"]:nth-child(1) [role="spinbutton"] + * button,
    [data-testid="column"]:nth-child(1) input[type="number"] ~ button,
    [data-testid="column"]:nth-child(1) input[type="number"] + * button {{
        background-color: var(--col1-color) !important;
        background: var(--col1-color) !important;
    }}

    [data-testid="column"]:nth-child(2) [role="spinbutton"] ~ button,
    [data-testid="column"]:nth-child(2) [role="spinbutton"] + * button,
    [data-testid="column"]:nth-child(2) input[type="number"] ~ button,
    [data-testid="column"]:nth-child(2) input[type="number"] + * button {{
        background-color: var(--col2-color) !important;
        background: var(--col2-color) !important;
    }}

    [data-testid="column"]:nth-child(3) [role="spinbutton"] ~ button,
    [data-testid="column"]:nth-child(3) [role="spinbutton"] + * button,
    [data-testid="column"]:nth-child(3) input[type="number"] ~ button,
    [data-testid="column"]:nth-child(3) input[type="number"] + * button {{
        background-color: var(--col3-color) !important;
        background: var(--col3-color) !important;
    }}

    /* 终极解决方案 - 使用更强力的CSS选择器 */

    /* 通过容器div来定位按钮 */
    div[data-testid="column"]:nth-child(1) [data-testid="stNumberInput"] button {{
        background-color: #20b2aa !important;
        background: #20b2aa !important;
        color: white !important;
        border: none !important;
        border-radius: 4px !important;
    }}

    div[data-testid="column"]:nth-child(2) [data-testid="stNumberInput"] button {{
        background-color: #daa520 !important;
        background: #daa520 !important;
        color: white !important;
        border: none !important;
        border-radius: 4px !important;
    }}

    div[data-testid="column"]:nth-child(3) [data-testid="stNumberInput"] button {{
        background-color: #cd5c5c !important;
        background: #cd5c5c !important;
        color: white !important;
        border: none !important;
        border-radius: 4px !important;
    }}

    /* 备用方案：直接通过按钮位置 */
    [data-testid="stNumberInput"]:nth-of-type(1) button,
    [data-testid="stNumberInput"]:nth-of-type(2) button,
    [data-testid="stNumberInput"]:nth-of-type(3) button {{
        background-color: #20b2aa !important;
        background: #20b2aa !important;
        color: white !important;
    }}

    [data-testid="stNumberInput"]:nth-of-type(4) button,
    [data-testid="stNumberInput"]:nth-of-type(5) button,
    [data-testid="stNumberInput"]:nth-of-type(6) button {{
        background-color: #daa520 !important;
        background: #daa520 !important;
        color: white !important;
    }}

    [data-testid="stNumberInput"]:nth-of-type(7) button,
    [data-testid="stNumberInput"]:nth-of-type(8) button,
    [data-testid="stNumberInput"]:nth-of-type(9) button {{
        background-color: #cd5c5c !important;
        background: #cd5c5c !important;
        color: white !important;
    }}

    /* 最强力的覆盖 - 使用CSS动画 */
    @keyframes forceGreen {{
        0%, 100% {{ background-color: #20b2aa !important; }}
    }}

    @keyframes forceGold {{
        0%, 100% {{ background-color: #daa520 !important; }}
    }}

    @keyframes forceRed {{
        0%, 100% {{ background-color: #cd5c5c !important; }}
    }}

    /* 应用动画到特定列 */
    div[data-testid="column"]:nth-child(1) button {{
        animation: forceGreen 0.1s infinite !important;
        color: white !important;
    }}

    div[data-testid="column"]:nth-child(2) button {{
        animation: forceGold 0.1s infinite !important;
        color: white !important;
    }}

    div[data-testid="column"]:nth-child(3) button {{
        animation: forceRed 0.1s infinite !important;
        color: white !important;
    }}

    /* 通过自定义属性强制设置 */
    button[data-forced-color="green"] {{
        background-color: #20b2aa !important;
        background: #20b2aa !important;
        color: white !important;
    }}

    button[data-forced-color="gold"] {{
        background-color: #daa520 !important;
        background: #daa520 !important;
        color: white !important;
    }}

    button[data-forced-color="red"] {{
        background-color: #cd5c5c !important;
        background: #cd5c5c !important;
        color: white !important;
    }}
    </style>

    <script>
    // 强制设置所有数字输入框按钮为青绿色
    function forceButtonColors() {{
        // 获取所有数字输入框按钮
        const allButtons = document.querySelectorAll(`
            [data-testid="stNumberInput"] button,
            .stNumberInput button,
            button[aria-label="Increment"],
            button[aria-label="Decrement"],
            button[title="Increment"],
            button[title="Decrement"],
            div[data-baseweb="input"] button
        `);

        allButtons.forEach(btn => {{
            btn.style.setProperty('background-color', '#20b2aa', 'important');
            btn.style.setProperty('background', '#20b2aa', 'important');
            btn.style.setProperty('background-image', 'none', 'important');
            btn.style.setProperty('border-color', '#20b2aa', 'important');
            btn.style.setProperty('color', 'white', 'important');
        }});

        console.log('强制设置了', allButtons.length, '个按钮为青绿色');
    }}

    // 立即执行一次
    setTimeout(forceButtonColors, 500);
    // 再次执行确保生效
    setTimeout(forceButtonColors, 1500);
    // 监听DOM变化，确保新按钮也被设置
    setTimeout(forceButtonColors, 3000);








    </script>
    """, unsafe_allow_html=True)

    # 显示当前选择的模型
    st.markdown(f"""
    <div style="text-align: center; margin-top: 20px; padding: 10px; background: rgba(255,255,255,0.1); border-radius: 10px; backdrop-filter: blur(5px);">
        <h4 style="color: white; margin: 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.8);">当前模型：{selected_model}</h4>
    </div>
    """, unsafe_allow_html=True)







class ModelPredictor:
    """根据图片特征统计信息正确调整的预测器类"""
    
    def __init__(self, target_model="Char Yield"):
        self.target_name = target_model
        
        # 根据图片中的特征统计信息，按照正确顺序定义特征名称
        self.feature_names = [
            'M(wt%)',           # 水分
            'Ash(wt%)',         # 灰分  
            'VM(wt%)',          # 挥发分
            'O/C',              # 氧碳比
            'H/C',              # 氢碳比
            'N/C',              # 氮碳比
            'FT(℃)',           # 热解温度
            'HR(℃/min)',       # 升温速率
            'FR(mL/min)'        # 流量
        ]
        
        # 根据图片中的统计信息设置训练范围
        self.training_ranges = {
            'M(wt%)': {'min': 2.750, 'max': 11.630},
            'Ash(wt%)': {'min': 0.410, 'max': 11.600},
            'VM(wt%)': {'min': 65.700, 'max': 89.500},
            'O/C': {'min': 0.301, 'max': 0.988},
            'H/C': {'min': 1.212, 'max': 1.895},
            'N/C': {'min': 0.003, 'max': 0.129},
            'FT(℃)': {'min': 300.000, 'max': 900.000},
            'HR(℃/min)': {'min': 5.000, 'max': 100.000},
            'FR(mL/min)': {'min': 0.000, 'max': 600.000}
        }
        
        # UI显示的特征映射（处理温度符号）
        self.ui_to_model_mapping = {
            'FT(°C)': 'FT(℃)',
            'HR(°C/min)': 'HR(℃/min)'
        }
        
        self.last_features = {}  # 存储上次的特征值
        self.last_result = None  # 存储上次的预测结果
        
        # 使用缓存加载模型，避免重复加载相同模型
        self.pipeline = self._get_cached_model()
        self.model_loaded = self.pipeline is not None
        
        if not self.model_loaded:
            log(f"从缓存未找到模型，尝试加载{self.target_name}模型")
            # 查找并加载模型
            self.model_path = self._find_model_file()
            if self.model_path:
                self._load_pipeline()
    
    def _get_cached_model(self):
        """从缓存中获取模型"""
        if self.target_name in st.session_state.model_cache:
            log(f"从缓存加载{self.target_name}模型")
            return st.session_state.model_cache[self.target_name]
        return None
        
    def _find_model_file(self):
        """查找模型文件"""
        # 根据训练代码的模型保存路径
        model_file_patterns = {
            "Char Yield": [
                "GBDT-Char Yield-improved.joblib",
                "GBDT-Char-improved.joblib",
                "*char*.joblib",
                "*炭产率*.joblib"
            ],
            "Oil Yield": [
                "GBDT-Oil Yield-improved.joblib", 
                "GBDT-Oil-improved.joblib",
                "*oil*.joblib",
                "*油产率*.joblib"
            ],
            "Gas Yield": [
                "GBDT-Gas Yield-improved.joblib",
                "GBDT-Gas-improved.joblib", 
                "*gas*.joblib",
                "*气产率*.joblib"
            ]
        }
        
        # 搜索目录
        search_dirs = [
            ".", "./models", "../models", "/app/models", "/app",
            "./炭产率", "./油产率", "./气产率",
            "../炭产率", "../油产率", "../气产率"
        ]
        
        patterns = model_file_patterns.get(self.target_name, [])
        log(f"搜索{self.target_name}模型文件，模式: {patterns}")
        
        for directory in search_dirs:
            if not os.path.exists(directory):
                continue
                
            try:
                for pattern in patterns:
                    # 使用glob匹配文件
                    matches = glob.glob(os.path.join(directory, pattern))
                    for match in matches:
                        if os.path.isfile(match):
                            log(f"找到模型文件: {match}")
                            return match
                            
                # 也检查目录中的所有.joblib文件
                for file in os.listdir(directory):
                    if file.endswith('.joblib'):
                        model_id = self.target_name.split(" ")[0].lower()
                        if model_id in file.lower():
                            model_path = os.path.join(directory, file)
                            log(f"找到匹配的模型文件: {model_path}")
                            return model_path
            except Exception as e:
                log(f"搜索目录{directory}时出错: {str(e)}")
        
        log(f"未找到{self.target_name}模型文件")
        return None
    
    def _load_pipeline(self):
        """加载Pipeline模型"""
        if not self.model_path:
            log("模型路径为空，无法加载")
            return False
        
        try:
            log(f"加载Pipeline模型: {self.model_path}")
            self.pipeline = joblib.load(self.model_path)
            
            # 验证Pipeline结构
            if hasattr(self.pipeline, 'predict') and hasattr(self.pipeline, 'named_steps'):
                log(f"Pipeline加载成功，组件: {list(self.pipeline.named_steps.keys())}")
                
                # 验证Pipeline包含scaler和model
                if 'scaler' in self.pipeline.named_steps and 'model' in self.pipeline.named_steps:
                    scaler_type = type(self.pipeline.named_steps['scaler']).__name__
                    model_type = type(self.pipeline.named_steps['model']).__name__
                    log(f"Scaler类型: {scaler_type}, Model类型: {model_type}")
                    
                    self.model_loaded = True
                    # 将模型保存到缓存中
                    st.session_state.model_cache[self.target_name] = self.pipeline
                    return True
                else:
                    log("Pipeline结构不符合预期，缺少scaler或model组件")
                    return False
            else:
                log("加载的对象不是有效的Pipeline")
                return False
                
        except Exception as e:
            log(f"加载模型出错: {str(e)}")
            log(traceback.format_exc())
            self.model_loaded = False
            return False
    
    def check_input_range(self, features):
        """检查输入值是否在训练数据范围内"""
        warnings = []
        
        for feature, value in features.items():
            # 获取映射后的特征名
            mapped_feature = self.ui_to_model_mapping.get(feature, feature)
            range_info = self.training_ranges.get(mapped_feature)
            
            if range_info:
                if value < range_info['min'] or value > range_info['max']:
                    warning = f"{feature}: {value:.3f} (超出训练范围 {range_info['min']:.3f} - {range_info['max']:.3f})"
                    warnings.append(warning)
                    log(f"警告: {warning}")
        
        return warnings
    
    def _prepare_features(self, features):
        """准备特征，确保顺序与训练时一致"""
        # 创建特征字典，按训练时的顺序
        model_features = {}
        
        # 首先将UI特征映射到模型特征名称
        for ui_feature, value in features.items():
            model_feature = self.ui_to_model_mapping.get(ui_feature, ui_feature)
            if model_feature in self.feature_names:
                model_features[model_feature] = value
                if ui_feature != model_feature:
                    log(f"特征映射: '{ui_feature}' -> '{model_feature}'")
        
        # 确保所有特征都存在，缺失的设为均值（根据图片统计信息）
        feature_defaults = {
            'M(wt%)': 6.430226,
            'Ash(wt%)': 4.498340,
            'VM(wt%)': 75.375509,
            'O/C': 0.715385,
            'H/C': 1.534106,
            'N/C': 0.034083,
            'FT(℃)': 505.811321,
            'HR(℃/min)': 29.011321,
            'FR(mL/min)': 93.962264
        }
        
        for feature in self.feature_names:
            if feature not in model_features:
                default_value = feature_defaults.get(feature, 0.0)
                model_features[feature] = default_value
                log(f"警告: 特征 '{feature}' 缺失，设为默认值: {default_value}")
        
        # 创建DataFrame并按照正确顺序排列列
        df = pd.DataFrame([model_features])
        df = df[self.feature_names]  # 确保列顺序与训练时一致
        
        log(f"准备好的特征DataFrame形状: {df.shape}, 列: {list(df.columns)}")
        return df
    
    def predict(self, features):
        """预测方法 - 使用Pipeline进行预测"""
        # 检查输入是否有变化
        features_changed = False
        if self.last_features:
            for feature, value in features.items():
                if feature not in self.last_features or abs(self.last_features[feature] - value) > 0.001:
                    features_changed = True
                    break
        else:
            features_changed = True
        
        # 如果输入没有变化且有上次结果，直接返回上次结果
        if not features_changed and self.last_result is not None:
            log("输入未变化，使用上次的预测结果")
            return self.last_result
        
        # 保存当前特征
        self.last_features = features.copy()
        
        # 准备特征数据
        log(f"开始准备{len(features)}个特征数据进行预测")
        features_df = self._prepare_features(features)
        
        # 使用Pipeline进行预测
        if self.model_loaded and self.pipeline is not None:
            try:
                log("使用Pipeline进行预测（包含RobustScaler预处理）")
                # Pipeline会自动进行预处理（RobustScaler）然后预测
                result = float(self.pipeline.predict(features_df)[0])
                log(f"预测成功: {result:.4f}")
                self.last_result = result
                return result
            except Exception as e:
                log(f"Pipeline预测失败: {str(e)}")
                log(traceback.format_exc())
                
                # 尝试重新加载模型
                if self._find_model_file() and self._load_pipeline():
                    try:
                        result = float(self.pipeline.predict(features_df)[0])
                        log(f"重新加载后预测成功: {result:.4f}")
                        self.last_result = result
                        return result
                    except Exception as new_e:
                        log(f"重新加载后预测仍然失败: {str(new_e)}")
        
        # 如果到这里，说明预测失败
        log("所有预测尝试都失败")
        raise ValueError(f"模型预测失败。请确保模型文件存在且格式正确。当前模型: {self.target_name}")
    
    def get_model_info(self):
        """获取模型信息摘要"""
        info = {
            "模型类型": "GBDT Pipeline (RobustScaler + GradientBoostingRegressor)",
            "目标变量": self.target_name,
            "特征数量": len(self.feature_names),
            "模型状态": "已加载" if self.model_loaded else "未加载"
        }
        
        if self.model_loaded and hasattr(self.pipeline, 'named_steps'):
            pipeline_steps = list(self.pipeline.named_steps.keys())
            info["Pipeline组件"] = " → ".join(pipeline_steps)
            
            # 如果有模型组件，显示其参数
            if 'model' in self.pipeline.named_steps:
                model = self.pipeline.named_steps['model']
                model_type = type(model).__name__
                info["回归器类型"] = model_type
                
                # 显示部分关键超参数
                if hasattr(model, 'n_estimators'):
                    info["树的数量"] = model.n_estimators
                if hasattr(model, 'max_depth'):
                    info["最大深度"] = model.max_depth
                if hasattr(model, 'learning_rate'):
                    info["学习率"] = f"{model.learning_rate:.3f}"
                    
        return info

# 初始化预测器 - 使用当前选择的模型
predictor = ModelPredictor(target_model=st.session_state.selected_model)

# 根据当前页面显示不同内容
if st.session_state.current_page == "模型信息":
    # 只显示模型信息内容，不显示标题和其他内容
    model_info = predictor.get_model_info()

    # 构建完整的HTML内容
    info_content = '<div class="page-content">'
    for key, value in model_info.items():
        info_content += f"<p><strong>{key}</strong>: {value}</p>"
    info_content += '</div>'

    st.markdown(info_content, unsafe_allow_html=True)

elif st.session_state.current_page == "执行日志":
    # 只显示执行日志内容，不显示标题和其他内容
    if st.session_state.log_messages:
        # 将所有日志消息合并成一个完整的白色半透明背景显示
        log_content = "<br>".join(st.session_state.log_messages[-50:])  # 显示最近50条日志
        st.markdown(
            f'<div class="log-container">{log_content}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown('<div class="log-container">暂无日志记录</div>', unsafe_allow_html=True)

elif st.session_state.current_page == "技术说明":
    # 只显示技术说明内容，不显示标题和其他内容
    tech_content = """
    <div class="page-content">
    <h3>模型架构</h3>
    <p>本系统采用GBDT（Gradient Boosting Decision Tree）集成学习算法，结合RobustScaler数据预处理技术。</p>

    <h3>特征工程</h3>
    <ul>
    <li><strong>工业分析</strong>: 水分(M)、灰分(Ash)、挥发分(VM)</li>
    <li><strong>元素分析</strong>: O/C、H/C、N/C原子比</li>
    <li><strong>热解条件</strong>: 最终温度(FT)、升温速率(HR)、载气流量(FR)</li>
    </ul>

    <h3>模型性能</h3>
    <ul>
    <li>训练集R²: > 0.95</li>
    <li>测试集R²: > 0.90</li>
    <li>平均绝对误差: < 2%</li>
    </ul>
    </div>
    """
    st.markdown(tech_content, unsafe_allow_html=True)

elif st.session_state.current_page == "使用指南":
    # 只显示使用指南内容，不显示标题和其他内容
    guide_content = """
    <div class="page-content">
    <h3>操作步骤</h3>
    <ol>
    <li>在左侧导航栏选择"预测模型"</li>
    <li>输入生物质的工业分析数据</li>
    <li>输入元素分析数据</li>
    <li>设置热解工艺条件</li>
    <li>点击"预测"按钮获得结果</li>
    </ol>

    <h3>数据要求</h3>
    <ul>
    <li>所有数值应为正数</li>
    <li>工业分析数据单位为wt%</li>
    <li>温度单位为°C</li>
    <li>流量单位为mL/min</li>
    </ul>

    <h3>注意事项</h3>
    <ul>
    <li>确保输入数据在合理范围内</li>
    <li>模型适用于常见生物质原料</li>
    <li>预测结果仅供参考</li>
    </ul>
    </div>
    """
    st.markdown(guide_content, unsafe_allow_html=True)

elif st.session_state.current_page == "预测模型":
    # 显示预测模型页面（原有的主要功能）

    # 初始化会话状态
    if 'clear_pressed' not in st.session_state:
        st.session_state.clear_pressed = False
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    if 'warnings' not in st.session_state:
        st.session_state.warnings = []
    if 'prediction_error' not in st.session_state:
        st.session_state.prediction_error = None
    if 'feature_values' not in st.session_state:
        st.session_state.feature_values = {}

    # 创建三栏布局：左侧输入区、中间预测区、右侧参数显示区
    col_left, col_center, col_right = st.columns([1.2, 1, 0.8])

    # 左侧输入区域
    with col_left:
        # 根据图片特征统计信息定义默认值（使用均值）
        default_values = {
            "M(wt%)": 6.430,
            "Ash(wt%)": 4.498,
            "VM(wt%)": 75.376,
            "O/C": 0.715,
            "H/C": 1.534,
            "N/C": 0.034,
            "FT(°C)": 505.811,
            "HR(°C/min)": 29.011,
            "FR(mL/min)": 93.962
        }

        # 左侧输入区标题
        st.markdown("""
        <div style="text-align: center; margin-bottom: 20px; padding: 10px; background: rgba(255,255,255,0.1); border-radius: 10px; backdrop-filter: blur(5px);">
            <h4 style="color: white; margin: 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.8);">参数输入</h4>
        </div>
        """, unsafe_allow_html=True)

        # 保持原有的特征分类名称
        feature_categories = {
            "Proximate Analysis": ["M(wt%)", "Ash(wt%)", "VM(wt%)"],
            "Ultimate Analysis": ["O/C", "H/C", "N/C"],
            "Pyrolysis Conditions": ["FT(°C)", "HR(°C/min)", "FR(mL/min)"]
        }

        # 颜色配置 - 根据用户要求的颜色配置
        category_colors = {
            "Proximate Analysis": "#20b2aa",  # 青绿色
            "Ultimate Analysis": "#daa520",   # 金黄色
            "Pyrolysis Conditions": "#cd5c5c" # 橙红色
        }

        # 使用字典存储所有输入值
        features = {}

        # 为每个类别创建输入区域
        for category, feature_list in feature_categories.items():
            color = category_colors[category]

            # 类别标题
            st.markdown(f"""
            <div style="text-align: center; margin: 15px 0 10px 0; padding: 8px; background: {color}; border-radius: 8px; color: white; font-weight: bold; font-size: 16px;">
                {category}
            </div>
            """, unsafe_allow_html=True)

            # 为每个特征创建输入
            for feature in feature_list:
                if st.session_state.clear_pressed:
                    value = default_values[feature]
                else:
                    value = st.session_state.feature_values.get(feature, default_values[feature])

                # 使用number_input让用户可以直接输入
                new_value = st.number_input(
                    f"{feature}",
                    value=float(value),
                    step=0.001,
                    format="%.3f",
                    key=f"input_{category}_{feature}"
                )
                # 更新会话状态中的值
                st.session_state.feature_values[feature] = new_value
                # 存储特征值
                features[feature] = st.session_state.feature_values.get(feature, default_values[feature])

    # 中间预测区域
    with col_center:
        # 中间区标题
        st.markdown("""
        <div style="text-align: center; margin-bottom: 20px; padding: 10px; background: rgba(255,255,255,0.1); border-radius: 10px; backdrop-filter: blur(5px);">
            <h4 style="color: white; margin: 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.8);">预测结果</h4>
        </div>
        """, unsafe_allow_html=True)

        # 预测按钮
        predict_clicked = st.button("🔮 运行预测", use_container_width=True, type="primary")

        # 重置按钮
        if st.button("🔄 重置输入", use_container_width=True):
            log("重置所有输入值")
            st.session_state.clear_pressed = True
            st.session_state.prediction_result = None
            st.session_state.warnings = []
            st.session_state.prediction_error = None
            st.rerun()

        # 处理预测逻辑
        if predict_clicked:
            log("开始预测流程...")

            # 切换模型后需要重新初始化预测器
            if predictor.target_name != st.session_state.selected_model:
                log(f"检测到模型变更，重新初始化预测器: {st.session_state.selected_model}")
                predictor = ModelPredictor(target_model=st.session_state.selected_model)

            # 保存当前输入到会话状态
            st.session_state.feature_values = features.copy()

            log(f"开始{st.session_state.selected_model}预测，输入特征数: {len(features)}")

            # 检查输入范围
            warnings = predictor.check_input_range(features)
            st.session_state.warnings = warnings

            # 执行预测
            try:
                # 确保预测器已正确加载
                if not predictor.model_loaded:
                    log("模型未加载，尝试重新加载")
                    if predictor._find_model_file() and predictor._load_pipeline():
                        log("重新加载模型成功")
                    else:
                        error_msg = f"无法加载{st.session_state.selected_model}模型。请确保模型文件存在于正确位置。"
                        st.error(error_msg)
                        st.session_state.prediction_error = error_msg
                        st.rerun()

                # 执行预测
                result = predictor.predict(features)
                if result is not None:
                    st.session_state.prediction_result = float(result)
                    log(f"预测成功: {st.session_state.prediction_result:.4f}")
                    st.session_state.prediction_error = None
                else:
                    log("警告: 预测结果为空")
                    st.session_state.prediction_error = "预测结果为空"

            except Exception as e:
                error_msg = f"预测过程中发生错误: {str(e)}"
                st.session_state.prediction_error = error_msg
                log(f"预测错误: {str(e)}")
                log(traceback.format_exc())
                st.error(error_msg)

        # 显示预测结果
        if st.session_state.prediction_result is not None:
            # 显示主预测结果
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #20b2aa, #17a2b8); color: white; padding: 20px; border-radius: 15px; text-align: center; margin: 20px 0; box-shadow: 0 8px 25px rgba(32, 178, 170, 0.3);">
                    <h3 style="margin: 0; font-size: 24px;">{st.session_state.selected_model}</h3>
                    <h1 style="margin: 10px 0 0 0; font-size: 36px; font-weight: bold;">{st.session_state.prediction_result:.2f} wt%</h1>
                </div>
                """,
                unsafe_allow_html=True
            )

            # 显示警告
            if st.session_state.warnings:
                warnings_html = "<div style='background: rgba(255, 193, 7, 0.1); border-left: 4px solid #ffc107; padding: 10px; margin: 10px 0; border-radius: 5px; color: white;'><b>⚠️ 输入警告</b><ul>"
                for warning in st.session_state.warnings:
                    warnings_html += f"<li>{warning}</li>"
                warnings_html += "</ul><p><i>建议调整输入值以获得更准确的预测结果。</i></p></div>"
                st.markdown(warnings_html, unsafe_allow_html=True)

        elif st.session_state.prediction_error is not None:
            st.markdown(
                f"""
                <div style="background: rgba(220, 53, 69, 0.1); border-left: 4px solid #dc3545; padding: 15px; margin: 10px 0; border-radius: 5px; color: white;">
                    <b>❌ 预测错误</b><br>
                    {st.session_state.prediction_error}
                </div>
                """,
                unsafe_allow_html=True
            )

    # 右侧参数显示区域
    with col_right:
        # 右侧区标题
        st.markdown("""
        <div style="text-align: center; margin-bottom: 20px; padding: 10px; background: rgba(255,255,255,0.1); border-radius: 10px; backdrop-filter: blur(5px);">
            <h4 style="color: white; margin: 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.8);">当前参数</h4>
        </div>
        """, unsafe_allow_html=True)

        # 显示当前输入的参数值
        for category, feature_list in feature_categories.items():
            color = category_colors[category]

            # 类别标题
            st.markdown(f"""
            <div style="text-align: center; margin: 10px 0 5px 0; padding: 6px; background: {color}; border-radius: 6px; color: white; font-weight: bold; font-size: 12px;">
                {category}
            </div>
            """, unsafe_allow_html=True)

            # 显示该类别下的所有参数
            for feature in feature_list:
                current_value = st.session_state.feature_values.get(feature, default_values[feature])
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.9); border-radius: 6px; padding: 6px 8px; margin: 3px 0; display: flex; justify-content: space-between; align-items: center; font-size: 11px;">
                    <span style="color: #333; font-weight: bold;">{feature}</span>
                    <span style="color: {color}; font-weight: bold;">{current_value:.3f}</span>
                </div>
                """, unsafe_allow_html=True)

        # 显示模型信息
        if st.session_state.prediction_result is not None:
            st.markdown("""
            <div style="text-align: center; margin: 15px 0 5px 0; padding: 6px; background: rgba(255,255,255,0.2); border-radius: 6px; color: white; font-weight: bold; font-size: 12px;">
                预测信息
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.9); border-radius: 6px; padding: 8px; margin: 3px 0; color: #333; font-size: 11px;">
                <div style="margin: 2px 0;"><strong>目标变量:</strong> {st.session_state.selected_model}</div>
                <div style="margin: 2px 0;"><strong>预测结果:</strong> {st.session_state.prediction_result:.4f} wt%</div>
                <div style="margin: 2px 0;"><strong>模型类型:</strong> GBDT Pipeline</div>
                <div style="margin: 2px 0;"><strong>预处理:</strong> RobustScaler</div>
                <div style="margin: 2px 0;"><strong>特征数量:</strong> {len(predictor.feature_names)}</div>
                <div style="margin: 2px 0;"><strong>加载状态:</strong> {'✅ 正常' if predictor.model_loaded else '❌ 失败'}</div>
            </div>
            """, unsafe_allow_html=True)

        # 重置状态
        if st.session_state.clear_pressed:
            st.session_state.feature_values = {}
            st.session_state.clear_pressed = False

    elif st.session_state.prediction_error is not None:
        st.markdown(
            f"""
            <div style="background: rgba(220, 53, 69, 0.1); border-left: 4px solid #dc3545; padding: 15px; margin: 10px 0; border-radius: 5px; color: white;">
                <b>❌ 预测错误</b><br>
                {st.session_state.prediction_error}
            </div>
            """,
            unsafe_allow_html=True
        )
    [data-testid="stNumberInput"]:nth-child(1) button,
    [data-testid="stNumberInput"]:nth-child(2) button,
    [data-testid="stNumberInput"]:nth-child(3) button,
    div:nth-child(1) [data-testid="stNumberInput"] button,
    div:nth-child(1) button[aria-label*="crement"],
    div:nth-child(1) button[title*="crement"] {
        background-color: #20b2aa !important;
        background: #20b2aa !important;
        color: white !important;
        border: 2px solid #20b2aa !important;
    }

    /* 专门针对第一列的CSS类 - 超高优先级 */
    html body .first-column-input button,
    html body .first-column-input [data-testid="stNumberInput"] button,
    html body .first-column-input button[aria-label="Increment"],
    html body .first-column-input button[aria-label="Decrement"],
    html body .first-column-input button[title="Increment"],
    html body .first-column-input button[title="Decrement"],
    html body div.first-column-input button,
    html body div.first-column-input [data-testid="stNumberInput"] button {
        background-color: #20b2aa !important;
        background: #20b2aa !important;
        background-image: none !important;
        color: white !important;
        border: 2px solid #20b2aa !important;
        border-color: #20b2aa !important;
    }

    /* 通过位置直接选择第一列的按钮 - 终极方案 */
    [data-testid="column"]:first-child [data-testid="stNumberInput"] button {
        background-color: #20b2aa !important;
        background: #20b2aa !important;
        background-image: none !important;
        color: white !important;
        border: 2px solid #20b2aa !important;
        border-color: #20b2aa !important;
    }

    /* 终极覆盖 - 针对所有可能的Streamlit按钮选择器 */
    html body [data-testid="stNumberInput"] button:first-of-type,
    html body [data-testid="stNumberInput"] button:last-of-type {
        background-color: #20b2aa !important;
        background: #20b2aa !important;
        background-image: none !important;
        color: white !important;
        border: 2px solid #20b2aa !important;
    }

    /* 专门为JavaScript添加的强制绿色类 */
    .force-green-button {
        background-color: #20b2aa !important;
        background: #20b2aa !important;
        background-image: none !important;
        color: white !important;
        border: 2px solid #20b2aa !important;
        border-color: #20b2aa !important;
    }

    /* 确保主要按钮可见且样式正常 */
    .main-buttons .stButton {
        display: block !important;
    }

    .main-buttons .stButton button {
        display: block !important;
        width: 100% !important;
        height: auto !important;
        padding: 12px 20px !important;
        font-size: 18px !important;
        border-radius: 8px !important;
    }

    /* 移除列容器的背景，让参数行独立显示 */
    div[data-testid="column"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 10px !important;
    }

    /* 确保容器内的元素垂直对齐 */
    .stContainer > div {
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;
    }

    /* 强制对齐修复 */
    .feature-container {
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;
        background: rgba(255, 255, 255, 0.85) !important;
        border-radius: 10px !important;
        padding: 8px 12px !important;
        margin: 8px 0 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
        backdrop-filter: blur(5px) !important;
        min-height: 50px !important;
    }

    /* 确保输入框容器内的元素正确对齐 */
    .feature-row > div:last-child {
        display: flex !important;
        align-items: center !important;
        flex: 1 !important;
    }

    /* 修复Streamlit默认的margin和padding */
    .feature-row .stNumberInput > div {
        margin: 0 !important;
        padding: 0 !important;
    }

    /* 确保输入框和按钮在同一行 */
    .feature-row .stNumberInput > div > div {
        display: flex !important;
        align-items: center !important;
        margin: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # 颜色配置 - 根据用户要求的颜色配置
    category_colors = {
        "Proximate Analysis": "#20b2aa",  # 青绿色 (第一列)
        "Ultimate Analysis": "#daa520",   # 金黄色 (第二列)
        "Pyrolysis Conditions": "#cd5c5c" # 橙红色 (第三列)
    }

    # 创建三列布局
    col1, col2, col3 = st.columns(3)

    # 使用字典存储所有输入值
    features = {}

    # Proximate Analysis - 第一列
    with col1:
        category = "Proximate Analysis"
        color = category_colors[category]

        # 为每个特征创建独立的参数行
        for feature in feature_categories[category]:
            if st.session_state.clear_pressed:
                value = default_values[feature]
            else:
                value = st.session_state.feature_values.get(feature, default_values[feature])

            # 创建水平布局：标签和输入框在同一行
            label_col, input_col = st.columns([1, 1])

            with label_col:
                # 创建标签
                st.markdown(f"""
                <div style='background-color: {color}; width: 100%; text-align: center; margin: 0; padding: 12px 8px; border-radius: 6px; color: white; font-weight: bold; font-size: 14px; margin-bottom: 10px;'>
                    {feature}
                </div>
                """, unsafe_allow_html=True)

            with input_col:
                # 使用number_input让用户可以直接输入
                new_value = st.number_input(
                    f"{feature}",
                    value=float(value),
                    step=0.001,
                    format="%.3f",
                    key=f"input_{category}_{feature}",
                    label_visibility="collapsed"
                )
                # 更新会话状态中的值
                st.session_state.feature_values[feature] = new_value

            # 存储特征值
            features[feature] = st.session_state.feature_values.get(feature, default_values[feature])

    # Ultimate Analysis - 第二列
    with col2:
        category = "Ultimate Analysis"
        color = category_colors[category]

        # 为每个特征创建独立的参数行
        for feature in feature_categories[category]:
            if st.session_state.clear_pressed:
                value = default_values[feature]
            else:
                value = st.session_state.feature_values.get(feature, default_values[feature])

            # 创建水平布局：标签和输入框在同一行
            label_col, input_col = st.columns([1, 1])

            with label_col:
                # 创建标签
                st.markdown(f"""
                <div style='background-color: {color}; width: 100%; text-align: center; margin: 0; padding: 12px 8px; border-radius: 6px; color: white; font-weight: bold; font-size: 14px; margin-bottom: 10px;'>
                    {feature}
                </div>
                """, unsafe_allow_html=True)

            with input_col:
                # 使用number_input让用户可以直接输入
                new_value = st.number_input(
                    f"{feature}",
                    value=float(value),
                    step=0.001,
                    format="%.3f",
                    key=f"input_{category}_{feature}",
                    label_visibility="collapsed"
                )
                # 更新会话状态中的值
                st.session_state.feature_values[feature] = new_value

            # 存储特征值
            features[feature] = st.session_state.feature_values.get(feature, default_values[feature])

    # Pyrolysis Conditions - 第三列
    with col3:
        category = "Pyrolysis Conditions"
        color = category_colors[category]

        # 为每个特征创建独立的参数行
        for feature in feature_categories[category]:
            if st.session_state.clear_pressed:
                value = default_values[feature]
            else:
                value = st.session_state.feature_values.get(feature, default_values[feature])

            # 创建水平布局：标签和输入框在同一行
            label_col, input_col = st.columns([1, 1])

            with label_col:
                # 创建标签
                st.markdown(f"""
                <div style='background-color: {color}; width: 100%; text-align: center; margin: 0; padding: 12px 8px; border-radius: 6px; color: white; font-weight: bold; font-size: 14px; margin-bottom: 10px;'>
                    {feature}
                </div>
                """, unsafe_allow_html=True)

            with input_col:
                # 使用number_input让用户可以直接输入
                new_value = st.number_input(
                    f"{feature}",
                    value=float(value),
                    step=0.001,
                    format="%.3f",
                    key=f"input_{category}_{feature}",
                    label_visibility="collapsed"
                )
                # 更新会话状态中的值
                st.session_state.feature_values[feature] = new_value

            # 存储特征值
            features[feature] = st.session_state.feature_values.get(feature, default_values[feature])



    # 移除原来的调试信息显示，将在右侧边栏显示

    # 重置状态
    if st.session_state.clear_pressed:
        st.session_state.feature_values = {}
        st.session_state.clear_pressed = False

    # 预测结果显示区域
    result_container = st.container()

    # 预测按钮区域
    st.markdown('<div class="main-buttons">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])

    with col1:
        predict_clicked = st.button("🔮 运行预测", use_container_width=True, type="primary")
        if predict_clicked:
            log("开始预测流程...")

            # 切换模型后需要重新初始化预测器
            if predictor.target_name != st.session_state.selected_model:
                log(f"检测到模型变更，重新初始化预测器: {st.session_state.selected_model}")
                predictor = ModelPredictor(target_model=st.session_state.selected_model)

            # 保存当前输入到会话状态
            st.session_state.feature_values = features.copy()

            log(f"开始{st.session_state.selected_model}预测，输入特征数: {len(features)}")

            # 检查输入范围
            warnings = predictor.check_input_range(features)
            st.session_state.warnings = warnings

            # 执行预测
            try:
                # 确保预测器已正确加载
                if not predictor.model_loaded:
                    log("模型未加载，尝试重新加载")
                    if predictor._find_model_file() and predictor._load_pipeline():
                        log("重新加载模型成功")
                    else:
                        error_msg = f"无法加载{st.session_state.selected_model}模型。请确保模型文件存在于正确位置。"
                        st.error(error_msg)
                        st.session_state.prediction_error = error_msg
                        st.rerun()

                # 执行预测
                result = predictor.predict(features)
                if result is not None:
                    st.session_state.prediction_result = float(result)
                    log(f"预测成功: {st.session_state.prediction_result:.4f}")
                    st.session_state.prediction_error = None
                else:
                    log("警告: 预测结果为空")
                    st.session_state.prediction_error = "预测结果为空"

            except Exception as e:
                error_msg = f"预测过程中发生错误: {str(e)}"
                st.session_state.prediction_error = error_msg
                log(f"预测错误: {str(e)}")
                log(traceback.format_exc())
                st.error(error_msg)

    with col2:
        if st.button("🔄 重置输入", use_container_width=True):
            log("重置所有输入值")
            st.session_state.clear_pressed = True
            st.session_state.prediction_result = None
            st.session_state.warnings = []
            st.session_state.prediction_error = None
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # 显示预测结果
    if st.session_state.prediction_result is not None:
        st.markdown("---")

        # 显示主预测结果
        result_container.markdown(
            f"<div class='yield-result'>{st.session_state.selected_model}: {st.session_state.prediction_result:.2f} wt%</div>",
            unsafe_allow_html=True
        )

        # 显示模型状态
        if not predictor.model_loaded:
            result_container.markdown(
                "<div class='error-box'><b>⚠️ 错误：</b> 模型未成功加载，无法执行预测。请检查模型文件是否存在。</div>",
                unsafe_allow_html=True
            )

        # 显示警告
        if st.session_state.warnings:
            warnings_html = "<div class='warning-box'><b>⚠️ 输入警告</b><ul>"
            for warning in st.session_state.warnings:
                warnings_html += f"<li>{warning}</li>"
            warnings_html += "</ul><p><i>建议调整输入值以获得更准确的预测结果。</i></p></div>"
            result_container.markdown(warnings_html, unsafe_allow_html=True)

        # 显示预测详情
        with st.expander("📈 预测详情", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                **预测信息:**
                - 目标变量: {st.session_state.selected_model}
                - 预测结果: {st.session_state.prediction_result:.4f} wt%
                - 模型类型: GBDT Pipeline
                - 预处理: RobustScaler
                """)
            with col2:
                st.markdown(f"""
                **模型状态:**
                - 加载状态: {'✅ 正常' if predictor.model_loaded else '❌ 失败'}
                - 特征数量: {len(predictor.feature_names)}
                - 警告数量: {len(st.session_state.warnings)}
                """)

    elif st.session_state.prediction_error is not None:
        st.markdown("---")
        error_html = f"""
        <div class='error-box'>
            <h3>❌ 预测失败</h3>
            <p><b>错误信息:</b> {st.session_state.prediction_error}</p>
            <p><b>可能的解决方案:</b></p>
            <ul>
                <li>确保模型文件 (.joblib) 存在于应用目录中</li>
                <li>检查模型文件名是否包含对应的关键词 (char/oil/gas)</li>
                <li>验证输入数据格式是否正确</li>
                <li>确认特征顺序：M, Ash, VM, O/C, H/C, N/C, FT, HR, FR</li>
            </ul>
        </div>
        """
        st.markdown(error_html, unsafe_allow_html=True)

    # 右侧预测结果显示区域
    with col_right:
        # 添加右侧侧边栏的CSS样式
        st.markdown("""
        <style>
        /* 右侧侧边栏样式 */
        .right-sidebar {
            position: fixed;
            right: 20px;
            top: 120px;
            width: 300px;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            z-index: 1000;
        }

        /* 预测结果标题 */
        .prediction-title {
            background: linear-gradient(135deg, #20b2aa, #17a2b8);
            color: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            box-shadow: 0 4px 8px rgba(32, 178, 170, 0.3);
        }

        /* 预测信息区域 */
        .prediction-info {
            background: rgba(240, 248, 255, 0.8);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            border-left: 4px solid #20b2aa;
        }

        .prediction-info h4 {
            color: #333;
            margin: 0 0 10px 0;
            font-size: 16px;
            font-weight: bold;
        }

        .prediction-info-item {
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            padding: 5px 0;
            border-bottom: 1px solid rgba(0,0,0,0.1);
        }

        .prediction-info-item:last-child {
            border-bottom: none;
        }

        .prediction-info-label {
            color: #666;
            font-weight: 500;
        }

        .prediction-info-value {
            color: #20b2aa;
            font-weight: bold;
        }

        /* 预测状态区域 */
        .prediction-status {
            background: rgba(240, 255, 240, 0.8);
            border-radius: 10px;
            padding: 15px;
            border-left: 4px solid #28a745;
        }

        .prediction-status h4 {
            color: #333;
            margin: 0 0 10px 0;
            font-size: 16px;
            font-weight: bold;
        }

        .status-item {
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            padding: 5px 0;
            border-bottom: 1px solid rgba(0,0,0,0.1);
        }

        .status-item:last-child {
            border-bottom: none;
        }

        .status-label {
            color: #666;
            font-weight: 500;
        }

        .status-value {
            font-weight: bold;
        }

        .status-normal {
            color: #28a745;
        }

        .status-warning {
            color: #ffc107;
        }

        .status-error {
            color: #dc3545;
        }
        </style>
        """, unsafe_allow_html=True)

        # 获取当前预测结果和状态
        prediction_result = st.session_state.get('prediction_result')
        selected_model = st.session_state.selected_model
        warnings = st.session_state.get('warnings', [])
        prediction_error = st.session_state.get('prediction_error')

        # 预测结果标题
        if prediction_result is not None:
            result_text = f"{selected_model}: {prediction_result:.2f} wt%"
        else:
            result_text = f"{selected_model}: -- wt%"

        st.markdown(f"""
        <div class="right-sidebar">
            <div class="prediction-title">
                预测结果
            </div>

            <div style="text-align: center; font-size: 24px; font-weight: bold; color: #20b2aa; margin-bottom: 20px;">
                {result_text}
            </div>

            <div class="prediction-info">
                <h4>预测信息</h4>
                <div class="prediction-info-item">
                    <span class="prediction-info-label">• 目标变量：</span>
                    <span class="prediction-info-value">{selected_model}</span>
                </div>
                <div class="prediction-info-item">
                    <span class="prediction-info-label">• 预测结果：</span>
                    <span class="prediction-info-value">{"%.4f wt%" % prediction_result if prediction_result is not None else "-- wt%"}</span>
                </div>
                <div class="prediction-info-item">
                    <span class="prediction-info-label">• 模型类型：</span>
                    <span class="prediction-info-value">RobustScaler Pipeline</span>
                </div>
                <div class="prediction-info-item">
                    <span class="prediction-info-label">• 预处理：</span>
                    <span class="prediction-info-value">RobustScaler</span>
                </div>
            </div>

            <div class="prediction-status">
                <h4>预测状态</h4>
                <div class="status-item">
                    <span class="status-label">• 预测状态：</span>
                    <span class="status-value {'status-normal' if prediction_result is not None and not prediction_error else 'status-error'}">
                        {'✓ 正常' if prediction_result is not None and not prediction_error else '✗ 异常'}
                    </span>
                </div>
                <div class="status-item">
                    <span class="status-label">• 特征数量：</span>
                    <span class="status-value status-normal">{len(predictor.feature_names)}</span>
                </div>
                <div class="status-item">
                    <span class="status-label">• 警告数量：</span>
                    <span class="status-value {'status-normal' if len(warnings) == 0 else 'status-warning'}">{len(warnings)}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)