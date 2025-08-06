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

    /* 侧边栏背景 */
    .css-1d391kg, .css-1lcbmhc, .css-1outpf7, section[data-testid="stSidebar"] {
        background-color: #f8f9fa !important;
    }

    /* 侧边栏内容文字颜色 */
    section[data-testid="stSidebar"] * {
        color: #333333 !important;
    }

    /* 侧边栏标题颜色 */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #333333 !important;
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
    
    /* 显示当前输入值expander的特殊样式 */
    /* expander标题部分 - 透明背景（与当前模型样式一致） */
    div[data-testid="stExpander"] summary[aria-expanded] {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* expander内容部分 - 白色轻微透明背景 */
    div[data-testid="stExpander"] .streamlit-expanderContent {
        background-color: rgba(255, 255, 255, 0.8) !important;
        border-radius: 8px !important;
        padding: 15px !important;
        margin-top: 5px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
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

# 创建侧边栏导航
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
        background: rgba(255,255,255,0.8) !important;
        border: 2px solid rgba(255,255,255,0.3) !important;
        border-radius: 15px !important;
        padding: 20px !important;
        height: auto !important;
        min-height: 120px !important;
        color: #333 !important;
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
        background: rgba(255,255,255,0.9) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 40px rgba(255,255,255,0.2) !important;
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
        background: rgba(255,255,255,0.8) !important;
        border: 2px solid rgba(255,255,255,0.3) !important;
        color: #333 !important;
        transition: all 0.3s ease !important;
    }}

    button[kind="primary"],
    .stButton > button[kind="primary"],
    [data-testid="stButton"] > button[kind="primary"] {{
        background: linear-gradient(135deg, #20b2aa, #17a2b8) !important;
        border: 3px solid #20b2aa !important;
        color: white !important;
        box-shadow: 0 8px 25px rgba(32, 178, 170, 0.4) !important;
        transform: translateY(-2px) !important;
        font-weight: 600 !important;
    }}

    /* 数字输入框按钮的强制样式 - 使用更强的选择器 */
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
    }}

    /* 第一列按钮 - 青绿色 (Proximate Analysis) */
    [data-testid="column"]:nth-child(1) button[aria-label="Increment"],
    [data-testid="column"]:nth-child(1) button[aria-label="Decrement"],
    [data-testid="column"]:nth-child(1) button[title="Increment"],
    [data-testid="column"]:nth-child(1) button[title="Decrement"],
    [data-testid="column"]:nth-child(1) [data-testid="stNumberInput"] button,
    [data-testid="column"]:nth-child(1) button:has(svg) {{
        background-color: #20b2aa !important;
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
    // DOM结构调试和按钮颜色设置脚本
    function debugAndSetButtonColors() {{
        console.log('=== 开始DOM结构调试 ===');

        // 1. 详细分析DOM结构
        const allButtons = document.querySelectorAll('button');
        console.log(`页面总按钮数: ${{allButtons.length}}`);

        // 打印每个按钮的详细信息
        allButtons.forEach((btn, index) => {{
            const text = btn.textContent.trim();
            const ariaLabel = btn.getAttribute('aria-label') || '';
            const title = btn.getAttribute('title') || '';
            const className = btn.className || '';
            const parentClass = btn.parentElement ? btn.parentElement.className : '';
            const computedStyle = window.getComputedStyle(btn);

            console.log(`按钮${{index + 1}}:`, {{
                text: text,
                ariaLabel: ariaLabel,
                title: title,
                className: className,
                parentClass: parentClass,
                backgroundColor: computedStyle.backgroundColor,
                element: btn
            }});
        }});

        // 2. 查找数字输入框
        const numberInputs = document.querySelectorAll('[data-testid="stNumberInput"]');
        console.log(`找到${{numberInputs.length}}个数字输入框`);

        numberInputs.forEach((input, index) => {{
            const buttons = input.querySelectorAll('button');
            console.log(`数字输入框${{index + 1}}包含${{buttons.length}}个按钮`);

            buttons.forEach((btn, btnIndex) => {{
                console.log(`  按钮${{btnIndex + 1}}: "${{btn.textContent}}" - ${{btn.getAttribute('aria-label')}}`);
            }});
        }});

        // 3. 查找列容器
        const columns = document.querySelectorAll('[data-testid="column"]');
        console.log(`找到${{columns.length}}个列容器`);

        columns.forEach((column, colIndex) => {{
            const buttons = column.querySelectorAll('button');
            console.log(`列${{colIndex + 1}}包含${{buttons.length}}个按钮`);
        }});

        // 4. 强制设置按钮颜色 - 使用最直接的方法
        console.log('=== 开始强制设置按钮颜色 ===');

        const colors = ['#20b2aa', '#daa520', '#cd5c5c']; // 青绿、金黄、橙红

        // 方法1: 通过数字输入框设置
        numberInputs.forEach((input, inputIndex) => {{
            const columnIndex = Math.floor(inputIndex / 3);
            if (columnIndex < 3) {{
                const color = colors[columnIndex];
                const buttons = input.querySelectorAll('button');

                buttons.forEach(btn => {{
                    // 超强力设置
                    btn.style.cssText = `
                        background-color: ${{color}} !important;
                        background: ${{color}} !important;
                        color: white !important;
                        border: none !important;
                        border-radius: 4px !important;
                    `;

                    // 添加标识
                    btn.setAttribute('data-forced-color', color);
                    btn.setAttribute('data-column', columnIndex + 1);

                    console.log(`强制设置输入框${{inputIndex + 1}}的按钮为${{color}}`);
                }});
            }}
        }});

        // 方法2: 直接遍历所有+-按钮
        const plusMinusButtons = Array.from(allButtons).filter(btn => {{
            const text = btn.textContent.trim();
            return text === '+' || text === '−' || text === '-' || text === '＋' || text === '－';
        }});

        console.log(`找到${{plusMinusButtons.length}}个+-按钮`);

        plusMinusButtons.forEach((btn, index) => {{
            const columnIndex = Math.floor(index / 6); // 每列6个按钮
            if (columnIndex < 3) {{
                const color = colors[columnIndex];

                // 最强力的设置方法
                btn.style.cssText = `
                    background-color: ${{color}} !important;
                    background: ${{color}} !important;
                    background-image: none !important;
                    color: white !important;
                    border: none !important;
                    border-radius: 4px !important;
                `;

                btn.setAttribute('data-forced-color', color);
                btn.setAttribute('data-column', columnIndex + 1);

                console.log(`强制设置+-按钮${{index + 1}}("${{btn.textContent}}")为${{color}}`);
            }}
        }});

        console.log('=== DOM调试和颜色设置完成 ===');
    }}

    // 立即执行多次调试和设置函数
    setTimeout(debugAndSetButtonColors, 50);
    setTimeout(debugAndSetButtonColors, 100);
    setTimeout(debugAndSetButtonColors, 200);
    setTimeout(debugAndSetButtonColors, 500);
    setTimeout(debugAndSetButtonColors, 1000);
    setTimeout(debugAndSetButtonColors, 2000);
    setTimeout(debugAndSetButtonColors, 3000);

    // 持续监听和重新应用
    const observer = new MutationObserver(function(mutations) {{
        let shouldReapply = false;
        mutations.forEach(function(mutation) {{
            if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {{
                // 检查是否有新的按钮或输入框
                const hasNewButtons = Array.from(mutation.addedNodes).some(node => {{
                    if (node.nodeType === 1) {{ // Element node
                        return node.tagName === 'BUTTON' ||
                               node.querySelector && node.querySelector('button') ||
                               node.getAttribute && node.getAttribute('data-testid') === 'stNumberInput';
                    }}
                    return false;
                }});

                if (hasNewButtons) {{
                    shouldReapply = true;
                }}
            }}
        }});

        if (shouldReapply) {{
            console.log('检测到DOM变化，重新应用按钮颜色');
            setTimeout(debugAndSetButtonColors, 50);
            setTimeout(debugAndSetButtonColors, 200);
        }}
    }});

    // 开始观察
    observer.observe(document.body, {{
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ['style', 'class']
    }});

    // 定期强制重新应用（每5秒）
    setInterval(function() {{
        console.log('定期重新应用按钮颜色');
        debugAndSetButtonColors();
    }}, 5000);

    // 添加诊断函数
    function diagnoseButtons() {{
        console.log('=== 按钮诊断开始 ===');

        // 1. 检查所有按钮
        const allButtons = document.querySelectorAll('button');
        console.log('总按钮数量:', allButtons.length);

        // 2. 检查+-按钮
        const plusMinusButtons = Array.from(allButtons).filter(btn =>
            btn.textContent === '+' || btn.textContent === '−' || btn.textContent === '-'
        );
        console.log('+-按钮数量:', plusMinusButtons.length);

        // 3. 检查每个+-按钮的当前样式
        plusMinusButtons.forEach((btn, index) => {{
            const computedStyle = window.getComputedStyle(btn);
            console.log(`按钮${{index + 1}} "${{btn.textContent}}": 背景色=${{computedStyle.backgroundColor}}, 内联样式=${{btn.style.backgroundColor}}`);
        }});

        // 4. 强制设置红色测试
        console.log('=== 测试强制设置红色 ===');
        plusMinusButtons.forEach((btn, index) => {{
            btn.style.setProperty('background-color', '#ff0000', 'important');
            console.log(`按钮${{index + 1}}设置红色后: ${{btn.style.backgroundColor}}`);
        }});

        // 5. 1秒后检查是否被覆盖
        setTimeout(() => {{
            console.log('=== 1秒后检查是否被覆盖 ===');
            plusMinusButtons.forEach((btn, index) => {{
                const computedStyle = window.getComputedStyle(btn);
                console.log(`按钮${{index + 1}} 1秒后: 计算样式=${{computedStyle.backgroundColor}}, 内联样式=${{btn.style.backgroundColor}}`);
            }});
        }}, 1000);
    }}

    // 延迟执行诊断
    setTimeout(diagnoseButtons, 2000);


    // 最终解决方案：暴力覆盖所有按钮样式
    function bruteForceButtonColors() {{
        console.log('=== 暴力设置按钮颜色开始 ===');

        // 获取所有按钮
        const allButtons = document.querySelectorAll('button');
        console.log(`找到 ${{allButtons.length}} 个按钮`);

        // 定义颜色
        const colors = ['#20b2aa', '#daa520', '#cd5c5c']; // 青绿、金黄、橙红

        // 找到所有+/-按钮
        const incrementDecrementButtons = [];
        allButtons.forEach(btn => {{
            const text = btn.textContent.trim();
            if (text === '+' || text === '−' || text === '-' || text === '＋' || text === '－') {{
                incrementDecrementButtons.push(btn);
            }}
        }});

        console.log(`找到 ${{incrementDecrementButtons.length}} 个+/-按钮`);

        // 为每个+/-按钮设置颜色 - 使用更直接的方法
        incrementDecrementButtons.forEach((btn, index) => {{
            let color = '#666666'; // 默认颜色

            // 通过检查按钮所在的列容器来确定颜色
            let parent = btn.parentElement;
            let columnIndex = -1;

            // 向上遍历DOM树，寻找列容器
            while (parent && columnIndex === -1) {{
                if (parent.getAttribute && parent.getAttribute('data-testid') === 'column') {{
                    // 找到列容器，确定它是第几列
                    const allColumns = document.querySelectorAll('[data-testid="column"]');
                    columnIndex = Array.from(allColumns).indexOf(parent);
                    break;
                }}
                parent = parent.parentElement;
            }}

            // 根据列索引分配颜色
            if (columnIndex === 0) {{
                color = '#20b2aa'; // 第一列 - 青绿色
            }} else if (columnIndex === 1) {{
                color = '#daa520'; // 第二列 - 金黄色
            }} else if (columnIndex === 2) {{
                color = '#cd5c5c'; // 第三列 - 橙红色
            }} else {{
                // 如果无法确定列，使用简单的索引分配
                if (index < 6) {{
                    color = '#20b2aa'; // 第一列 - 青绿色
                }} else if (index < 12) {{
                    color = '#daa520'; // 第二列 - 金黄色
                }} else {{
                    color = '#cd5c5c'; // 第三列 - 橙红色
                }}
            }}

            // 最强力的样式设置
            btn.style.cssText = `
                background: ${{color}} !important;
                background-color: ${{color}} !important;
                background-image: none !important;
                color: white !important;
                border: none !important;
                border-radius: 4px !important;
                box-shadow: none !important;
                text-shadow: none !important;
                font-weight: bold !important;
                min-width: 24px !important;
                min-height: 24px !important;
            `;

            // 移除所有可能的类名
            btn.className = '';

            // 添加自定义属性
            btn.setAttribute('data-custom-color', color);
            btn.setAttribute('data-button-index', index);

            console.log(`按钮 ${{index}}: "${{btn.textContent}}" -> ${{color}}`);
        }});

        console.log('=== 暴力设置按钮颜色完成 ===');
    }}

    // 立即执行多次
    setTimeout(bruteForceButtonColors, 100);
    setTimeout(bruteForceButtonColors, 300);
    setTimeout(bruteForceButtonColors, 500);
    setTimeout(bruteForceButtonColors, 1000);
    setTimeout(bruteForceButtonColors, 2000);
    setTimeout(bruteForceButtonColors, 3000);
    setTimeout(bruteForceButtonColors, 5000);

    // 每隔2秒强制执行一次
    setInterval(bruteForceButtonColors, 2000);

    // 监听任何DOM变化
    const bruteForceMutationObserver = new MutationObserver(function(mutations) {{
        let needsUpdate = false;
        mutations.forEach(function(mutation) {{
            if (mutation.type === 'childList' || mutation.type === 'attributes') {{
                needsUpdate = true;
            }}
        }});

        if (needsUpdate) {{
            setTimeout(bruteForceButtonColors, 50);
        }}
    }});

    bruteForceMutationObserver.observe(document.body, {{
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ['style', 'class']
    }});

    console.log('暴力按钮颜色系统已启动');

    </script>
    """, unsafe_allow_html=True)

    # 显示当前选择的模型
    st.markdown(f"""
    <div style="text-align: center; margin-top: 20px; padding: 10px; background: rgba(255,255,255,0.1) !important; border-radius: 10px; backdrop-filter: blur(3px); box-shadow: none;">
        <h4 style="color: white; margin: 0; text-shadow: none; font-weight: bold;">当前模型：{selected_model}</h4>
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

    # 保持原有的特征分类名称
    feature_categories = {
        "Proximate Analysis": ["M(wt%)", "Ash(wt%)", "VM(wt%)"],
        "Ultimate Analysis": ["O/C", "H/C", "N/C"],
        "Pyrolysis Conditions": ["FT(°C)", "HR(°C/min)", "FR(mL/min)"]
    }

    # 添加新的参数行样式CSS - 修复对齐问题
    st.markdown("""
    <style>
    /* 特征行样式 - 每个特征标签和输入框在一行对齐 */
    .feature-row {
        display: flex;
        align-items: center;
        gap: 10px;
        background: rgba(255, 255, 255, 0.85);
        border-radius: 10px;
        padding: 8px 12px;
        margin: 8px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.3);
        backdrop-filter: blur(5px);
        min-height: 50px;
    }

    /* 参数标签样式 - 彩色背景，固定宽度，垂直居中 */
    .param-label {
        color: white;
        font-weight: bold;
        font-size: 14px;
        padding: 8px 12px;
        border-radius: 6px;
        text-align: center;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        display: inline-block;
        width: 80px;
        flex-shrink: 0;
        margin: 0;
        line-height: 1.2;
    }

    /* 隐藏number_input的标签 */
    .stNumberInput label {
        display: none !important;
    }

    /* 调整number_input的样式 */
    .stNumberInput {
        flex: 1;
        margin: 0 !important;
    }

    .stNumberInput input {
        background-color: white !important;
        color: #333 !important;
        border: 1px solid #ddd !important;
        border-radius: 6px !important;
        text-align: center !important;
        font-weight: bold !important;
        font-size: 14px !important;
        padding: 8px 12px !important;
        width: 100% !important;
        margin: 0 !important;
    }

    .stNumberInput input:focus {
        border-color: #20b2aa !important;
        box-shadow: 0 0 5px rgba(32, 178, 170, 0.3) !important;
    }

    /* number_input的加减按钮样式 - 通用样式 */
    .stNumberInput button {
        border: none !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 4px !important;
        margin: 0 !important;
    }

    /* 第一列 Proximate Analysis 按钮颜色 - 青绿色 */
    .stColumn:nth-child(1) .stNumberInput button {
        background-color: #20b2aa !important;
    }

    /* 第二列 Ultimate Analysis 按钮颜色 - 金黄色 */
    .stColumn:nth-child(2) .stNumberInput button {
        background-color: #daa520 !important;
    }

    /* 第三列 Pyrolysis Conditions 按钮颜色 - 橙红色 */
    .stColumn:nth-child(3) .stNumberInput button {
        background-color: #cd5c5c !important;
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



    # 调试信息：显示所有当前输入值
    with st.expander("📊 显示当前输入值", expanded=False):
        debug_info = "<div style='columns: 3; column-gap: 20px;'>"
        for feature, value in features.items():
            debug_info += f"<p><b>{feature}</b>: {value:.3f}</p>"
        debug_info += "</div>"
        st.markdown(debug_info, unsafe_allow_html=True)

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