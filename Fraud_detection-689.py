# -*- coding: utf-8 -*-
"""
Biomass Pyrolysis Yield Forecast using GBDT Ensemble Models
三栏布局版本 - 左侧输入、中间预测、右侧参数显示
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

# 自定义样式（添加背景图片和三栏布局）
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

# 记录启动日志
log("应用启动 - 三栏布局版本")
log("特征顺序：M, Ash, VM, O/C, H/C, N/C, FT, HR, FR")

# 初始化会话状态 - 添加模型选择功能
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Char Yield"  # 默认选择Char产率模型
    log(f"初始化选定模型: {st.session_state.selected_model}")

# 添加模型缓存 - 避免重复加载相同模型
if 'model_cache' not in st.session_state:
    st.session_state.model_cache = {}
