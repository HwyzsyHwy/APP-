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

    /* 模型卡片按钮样式 - 只影响主区域的模型选择按钮 */
    div[data-testid="stHorizontalBlock"] .stButton > button {
        background: rgba(255,255,255,0.85) !important;
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

    div[data-testid="stHorizontalBlock"] .stButton > button:hover {
        background: rgba(255,255,255,0.95) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 40px rgba(0,0,0,0.15) !important;
    }

    /* 选中状态的模型卡片 - 绿色背景 */
    .model-card-selected {
        background: linear-gradient(135deg, #00d2d3, #01a3a4) !important;
        color: white !important;
        border: 3px solid #00d2d3 !important;
    }

    </style>
    """, unsafe_allow_html=True)

    # 模型选择卡片（合并成完整的可点击卡片）
    col1, col2, col3 = st.columns(3)

    with col1:
        # Char Yield合并卡片
        if st.button("🔥\n\nChar Yield", key="char_card", use_container_width=True):
            if st.session_state.selected_model != "Char Yield":
                st.session_state.selected_model = "Char Yield"
                st.session_state.prediction_result = None
                st.session_state.warnings = []
                log(f"切换到模型: {st.session_state.selected_model}")
                st.rerun()

    with col2:
        # Oil Yield合并卡片
        if st.button("�️\n\nOil Yield", key="oil_card", use_container_width=True):
            if st.session_state.selected_model != "Oil Yield":
                st.session_state.selected_model = "Oil Yield"
                st.session_state.prediction_result = None
                st.session_state.warnings = []
                log(f"切换到模型: {st.session_state.selected_model}")
                st.rerun()

    with col3:
        # Gas Yield合并卡片
        if st.button("💨\n\nGas Yield", key="gas_card", use_container_width=True):
            if st.session_state.selected_model != "Gas Yield":
                st.session_state.selected_model = "Gas Yield"
                st.session_state.prediction_result = None
                st.session_state.warnings = []
                log(f"切换到模型: {st.session_state.selected_model}")
                st.rerun()

    # 动态设置选中状态的样式
    selected_model = st.session_state.selected_model
    st.markdown(f"""
    <script>
    setTimeout(function() {{
        // 重置所有模型卡片按钮样式
        var modelButtons = document.querySelectorAll('div[data-testid="stHorizontalBlock"] [data-testid="stButton"] button');
        modelButtons.forEach(function(btn) {{
            btn.style.background = 'rgba(255,255,255,0.85)';
            btn.style.border = '2px solid rgba(255,255,255,0.3)';
            btn.style.color = '#333';
        }});

        // 设置选中按钮的绿色样式
        var selectedModel = '{selected_model}';
        modelButtons.forEach(function(btn) {{
            if ((selectedModel === 'Char Yield' && btn.textContent.includes('Char Yield')) ||
                (selectedModel === 'Oil Yield' && btn.textContent.includes('Oil Yield')) ||
                (selectedModel === 'Gas Yield' && btn.textContent.includes('Gas Yield'))) {{
                btn.style.background = 'linear-gradient(135deg, #00d2d3, #01a3a4)';
                btn.style.border = '3px solid #00d2d3';
                btn.style.color = 'white';
            }}
        }});
    }}, 100);
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

    # 颜色配置
    category_colors = {
        "Ultimate Analysis": "#501d8a",
        "Proximate Analysis": "#1c8041",
        "Pyrolysis Conditions": "#e55709"
    }

    # 创建三列布局
    col1, col2, col3 = st.columns(3)

    # 使用字典存储所有输入值
    features = {}

    # Proximate Analysis - 第一列
    with col1:
        category = "Proximate Analysis"
        color = category_colors[category]
        st.markdown(f"<div class='section-header' style='background-color: {color};'>{category}</div>", unsafe_allow_html=True)

        for feature in feature_categories[category]:
            if st.session_state.clear_pressed:
                value = default_values[feature]
            else:
                value = st.session_state.feature_values.get(feature, default_values[feature])

            col_a, col_b = st.columns([1, 0.5])
            with col_a:
                st.markdown(f"<div class='input-label' style='background-color: {color};'>{feature}</div>", unsafe_allow_html=True)
            with col_b:
                features[feature] = st.number_input(
                    "",
                    value=float(value),
                    step=0.01,
                    key=f"{category}_{feature}",
                    format="%.3f",
                    label_visibility="collapsed"
                )

    # Ultimate Analysis - 第二列
    with col2:
        category = "Ultimate Analysis"
        color = category_colors[category]
        st.markdown(f"<div class='section-header' style='background-color: {color};'>{category}</div>", unsafe_allow_html=True)

        for feature in feature_categories[category]:
            if st.session_state.clear_pressed:
                value = default_values[feature]
            else:
                value = st.session_state.feature_values.get(feature, default_values[feature])

            col_a, col_b = st.columns([1, 0.5])
            with col_a:
                st.markdown(f"<div class='input-label' style='background-color: {color};'>{feature}</div>", unsafe_allow_html=True)
            with col_b:
                features[feature] = st.number_input(
                    "",
                    value=float(value),
                    step=0.001,
                    key=f"{category}_{feature}",
                    format="%.3f",
                    label_visibility="collapsed"
                )

    # Pyrolysis Conditions - 第三列
    with col3:
        category = "Pyrolysis Conditions"
        color = category_colors[category]
        st.markdown(f"<div class='section-header' style='background-color: {color};'>{category}</div>", unsafe_allow_html=True)

        for feature in feature_categories[category]:
            if st.session_state.clear_pressed:
                value = default_values[feature]
            else:
                value = st.session_state.feature_values.get(feature, default_values[feature])

            col_a, col_b = st.columns([1, 0.5])
            with col_a:
                st.markdown(f"<div class='input-label' style='background-color: {color};'>{feature}</div>", unsafe_allow_html=True)
            with col_b:
                # 不同特征使用不同的步长
                if feature == "FT(°C)":
                    step = 1.0
                    format_str = "%.1f"
                elif feature == "FR(mL/min)":
                    step = 1.0
                    format_str = "%.1f"
                else:  # HR(°C/min)
                    step = 0.1
                    format_str = "%.2f"

                features[feature] = st.number_input(
                    "",
                    value=float(value),
                    step=step,
                    key=f"{category}_{feature}",
                    format=format_str,
                    label_visibility="collapsed"
                )

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