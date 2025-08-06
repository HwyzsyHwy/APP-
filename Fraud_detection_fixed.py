# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import traceback
from datetime import datetime

# 页面配置
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
        background-image: url('https://raw.githubusercontent.com/HwyzsyHwy/APP-/main/%E8%83%8C%E6%99%AF.png');
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

    /* 移除整体白色背景，保持透明 */
    .main .block-container {
        background-color: transparent !important;
        backdrop-filter: none !important;
        border-radius: 20px !important;
        padding: 30px !important;
        margin: 20px auto !important;
        max-width: 1200px !important;
        box-shadow: none !important;
        border: none !important;
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
        background: rgba(255,255,255,0.9) !important;
        border: 2px solid rgba(255,255,255,0.3) !important;
        border-radius: 15px !important;
        padding: 20px !important;
        height: auto !important;
        min-height: 120px !important;
        color: black !important;
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

    /* 数字输入框按钮的强制样式 */
    .stNumberInput button {
        color: white !important;
        border: none !important;
        font-weight: bold !important;
        border-radius: 4px !important;
        margin: 0 !important;
    }

    .stColumn:nth-child(1) .stNumberInput button {
        background-color: #20b2aa !important;
    }
    .stColumn:nth-child(2) .stNumberInput button {
        background-color: #daa520 !important;
    }
    .stColumn:nth-child(3) .stNumberInput button {
        background-color: #cd5c5c !important;
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
    </style>
    """,
    unsafe_allow_html=True
)

# 初始化会话状态
if 'current_page' not in st.session_state:
    st.session_state.current_page = "预测模型"

if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Char Yield"

if 'bottom_button_selected' not in st.session_state:
    st.session_state.bottom_button_selected = "predict"

if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []

def log(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    st.session_state.log_messages.append(log_entry)
    if len(st.session_state.log_messages) > 100:
        st.session_state.log_messages = st.session_state.log_messages[-100:]

# 侧边栏导航
with st.sidebar:
    # 用户信息区域
    st.markdown("""
    <div class="user-info">
        <img src="https://raw.githubusercontent.com/HwyzsyHwy/APP-/main/用户.png" class="user-avatar" alt="用户头像">
        <p class="user-name">用户：wy1122</p>
    </div>
    """, unsafe_allow_html=True)

    current_page = st.session_state.current_page

    if st.button("预测模型", key="nav_predict", use_container_width=True, type="primary" if current_page == "预测模型" else "secondary"):
        st.session_state.current_page = "预测模型"
        st.rerun()

    if st.button("执行日志", key="nav_log", use_container_width=True, type="primary" if current_page == "执行日志" else "secondary"):
        st.session_state.current_page = "执行日志"
        st.rerun()

    if st.button("模型信息", key="nav_info", use_container_width=True, type="primary" if current_page == "模型信息" else "secondary"):
        st.session_state.current_page = "模型信息"
        st.rerun()

    if st.button("技术说明", key="nav_tech", use_container_width=True, type="primary" if current_page == "技术说明" else "secondary"):
        st.session_state.current_page = "技术说明"
        st.rerun()

    if st.button("使用指南", key="nav_guide", use_container_width=True, type="primary" if current_page == "使用指南" else "secondary"):
        st.session_state.current_page = "使用指南"
        st.rerun()

# 主要内容
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

    st.markdown("<h3 style='color: white; text-align: center; margin-bottom: 30px;'>选择预测目标</h3>", unsafe_allow_html=True)
    
    # 模型选择
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔥 Char Yield", key="char_card", use_container_width=True,
                    type="primary" if st.session_state.selected_model == "Char Yield" else "secondary"):
            if st.session_state.selected_model != "Char Yield":
                st.session_state.selected_model = "Char Yield"
                st.session_state.prediction_result = None
                log("切换到模型: Char Yield")
                st.rerun()
    
    with col2:
        if st.button("🛢️ Oil Yield", key="oil_card", use_container_width=True,
                    type="primary" if st.session_state.selected_model == "Oil Yield" else "secondary"):
            if st.session_state.selected_model != "Oil Yield":
                st.session_state.selected_model = "Oil Yield"
                st.session_state.prediction_result = None
                log("切换到模型: Oil Yield")
                st.rerun()
    
    with col3:
        if st.button("💨 Gas Yield", key="gas_card", use_container_width=True,
                    type="primary" if st.session_state.selected_model == "Gas Yield" else "secondary"):
            if st.session_state.selected_model != "Gas Yield":
                st.session_state.selected_model = "Gas Yield"
                st.session_state.prediction_result = None
                log("切换到模型: Gas Yield")
                st.rerun()
    
    st.markdown("---")
    
    # 参数输入
    st.markdown('<h4 style="color: white; text-align: center; margin-bottom: 20px;">输入参数</h4>', unsafe_allow_html=True)
    
    # 默认值
    default_values = {
        "M(wt%)": 7.542,
        "Ash(wt%)": 1.542,
        "VM(wt%)": 82.542,
        "O/C": 0.542,
        "H/C": 1.542,
        "N/C": 0.034,
        "FT(C)": 505.811,
        "HR(C/min)": 29.011,
        "FR(mL/min)": 93.962
    }
    
    feature_categories = {
        "Proximate Analysis": ["M(wt%)", "Ash(wt%)", "VM(wt%)"],
        "Ultimate Analysis": ["O/C", "H/C", "N/C"],
        "Pyrolysis Conditions": ["FT(C)", "HR(C/min)", "FR(mL/min)"]
    }
    
    category_colors = {
        "Proximate Analysis": "#20b2aa",
        "Ultimate Analysis": "#daa520",
        "Pyrolysis Conditions": "#cd5c5c"
    }
    
    col1, col2, col3 = st.columns(3)
    features = {}
    
    # 创建页面内容容器
    st.markdown('<div class="page-content">', unsafe_allow_html=True)

    # 第一列
    with col1:
        st.markdown('<div style="background: white; color: #333; padding: 12px 20px; border-radius: 25px; text-align: center; font-weight: bold; font-size: 16px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">Proximate Analysis</div>', unsafe_allow_html=True)

        for feature in feature_categories["Proximate Analysis"]:
            value = st.number_input(
                feature,
                value=default_values[feature],
                step=0.001,
                format="%.3f",
                key=f"input_{feature}"
            )
            features[feature] = value

    # 第二列
    with col2:
        st.markdown('<div style="background: white; color: #333; padding: 12px 20px; border-radius: 25px; text-align: center; font-weight: bold; font-size: 16px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">Ultimate Analysis</div>', unsafe_allow_html=True)

        for feature in feature_categories["Ultimate Analysis"]:
            value = st.number_input(
                feature,
                value=default_values[feature],
                step=0.001,
                format="%.3f",
                key=f"input_{feature}"
            )
            features[feature] = value

    # 第三列
    with col3:
        st.markdown('<div style="background: white; color: #333; padding: 12px 20px; border-radius: 25px; text-align: center; font-weight: bold; font-size: 16px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">Pyrolysis Conditions</div>', unsafe_allow_html=True)

        for feature in feature_categories["Pyrolysis Conditions"]:
            value = st.number_input(
                feature,
                value=default_values[feature],
                step=0.001,
                format="%.3f",
                key=f"input_{feature}"
            )
            features[feature] = value

    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 底部按钮
    col1, col2 = st.columns([1, 1])
    
    with col1:
        predict_clicked = st.button("运行预测", use_container_width=True, 
                                   type="primary" if st.session_state.bottom_button_selected == "predict" else "secondary")
        if predict_clicked:
            st.session_state.bottom_button_selected = "predict"
            log("开始预测流程...")
            # 这里可以添加实际的预测逻辑
            st.session_state.prediction_result = 42.5  # 示例结果
            st.rerun()
    
    with col2:
        reset_clicked = st.button("重置输入", use_container_width=True,
                                 type="primary" if st.session_state.bottom_button_selected == "reset" else "secondary")
        if reset_clicked:
            st.session_state.bottom_button_selected = "reset"
            log("重置所有输入值")
            st.session_state.prediction_result = None
            st.rerun()
    
    # 显示预测结果
    if st.session_state.prediction_result is not None:
        st.markdown("---")
        st.markdown(f'<div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;"><h3 style="color: white; margin: 0;">预测结果: {st.session_state.prediction_result:.3f} wt%</h3></div>', unsafe_allow_html=True)

elif st.session_state.current_page == "执行日志":
    st.markdown('<h2 style="color: white; text-align: center; margin-bottom: 30px;">执行日志</h2>', unsafe_allow_html=True)

    if st.session_state.log_messages:
        log_content = "<br>".join(st.session_state.log_messages[-50:])
        st.markdown(f'<div class="log-container">{log_content}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="log-container" style="text-align: center;">暂无日志记录</div>', unsafe_allow_html=True)

elif st.session_state.current_page == "模型信息":
    st.markdown('<h2 style="color: white; text-align: center; margin-bottom: 30px;">模型信息</h2>', unsafe_allow_html=True)

    st.markdown('<div class="page-content">', unsafe_allow_html=True)
    st.markdown("### 当前模型：" + st.session_state.selected_model)
    st.markdown("**模型类型：** GBDT (Gradient Boosting Decision Tree)")
    st.markdown("**特征数量：** 9个")
    st.markdown("**训练数据：** 生物质热解产率数据集")
    st.markdown("**预处理：** RobustScaler标准化")
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.current_page == "技术说明":
    st.markdown('<h2 style="color: white; text-align: center; margin-bottom: 30px;">技术说明</h2>', unsafe_allow_html=True)

    st.markdown('<div class="page-content">', unsafe_allow_html=True)
    st.markdown("### 生物质热解产率预测系统")
    st.markdown("本系统基于机器学习技术，能够预测生物质在不同条件下的热解产率。")
    st.markdown("**支持的预测目标：**")
    st.markdown("- Char Yield (炭产率)")
    st.markdown("- Oil Yield (油产率)")
    st.markdown("- Gas Yield (气产率)")
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.current_page == "使用指南":
    st.markdown('<h2 style="color: white; text-align: center; margin-bottom: 30px;">使用指南</h2>', unsafe_allow_html=True)

    st.markdown('<div class="page-content">', unsafe_allow_html=True)
    st.markdown("### 如何使用本系统")
    st.markdown("1. **选择预测目标**：点击相应的模型卡片")
    st.markdown("2. **输入参数**：在三个类别中输入相应的数值")
    st.markdown("3. **运行预测**：点击"运行预测"按钮")
    st.markdown("4. **查看结果**：预测结果将显示在页面底部")
    st.markdown("5. **重置输入**：如需重新输入，点击"重置输入"按钮")
    st.markdown('</div>', unsafe_allow_html=True)
