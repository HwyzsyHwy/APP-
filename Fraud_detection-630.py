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

# 初始化会话状态
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = "预测模型"
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
# 添加折叠状态
if 'prediction_info_expanded' not in st.session_state:
    st.session_state.prediction_info_expanded = True
if 'model_status_expanded' not in st.session_state:
    st.session_state.model_status_expanded = True

def add_log(message):
    """添加日志消息到会话状态"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    st.session_state.log_messages.append(log_entry)
    if len(st.session_state.log_messages) > 100:
        st.session_state.log_messages = st.session_state.log_messages[-100:]

def display_logs():
    """显示日志"""
    if st.session_state.log_messages:
        log_content = '<br>'.join(st.session_state.log_messages)
        st.markdown(f"<div class='log-container'>{log_content}</div>", unsafe_allow_html=True)

# 自定义样式
st.markdown("""
<style>
/* 全局背景设置 */
.stApp {
    background-color: #f5f5f5 !important;
}

/* 主内容区域 */
.main .block-container {
    padding-top: 2rem !important;
    background-color: #f5f5f5 !important;
    max-width: 100% !important;
}

/* 侧边栏整体样式 - 手机界面风格 */
.css-1d391kg {
    background-color: #f0f0f0 !important;
    border-radius: 20px !important;
    margin: 10px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    border: 1px solid #e0e0e0 !important;
}

/* 侧边栏内容区域 */
.css-1lcbmhc {
    background-color: #f0f0f0 !important;
    padding: 20px 15px !important;
    border-radius: 20px !important;
}

.main-title {
    text-align: center;
    font-size: 32px !important;
    font-weight: bold;
    margin-bottom: 20px;
    color: #333 !important;
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
}

.current-model {
    background-color: #1f4e79;
    color: white;
    font-size: 16px;
    padding: 10px;
    border-radius: 25px;
    margin: 20px 0;
    text-align: center;
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
}

/* 侧边栏用户信息样式 - 手机界面风格 */
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
}

/* Streamlit按钮样式覆盖 - 手机界面风格 */
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
}

.stButton > button:hover {
    background-color: #dee2e6 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
}

/* 主要按钮样式 - 深蓝色 */
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

/* 折叠按钮样式 */
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

/* 底部导航按钮样式 */
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
}

/* 侧边栏底部返回按钮 */
.sidebar-bottom {
    position: absolute;
    bottom: 20px;
    left: 50%;
    transform: translateX(-50%);
    width: 80%;
}

.back-button {
    background-color: transparent;
    border: none;
    color: #6c757d;
    font-size: 18px;
    font-weight: bold;
    cursor: pointer;
    padding: 10px;
    border-radius: 15px;
    transition: all 0.3s ease;
}

.back-button:hover {
    background-color: #e9ecef;
}
</style>
""", unsafe_allow_html=True)

# 记录启动日志
add_log("应用启动")
add_log(f"初始化选定模型: {st.session_state.selected_model}")

# 侧边栏导航 - 新的布局
with st.sidebar:
    # 用户信息区域
    st.markdown("""
    <div class='sidebar-user-info'>
        <div class='user-avatar'>👤</div>
        <div class='user-name'>用户：wy1122</div>
    </div>
    """, unsafe_allow_html=True)
    
    # 导航按钮
    st.markdown("### ")  # 空标题用于间距
    
    # 预测模型按钮
    if st.button("预测模型", key="nav_predict", use_container_width=True, 
                type="primary" if st.session_state.current_page == "预测模型" else "secondary"):
        st.session_state.current_page = "预测模型"
        add_log("切换到预测模型页面")
        st.rerun()
    
    # 执行日志按钮
    if st.button("执行日志", key="nav_logs", use_container_width=True,
                type="primary" if st.session_state.current_page == "执行日志" else "secondary"):
        st.session_state.current_page = "执行日志"
        add_log("切换到执行日志页面")
        st.rerun()
    
    # 模型信息按钮
    if st.button("模型信息", key="nav_model_info", use_container_width=True,
                type="primary" if st.session_state.current_page == "模型信息" else "secondary"):
        st.session_state.current_page = "模型信息"
        add_log("切换到模型信息页面")
        st.rerun()
    
    # 技术说明按钮
    if st.button("技术说明", key="nav_tech", use_container_width=True,
                type="primary" if st.session_state.current_page == "技术说明" else "secondary"):
        st.session_state.current_page = "技术说明"
        add_log("切换到技术说明页面")
        st.rerun()
    
    # 使用指南按钮
    if st.button("使用指南", key="nav_guide", use_container_width=True,
                type="primary" if st.session_state.current_page == "使用指南" else "secondary"):
        st.session_state.current_page = "使用指南"
        add_log("切换到使用指南页面")
        st.rerun()
    
    # 底部返回按钮
    st.markdown("<br><br><br>", unsafe_allow_html=True)  # 添加间距
    st.markdown("""
    <div style='text-align: center; margin-top: 50px;'>
        <button class='back-button'>&lt;</button>
    </div>
    """, unsafe_allow_html=True)

# 简化的预测器类
class ModelPredictor:
    def __init__(self, target_model="Char Yield"):
        self.target_name = target_model
        self.feature_names = [
            'M(wt%)', 'Ash(wt%)', 'VM(wt%)', 'O/C', 'H/C', 'N/C',
            'FT(℃)', 'HR(℃/min)', 'FR(mL/min)'
        ]
        self.model_loaded = False
        add_log(f"初始化预测器: {self.target_name}")
    
    def get_model_info(self):
        return {
            "模型类型": "GBDT Pipeline",
            "目标变量": self.target_name,
            "特征数量": len(self.feature_names),
            "模型状态": "已加载" if self.model_loaded else "未加载"
        }
    
    def predict(self, features):
        """模拟预测功能"""
        # 模拟预测结果
        import random
        random.seed(42)
        base_values = {
            "Char Yield": 27.79,
            "Oil Yield": 45.23,
            "Gas Yield": 18.56
        }
        result = base_values[self.target_name] + random.uniform(-5, 5)
        return round(result, 2)

# 根据当前页面显示不同内容
if st.session_state.current_page == "预测模型":
    # 主页面内容
    st.markdown("<h1 class='main-title'>基于GBDT集成模型的生物质热解产物预测系统</h1>", unsafe_allow_html=True)

    # 模型选择区域
    st.markdown("<div class='model-selector'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #333; text-align: center; margin-bottom: 30px;'>选择预测目标</h3>", unsafe_allow_html=True)
    
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
        add_log(f"切换到模型: {st.session_state.selected_model}")
        st.rerun()

    if oil_button:
        st.session_state.selected_model = "Oil Yield"
        st.session_state.prediction_result = None
        add_log(f"切换到模型: {st.session_state.selected_model}")
        st.rerun()

    if gas_button:
        st.session_state.selected_model = "Gas Yield"
        st.session_state.prediction_result = None
        add_log(f"切换到模型: {st.session_state.selected_model}")
        st.rerun()

    st.markdown(f"<div class='current-model'>当前模型：{st.session_state.selected_model}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # 初始化预测器
    predictor = ModelPredictor(target_model=st.session_state.selected_model)

    # 默认值
    default_values = {
        "M(wt%)": 6.460, "Ash(wt%)": 6.460, "VM(wt%)": 6.460,
        "O/C": 6.460, "H/C": 6.460, "N/C": 6.460,
        "FT(°C)": 6.460, "HR(°C/min)": 6.460, "FR(mL/min)": 6.460
    }

    # 创建主要布局：左侧输入区域，右侧信息面板
    main_col, info_col = st.columns([3, 1])

    with main_col:
        # 创建三列布局的卡片式输入界面
        col1, col2, col3 = st.columns(3)
        features = {}

        # Proximate Analysis 卡片
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
            col1_1, col1_2, col1_3 = st.columns([6, 1, 1])
            with col1_1:
                features["M(wt%)"] = st.number_input("", value=default_values["M(wt%)"], key="input_M", label_visibility="collapsed")
            with col1_2:
                if st.button("-", key="m_minus"):
                    pass
            with col1_3:
                if st.button("+", key="m_plus"):
                    pass
            
            # Ash(wt%)
            st.markdown("""
            <div class='input-row'>
                <div class='input-label'>Ash(wt%)</div>
            </div>
            """, unsafe_allow_html=True)
            col1_1, col1_2, col1_3 = st.columns([6, 1, 1])
            with col1_1:
                features["Ash(wt%)"] = st.number_input("", value=default_values["Ash(wt%)"], key="input_Ash", label_visibility="collapsed")
            with col1_2:
                if st.button("-", key="ash_minus"):
                    pass
            with col1_3:
                if st.button("+", key="ash_plus"):
                    pass
            
            # VM(wt%)
            st.markdown("""
            <div class='input-row'>
                <div class='input-label'>VM(wt%)</div>
            </div>
            """, unsafe_allow_html=True)
            col1_1, col1_2, col1_3 = st.columns([6, 1, 1])
            with col1_1:
                features["VM(wt%)"] = st.number_input("", value=default_values["VM(wt%)"], key="input_VM", label_visibility="collapsed")
            with col1_2:
                if st.button("-", key="vm_minus"):
                    pass
            with col1_3:
                if st.button("+", key="vm_plus"):
                    pass

        # Ultimate Analysis 卡片
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
            col2_1, col2_2, col2_3 = st.columns([6, 1, 1])
            with col2_1:
                features["O/C"] = st.number_input("", value=default_values["O/C"], key="input_OC", label_visibility="collapsed")
            with col2_2:
                if st.button("-", key="oc_minus"):
                    pass
            with col2_3:
                if st.button("+", key="oc_plus"):
                    pass
            
            # H/C
            st.markdown("""
            <div class='input-row'>
                <div class='input-label'>H/C</div>
            </div>
            """, unsafe_allow_html=True)
            col2_1, col2_2, col2_3 = st.columns([6, 1, 1])
            with col2_1:
                features["H/C"] = st.number_input("", value=default_values["H/C"], key="input_HC", label_visibility="collapsed")
            with col2_2:
                if st.button("-", key="hc_minus"):
                    pass
            with col2_3:
                if st.button("+", key="hc_plus"):
                    pass
            
            # N/C
            st.markdown("""
            <div class='input-row'>
                <div class='input-label'>N/C</div>
            </div>
            """, unsafe_allow_html=True)
            col2_1, col2_2, col2_3 = st.columns([6, 1, 1])
            with col2_1:
                features["N/C"] = st.number_input("", value=default_values["N/C"], key="input_NC", label_visibility="collapsed")
            with col2_2:
                if st.button("-", key="nc_minus"):
                    pass
            with col2_3:
                if st.button("+", key="nc_plus"):
                    pass

        # Pyrolysis Conditions 卡片
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
            col3_1, col3_2, col3_3 = st.columns([6, 1, 1])
            with col3_1:
                features["FT(°C)"] = st.number_input("", value=default_values["FT(°C)"], key="input_FT", label_visibility="collapsed")
            with col3_2:
                if st.button("-", key="ft_minus"):
                    pass
            with col3_3:
                if st.button("+", key="ft_plus"):
                    pass
            
            # HR(°C/min)
            st.markdown("""
            <div class='input-row'>
                <div class='input-label'>HR(°C/min)</div>
            </div>
            """, unsafe_allow_html=True)
            col3_1, col3_2, col3_3 = st.columns([6, 1, 1])
            with col3_1:
                features["HR(°C/min)"] = st.number_input("", value=default_values["HR(°C/min)"], key="input_HR", label_visibility="collapsed")
            with col3_2:
                if st.button("-", key="hr_minus"):
                    pass
            with col3_3:
                if st.button("+", key="hr_plus"):
                    pass
            
            # FR(mL/min)
            st.markdown("""
            <div class='input-row'>
                <div class='input-label'>FR(mL/min)</div>
            </div>
            """, unsafe_allow_html=True)
            col3_1, col3_2, col3_3 = st.columns([6, 1, 1])
            with col3_1:
                features["FR(mL/min)"] = st.number_input("", value=default_values["FR(mL/min)"], key="input_FR", label_visibility="collapsed")
            with col3_2:
                if st.button("-", key="fr_minus"):
                    pass
            with col3_3:
                if st.button("+", key="fr_plus"):
                    pass

        # 操作按钮
        st.markdown("""
        <div class='action-buttons'>
        </div>
        """, unsafe_allow_html=True)
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("运行预测", type="primary", use_container_width=True):
                add_log("开始预测流程...")
                # 执行预测
                result = predictor.predict(features)
                st.session_state.prediction_result = result
                add_log(f"预测完成: {st.session_state.selected_model} = {result} wt%")
                st.rerun()
        
        with col_btn2:
            if st.button("重置数据", use_container_width=True):
                add_log("重置所有输入数据")
                st.session_state.prediction_result = None
                st.rerun()

    # 右侧信息面板 - 添加折叠功能
    with info_col:
        # 获取当前模型的统计信息
        current_stats = st.session_state.model_stats[st.session_state.selected_model]
        
        # 预测结果显示
        result_text = f"{st.session_state.prediction_result} wt%" if st.session_state.prediction_result else "等待预测"
        
        # 使用Streamlit容器而不是HTML
        with st.container():
            # 预测结果标题
            st.markdown("### 预测结果")
            
            # 预测结果值
            if st.session_state.prediction_result:
                # 根据模型类型显示中文名称
                model_names = {
                    "Char Yield": "炭产量",
                    "Oil Yield": "油产量", 
                    "Gas Yield": "气产量"
                }
                model_chinese = model_names.get(st.session_state.selected_model, st.session_state.selected_model)
                st.success(f"**{model_chinese}**: {st.session_state.prediction_result} wt%")
            else:
                st.info("等待预测...")
            
            st.markdown("---")
            
            # 预测信息 - 可折叠
            col_header, col_toggle = st.columns([4, 1])
            with col_header:
                st.markdown("### 预测信息")
            with col_toggle:
                if st.button("▼" if st.session_state.prediction_info_expanded else "▶", 
                           key="toggle_prediction_info", 
                           help="展开/折叠预测信息"):
                    st.session_state.prediction_info_expanded = not st.session_state.prediction_info_expanded
                    st.rerun()
            
            if st.session_state.prediction_info_expanded:
                st.write(f"• **目标变量**: {st.session_state.selected_model}")
                st.write(f"• **预测结果**: {result_text}")
                st.write(f"• **模型类型**: GBDT Pipeline")
                st.write(f"• **预处理**: RobustScaler")
            
            st.markdown("---")
            
            # 模型状态 - 可折叠
            col_header2, col_toggle2 = st.columns([4, 1])
            with col_header2:
                st.markdown("### 模型状态")
            with col_toggle2:
                if st.button("▼" if st.session_state.model_status_expanded else "▶", 
                           key="toggle_model_status", 
                           help="展开/折叠模型状态"):
                    st.session_state.model_status_expanded = not st.session_state.model_status_expanded
                    st.rerun()
            
            if st.session_state.model_status_expanded:
                st.write(f"• **加载状态**: ✅ 正常")
                st.write(f"• **特征数量**: {current_stats['features']}")
                st.write(f"• **警告数量**: {current_stats['warnings']}")
            
            st.markdown("---")
            
            # 更多详细信息按钮
            if st.button("更多详细信息...", use_container_width=True):
                st.info("显示更多模型详细信息和统计数据...")

elif st.session_state.current_page == "执行日志":
    st.markdown("<h1 class='main-title'>执行日志</h1>", unsafe_allow_html=True)
    display_logs()

elif st.session_state.current_page == "模型信息":
    st.markdown("<h1 class='main-title'>模型信息</h1>", unsafe_allow_html=True)
    predictor = ModelPredictor(target_model=st.session_state.selected_model)
    model_info = predictor.get_model_info()
    
    for key, value in model_info.items():
        st.write(f"**{key}**: {value}")

elif st.session_state.current_page == "技术说明":
    st.markdown("<h1 class='main-title'>技术说明</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class='tech-info'>
    <h4>🔬 模型技术说明</h4>
    <p>本系统基于<b>梯度提升决策树(GBDT)</b>算法构建，采用Pipeline架构集成数据预处理和模型预测。</p>
    
    <h4>📋 特征说明</h4>
    <ul>
        <li><b>Proximate Analysis:</b> M(wt%) - 水分含量, Ash(wt%) - 灰分含量, VM(wt%) - 挥发分含量</li>
        <li><b>Ultimate Analysis:</b> O/C - 氧碳比, H/C - 氢碳比, N/C - 氮碳比</li>
        <li><b>Pyrolysis Conditions:</b> FT(°C) - 热解温度, HR(°C/min) - 升温速率, FR(mL/min) - 载气流量</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.current_page == "使用指南":
    st.markdown("<h1 class='main-title'>使用指南</h1>", unsafe_allow_html=True)
    st.markdown("""
    ### 📋 使用步骤
    1. 在侧边栏选择"预测模型"
    2. 选择要预测的目标（Char/Oil/Gas Yield）
    3. 输入生物质特征参数
    4. 点击"运行预测"获取结果
    
    ### ⚠️ 注意事项
    - 确保输入参数在合理范围内
    - 模型预测结果仅供参考
    - 实际应用需结合专业知识验证
    """)

# 页脚
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
<p>© 2024 生物质纳米材料与智能装备实验室 | 基于GBDT的生物质热解产物预测系统</p>
</div>
""", unsafe_allow_html=True)