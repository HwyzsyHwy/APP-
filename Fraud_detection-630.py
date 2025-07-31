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
    st.session_state.current_page = "预测"
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Char Yield"
if 'model_cache' not in st.session_state:
    st.session_state.model_cache = {}

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
.main-title {
    text-align: center;
    font-size: 32px !important;
    font-weight: bold;
    margin-bottom: 20px;
    color: white !important;
}
.section-header {
    color: white;
    font-weight: bold;
    font-size: 22px;
    text-align: center;
    padding: 10px;
    border-radius: 8px;
    margin-bottom: 15px;
}
.input-label {
    padding: 5px;
    border-radius: 5px;
    margin-bottom: 5px;
    font-size: 18px;
    color: white;
}
.yield-result {
    background-color: #1E1E1E;
    color: white;
    font-size: 36px;
    font-weight: bold;
    text-align: center;
    padding: 15px;
    border-radius: 8px;
    margin-top: 20px;
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
    background-color: #2E2E2E;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
}
.tech-info {
    background-color: #2E2E2E;
    padding: 15px;
    border-radius: 8px;
    color: white;
}
.model-selector {
    background-color: #2E2E2E;
    padding: 15px;
    border-radius: 8px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# 记录启动日志
add_log("应用启动")
add_log(f"初始化选定模型: {st.session_state.selected_model}")

# 侧边栏导航
with st.sidebar:
    st.markdown("### 🧭 导航菜单")
    
    if st.button("🔮 预测", use_container_width=True):
        st.session_state.current_page = "预测"
        st.rerun()
    
    st.markdown("### 📋 执行日志")
    display_logs()

# 主页面内容
st.markdown("<h1 class='main-title'>基于GBDT集成模型的生物质热解产物预测系统</h1>", unsafe_allow_html=True)

# 模型选择区域
st.markdown("<div class='model-selector'>", unsafe_allow_html=True)
st.markdown("<h3>选择预测目标</h3>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    char_button = st.button("🔥 Char Yield", key="char_button", use_container_width=True)
with col2:
    oil_button = st.button("🛢️ Oil Yield", key="oil_button", use_container_width=True)
with col3:
    gas_button = st.button("💨 Gas Yield", key="gas_button", use_container_width=True)

if char_button:
    st.session_state.selected_model = "Char Yield"
    add_log(f"切换到模型: {st.session_state.selected_model}")
    st.rerun()

if oil_button:
    st.session_state.selected_model = "Oil Yield"
    add_log(f"切换到模型: {st.session_state.selected_model}")
    st.rerun()

if gas_button:
    st.session_state.selected_model = "Gas Yield"
    add_log(f"切换到模型: {st.session_state.selected_model}")
    st.rerun()

st.markdown(f"<p style='text-align:center;'>当前模型: <b>{st.session_state.selected_model}</b></p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

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

# 初始化预测器
predictor = ModelPredictor(target_model=st.session_state.selected_model)

# 显示模型信息
model_info = predictor.get_model_info()
st.sidebar.markdown("### 模型信息")
for key, value in model_info.items():
    st.sidebar.write(f"**{key}**: {value}")

# 默认值
default_values = {
    "M(wt%)": 6.430, "Ash(wt%)": 4.498, "VM(wt%)": 75.376,
    "O/C": 0.715, "H/C": 1.534, "N/C": 0.034,
    "FT(°C)": 505.811, "HR(°C/min)": 29.011, "FR(mL/min)": 93.962
}

feature_categories = {
    "Proximate Analysis": ["M(wt%)", "Ash(wt%)", "VM(wt%)"],
    "Ultimate Analysis": ["O/C", "H/C", "N/C"],
    "Pyrolysis Conditions": ["FT(°C)", "HR(°C/min)", "FR(mL/min)"]
}

category_colors = {
    "Ultimate Analysis": "#501d8a",  
    "Proximate Analysis": "#1c8041",  
    "Pyrolysis Conditions": "#e55709" 
}

# 创建输入界面
col1, col2, col3 = st.columns(3)
features = {}

# Proximate Analysis
with col1:
    category = "Proximate Analysis"
    color = category_colors[category]
    st.markdown(f"<div class='section-header' style='background-color: {color};'>{category}</div>", unsafe_allow_html=True)
    
    for feature in feature_categories[category]:
        st.markdown(f"<div class='input-label' style='background-color: {color};'>{feature}</div>", unsafe_allow_html=True)
        features[feature] = st.number_input("", value=default_values[feature], key=f"input_{feature}")

# Ultimate Analysis
with col2:
    category = "Ultimate Analysis"
    color = category_colors[category]
    st.markdown(f"<div class='section-header' style='background-color: {color};'>{category}</div>", unsafe_allow_html=True)
    
    for feature in feature_categories[category]:
        st.markdown(f"<div class='input-label' style='background-color: {color};'>{feature}</div>", unsafe_allow_html=True)
        features[feature] = st.number_input("", value=default_values[feature], key=f"input_{feature}")

# Pyrolysis Conditions
with col3:
    category = "Pyrolysis Conditions"
    color = category_colors[category]
    st.markdown(f"<div class='section-header' style='background-color: {color};'>{category}</div>", unsafe_allow_html=True)
    
    for feature in feature_categories[category]:
        st.markdown(f"<div class='input-label' style='background-color: {color};'>{feature}</div>", unsafe_allow_html=True)
        features[feature] = st.number_input("", value=default_values[feature], key=f"input_{feature}")

# 预测按钮
if st.button("🔮 运行预测", type="primary"):
    add_log("开始预测流程...")
    st.success(f"模拟预测结果: {st.session_state.selected_model} = 25.67 wt%")

# 页脚
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
<p>© 2024 生物质纳米材料与智能装备实验室 | 基于GBDT的生物质热解产物预测系统</p>
</div>
""", unsafe_allow_html=True)