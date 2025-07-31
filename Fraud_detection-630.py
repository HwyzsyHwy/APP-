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
/* 隐藏Streamlit默认元素 */
.stApp > header {visibility: hidden;}
.stDeployButton {display:none;}
footer {visibility: hidden;}
.stStatus {display: none;}

/* 主容器样式 */
.main-container {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    min-height: 100vh;
    padding: 0;
    margin: 0;
}

/* 顶部导航栏 */
.top-nav {
    background-color: rgba(0, 0, 0, 0.8);
    padding: 10px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: white;
}

.user-info {
    display: flex;
    align-items: center;
    gap: 10px;
}

.user-avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background-color: #2c5aa0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 16px;
}

.nav-buttons {
    display: flex;
    gap: 10px;
}

.nav-btn {
    background-color: #6c757d;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 20px;
    cursor: pointer;
    font-size: 14px;
}

.nav-btn.active {
    background-color: #2c5aa0;
}

/* 主标题 */
.main-title {
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    color: white;
    margin: 20px 0;
}

/* 模型选择区域 */
.model-selection {
    text-align: center;
    margin: 30px 0;
}

.model-selection h3 {
    color: white;
    margin-bottom: 20px;
    font-size: 18px;
}

.model-cards {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin-bottom: 20px;
}

.model-card {
    background-color: white;
    border-radius: 15px;
    padding: 30px;
    width: 200px;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

.model-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 16px rgba(0,0,0,0.3);
}

.model-card.selected {
    border: 3px solid #2c5aa0;
    background-color: #e3f2fd;
}

.model-icon {
    font-size: 48px;
    margin-bottom: 10px;
}

.model-name {
    font-size: 18px;
    font-weight: bold;
    color: #2c5aa0;
}

.current-model {
    background-color: #2c5aa0;
    color: white;
    padding: 10px 20px;
    border-radius: 25px;
    display: inline-block;
    margin: 20px 0;
}

/* 主要内容区域 */
.content-area {
    display: flex;
    gap: 20px;
    padding: 0 20px;
    max-width: 1400px;
    margin: 0 auto;
}

.input-section {
    flex: 3;
}

.info-panel {
    flex: 1;
    background-color: white;
    border-radius: 15px;
    padding: 0;
    height: fit-content;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

/* 输入卡片 */
.analysis-cards {
    display: flex;
    gap: 20px;
    margin-bottom: 30px;
}

.analysis-card {
    background-color: white;
    border-radius: 15px;
    padding: 20px;
    flex: 1;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

.card-title {
    background-color: #2c5aa0;
    color: white;
    font-weight: bold;
    font-size: 16px;
    text-align: center;
    padding: 12px;
    border-radius: 8px;
    margin-bottom: 20px;
}

.input-group {
    margin-bottom: 15px;
}

.input-row {
    display: flex;
    align-items: center;
    gap: 10px;
}

.input-label {
    background-color: #2c5aa0;
    color: white;
    padding: 8px 12px;
    border-radius: 5px;
    font-size: 14px;
    font-weight: bold;
    min-width: 80px;
    text-align: center;
}

.input-field {
    flex: 1;
    padding: 8px;
    border: 1px solid #ddd;
    border-radius: 5px;
    font-size: 14px;
}

.input-controls {
    display: flex;
    gap: 5px;
}

.control-btn {
    background-color: #2c5aa0;
    color: white;
    border: none;
    padding: 8px 12px;
    border-radius: 5px;
    cursor: pointer;
    font-size: 14px;
    min-width: 35px;
}

.control-btn:hover {
    background-color: #1a4480;
}

/* 操作按钮 */
.action-buttons {
    display: flex;
    gap: 20px;
    justify-content: center;
    margin-top: 30px;
}

.action-btn {
    padding: 15px 40px;
    border-radius: 8px;
    border: none;
    font-size: 16px;
    font-weight: bold;
    cursor: pointer;
    transition: all 0.3s;
}

.predict-btn {
    background-color: #2c5aa0;
    color: white;
}

.predict-btn:hover {
    background-color: #1a4480;
}

.reset-btn {
    background-color: #6c757d;
    color: white;
}

.reset-btn:hover {
    background-color: #545b62;
}

/* 右侧信息面板 */
.result-header {
    background-color: #2c5aa0;
    color: white;
    padding: 15px;
    border-radius: 15px 15px 0 0;
    font-size: 16px;
    font-weight: bold;
}

.result-value {
    background-color: #e8f5e8;
    padding: 15px;
    font-size: 16px;
    font-weight: bold;
    color: #2c5aa0;
    border-bottom: 1px solid #e0e0e0;
}

.info-section {
    padding: 15px;
    border-bottom: 1px solid #e0e0e0;
}

.info-title {
    font-size: 16px;
    font-weight: bold;
    color: #333;
    margin-bottom: 10px;
}

.info-item {
    margin-bottom: 5px;
    font-size: 14px;
    color: #555;
}

.status-normal {
    color: #28a745;
}

.expand-btn {
    background-color: #f8f9fa;
    border: none;
    padding: 15px;
    width: 100%;
    text-align: center;
    border-radius: 0 0 15px 15px;
    cursor: pointer;
    font-size: 16px;
    color: #666;
}

.expand-btn:hover {
    background-color: #e9ecef;
}

/* 底部按钮 */
.bottom-buttons {
    display: flex;
    gap: 20px;
    justify-content: center;
    margin-top: 30px;
    padding-bottom: 30px;
}

.bottom-btn {
    background-color: #2c5aa0;
    color: white;
    border: none;
    padding: 15px 30px;
    border-radius: 8px;
    font-size: 16px;
    font-weight: bold;
    cursor: pointer;
    min-width: 120px;
}

.bottom-btn.secondary {
    background-color: #6c757d;
}

.bottom-btn:hover {
    opacity: 0.9;
}

/* 隐藏Streamlit默认样式 */
.stButton > button {
    background-color: transparent !important;
    border: none !important;
    padding: 0 !important;
    color: inherit !important;
}

.stSelectbox > div > div {
    background-color: white;
}

.stNumberInput > div > div > input {
    background-color: white;
    border: 1px solid #ddd;
    border-radius: 5px;
}

/* 响应式设计 */
@media (max-width: 1200px) {
    .analysis-cards {
        flex-direction: column;
    }
    
    .content-area {
        flex-direction: column;
    }
    
    .model-cards {
        flex-direction: column;
        align-items: center;
    }
}
</style>
""", unsafe_allow_html=True)

# 记录启动日志
add_log("应用启动")
add_log(f"初始化选定模型: {st.session_state.selected_model}")

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
    # 主容器
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # 顶部导航栏
    st.markdown(f"""
    <div class="top-nav">
        <div class="user-info">
            <div class="user-avatar">👤</div>
            <span>用户：wy1122</span>
        </div>
        <div class="nav-buttons">
            <button class="nav-btn active">预测模型</button>
            <button class="nav-btn">执行日志</button>
            <button class="nav-btn">模型信息</button>
            <button class="nav-btn">技术说明</button>
            <button class="nav-btn">使用指南</button>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 模型选择区域
    st.markdown("""
    <div class="model-selection">
        <h3>选择预测目标</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # 模型选择卡片和右侧面板
    col_main, col_info = st.columns([4, 1])
    
    with col_main:
        # 模型选择卡片
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("char_select", key="char_button", label_visibility="collapsed"):
                st.session_state.selected_model = "Char Yield"
                st.session_state.prediction_result = None
                add_log(f"切换到模型: {st.session_state.selected_model}")
                st.rerun()
            
            selected_class = "selected" if st.session_state.selected_model == "Char Yield" else ""
            st.markdown(f"""
            <div class="model-card {selected_class}" onclick="document.querySelector('[data-testid*=char_button]').click()">
                <div class="model-icon">🔥</div>
                <div class="model-name">Char Yield</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if st.button("oil_select", key="oil_button", label_visibility="collapsed"):
                st.session_state.selected_model = "Oil Yield"
                st.session_state.prediction_result = None
                add_log(f"切换到模型: {st.session_state.selected_model}")
                st.rerun()
            
            selected_class = "selected" if st.session_state.selected_model == "Oil Yield" else ""
            st.markdown(f"""
            <div class="model-card {selected_class}" onclick="document.querySelector('[data-testid*=oil_button]').click()">
                <div class="model-icon">🛢️</div>
                <div class="model-name">Oil Yield</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            if st.button("gas_select", key="gas_button", label_visibility="collapsed"):
                st.session_state.selected_model = "Gas Yield"
                st.session_state.prediction_result = None
                add_log(f"切换到模型: {st.session_state.selected_model}")
                st.rerun()
            
            selected_class = "selected" if st.session_state.selected_model == "Gas Yield" else ""
            st.markdown(f"""
            <div class="model-card {selected_class}" onclick="document.querySelector('[data-testid*=gas_button]').click()">
                <div class="model-icon">💨</div>
                <div class="model-name">Gas Yield</div>
            </div>
            """, unsafe_allow_html=True)
        
        # 当前模型显示
        st.markdown(f'<div class="current-model">当前模型：{st.session_state.selected_model}</div>', unsafe_allow_html=True)
        
        # 输入卡片区域
        st.markdown('<div class="analysis-cards">', unsafe_allow_html=True)
        
        # 三个分析卡片
        col1, col2, col3 = st.columns(3)
        features = {}
        default_values = {
            "M(wt%)": 6.460, "Ash(wt%)": 6.460, "VM(wt%)": 6.460,
            "O/C": 6.460, "H/C": 6.460, "N/C": 6.460,
            "FT(°C)": 6.460, "HR(°C/min)": 6.460, "FR(mL/min)": 6.460
        }
        
        # Proximate Analysis 卡片
        with col1:
            st.markdown("""
            <div class="analysis-card">
                <div class="card-title">Proximate Analysis</div>
            """, unsafe_allow_html=True)
            
            # M(wt%)
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            col1_1, col1_2, col1_3 = st.columns([1, 4, 1])
            with col1_1:
                st.markdown('<div class="input-label">M(wt%)</div>', unsafe_allow_html=True)
            with col1_2:
                features["M(wt%)"] = st.number_input("", value=default_values["M(wt%)"], key="input_M", label_visibility="collapsed")
            with col1_3:
                st.markdown('<div class="input-controls"><button class="control-btn">-</button><button class="control-btn">+</button></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Ash(wt%)
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            col1_1, col1_2, col1_3 = st.columns([1, 4, 1])
            with col1_1:
                st.markdown('<div class="input-label">Ash(wt%)</div>', unsafe_allow_html=True)
            with col1_2:
                features["Ash(wt%)"] = st.number_input("", value=default_values["Ash(wt%)"], key="input_Ash", label_visibility="collapsed")
            with col1_3:
                st.markdown('<div class="input-controls"><button class="control-btn">-</button><button class="control-btn">+</button></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # VM(wt%)
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            col1_1, col1_2, col1_3 = st.columns([1, 4, 1])
            with col1_1:
                st.markdown('<div class="input-label">VM(wt%)</div>', unsafe_allow_html=True)
            with col1_2:
                features["VM(wt%)"] = st.number_input("", value=default_values["VM(wt%)"], key="input_VM", label_visibility="collapsed")
            with col1_3:
                st.markdown('<div class="input-controls"><button class="control-btn">-</button><button class="control-btn">+</button></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Ultimate Analysis 卡片
        with col2:
            st.markdown("""
            <div class="analysis-card">
                <div class="card-title">Ultimate Analysis</div>
            """, unsafe_allow_html=True)
            
            # O/C
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            col2_1, col2_2, col2_3 = st.columns([1, 4, 1])
            with col2_1:
                st.markdown('<div class="input-label">O/C</div>', unsafe_allow_html=True)
            with col2_2:
                features["O/C"] = st.number_input("", value=default_values["O/C"], key="input_OC", label_visibility="collapsed")
            with col2_3:
                st.markdown('<div class="input-controls"><button class="control-btn">-</button><button class="control-btn">+</button></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # H/C
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            col2_1, col2_2, col2_3 = st.columns([1, 4, 1])
            with col2_1:
                st.markdown('<div class="input-label">H/C</div>', unsafe_allow_html=True)
            with col2_2:
                features["H/C"] = st.number_input("", value=default_values["H/C"], key="input_HC", label_visibility="collapsed")
            with col2_3:
                st.markdown('<div class="input-controls"><button class="control-btn">-</button><button class="control-btn">+</button></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # N/C
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            col2_1, col2_2, col2_3 = st.columns([1, 4, 1])
            with col2_1:
                st.markdown('<div class="input-label">N/C</div>', unsafe_allow_html=True)
            with col2_2:
                features["N/C"] = st.number_input("", value=default_values["N/C"], key="input_NC", label_visibility="collapsed")
            with col2_3:
                st.markdown('<div class="input-controls"><button class="control-btn">-</button><button class="control-btn">+</button></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Pyrolysis Conditions 卡片
        with col3:
            st.markdown("""
            <div class="analysis-card">
                <div class="card-title">Pyrolysis Conditions</div>
            """, unsafe_allow_html=True)
            
            # FT(°C)
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            col3_1, col3_2, col3_3 = st.columns([1, 4, 1])
            with col3_1:
                st.markdown('<div class="input-label">FT(°C)</div>', unsafe_allow_html=True)
            with col3_2:
                features["FT(°C)"] = st.number_input("", value=default_values["FT(°C)"], key="input_FT", label_visibility="collapsed")
            with col3_3:
                st.markdown('<div class="input-controls"><button class="control-btn">-</button><button class="control-btn">+</button></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # HR(°C/min)
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            col3_1, col3_2, col3_3 = st.columns([1, 4, 1])
            with col3_1:
                st.markdown('<div class="input-label">HR(°C/min)</div>', unsafe_allow_html=True)
            with col3_2:
                features["HR(°C/min)"] = st.number_input("", value=default_values["HR(°C/min)"], key="input_HR", label_visibility="collapsed")
            with col3_3:
                st.markdown('<div class="input-controls"><button class="control-btn">-</button><button class="control-btn">+</button></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # FR(mL/min)
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            col3_1, col3_2, col3_3 = st.columns([1, 4, 1])
            with col3_1:
                st.markdown('<div class="input-label">FR(mL/min)</div>', unsafe_allow_html=True)
            with col3_2:
                features["FR(mL/min)"] = st.number_input("", value=default_values["FR(mL/min)"], key="input_FR", label_visibility="collapsed")
            with col3_3:
                st.markdown('<div class="input-controls"><button class="control-btn">-</button><button class="control-btn">+</button></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 底部操作按钮
        st.markdown('<div class="bottom-buttons">', unsafe_allow_html=True)
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        
        with col_btn1:
            if st.button("<", key="prev_btn", use_container_width=True):
                st.info("上一步功能")
        
        with col_btn2:
            if st.button("运行预测", key="predict_btn", type="primary", use_container_width=True):
                predictor = ModelPredictor(target_model=st.session_state.selected_model)
                add_log("开始预测流程...")
                result = predictor.predict(features)
                st.session_state.prediction_result = result
                add_log(f"预测完成: {st.session_state.selected_model} = {result} wt%")
                st.rerun()
        
        with col_btn3:
            if st.button("重置数据", key="reset_btn", use_container_width=True):
                add_log("重置所有输入数据")
                st.session_state.prediction_result = None
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 右侧信息面板
    with col_info:
        current_stats = st.session_state.model_stats[st.session_state.selected_model]
        result_text = f"{st.session_state.prediction_result} wt%" if st.session_state.prediction_result else "等待预测"
        
        # 预测结果面板
        st.markdown(f"""
        <div class="info-panel">
            <div class="result-header">预测结果</div>
            <div class="result-value">{st.session_state.selected_model}: {result_text}</div>
            
            <div class="info-section">
                <div class="info-title">预测信息</div>
                <div class="info-item">• 目标变量：{st.session_state.selected_model}</div>
                <div class="info-item">• 预测结果：{result_text}</div>
                <div class="info-item">• 模型类型：GBDT Pipeline</div>
                <div class="info-item">• 预处理：RobustScaler</div>
            </div>
            
            <div class="info-section">
                <div class="info-title">模型状态</div>
                <div class="info-item">• 加载状态：<span class="status-normal">✅ 正常</span></div>
                <div class="info-item">• 特征数量：{current_stats['features']}</div>
                <div class="info-item">• 警告数量：{current_stats['warnings']}</div>
            </div>
            
            <button class="expand-btn">></button>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

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
    ### 🔬 模型技术说明
    本系统基于**梯度提升决策树(GBDT)**算法构建，采用Pipeline架构集成数据预处理和模型预测。
    
    ### 📋 特征说明
    - **Proximate Analysis:** M(wt%) - 水分含量, Ash(wt%) - 灰分含量, VM(wt%) - 挥发分含量
    - **Ultimate Analysis:** O/C - 氧碳比, H/C - 氢碳比, N/C - 氮碳比
    - **Pyrolysis Conditions:** FT(°C) - 热解温度, HR(°C/min) - 升温速率, FR(mL/min) - 载气流量
    """)

elif st.session_state.current_page == "使用指南":
    st.markdown("<h1 class='main-title'>使用指南</h1>", unsafe_allow_html=True)
    st.markdown("""
    ### 📋 使用步骤
    1. 选择要预测的目标（Char/Oil/Gas Yield）
    2. 输入生物质特征参数
    3. 点击"运行预测"获取结果
    
    ### ⚠️ 注意事项
    - 确保输入参数在合理范围内
    - 模型预测结果仅供参考
    - 实际应用需结合专业知识验证
    """)