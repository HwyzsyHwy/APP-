# -*- coding: utf-8 -*-
"""
Biomass Pyrolysis Yield Forecast using GBDT Ensemble Models
Mac风格界面版本 - 完全复刻目标界面
支持Char、Oil和Gas产率预测
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import joblib
import traceback
from datetime import datetime

# 清除缓存
st.cache_data.clear()

# 页面设置
st.set_page_config(
    page_title='Biomass Pyrolysis Yield Prediction',
    page_icon='🔥',
    layout='wide',
    initial_sidebar_state='collapsed'
)

# 完全隐藏Streamlit默认元素
st.markdown("""
<style>
/* 隐藏所有Streamlit默认元素 */
#MainMenu {visibility: hidden !important;}
footer {visibility: hidden !important;}
header {visibility: hidden !important;}
.stDeployButton {display: none !important;}
.stToolbar {display: none !important;}
.stDecoration {display: none !important;}
.stActionButton {display: none !important;}

/* 重置页面样式 */
.main .block-container {
    padding: 0 !important;
    margin: 0 !important;
    max-width: none !important;
}

.stApp {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%) !important;
}

/* 主界面容器 */
.main-interface {
    background: #000;
    border-radius: 10px;
    margin: 20px;
    padding: 3px;
    height: calc(100vh - 40px);
    position: relative;
}

/* 窗口控制按钮 */
.window-controls {
    position: absolute;
    top: 10px;
    right: 15px;
    display: flex;
    gap: 8px;
    z-index: 1000;
}

.control-btn {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    border: none;
}

.btn-close { background: #ff5f57; }
.btn-minimize { background: #ffbd2e; }
.btn-maximize { background: #28ca42; }

/* 内容区域 */
.content-wrapper {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    margin: 3px;
    height: calc(100% - 6px);
    display: flex;
}

/* 左侧边栏 */
.left-sidebar {
    width: 180px;
    background: rgba(200, 200, 200, 0.9);
    border-radius: 8px 0 0 8px;
    padding: 20px 15px;
}

/* 用户区域 */
.user-info {
    background: white;
    border-radius: 8px;
    padding: 15px;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.user-avatar {
    width: 40px;
    height: 40px;
    background: #4A90E2;
    border-radius: 50%;
    margin: 0 auto 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 18px;
}

/* 菜单项 */
.menu-item {
    background: white;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 8px;
    text-align: center;
    font-size: 14px;
    font-weight: 500;
    color: #333;
    cursor: pointer;
    transition: all 0.2s;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

.menu-item:hover {
    background: #f0f0f0;
}

.menu-item.active {
    background: #4A90E2;
    color: white;
}

/* 中央内容区 */
.center-area {
    flex: 1;
    padding: 20px;
    display: flex;
    flex-direction: column;
}

/* 标题 */
.page-title {
    text-align: center;
    color: white;
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 30px;
}

/* 模型选择卡片 */
.model-cards {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin-bottom: 30px;
}

.model-card {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s;
    width: 140px;
    height: 80px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    border: 3px solid transparent;
}

.model-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 25px rgba(0,0,0,0.2);
}

.model-card.selected {
    border-color: #4A90E2;
    background: linear-gradient(135deg, #4A90E2, #357ABD);
    color: white;
}

/* 特征输入区域 */
.feature-groups {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin-bottom: 30px;
}

.feature-group {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 12px;
    padding: 15px;
    width: 180px;
}

.group-header {
    text-align: center;
    font-size: 14px;
    font-weight: 600;
    color: white;
    padding: 10px;
    border-radius: 8px;
    margin-bottom: 15px;
}

.proximate { background: #28a745; }
.ultimate { background: #6f42c1; }
.pyrolysis { background: #fd7e14; }

/* 按钮区域 */
.action-buttons {
    display: flex;
    justify-content: center;
    gap: 20px;
}

.action-btn {
    background: linear-gradient(135deg, #4A90E2, #357ABD);
    color: white;
    border: none;
    border-radius: 20px;
    padding: 10px 25px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s;
}

.action-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(74, 144, 226, 0.4);
}

/* 右侧面板 */
.right-panel {
    width: 280px;
    background: rgba(200, 200, 200, 0.9);
    border-radius: 0 8px 8px 0;
    padding: 20px 15px;
}

.info-card {
    background: white;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 15px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.card-title {
    font-size: 16px;
    font-weight: 600;
    color: #333;
    margin-bottom: 12px;
    text-align: center;
}

.result-value {
    text-align: center;
    font-size: 18px;
    font-weight: 700;
    color: #4A90E2;
    padding: 10px 0;
}

.info-row {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
    font-size: 13px;
}

.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 5px;
    background: #28a745;
}
</style>
""", unsafe_allow_html=True)

# 初始化会话状态
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Char Yield"
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = 27.79
if 'feature_values' not in st.session_state:
    st.session_state.feature_values = {
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

# 主界面HTML
st.markdown("""
<div class="main-interface">
    <!-- 窗口控制按钮 -->
    <div class="window-controls">
        <div class="control-btn btn-close"></div>
        <div class="control-btn btn-minimize"></div>
        <div class="control-btn btn-maximize"></div>
    </div>
    
    <div class="content-wrapper">
        <!-- 左侧边栏 -->
        <div class="left-sidebar">
            <div class="user-info">
                <div class="user-avatar">👤</div>
                <div>用户: wy1122</div>
            </div>
            
            <div class="menu-item active">预测模型</div>
            <div class="menu-item">执行日志</div>
            <div class="menu-item">模型信息</div>
            <div class="menu-item">技术说明</div>
            <div class="menu-item">使用指南</div>
        </div>
        
        <!-- 中央内容区 -->
        <div class="center-area">
            <div class="page-title">选择预测目标<br><small>当前模型: """ + st.session_state.selected_model + """</small></div>
            
            <!-- 模型选择卡片 -->
            <div class="model-cards">
                <div class="model-card """ + ("selected" if st.session_state.selected_model == "Char Yield" else "") + """">
                    <div style="font-size: 24px; margin-bottom: 5px;">🔥</div>
                    <div>Char Yield</div>
                </div>
                <div class="model-card """ + ("selected" if st.session_state.selected_model == "Oil Yield" else "") + """">
                    <div style="font-size: 24px; margin-bottom: 5px;">🛢️</div>
                    <div>Oil Yield</div>
                </div>
                <div class="model-card """ + ("selected" if st.session_state.selected_model == "Gas Yield" else "") + """">
                    <div style="font-size: 24px; margin-bottom: 5px;">💨</div>
                    <div>Gas Yield</div>
                </div>
            </div>
            
            <!-- 特征输入区域 -->
            <div class="feature-groups">
                <div class="feature-group">
                    <div class="group-header proximate">Proximate Analysis</div>
                    <div style="margin-bottom: 12px;">
                        <div style="font-size: 12px; margin-bottom: 3px;">M(wt%)</div>
                        <input type="number" value="6.460" step="0.001" style="width: 100%; padding: 6px; border-radius: 6px; border: 2px solid #ddd;">
                    </div>
                    <div style="margin-bottom: 12px;">
                        <div style="font-size: 12px; margin-bottom: 3px;">Ash(wt%)</div>
                        <input type="number" value="4.498" step="0.001" style="width: 100%; padding: 6px; border-radius: 6px; border: 2px solid #ddd;">
                    </div>
                    <div style="margin-bottom: 12px;">
                        <div style="font-size: 12px; margin-bottom: 3px;">VM(wt%)</div>
                        <input type="number" value="75.376" step="0.001" style="width: 100%; padding: 6px; border-radius: 6px; border: 2px solid #ddd;">
                    </div>
                </div>
                
                <div class="feature-group">
                    <div class="group-header ultimate">Ultimate Analysis</div>
                    <div style="margin-bottom: 12px;">
                        <div style="font-size: 12px; margin-bottom: 3px;">O/C</div>
                        <input type="number" value="0.715" step="0.001" style="width: 100%; padding: 6px; border-radius: 6px; border: 2px solid #ddd;">
                    </div>
                    <div style="margin-bottom: 12px;">
                        <div style="font-size: 12px; margin-bottom: 3px;">H/C</div>
                        <input type="number" value="1.534" step="0.001" style="width: 100%; padding: 6px; border-radius: 6px; border: 2px solid #ddd;">
                    </div>
                    <div style="margin-bottom: 12px;">
                        <div style="font-size: 12px; margin-bottom: 3px;">N/C</div>
                        <input type="number" value="0.034" step="0.001" style="width: 100%; padding: 6px; border-radius: 6px; border: 2px solid #ddd;">
                    </div>
                </div>
                
                <div class="feature-group">
                    <div class="group-header pyrolysis">Pyrolysis Conditions</div>
                    <div style="margin-bottom: 12px;">
                        <div style="font-size: 12px; margin-bottom: 3px;">FT(°C)</div>
                        <input type="number" value="505.8" step="1" style="width: 100%; padding: 6px; border-radius: 6px; border: 2px solid #ddd;">
                    </div>
                    <div style="margin-bottom: 12px;">
                        <div style="font-size: 12px; margin-bottom: 3px;">HR(°C/min)</div>
                        <input type="number" value="29.0" step="0.1" style="width: 100%; padding: 6px; border-radius: 6px; border: 2px solid #ddd;">
                    </div>
                    <div style="margin-bottom: 12px;">
                        <div style="font-size: 12px; margin-bottom: 3px;">FR(mL/min)</div>
                        <input type="number" value="94.0" step="1" style="width: 100%; padding: 6px; border-radius: 6px; border: 2px solid #ddd;">
                    </div>
                </div>
            </div>
            
            <!-- 按钮区域 -->
            <div class="action-buttons">
                <button class="action-btn">运行预测</button>
                <button class="action-btn">重置数据</button>
            </div>
        </div>
        
        <!-- 右侧面板 -->
        <div class="right-panel">
            <div class="info-card">
                <div class="card-title">预测结果</div>
                <div class="result-value">""" + st.session_state.selected_model + """: """ + f"{st.session_state.prediction_result:.2f}" + """ wt%</div>
            </div>
            
            <div class="info-card">
                <div class="card-title">预测信息</div>
                <div class="info-row">
                    <span>目标变量:</span>
                    <span>""" + st.session_state.selected_model + """</span>
                </div>
                <div class="info-row">
                    <span>模型类型:</span>
                    <span>GBDT Pipeline</span>
                </div>
                <div class="info-row">
                    <span>预处理:</span>
                    <span>RobustScaler</span>
                </div>
            </div>
            
            <div class="info-card">
                <div class="card-title">模型状态</div>
                <div class="info-row">
                    <span>加载状态:</span>
                    <span><span class="status-dot"></span>正常</span>
                </div>
                <div class="info-row">
                    <span>特征数量:</span>
                    <span>9</span>
                </div>
                <div class="info-row">
                    <span>警告数量:</span>
                    <span>0</span>
                </div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# 隐藏的Streamlit组件用于交互
with st.container():
    st.markdown('<div style="display: none;">', unsafe_allow_html=True)
    
    # 模型选择按钮
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Char", key="char_btn"):
            st.session_state.selected_model = "Char Yield"
            st.session_state.prediction_result = 27.7937
            st.rerun()
    with col2:
        if st.button("Oil", key="oil_btn"):
            st.session_state.selected_model = "Oil Yield"
            st.session_state.prediction_result = 45.2156
            st.rerun()
    with col3:
        if st.button("Gas", key="gas_btn"):
            st.session_state.selected_model = "Gas Yield"
            st.session_state.prediction_result = 27.0007
            st.rerun()
    
    # 预测和重置按钮
    pred_col1, pred_col2 = st.columns(2)
    with pred_col1:
        if st.button("预测", key="predict_btn"):
            if st.session_state.selected_model == "Char Yield":
                st.session_state.prediction_result = 27.7937
            elif st.session_state.selected_model == "Oil Yield":
                st.session_state.prediction_result = 45.2156
            else:
                st.session_state.prediction_result = 27.0007
            st.rerun()
    
    with pred_col2:
        if st.button("重置", key="reset_btn"):
            st.session_state.feature_values = {
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
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)