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

# 自定义样式
st.markdown(
    """
    <style>
    /* 全局字体设置 */
    html, body, [class*="css"] {
        font-size: 16px !important;
    }
    
    /* 主容器样式 */
    .main-container {
        display: flex;
        height: 100vh;
        background-color: #f0f0f0;
    }
    
    /* 左侧边栏样式 */
    .left-sidebar {
        width: 200px;
        background-color: #8B7D6B;
        padding: 20px;
        color: white;
    }
    
    .sidebar-item {
        background-color: rgba(255,255,255,0.2);
        padding: 10px;
        margin: 10px 0;
        border-radius: 8px;
        text-align: center;
        cursor: pointer;
    }
    
    .sidebar-item:hover {
        background-color: rgba(255,255,255,0.3);
    }
    
    /* 中间内容区域 */
    .center-content {
        flex: 1;
        padding: 20px;
        background-color: #D4C4B0;
    }
    
    /* 模型选择区域 */
    .model-selection {
        text-align: center;
        margin-bottom: 20px;
    }
    
    .model-buttons {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin: 20px 0;
    }
    
    .model-card {
        background-color: rgba(255,255,255,0.8);
        padding: 40px 60px;
        border-radius: 15px;
        cursor: pointer;
        transition: all 0.3s;
        min-width: 150px;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
    }
    
    .model-card:hover {
        background-color: rgba(255,255,255,0.9);
        transform: translateY(-2px);
    }
    
    .model-card.selected {
        background-color: #4A90E2;
        color: white;
    }
    
    /* 特征输入区域 */
    .feature-sections {
        display: flex;
        gap: 20px;
        margin: 20px 0;
    }
    
    .feature-section {
        flex: 1;
        background-color: rgba(255,255,255,0.8);
        padding: 20px;
        border-radius: 15px;
    }
    
    .section-title {
        text-align: center;
        font-weight: bold;
        font-size: 18px;
        margin-bottom: 15px;
        padding: 10px;
        border-radius: 8px;
        color: white;
    }
    
    .proximate { background-color: #1c8041; }
    .ultimate { background-color: #501d8a; }
    .pyrolysis { background-color: #e55709; }
    
    /* 输入框样式 */
    .feature-input {
        display: flex;
        align-items: center;
        margin: 10px 0;
    }
    
    .feature-label {
        flex: 1;
        font-weight: bold;
        margin-right: 10px;
    }
    
    .feature-value {
        width: 80px;
    }
    
    /* 右侧结果区域 */
    .right-panel {
        width: 300px;
        background-color: #A69B8A;
        padding: 20px;
        color: white;
    }
    
    .result-card {
        background-color: rgba(255,255,255,0.2);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    
    .result-value {
        font-size: 24px;
        font-weight: bold;
        color: #FFD700;
    }
    
    .info-section {
        background-color: rgba(255,255,255,0.1);
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
    }
    
    /* 底部按钮区域 */
    .bottom-buttons {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin: 30px 0;
    }
    
    .action-button {
        padding: 15px 40px;
        border-radius: 10px;
        font-size: 16px;
        font-weight: bold;
        cursor: pointer;
        border: none;
        transition: all 0.3s;
    }
    
    .predict-btn {
        background-color: #4A90E2;
        color: white;
    }
    
    .reset-btn {
        background-color: #E74C3C;
        color: white;
    }
    
    .action-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* 隐藏Streamlit默认元素 */
    .stApp > header {
        background-color: transparent;
    }
    
    .stApp {
        margin-top: -80px;
    }
    
    /* 输入框样式 */
    [data-testid="stNumberInput"] input {
        background-color: white !important;
        color: black !important;
        border-radius: 5px;
        border: 1px solid #ccc;
        padding: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 初始化日志
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []

def log(message):
    """记录日志"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    st.session_state.log_messages.append(log_entry)
    if len(st.session_state.log_messages) > 100:
        st.session_state.log_messages = st.session_state.log_messages[-100:]

# 初始化会话状态
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Char Yield"
if 'model_cache' not in st.session_state:
    st.session_state.model_cache = {}
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'warnings' not in st.session_state:
    st.session_state.warnings = []
if 'feature_values' not in st.session_state:
    st.session_state.feature_values = {}

log("应用启动 - 根据图片特征统计信息正确修复版本")

class ModelPredictor:
    """预测器类"""
    
    def __init__(self, target_model="Char Yield"):
        self.target_name = target_model
        self.feature_names = [
            'M(wt%)', 'Ash(wt%)', 'VM(wt%)', 'O/C', 'H/C', 'N/C',
            'FT(℃)', 'HR(℃/min)', 'FR(mL/min)'
        ]
        
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
        
        self.ui_to_model_mapping = {
            'FT(°C)': 'FT(℃)',
            'HR(°C/min)': 'HR(℃/min)'
        }
        
        self.model_loaded = False
        self.pipeline = None
        self.model_file = None
        
        # 尝试加载模型
        if self._find_model_file():
            self._load_pipeline()
    
    def _find_model_file(self):
        """查找模型文件"""
        target_keywords = {
            "Char Yield": ["char"],
            "Oil Yield": ["oil"],
            "Gas Yield": ["gas"]
        }
        
        keywords = target_keywords.get(self.target_name, [])
        joblib_files = glob.glob("*.joblib")
        
        for keyword in keywords:
            for file in joblib_files:
                if keyword.lower() in file.lower():
                    self.model_file = file
                    log(f"找到模型文件: {file}")
                    return True
        
        log(f"未找到{self.target_name}对应的模型文件")
        return False
    
    def _load_pipeline(self):
        """加载Pipeline模型"""
        if not self.model_file:
            return False
        
        try:
            self.pipeline = joblib.load(self.model_file)
            self.model_loaded = True
            log(f"成功加载Pipeline模型: {self.model_file}")
            return True
        except Exception as e:
            log(f"加载模型失败: {str(e)}")
            return False
    
    def _prepare_features(self, features):
        """准备特征数据"""
        model_features = {}
        
        for ui_feature, value in features.items():
            model_feature = self.ui_to_model_mapping.get(ui_feature, ui_feature)
            if model_feature in self.feature_names:
                model_features[model_feature] = value
        
        feature_defaults = {
            'M(wt%)': 6.430226, 'Ash(wt%)': 4.498340, 'VM(wt%)': 75.375509,
            'O/C': 0.715385, 'H/C': 1.534106, 'N/C': 0.034083,
            'FT(℃)': 505.811321, 'HR(℃/min)': 29.011321, 'FR(mL/min)': 93.962264
        }
        
        for feature_name in self.feature_names:
            if feature_name not in model_features:
                model_features[feature_name] = feature_defaults[feature_name]
        
        feature_array = [model_features[name] for name in self.feature_names]
        return pd.DataFrame([feature_array], columns=self.feature_names)
    
    def check_input_range(self, features):
        """检查输入范围"""
        warnings = []
        for feature, value in features.items():
            model_feature = self.ui_to_model_mapping.get(feature, feature)
            if model_feature in self.training_ranges:
                range_info = self.training_ranges[model_feature]
                if value < range_info['min'] or value > range_info['max']:
                    warnings.append(f"{feature}: {value:.3f} 超出训练范围 [{range_info['min']:.3f}, {range_info['max']:.3f}]")
        return warnings
    
    def predict(self, features):
        """执行预测"""
        if not self.model_loaded or self.pipeline is None:
            return None
        
        try:
            features_df = self._prepare_features(features)
            result = float(self.pipeline.predict(features_df)[0])
            return result
        except Exception as e:
            log(f"预测失败: {str(e)}")
            return None

# 初始化预测器
predictor = ModelPredictor(target_model=st.session_state.selected_model)

# 默认值
default_values = {
    "M(wt%)": 6.430, "Ash(wt%)": 4.498, "VM(wt%)": 75.376,
    "O/C": 0.715, "H/C": 1.534, "N/C": 0.034,
    "FT(°C)": 505.811, "HR(°C/min)": 29.011, "FR(mL/min)": 93.962
}

# 特征分类
feature_categories = {
    "Proximate Analysis": ["M(wt%)", "Ash(wt%)", "VM(wt%)"],
    "Ultimate Analysis": ["O/C", "H/C", "N/C"],
    "Pyrolysis Conditions": ["FT(°C)", "HR(°C/min)", "FR(mL/min)"]
}

# 创建主布局
col_left, col_center, col_right = st.columns([1, 4, 2])

# 左侧边栏
with col_left:
    st.markdown("""
    <div class="left-sidebar">
        <div style="text-align: center; margin-bottom: 20px;">
            <h4>用户: wy1122</h4>
        </div>
        <div class="sidebar-item">预测模型</div>
        <div class="sidebar-item">执行日志</div>
        <div class="sidebar-item">模型信息</div>
        <div class="sidebar-item">技术说明</div>
        <div class="sidebar-item">使用指南</div>
    </div>
    """, unsafe_allow_html=True)

# 中间内容区域
with col_center:
    st.markdown('<div class="center-content">', unsafe_allow_html=True)
    
    # 标题
    st.markdown('<div class="model-selection"><h2>选择预测目标</h2></div>', unsafe_allow_html=True)
    
    # 模型选择按钮
    col1, col2, col3 = st.columns(3)
    with col1:
        char_selected = st.button("Char Yield", key="char_btn", use_container_width=True)
    with col2:
        oil_selected = st.button("Oil Yield", key="oil_btn", use_container_width=True)
    with col3:
        gas_selected = st.button("Gas Yield", key="gas_btn", use_container_width=True)
    
    # 处理模型选择
    if char_selected and st.session_state.selected_model != "Char Yield":
        st.session_state.selected_model = "Char Yield"
        predictor = ModelPredictor(target_model=st.session_state.selected_model)
        st.rerun()
    elif oil_selected and st.session_state.selected_model != "Oil Yield":
        st.session_state.selected_model = "Oil Yield"
        predictor = ModelPredictor(target_model=st.session_state.selected_model)
        st.rerun()
    elif gas_selected and st.session_state.selected_model != "Gas Yield":
        st.session_state.selected_model = "Gas Yield"
        predictor = ModelPredictor(target_model=st.session_state.selected_model)
        st.rerun()
    
    st.markdown(f'<div style="text-align: center; margin: 10px 0;">当前模型: <b>{st.session_state.selected_model}</b></div>', unsafe_allow_html=True)
    
    # 特征输入区域
    st.markdown('<div class="feature-sections">', unsafe_allow_html=True)
    
    features = {}
    
    # 三个特征输入列
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="section-title proximate">Proximate Analysis</div>', unsafe_allow_html=True)
        for feature in feature_categories["Proximate Analysis"