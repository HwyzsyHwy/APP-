# -*- coding: utf-8 -*-
"""
Biomass Pyrolysis Yield Forecast using GBDT Ensemble Models
彻底重构版本 - 解决参数输入无效和标准化器识别问题
支持Char、Oil和Gas产率预测
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import joblib
import json
import traceback
import matplotlib.pyplot as plt
from datetime import datetime
import io
from PIL import Image
import pickle
import sys
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler

# 清除缓存，强制重新渲染
if "debug" not in st.session_state:
    st.cache_data.clear()
    st.session_state.debug = True

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
    
    /* 标题 */
    .main-title {
        text-align: center;
        font-size: 32px !important;
        font-weight: bold;
        margin-bottom: 20px;
        color: white !important;
    }
    
    /* 区域样式 */
    .section-header {
        color: white;
        font-weight: bold;
        font-size: 22px;
        text-align: center;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    
    /* 输入标签样式 */
    .input-label {
        padding: 5px;
        border-radius: 5px;
        margin-bottom: 5px;
        font-size: 18px;
        color: white;
    }
    
    /* 结果显示样式 */
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
        background-color: rgba(255, 165, 0, 0.2);
        border-left: 5px solid orange;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    
    /* 错误样式 */
    .error-box {
        background-color: rgba(255, 0, 0, 0.2);
        border-left: 5px solid red;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    
    /* 成功样式 */
    .success-box {
        background-color: rgba(0, 128, 0, 0.2);
        border-left: 5px solid green;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    
    /* 日志样式 */
    .log-container {
        height: 300px;
        overflow-y: auto;
        background-color: #1E1E1E;
        color: #00FF00;
        font-family: 'Courier New', monospace;
        padding: 10px;
        border-radius: 5px;
        font-size: 14px !important;
    }
    
    /* 模型选择器样式 */
    .model-selector {
        background-color: #2E2E2E;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
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

# 创建侧边栏日志区域
log_container = st.sidebar.container()
log_container.markdown("<h3>执行日志</h3>", unsafe_allow_html=True)
log_text = st.sidebar.empty()

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
    
    # 更新日志显示
    log_text.markdown(
        f"<div class='log-container'>{'<br>'.join(st.session_state.log_messages)}</div>", 
        unsafe_allow_html=True
    )

# 记录启动日志
log("应用启动 - 彻底重构版本")
log("采用新方法解决参数输入无效问题")

# 初始化会话状态 - 添加模型选择功能
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Char Yield"  # 默认选择Char产率模型
    log(f"初始化选定模型: {st.session_state.selected_model}")

# 更新主标题以显示当前选定的模型
st.markdown("<h1 class='main-title'>基于GBDT集成模型的生物质热解产物预测系统</h1>", unsafe_allow_html=True)

# 添加模型选择区域 - 修改为三个按钮一排
st.markdown("<div class='model-selector'>", unsafe_allow_html=True)
st.markdown("<h3>选择预测目标</h3>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    char_button = st.button(" Char Yield", 
                           key="char_button", 
                           help="预测焦炭产率 (wt%)", 
                           use_container_width=True,
                           type="primary" if st.session_state.selected_model == "Char Yield" else "secondary")
with col2:
    oil_button = st.button(" Oil Yield", 
                          key="oil_button", 
                          help="预测生物油产率 (wt%)", 
                          use_container_width=True,
                          type="primary" if st.session_state.selected_model == "Oil Yield" else "secondary")
with col3:
    gas_button = st.button(" Gas Yield", 
                          key="gas_button", 
                          help="预测气体产率 (wt%)", 
                          use_container_width=True,
                          type="primary" if st.session_state.selected_model == "Gas Yield" else "secondary")

# 处理模型选择 - 修改为切换模型时不重置输入值
if char_button and st.session_state.selected_model != "Char Yield":
    st.session_state.selected_model = "Char Yield"
    st.session_state.prediction_result = None
    st.session_state.warnings = []
    log(f"切换到模型: {st.session_state.selected_model}")
    st.rerun()

if oil_button and st.session_state.selected_model != "Oil Yield":
    st.session_state.selected_model = "Oil Yield"
    st.session_state.prediction_result = None
    st.session_state.warnings = []
    log(f"切换到模型: {st.session_state.selected_model}")
    st.rerun()

if gas_button and st.session_state.selected_model != "Gas Yield":
    st.session_state.selected_model = "Gas Yield"
    st.session_state.prediction_result = None
    st.session_state.warnings = []
    log(f"切换到模型: {st.session_state.selected_model}")
    st.rerun()

st.markdown(f"<p style='text-align:center;'>当前模型: <b>{st.session_state.selected_model}</b></p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

class ModelLoader:
    """通用模型加载器"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.is_pipeline = False
        self.loaded = False
        
        self._load_model()
    
    def _load_model(self):
        """加载模型和标准化器"""
        try:
            log(f"尝试加载模型文件: {self.model_path}")
            
            # 尝试加载模型
            loaded_obj = joblib.load(self.model_path)
            log(f"成功加载文件: {type(loaded_obj).__name__}")
            
            # 检查是否为Pipeline
            if hasattr(loaded_obj, 'named_steps'):
                log("检测到Pipeline结构")
                self.is_pipeline = True
                
                # 遍历Pipeline步骤
                for name, step in loaded_obj.named_steps.items():
                    step_type = type(step).__name__
                    log(f"Pipeline组件: {name} (类型: {step_type})")
                    
                    # 检查是否为标准化器
                    if any(x in step_type.lower() for x in ['scaler', 'standard', 'robust', 'normalizer']):
                        self.scaler = step
                        log(f"找到标准化器: {step_type}")
                    
                    # 检查是否为模型
                    if any(x in step_type.lower() for x in ['regressor', 'boost', 'forest']):
                        self.model = step
                        log(f"找到模型: {step_type}")
                
                # 如果没有找到模型，使用整个Pipeline作为模型
                if self.model is None:
                    log("未找到独立模型组件，使用整个Pipeline")
                    self.model = loaded_obj
            else:
                # 直接将加载的对象作为模型
                log("加载的对象不是Pipeline，直接用作模型")
                self.model = loaded_obj
            
            # 确认加载状态
            self.loaded = self.model is not None
            log(f"模型加载状态: {'成功' if self.loaded else '失败'}")
            log(f"标准化器状态: {'已找到' if self.scaler else '未找到'}")
            
            return self.loaded
            
        except Exception as e:
            log(f"加载模型时出错: {str(e)}")
            log(traceback.format_exc())
            return False
    
    def predict(self, X):
        """预测函数"""
        if not self.loaded:
            log("错误: 模型未加载，无法预测")
            return None
        
        try:
            # 记录输入数据
            if isinstance(X, pd.DataFrame):
                for col in X.columns:
                    log(f"预测输入 {col}: {X[col].values[0]}")
            
            # 如果是Pipeline，直接使用
            if self.is_pipeline:
                log("使用Pipeline进行预测")
                return self.model.predict(X)
            
            # 如果有标准化器，先转换数据
            if self.scaler:
                log("使用标准化器转换数据")
                if isinstance(X, pd.DataFrame):
                    X_scaled = self.scaler.transform(X.values)
                else:
                    X_scaled = self.scaler.transform(X)
                log(f"数据标准化后形状: {X_scaled.shape}")
                
                # 使用模型预测
                log("使用模型预测标准化后的数据")
                return self.model.predict(X_scaled)
            else:
                # 直接使用模型预测
                log("直接使用模型预测原始数据")
                return self.model.predict(X)
                
        except Exception as e:
            log(f"预测时出错: {str(e)}")
            log(traceback.format_exc())
            return None

class GBDTPredictor:
    """GBDT模型预测器 - 支持多模型切换"""
    
    def __init__(self, target_model="Char Yield"):
        self.target_name = target_model  # 设置目标变量名称
        self.model_loader = None
        self.training_ranges = {}
        self.model_loaded = False
        self.feature_names = [
            'M(wt%)', 'Ash(wt%)', 'VM(wt%)', 'FC(wt%)', 
            'C(wt%)', 'H(wt%)', 'N(wt%)', 'O(wt%)', 
            'PS(mm)', 'SM(g)', 'FT(°C)', 'HR(°C/min)', 
            'FR(mL/min)', 'RT(min)'
        ]
        
        # 加载模型
        self.load_model()
    
    def find_model_file(self):
        """查找模型文件"""
        model_name = self.target_name.replace(' ', '-').lower()
        possible_names = [
            f"GBDT-{model_name}-improved.joblib",
            f"GBDT-{model_name}.joblib",
            f"GBDT_{model_name}.joblib",
            f"gbdt_{model_name}.joblib",
            f"gbdt-{model_name}.joblib"
        ]
        
        # 获取当前目录
        try:
            current_dir = os.getcwd()
            log(f"当前工作目录: {current_dir}")
        except Exception as e:
            log(f"获取当前目录时出错: {str(e)}")
            current_dir = "."
        
        # 可能的目录列表
        possible_dirs = [
            current_dir,
            os.path.join(current_dir, "models"),
            os.path.dirname(current_dir),
            "/mount/src/app",
            "/app",
            "."
        ]
        
        # 搜索所有可能的文件名和路径
        for directory in possible_dirs:
            if not os.path.exists(directory):
                continue
                
            log(f"搜索目录: {directory}")
            
            # 尝试具体的文件名
            for name in possible_names:
                file_path = os.path.join(directory, name)
                if os.path.exists(file_path):
                    log(f"找到模型文件: {file_path}")
                    return file_path
            
            # 列出目录中的所有.joblib文件
            try:
                joblib_files = [f for f in os.listdir(directory) if f.endswith('.joblib')]
                if joblib_files:
                    log(f"目录中的.joblib文件: {', '.join(joblib_files)}")
                    
                    # 尝试找到匹配当前模型的文件
                    model_type = self.target_name.split(' ')[0].lower()
                    for file in joblib_files:
                        if model_type in file.lower() and 'scaler' not in file.lower():
                            file_path = os.path.join(directory, file)
                            log(f"找到可能的模型文件: {file_path}")
                            return file_path
            except Exception as e:
                log(f"列出目录内容时出错: {str(e)}")
        
        log("未找到匹配的模型文件")
        return None
    
    def load_model(self):
        """加载模型"""
        model_path = self.find_model_file()
        
        if model_path:
            self.model_loader = ModelLoader(model_path)
            self.model_loaded = self.model_loader.loaded
        else:
            log(f"错误: 未找到{self.target_name}模型文件")
            self.model_loaded = False
        
        # 设置训练数据范围
        self.set_training_ranges()
    
    def set_training_ranges(self):
        """设置训练数据的范围"""
        self.training_ranges = {
            'M(wt%)': {'min': 2.750, 'max': 12.640},
            'Ash(wt%)': {'min': 0.780, 'max': 29.510},
            'VM(wt%)': {'min': 51.640, 'max': 89.500},
            'FC(wt%)': {'min': 0.100, 'max': 23.900},
            'C(wt%)': {'min': 22.490, 'max': 53.300},
            'H(wt%)': {'min': 3.303, 'max': 8.200},
            'N(wt%)': {'min': 0.170, 'max': 4.870},
            'O(wt%)': {'min': 34.000, 'max': 73.697},
            'PS(mm)': {'min': 0.075, 'max': 10.000},
            'SM(g)': {'min': 3.000, 'max': 125.000},
            'FT(°C)': {'min': 250.000, 'max': 900.000},
            'HR(°C/min)': {'min': 1.000, 'max': 100.000},
            'FR(mL/min)': {'min': 0.000, 'max': 600.000},
            'RT(min)': {'min': 15.000, 'max': 90.000}
        }
        
        log(f"已设置训练数据范围，共 {len(self.training_ranges)} 个特征")
    
    def check_input_range(self, input_df):
        """检查输入值是否在训练数据范围内"""
        warnings = []
        
        for feature, range_info in self.training_ranges.items():
            if feature in input_df.columns:
                value = input_df[feature].iloc[0]
                if value < range_info['min'] or value > range_info['max']:
                    warning = f"{feature}: {value:.2f} (超出训练范围 {range_info['min']:.2f} - {range_info['max']:.2f})"
                    warnings.append(warning)
                    log(f"警告: {warning}")
        
        return warnings
    
    def predict(self, input_features):
        """使用模型进行预测"""
        if not self.model_loaded:
            log(f"错误: {self.target_name}模型未加载")
            return np.array([0.0])
        
        # 确保输入特征包含所有必要特征
        missing_features = [f for f in self.feature_names if f not in input_features.columns]
        if missing_features:
            log(f"错误: 输入缺少特征: {', '.join(missing_features)}")
            return np.array([0.0])
        
        # 按照模型训练时的特征顺序重新排列
        input_ordered = input_features[self.feature_names].copy()
        log(f"创建输入数据框: {input_ordered.shape}")
        
        # 执行预测
        try:
            result = self.model_loader.predict(input_ordered)
            
            if result is not None:
                pred_value = float(result[0]) if isinstance(result, (np.ndarray, list)) else float(result)
                log(f"{self.target_name}预测结果: {pred_value:.2f}")
                return np.array([pred_value])
            else:
                log("预测返回空结果")
                return np.array([0.0])
                
        except Exception as e:
            log(f"预测过程中出错: {str(e)}")
            log(traceback.format_exc())
            return np.array([0.0])
    
    def get_model_info(self):
        """获取模型信息摘要"""
        info = {
            "模型类型": "GBDT模型",
            "目标变量": self.target_name,
            "特征数量": len(self.feature_names),
            "模型加载状态": "成功" if self.model_loaded else "失败"
        }
        
        if self.model_loaded:
            info["标准化器状态"] = "已找到" if self.model_loader.scaler else "未找到"
            info["模型类型"] = type(self.model_loader.model).__name__
            
            if self.model_loader.is_pipeline:
                info["模型结构"] = "Pipeline"
            else:
                info["模型结构"] = "独立模型"
        
        return info

# 初始化预测器 - 使用当前选择的模型
predictor = GBDTPredictor(target_model=st.session_state.selected_model)

# 如果模型加载失败，显示上传模型提示
if not predictor.model_loaded:
    st.error(f"错误: 未找到{st.session_state.selected_model}模型文件。请检查应用安装或联系管理员。")

# 在侧边栏添加模型信息
model_info = predictor.get_model_info()
model_info_html = "<div class='sidebar-model-info'><h3>关于模型</h3>"
for key, value in model_info.items():
    model_info_html += f"<p><b>{key}</b>: {value}</p>"

model_info_html += "</div>"
st.sidebar.markdown(model_info_html, unsafe_allow_html=True)

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
    # 初始化存储所有特征输入值的字典
    st.session_state.feature_values = {}

# 定义默认值 - 从图表中提取均值作为默认值
default_values = {
    "M(wt%)": 6.57,
    "Ash(wt%)": 5.87,
    "VM(wt%)": 74.22,
    "FC(wt%)": 13.32,
    "C(wt%)": 45.12,
    "H(wt%)": 5.95,
    "N(wt%)": 1.50,
    "O(wt%)": 47.40,
    "PS(mm)": 1.23,
    "SM(g)": 27.03,
    "FT(°C)": 505.24,
    "HR(°C/min)": 27.81,
    "FR(mL/min)": 87.42,
    "RT(min)": 36.88
}

# 特征分类
feature_categories = {
    "Proximate Analysis": ["M(wt%)", "Ash(wt%)", "VM(wt%)", "FC(wt%)"],
    "Ultimate Analysis": ["C(wt%)", "H(wt%)", "N(wt%)", "O(wt%)"],
    "Pyrolysis Conditions": ["PS(mm)", "SM(g)", "FT(°C)", "HR(°C/min)", "FR(mL/min)", "RT(min)"]
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
            # 先从会话状态获取值，如果不存在则使用默认值
            value = st.session_state.feature_values.get(feature, default_values[feature])
        
        col_a, col_b = st.columns([1, 0.5])
        with col_a:
            st.markdown(f"<div class='input-label' style='background-color: {color};'>{feature}</div>", unsafe_allow_html=True)
        with col_b:
            # 设置范围根据训练数据
            min_val = predictor.training_ranges[feature]['min']
            max_val = predictor.training_ranges[feature]['max']
            
            # 确保每个输入控件有唯一键名
            features[feature] = st.number_input(
                "", 
                min_value=float(min_val), 
                max_value=float(max_val), 
                value=float(value), 
                step=0.01,
                key=f"{category}_{feature}",
                format="%.2f",
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
            min_val = predictor.training_ranges[feature]['min']
            max_val = predictor.training_ranges[feature]['max']
            
            features[feature] = st.number_input(
                "", 
                min_value=float(min_val), 
                max_value=float(max_val), 
                value=float(value), 
                step=0.01,
                key=f"{category}_{feature}",
                format="%.2f",
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
        
        min_val = predictor.training_ranges[feature]['min']
        max_val = predictor.training_ranges[feature]['max']
        
        col_a, col_b = st.columns([1, 0.5])
        with col_a:
            st.markdown(f"<div class='input-label' style='background-color: {color};'>{feature}</div>", unsafe_allow_html=True)
        with col_b:
            features[feature] = st.number_input(
                "", 
                min_value=float(min_val), 
                max_value=float(max_val), 
                value=float(value), 
                step=0.01,
                key=f"{category}_{feature}",
                format="%.2f",
                label_visibility="collapsed"
            )

# 调试信息：显示所有当前输入值
debug_info = "<div style='background-color: #333; padding: 10px; border-radius: 5px; margin-top: 10px;'>"
debug_info += "<h4>当前输入值</h4><ul style='columns: 3;'>"
for feature, value in features.items():
    debug_info += f"<li>{feature}: {value:.2f}</li>"
debug_info += "</ul></div>"

# 可选的调试信息展示
with st.expander("显示当前输入值"):
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
        log("开始预测，获取最新输入值...")
        
        # 保存当前输入到会话状态
        st.session_state.feature_values = features.copy()
        
        log(f"开始{st.session_state.selected_model}预测，输入特征数: {len(features)}")
        
        # 创建输入数据框
        input_df = pd.DataFrame([features])
        
        # 检查输入范围
        warnings = predictor.check_input_range(input_df)
        st.session_state.warnings = warnings
        
        # 执行预测
        try:
            result = predictor.predict(input_df)
            if result is not None and len(result) > 0:
                st.session_state.prediction_result = float(result[0])
                log(f"预测成功: {st.session_state.prediction_result:.2f}")
            else:
                log("警告: 预测结果为空")
                st.session_state.prediction_result = 0.0
        except Exception as e:
            st.session_state.prediction_error = str(e)
            log(f"预测错误: {str(e)}")
            log(traceback.format_exc())
            st.error(f"预测过程中发生错误: {str(e)}")

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
    result_container.markdown(f"<div class='yield-result'>{st.session_state.selected_model}: {st.session_state.prediction_result:.2f} wt%</div>", unsafe_allow_html=True)
    
    # 显示警告
    if st.session_state.warnings:
        warnings_html = "<div class='warning-box'><b>⚠️ 警告：部分输入超出训练范围</b><ul>"
        for warning in st.session_state.warnings:
            warnings_html += f"<li>{warning}</li>"
        warnings_html += "</ul><p>预测结果可能不太可靠。</p></div>"
        result_container.markdown(warnings_html, unsafe_allow_html=True)
    
    # 标准化器状态
    if not predictor.model_loader or not predictor.model_loader.scaler:
        result_container.markdown(
            "<div class='warning-box'><b>⚠️ 注意：</b> 未找到标准化器，这可能影响预测精度。</div>", 
            unsafe_allow_html=True
        )
    
    # 技术说明部分 - 使用折叠式展示
    with st.expander("技术说明"):
        st.markdown("""
        <div class='tech-info'>
        <p>本模型基于GBDT（梯度提升决策树）算法创建，预测生物质热解产物分布。模型使用生物质的元素分析、近似分析数据和热解条件作为输入，计算热解炭、热解油和热解气体产量。</p>
       
        <p><b>特别提醒：</b></p>
        <ul>
            <li>输入参数应该满足设定好的范围内，因为这样符合模型训练数据的分布范围，可以保证软件的预测精度，如果超过范围，会有文字提醒</li>
            <li>由于模型训练时FC(wt%)通过100-Ash(wt%)-VM(wt%)公式转换得出，所以用户使用此软件进行预测时也需要使用100-Ash(wt%)-VM(wt%)公式对FC(wt%)进行转换，以保证预测的准确性。</li>
            <li>所有特征的输入范围都基于真实训练数据的统计信息，确保预测结果的可靠性。</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# 添加页脚
st.markdown("---")
footer = """
<div style='text-align: center;'>
<p>© 2023 生物质纳米材料与智能装备实验室. 版本: 4.0.0</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)