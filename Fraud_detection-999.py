# -*- coding: utf-8 -*-
"""
Biomass Pyrolysis Yield Forecast using GBDT Ensemble Models
修复版本 - 解决小数精度问题和子模型标准化器问题
添加多模型切换功能 - 支持Char、Oil和Gas产率预测
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

# 清除缓存，强制重新渲染
if "debug" not in st.session_state:
    st.cache_data.clear()
    st.session_state.debug = True
    st.session_state.decimal_test = 46.12  # 测试两位小数

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
log("应用启动 - 支持两位小数和多模型切换功能")

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
                           help="预测焦炭产率 (%)", 
                           use_container_width=True,
                           type="primary" if st.session_state.selected_model == "Char Yield" else "secondary")
with col2:
    oil_button = st.button(" Oil Yield", 
                          key="oil_button", 
                          help="预测生物油产率 (%)", 
                          use_container_width=True,
                          type="primary" if st.session_state.selected_model == "Oil Yield" else "secondary")
with col3:
    gas_button = st.button(" Gas Yield", 
                          key="gas_button", 
                          help="预测气体产率 (%)", 
                          use_container_width=True,
                          type="primary" if st.session_state.selected_model == "Gas Yield" else "secondary")

# 处理模型选择
if char_button:
    st.session_state.selected_model = "Char Yield"
    st.session_state.prediction_result = None
    st.session_state.warnings = []
    st.session_state.individual_predictions = []
    log(f"切换到模型: {st.session_state.selected_model}")
    st.rerun()

if oil_button:
    st.session_state.selected_model = "Oil Yield"
    st.session_state.prediction_result = None
    st.session_state.warnings = []
    st.session_state.individual_predictions = []
    log(f"切换到模型: {st.session_state.selected_model}")
    st.rerun()

if gas_button:
    st.session_state.selected_model = "Gas Yield"
    st.session_state.prediction_result = None
    st.session_state.warnings = []
    st.session_state.individual_predictions = []
    log(f"切换到模型: {st.session_state.selected_model}")
    st.rerun()

st.markdown(f"<p style='text-align:center;'>当前模型: <b>{st.session_state.selected_model}</b></p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

class GBDTPredictor:
    """GBDT模型预测器 - 支持多模型切换"""
    
    def __init__(self, target_model="Char Yield"):
        self.model = None
        self.scaler = None  # 标准化器
        self.target_name = target_model  # 设置目标变量名称
        self.metadata = {}
        self.model_dir = None
        self.feature_importance = None
        self.training_ranges = {}
        self.model_loaded = False  # 新增：标记模型加载状态
        self.feature_names = [
            'M(wt%)', 'Ash(wt%)', 'VM(wt%)', 'FC(wt%)', 
            'C(wt%)', 'H(wt%)', 'N(wt%)', 'O(wt%)', 
            'PS(mm)', 'SM(g)', 'FT(°C)', 'HR(°C/min)', 
            'FR(mL/min)', 'RT(min)'
        ]
        
        # 加载模型
        self.load_model()
    
    def find_model_files(self):
        """查找模型文件"""
        # 根据目标变量确定模型文件名
        model_name = self.target_name.replace(' ', '-')
        model_file = f"GBDT-{model_name}-improved.joblib"
        scaler_file = f"GBDT-{model_name}-scaler-improved.joblib"
        
        log(f"尝试查找模型文件: {model_file}")
        
        # 可能的路径列表
        possible_paths = [
            # 当前目录
            "./",
            # 父目录
            "../",
            # 应用根目录
            "./models/",
            "../models/",
            # 更多可能的位置
            "C:/Users/HWY/Desktop/方-3/",
            "/app/",
            "/app/models/",
            "/mount/src/",
            os.getcwd(),
            os.path.join(os.getcwd(), "models")
        ]
        
        model_path = None
        scaler_path = None
        
        # 搜索模型文件
        for path in possible_paths:
            potential_model_path = os.path.join(path, model_file)
            potential_scaler_path = os.path.join(path, scaler_file)
            
            if os.path.exists(potential_model_path):
                model_path = potential_model_path
                log(f"找到模型文件: {model_path}")
            
            if os.path.exists(potential_scaler_path):
                scaler_path = potential_scaler_path
                log(f"找到标准化器文件: {scaler_path}")
            
            if model_path and scaler_path:
                return model_path, scaler_path
        
        # 如果没有找到，尝试全局搜索
        try:
            log("在当前目录及子目录搜索模型文件...")
            model_matches = glob.glob(f"**/{model_file}", recursive=True)
            scaler_matches = glob.glob(f"**/{scaler_file}", recursive=True)
            
            if model_matches:
                model_path = model_matches[0]
                log(f"通过全局搜索找到模型文件: {model_path}")
            
            if scaler_matches:
                scaler_path = scaler_matches[0]
                log(f"通过全局搜索找到标准化器文件: {scaler_path}")
        except Exception as e:
            log(f"搜索模型文件时出错: {str(e)}")
        
        # 如果找不到文件，返回None
        if not model_path:
            log(f"严重警告: 未找到模型文件 {model_file}")
        if not scaler_path:
            log(f"严重警告: 未找到标准化器文件 {scaler_file}")
        
        return model_path, scaler_path
    
    def load_model(self):
        """加载模型和标准化器"""
        try:
            # 查找模型文件
            model_path, scaler_path = self.find_model_files()
            
            # 加载模型
            if model_path and os.path.exists(model_path):
                self.model = joblib.load(model_path)
                log(f"成功加载模型: {model_path}")
            else:
                log(f"错误: 未找到{self.target_name}模型文件")
                st.error(f"错误: 未找到{self.target_name}模型文件。请检查应用安装或联系管理员。")
                return False
            
            # 加载标准化器
            if scaler_path and os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                log(f"成功加载标准化器: {scaler_path}")
            else:
                log(f"警告: 未找到{self.target_name}标准化器文件")
                
            # 设置训练数据范围
            self.set_training_ranges()
            
            # 标记模型加载成功
            self.model_loaded = True
            return True
            
        except Exception as e:
            log(f"加载模型时出错: {str(e)}")
            log(traceback.format_exc())
            st.error(f"加载{self.target_name}模型时发生错误: {str(e)}")
            return False
    
    def set_training_ranges(self):
        """设置训练数据的范围"""
        # 根据截图中的特征统计数据设置范围
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
        
        if not self.training_ranges:
            log("警告: 没有训练数据范围信息，跳过范围检查")
            return warnings
        
        for feature, range_info in self.training_ranges.items():
            if feature in input_df.columns:
                value = input_df[feature].iloc[0]
                # 检查是否超出训练数据的真实范围
                if value < range_info['min'] or value > range_info['max']:
                    warning = f"{feature}: {value:.2f} (超出训练范围 {range_info['min']:.2f} - {range_info['max']:.2f})"
                    warnings.append(warning)
                    log(f"警告: {warning}")
        
        return warnings
    
    def predict(self, input_features):
        """使用模型和标准化器进行预测"""
        try:
            # 验证模型组件
            if not self.model_loaded or not self.model:
                log(f"错误: 没有加载{self.target_name}模型或模型加载失败")
                st.error(f"错误: {self.target_name}模型未正确加载。请检查应用安装或联系管理员。")
                return np.array([0.0])
            
            # 确保输入特征包含所有必要特征
            missing_features = []
            for feature in self.feature_names:
                if feature not in input_features.columns:
                    missing_features.append(feature)
            
            if missing_features:
                missing_str = ", ".join(missing_features)
                log(f"错误: 输入缺少以下特征: {missing_str}")
                st.error(f"输入数据缺少以下必要特征: {missing_str}")
                return np.array([0.0])
            
            # 按照模型训练时的特征顺序重新排列
            input_ordered = input_features[self.feature_names].copy()
            log(f"{self.target_name}模型: 输入特征已按照训练时的顺序排列")
            
            # 记录输入数据
            log(f"预测输入数据: {input_ordered.iloc[0].to_dict()}")
            
            # 使用标准化器（如果可用）
            if self.scaler:
                X_scaled = self.scaler.transform(input_ordered)
                log(f"已使用标准化器进行特征缩放")
            else:
                log(f"警告: 没有可用的标准化器，使用原始特征")
                X_scaled = input_ordered.values
            
            # 执行预测
            pred = self.model.predict(X_scaled)
            # 确保返回标量值
            pred_value = float(pred[0]) if isinstance(pred, (np.ndarray, list)) else float(pred)
            log(f"{self.target_name}预测结果: {pred_value:.2f}")
            
            return np.array([pred_value])
            
        except Exception as e:
            log(f"预测过程中出错: {str(e)}")
            log(traceback.format_exc())
            st.error(f"预测过程中发生错误: {str(e)}")
            return np.array([0.0])
    
    def get_model_info(self):
        """获取模型信息摘要"""
        info = {
            "模型类型": "GBDT模型",
            "目标变量": self.target_name,
            "特征数量": len(self.feature_names),
            "模型加载状态": "成功" if self.model_loaded else "失败",
            "标准化器状态": "已加载" if self.scaler else "未加载"
        }
        
        return info

# 初始化预测器 - 使用当前选择的模型
predictor = GBDTPredictor(target_model=st.session_state.selected_model)

# 在侧边栏添加模型信息
model_info = predictor.get_model_info()
model_info_html = "<div class='sidebar-model-info'><h3>关于模型</h3>"
for key, value in model_info.items():
    model_info_html += f"<p><b>{key}</b>: {value}</p>"

# 标准化器状态
model_info_html += "<h4>标准化器状态</h4>"
if predictor.scaler:
    model_info_html += f"<p style='color:green'>✅ 标准化器已正确加载</p>"
else:
    model_info_html += "<p style='color:red'>❌ 未找到标准化器，可能影响预测精度</p>"

model_info_html += "</div>"
st.sidebar.markdown(model_info_html, unsafe_allow_html=True)

# 性能指标显示区域（在预测后动态更新）
performance_container = st.sidebar.container()

# 初始化会话状态
if 'clear_pressed' not in st.session_state:
    st.session_state.clear_pressed = False
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'warnings' not in st.session_state:
    st.session_state.warnings = []
if 'individual_predictions' not in st.session_state:
    st.session_state.individual_predictions = []
if 'current_rmse' not in st.session_state:
    st.session_state.current_rmse = None
if 'current_r2' not in st.session_state:
    st.session_state.current_r2 = None
if 'prediction_error' not in st.session_state:
    st.session_state.prediction_error = None

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
            value = st.session_state.get(f"{category}_{feature}", default_values[feature])
        
        col_a, col_b = st.columns([1, 0.5])
        with col_a:
            st.markdown(f"<div class='input-label' style='background-color: {color};'>{feature}</div>", unsafe_allow_html=True)
        with col_b:
            # 设置范围根据训练数据
            min_val = predictor.training_ranges[feature]['min']
            max_val = predictor.training_ranges[feature]['max']
            
            features[feature] = st.number_input(
                "", 
                min_value=float(min_val), 
                max_value=float(max_val), 
                value=float(value), 
                step=0.01,  # 设置为0.01允许两位小数输入
                key=f"{category}_{feature}", 
                format="%.2f",  # 强制显示两位小数
                label_visibility="collapsed"
            )
            
            # 调试显示
            st.markdown(f"<span style='font-size:10px;color:gray;'>输入值: {features[feature]:.2f}</span>", unsafe_allow_html=True)

# Ultimate Analysis - 第二列
with col2:
    category = "Ultimate Analysis"
    color = category_colors[category]
    st.markdown(f"<div class='section-header' style='background-color: {color};'>{category}</div>", unsafe_allow_html=True)
    
    for feature in feature_categories[category]:
        if st.session_state.clear_pressed:
            value = default_values[feature]
        else:
            value = st.session_state.get(f"{category}_{feature}", default_values[feature])
        
        col_a, col_b = st.columns([1, 0.5])
        with col_a:
            st.markdown(f"<div class='input-label' style='background-color: {color};'>{feature}</div>", unsafe_allow_html=True)
        with col_b:
            # 设置范围根据训练数据
            min_val = predictor.training_ranges[feature]['min']
            max_val = predictor.training_ranges[feature]['max']
            
            features[feature] = st.number_input(
                "", 
                min_value=float(min_val), 
                max_value=float(max_val), 
                value=float(value), 
                step=0.01,  # 设置为0.01允许两位小数输入
                key=f"{category}_{feature}", 
                format="%.2f",  # 强制显示两位小数
                label_visibility="collapsed"
            )
            
            # 调试显示
            st.markdown(f"<span style='font-size:10px;color:gray;'>输入值: {features[feature]:.2f}</span>", unsafe_allow_html=True)

# Pyrolysis Conditions - 第三列
with col3:
    category = "Pyrolysis Conditions"
    color = category_colors[category]
    st.markdown(f"<div class='section-header' style='background-color: {color};'>{category}</div>", unsafe_allow_html=True)
    
    for feature in feature_categories[category]:
        if st.session_state.clear_pressed:
            value = default_values[feature]
        else:
            value = st.session_state.get(f"{category}_{feature}", default_values[feature])
        
        # 设置范围根据训练数据
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
                step=0.01,  # 设置为0.01允许两位小数输入
                key=f"{category}_{feature}", 
                format="%.2f",  # 强制显示两位小数
                label_visibility="collapsed"
            )
            
            # 调试显示
            st.markdown(f"<span style='font-size:10px;color:gray;'>输入值: {features[feature]:.2f}</span>", unsafe_allow_html=True)

# 重置状态
if st.session_state.clear_pressed:
    st.session_state.clear_pressed = False

# 预测结果显示区域
result_container = st.container()

# 预测按钮区域
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🔮 运行预测", use_container_width=True, type="primary"):
        log(f"开始{st.session_state.selected_model}预测")
        st.session_state.predictions_running = True
        st.session_state.prediction_error = None  # 清除之前的错误
        
        # 记录输入
        log(f"输入特征: {features}")
        
        # 创建输入数据框
        input_df = pd.DataFrame([features])
        
        # 检查输入范围
        warnings = predictor.check_input_range(input_df)
        st.session_state.warnings = warnings
        
        # 执行预测
        try:
            result = predictor.predict(input_df)
            # 确保结果不为空，修复预测值不显示的问题
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
        
        st.session_state.predictions_running = False
        st.rerun()

with col2:
    if st.button("🔄 重置输入", use_container_width=True):
        log("重置所有输入值")
        st.session_state.clear_pressed = True
        st.session_state.prediction_result = None
        st.session_state.warnings = []
        st.session_state.individual_predictions = []
        st.session_state.prediction_error = None
        st.rerun()

# 显示预测结果
if st.session_state.prediction_result is not None:
    st.markdown("---")
    
    # 显示主预测结果
    result_container.markdown(f"<div class='yield-result'>{st.session_state.selected_model}: {st.session_state.prediction_result:.2f}%</div>", unsafe_allow_html=True)
    
    # 显示警告
    if st.session_state.warnings:
        warnings_html = "<div class='warning-box'><b>⚠️ 警告：部分输入超出训练范围</b><ul>"
        for warning in st.session_state.warnings:
            warnings_html += f"<li>{warning}</li>"
        warnings_html += "</ul><p>预测结果可能不太可靠。</p></div>"
        result_container.markdown(warnings_html, unsafe_allow_html=True)
    
    # 标准化器状态
    if not predictor.scaler:
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
<p>© 2023 生物质纳米材料与智能装备实验室. 版本: 3.0.0</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)