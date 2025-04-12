# -*- coding: utf-8 -*-
"""
Biomass Pyrolysis Yield Forecast using GBDT Ensemble Models
修复版本 - 解决小数精度问题和子模型标准化器问题
添加多模型切换功能 - 支持Char、Oil和Gas产率预测
修复所有输入参数对预测结果的影响问题
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
log("修复版本 - 解决所有输入参数对预测结果的影响问题")

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
    st.session_state.individual_predictions = []
    log(f"切换到模型: {st.session_state.selected_model}")
    st.rerun()

if oil_button and st.session_state.selected_model != "Oil Yield":
    st.session_state.selected_model = "Oil Yield"
    st.session_state.prediction_result = None
    st.session_state.warnings = []
    st.session_state.individual_predictions = []
    log(f"切换到模型: {st.session_state.selected_model}")
    st.rerun()

if gas_button and st.session_state.selected_model != "Gas Yield":
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
        # 根据目标变量确定模型文件名 - 修改模型命名方式
        model_name = self.target_name.replace(' ', '-').lower()
        model_file = f"GBDT-{model_name}-improved.joblib"
        scaler_file = f"GBDT-{model_name}-scaler-improved.joblib"
        
        log(f"尝试查找模型文件: {model_file}")
        
        # 获取当前目录
        try:
            current_dir = os.getcwd()
            log(f"当前工作目录: {current_dir}")
        except Exception as e:
            log(f"获取当前目录时出错: {str(e)}")
            current_dir = "."
        
        # 直接在当前目录查找模型文件
        model_path = os.path.join(current_dir, model_file)
        scaler_path = os.path.join(current_dir, scaler_file)
        
        if os.path.exists(model_path):
            log(f"找到模型文件: {model_path}")
        else:
            log(f"当前目录未找到模型文件: {model_path}")
            model_path = None
            
        if os.path.exists(scaler_path):
            log(f"找到标准化器文件: {scaler_path}")
        else:
            log(f"当前目录未找到标准化器文件: {scaler_path}")
            scaler_path = None
            
        # 如果当前目录没找到，再尝试其他常见位置
        if not model_path or not scaler_path:
            # 替代名称 - 尝试不同命名格式
            alt_model_files = [
                f"GBDT-{model_name}-improved.joblib",
                f"GBDT-{self.target_name.replace(' ', '-')}-improved.joblib",
                f"GBDT-{self.target_name.split(' ')[0]}-improved.joblib"
            ]
            
            alt_scaler_files = [
                f"GBDT-{model_name}-scaler-improved.joblib",
                f"GBDT-{self.target_name.replace(' ', '-')}-scaler-improved.joblib",
                f"GBDT-{self.target_name.split(' ')[0]}-scaler-improved.joblib"
            ]
            
            # 可能的路径列表 - 根据常见部署位置添加
            possible_dirs = [
                ".",
                "./models",
                "../models",
                os.path.join(current_dir, "models"),
                os.path.dirname(current_dir)
            ]
            
            # 搜索模型和标准化器
            for directory in possible_dirs:
                for m_file in alt_model_files:
                    potential_path = os.path.join(directory, m_file)
                    if os.path.exists(potential_path):
                        model_path = potential_path
                        log(f"在目录 {directory} 中找到模型文件: {model_path}")
                        break
                
                for s_file in alt_scaler_files:
                    potential_path = os.path.join(directory, s_file)
                    if os.path.exists(potential_path):
                        scaler_path = potential_path
                        log(f"在目录 {directory} 中找到标准化器文件: {scaler_path}")
                        break
                
                if model_path and scaler_path:
                    break
        
        # 如果仍未找到，尝试查找模型文件的不区分大小写版本
        if not model_path or not scaler_path:
            log("使用不区分大小写方式搜索模型文件...")
            try:
                for directory in possible_dirs:
                    if os.path.exists(directory):
                        files = os.listdir(directory)
                        for file in files:
                            if file.lower().startswith("gbdt") and file.lower().endswith(".joblib"):
                                # 检查是否匹配目标模型类型
                                model_type = self.target_name.split(' ')[0].lower()
                                if model_type in file.lower():
                                    if "scaler" in file.lower() and not scaler_path:
                                        scaler_path = os.path.join(directory, file)
                                        log(f"通过不区分大小写搜索找到标准化器文件: {scaler_path}")
                                    elif "scaler" not in file.lower() and not model_path:
                                        model_path = os.path.join(directory, file)
                                        log(f"通过不区分大小写搜索找到模型文件: {model_path}")
            except Exception as e:
                log(f"在搜索模型文件时发生错误: {str(e)}")
        
        # 最后一次尝试: 搜索所有.joblib文件
        if not model_path:
            try:
                joblib_files = []
                for directory in possible_dirs:
                    if os.path.exists(directory):
                        for file in glob.glob(os.path.join(directory, "*.joblib")):
                            joblib_files.append(file)
                
                if joblib_files:
                    log(f"找到以下.joblib文件: {', '.join(joblib_files)}")
            except Exception as e:
                log(f"列出joblib文件时出错: {str(e)}")
        
        # 检查结果并返回
        if not model_path:
            log(f"错误: 未找到{self.target_name}模型文件，请确保模型文件与应用程序在同一目录")
        
        if not scaler_path:
            log(f"警告: 未找到{self.target_name}标准化器文件，将使用未标准化数据进行预测")
        
        return model_path, scaler_path
    
    def load_model(self):
        """加载模型和标准化器"""
        try:
            # 查找模型文件
            model_path, scaler_path = self.find_model_files()
            
            # 加载模型
            if model_path and os.path.exists(model_path):
                try:
                    loaded_model = joblib.load(model_path)
                    # 检查是否为Pipeline，如果是则获取模型部分
                    if hasattr(loaded_model, 'named_steps') and 'model' in loaded_model.named_steps:
                        self.model = loaded_model.named_steps['model']
                        log(f"从Pipeline加载模型组件: {model_path}")
                    else:
                        self.model = loaded_model
                        log(f"直接加载模型: {model_path}")
                    
                    log(f"成功加载模型: {model_path}")
                except Exception as e:
                    log(f"加载模型失败: {str(e)}")
                    return False
            else:
                st.error(f"错误: 未找到{self.target_name}模型文件。请检查应用安装或联系管理员。")
                return False
            
            # 加载标准化器
            if scaler_path and os.path.exists(scaler_path):
                try:
                    loaded_scaler = joblib.load(scaler_path)
                    # 检查是否直接是标准化器或在Pipeline中
                    if hasattr(loaded_scaler, 'transform'):
                        self.scaler = loaded_scaler
                    elif hasattr(loaded_scaler, 'named_steps') and 'scaler' in loaded_scaler.named_steps:
                        self.scaler = loaded_scaler.named_steps['scaler']
                    log(f"成功加载标准化器: {scaler_path}")
                except Exception as e:
                    log(f"加载标准化器失败: {str(e)}，将使用未标准化数据进行预测")
                    self.scaler = None
            else:
                log(f"未找到{self.target_name}标准化器文件，将使用未标准化数据进行预测")
                
            # 设置训练数据范围
            self.set_training_ranges()
            
            # 标记模型加载成功
            self.model_loaded = True if self.model is not None else False
            log(f"模型加载状态: {'成功' if self.model_loaded else '失败'}")
            return self.model_loaded
            
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
                try:
                    X_scaled = self.scaler.transform(input_ordered)
                    log(f"已使用标准化器进行特征缩放")
                except Exception as e:
                    log(f"标准化器转换失败: {str(e)}，使用原始特征")
                    X_scaled = input_ordered.values
            else:
                log(f"警告: 没有可用的标准化器，使用原始特征")
                X_scaled = input_ordered.values
            
            # 执行预测
            try:
                pred = self.model.predict(X_scaled)
                # 确保返回标量值
                pred_value = float(pred[0]) if isinstance(pred, (np.ndarray, list)) else float(pred)
                log(f"{self.target_name}预测结果: {pred_value:.2f}")
                
                return np.array([pred_value])
            except Exception as e:
                log(f"模型预测失败: {str(e)}")
                st.error(f"模型预测时出错: {str(e)}")
                return np.array([0.0])
            
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

# 如果模型加载失败，显示上传模型提示
if not predictor.model_loaded:
    st.error(f"错误: 未找到{st.session_state.selected_model}模型文件。请检查应用安装或联系管理员。")
    
    st.markdown("""
    <div class='error-box'>
    <h3>模型文件缺失</h3>
    <p>未能找到模型文件。请确保以下文件存在于应用程序目录:</p>
    <ul>
    <li>GBDT-char-yield-improved.joblib (Char Yield模型)</li>
    <li>GBDT-oil-yield-improved.joblib (Oil Yield模型)</li>
    <li>GBDT-gas-yield-improved.joblib (Gas Yield模型)</li>
    </ul>
    <p>以及对应的标准化器文件。</p>
    </div>
    """, unsafe_allow_html=True)

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
if 'feature_values' not in st.session_state:
    # 初始化存储所有特征输入值的字典
    st.session_state.feature_values = {}
if 'latest_input_values' not in st.session_state:
    # 存储最新的输入值用于预测
    st.session_state.latest_input_values = {}

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
                step=0.01,  # 设置为0.01允许两位小数输入
                key=f"{category}_{feature}",  # 使用类别和特征名组合的唯一键名
                format="%.2f",  # 强制显示两位小数
                label_visibility="collapsed"
            )
            
            # 显示输入值，方便调试
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
                step=0.01,  # 设置为0.01允许两位小数输入
                key=f"{category}_{feature}",  # 使用类别和特征名组合的唯一键名
                format="%.2f",  # 强制显示两位小数
                label_visibility="collapsed"
            )
            
            # 显示输入值，方便调试
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
            # 先从会话状态获取值，如果不存在则使用默认值
            value = st.session_state.feature_values.get(feature, default_values[feature])
        
        # 设置范围根据训练数据
        min_val = predictor.training_ranges[feature]['min']
        max_val = predictor.training_ranges[feature]['max']
        
        col_a, col_b = st.columns([1, 0.5])
        with col_a:
            st.markdown(f"<div class='input-label' style='background-color: {color};'>{feature}</div>", unsafe_allow_html=True)
        with col_b:
                # 确保每个输入控件有唯一键名
                features[feature] = st.number_input(
                    "", 
                    min_value=float(min_val), 
                    max_value=float(max_val), 
                    value=float(value), 
                    step=0.01,  # 设置为0.01允许两位小数输入
                    key=f"{category}_{feature}",  # 使用类别和特征名组合的唯一键名
                    format="%.2f",  # 强制显示两位小数
                    label_visibility="collapsed"
                )
                
                # 显示输入值，方便调试
                st.markdown(f"<span style='font-size:10px;color:gray;'>输入值: {features[feature]:.2f}</span>", unsafe_allow_html=True)

# 关键修复：将所有最新输入存储到会话状态，确保每个输入都能影响预测
# 这是解决只有PS(mm)影响预测的关键修复点
for feature, value in features.items():
    # 保存所有特征的当前值到会话状态
    st.session_state.latest_input_values[feature] = value

# 重置状态
if st.session_state.clear_pressed:
    # 如果按下重置按钮，清除所有保存的特征值
    st.session_state.feature_values = {}
    st.session_state.latest_input_values = {}
    st.session_state.clear_pressed = False

# 预测结果显示区域
result_container = st.container()

# 预测按钮区域
col1, col2 = st.columns([1, 1])

with col1:
    # 预测按钮 - 修复预测逻辑，确保每次使用最新输入值
    predict_clicked = st.button("🔮 运行预测", use_container_width=True, type="primary")
    if predict_clicked:
        # 确保使用当前页面上的最新输入值
        log("开始预测，获取当前最新输入值...")
        current_features = {}
        
        # 关键修复：直接从字典中获取所有输入值
        # 这样可以确保所有参数都被正确考虑，而不仅仅是PS(mm)
        for feature, value in features.items():
            current_features[feature] = value
            log(f"获取当前输入: {feature} = {current_features[feature]}")
        
        # 保存当前输入到会话状态供下次使用
        st.session_state.feature_values = current_features.copy()
        
        log(f"开始{st.session_state.selected_model}预测")
        
        # 创建输入数据框 - 使用完整的特征字典
        input_df = pd.DataFrame([current_features])
        
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
        st.session_state.individual_predictions = []
        st.session_state.prediction_error = None
        st.rerun()

# 显示预测结果
if st.session_state.prediction_result is not None:
    st.markdown("---")
    
    # 显示主预测结果 - 修改单位从%为wt%
    result_container.markdown(f"<div class='yield-result'>{st.session_state.selected_model}: {st.session_state.prediction_result:.2f} wt%</div>", unsafe_allow_html=True)
    
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
<p>© 2023 生物质纳米材料与智能装备实验室. 版本: 3.0.1</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)