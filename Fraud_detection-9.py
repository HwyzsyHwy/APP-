# -*- coding: utf-8 -*-
"""
Biomass Pyrolysis Yield Forecast using CatBoost Ensemble Models
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
        color: black !important;
    }
    
    /* 区域样式 */
    .section-header {
        color: black;
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
        color: black;
    }
    
    /* 结果显示样式 */
    .yield-result {
        background-color: #1E1E1E;
        color: black;
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
        color: black;
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
log_container.markdown("<h3>Execution Log</h3>", unsafe_allow_html=True)
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
log("App started - supports two decimal places and multi-model switching")

# 初始化会话状态 - 添加模型选择功能
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Char Yield(%)"  # 默认选择Char产率模型
    log(f"Initialized selected model: {st.session_state.selected_model}")

# 更新主标题以显示当前选定的模型
st.markdown("<h1 class='main-title'>Biomass Pyrolysis Product Prediction System Based on CatBoost Ensemble Model</h1>", unsafe_allow_html=True)

# 添加模型选择区域 - 修改为三个按钮一排
st.markdown("<div class='model-selector'>", unsafe_allow_html=True)
st.markdown("<h3>Select Prediction Target</h3>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    char_button = st.button(" Char Yield", 
                           key="char_button", 
                           help="Predict char yield (%)",
                           use_container_width=True,
                           type="primary" if st.session_state.selected_model == "Char Yield(%)" else "secondary")
with col2:
    oil_button = st.button(" Oil Yield", 
                          key="oil_button", 
                          help="Predict bio-oil yield (%)",
                          use_container_width=True,
                          type="primary" if st.session_state.selected_model == "Oil Yield(%)" else "secondary")
with col3:
    gas_button = st.button(" Gas Yield", 
                          key="gas_button", 
                          help="Predict gas yield (%)",
                          use_container_width=True,
                          type="primary" if st.session_state.selected_model == "Gas Yield(%)" else "secondary")

# 处理模型选择
if char_button:
    st.session_state.selected_model = "Char Yield(%)"
    st.session_state.prediction_result = None
    st.session_state.warnings = []
    st.session_state.individual_predictions = []
    log(f"Switched to model: {st.session_state.selected_model}")
    st.rerun()

if oil_button:
    st.session_state.selected_model = "Oil Yield(%)"
    st.session_state.prediction_result = None
    st.session_state.warnings = []
    st.session_state.individual_predictions = []
    log(f"Switched to model: {st.session_state.selected_model}")
    st.rerun()

if gas_button:
    st.session_state.selected_model = "Gas Yield(%)"
    st.session_state.prediction_result = None
    st.session_state.warnings = []
    st.session_state.individual_predictions = []
    log(f"Switched to model: {st.session_state.selected_model}")
    st.rerun()

st.markdown(f"<p style='text-align:center;'>Current Model: <b>{st.session_state.selected_model}</b></p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

class CorrectedEnsemblePredictor:
    """修复版集成模型预测器 - 解决子模型标准化器问题，支持多模型切换"""
    
    def __init__(self, target_model="Char Yield(%)"):
        self.models = []
        self.scalers = []  # 每个子模型的标准化器
        self.final_scaler = None  # 最终标准化器（备用）
        self.model_weights = None
        self.feature_names = None
        self.target_name = target_model  # 设置目标变量名称
        self.metadata = None
        self.model_dir = None
        self.feature_importance = None
        self.training_ranges = {}
        self.model_loaded = False  # 新增：标记模型加载状态
        
        # 加载模型
        self.load_model()
    
    def find_model_directory(self):
        """查找模型目录的多种方法，支持不同模型类型"""
        # 根据目标变量确定模型目录名称
        model_name = self.target_name.replace(' ', '_').replace('(', '').replace(')', '')
        log(f"Searching for model directory: {model_name}_Model")
        
        # 模型目录可能的路径 - 添加更多可能的路径以提高查找成功率
        possible_dirs = [
            # 当前目录和父目录
            f"./{model_name}_Model",
            f"../{model_name}_Model",
            # 应用根目录
            f"{model_name}_Model",
            # 更多可能的位置
            f"./models/{model_name}_Model",
            f"../models/{model_name}_Model",
            # 系统路径
            f"C:/Users/HWY/Desktop/方-3/{model_name}_Model",
            # 如果是在云服务上运行
            f"/app/{model_name}_Model",
            f"/app/models/{model_name}_Model",
            f"/mount/src/{model_name}_Model",
            # 应用当前工作目录
            os.path.join(os.getcwd(), f"{model_name}_Model"),
            # 特定路径 (从截图中看到的)
            f"/source/src/app/{model_name}_Model"
        ]
        
        # 尝试所有可能路径
        for dir_path in possible_dirs:
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                log(f"Found model directory: {dir_path}")
                return os.path.abspath(dir_path)
        
        # 如果找不到，尝试全局模糊搜索 (先搜索当前目录和子目录)
        try:
            log("Searching model files in current directory and subdirectories...")
            # 使用 ** 通配符进行递归搜索
            for pattern in [
                f"**/{model_name}_Model",
                f"**/models/{model_name}_Model",
                f"**/{model_name}_Model/**",
                f"**/models/**/{model_name}_Model"
            ]:
                matches = glob.glob(pattern, recursive=True)
                if matches:
                    for match in matches:
                        if os.path.isdir(match):
                            log(f"Found model directory via global search: {match}")
                            return os.path.abspath(match)
            
            # 如果上面的搜索失败，尝试根据模型文件反向查找目录
            model_files = glob.glob(f"**/{model_name}_Model/**/model_*.joblib", recursive=True)
            if model_files:
                model_dir = os.path.dirname(os.path.dirname(model_files[0]))
                log(f"Inferred model directory from model files: {model_dir}")
                return model_dir
        except Exception as e:
            log(f"Search error: {str(e)}")
        
        # 返回当前目录作为最后的退路，同时记录警告
        log(f"CRITICAL WARNING: Cannot find {self.target_name} model directory, using current directory. Predictions will return default values!")
        return os.getcwd()
    
    def load_feature_importance(self):
        """加载特征重要性数据"""
        try:
            # 尝试从CSV文件加载特征重要性
            importance_csv = os.path.join(self.model_dir, "feature_importance.csv")
            if os.path.exists(importance_csv):
                importance_df = pd.read_csv(importance_csv)
                self.feature_importance = importance_df
                log(f"Feature importance loaded: {len(importance_df)} features")
                return True
            
            # 如果CSV不存在，尝试从元数据中加载
            if self.metadata and 'feature_importance' in self.metadata:
                importance_data = self.metadata['feature_importance']
                self.feature_importance = pd.DataFrame(importance_data)
                log(f"Loaded feature importance from metadata")
                return True
            
            # 尝试通过加载的模型计算特征重要性
            if self.models and self.model_weights is not None and self.feature_names:
                log("Calculating feature importance from models")
                importance = np.zeros(len(self.feature_names))
                for i, model in enumerate(self.models):
                    try:
                        model_importance = model.get_feature_importance()
                        importance += model_importance * self.model_weights[i]
                    except Exception as e:
                        log(f"Error getting feature importance for model {i}: {str(e)}")
                
                self.feature_importance = pd.DataFrame({
                    'Feature': self.feature_names,
                    'Importance': importance
                }).sort_values('Importance', ascending=False)
                
                log(f"Feature importance calculated, top feature: {self.feature_importance['Feature'].iloc[0]}")
                return True
                
            log("WARNING: Cannot load or calculate feature importance")
            return False
        except Exception as e:
            log(f"Error loading feature importance: {str(e)}")
            return False
    
    def extract_training_ranges(self):
        """从元数据中提取训练数据真实范围"""
        # 修改：直接从元数据中加载真实的特征范围，而不是使用标准化器估算
        if self.metadata and 'feature_ranges' in self.metadata:
            self.training_ranges = self.metadata['feature_ranges']
            log(f"Loaded training data ranges from metadata: {len(self.training_ranges)} features")
            
            # 记录特征范围到日志
            for feature, range_info in self.training_ranges.items():
                log(f"Feature {feature} training range: {range_info['min']:.2f} - {range_info['max']:.2f}")
            
            return True
        else:
            log("WARNING: No feature range info in metadata")
            
            # 如果元数据中没有范围信息，尝试使用训练代码中的默认范围
            # 这些是从训练代码中提取的实际数据范围
            self.training_ranges = {
                'C(%)': {'min': 34.44, 'max': 64.23},
                'H(%)': {'min': 4.10, 'max': 7.30},
                'O(%)': {'min': 27.61, 'max': 59.92},
                'N(%)': {'min': 0.10, 'max': 6.90},
                'Ash(%)': {'min': 0.16, 'max': 15.14},
                'VM(%)': {'min': 71.93, 'max': 91.16},
                'FC(%)': {'min': 5.58, 'max': 23.30},
                'PT(°C)': {'min': 200.00, 'max': 900.00},
                'HR(℃/min)': {'min': 5.00, 'max': 65.00},
                'RT(min)': {'min': 10.00, 'max': 75.00}
            }
            
            log("Using default feature ranges extracted from training code")
            return False
    
    def load_model(self):
        """加载所有模型组件，包括每个子模型的标准化器"""
        try:
            # 清空之前的模型数据
            self.models = []
            self.scalers = []
            self.feature_importance = None
            self.training_ranges = {}
            self.model_loaded = False  # 重置加载状态
            
            # 1. 查找模型目录
            self.model_dir = self.find_model_directory()
            log(f"Using {self.target_name} model directory: {self.model_dir}")
            
            # 2. 加载元数据
            metadata_path = os.path.join(self.model_dir, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                
                # 获取特征名称和目标变量
                self.feature_names = self.metadata.get('feature_names', None)
                if self.metadata.get('target_name'):
                    self.target_name = self.metadata['target_name']
                
                log(f"Loaded feature list from metadata: {self.feature_names}")
                log(f"Target variable: {self.target_name}")
            else:
                log(f"WARNING: Metadata file not found: {metadata_path}")
                # 使用默认特征列表 - 必须与模型训练时完全一致
                self.feature_names = [
                    'C(%)', 'H(%)', 'O(%)', 'N(%)', 'Ash(%)', 'VM(%)', 'FC(%)', 
                    'PT(°C)', 'HR(℃/min)', 'RT(min)'
                ]
                log(f"Using default feature list: {self.feature_names}")
            
            # 3. 加载模型
            models_dir = os.path.join(self.model_dir, 'models')
            if os.path.exists(models_dir):
                model_files = sorted(glob.glob(os.path.join(models_dir, 'model_*.joblib')))
                if model_files:
                    for model_file in model_files:
                        model = joblib.load(model_file)
                        self.models.append(model)
                        log(f"Loaded model: {os.path.basename(model_file)}")
                else:
                    log(f"ERROR: No model files found in {models_dir}")
                    st.error(f"ERROR: No model files found for {self.target_name}. Please check the installation or contact the administrator.")
                    return False
            else:
                log(f"ERROR: Model directory does not exist: {models_dir}")
                st.error(f"ERROR: {self.target_name} model directory does not exist. Please check the installation or contact the administrator.")
                return False
            
            # 4. 加载每个子模型的标准化器 - 这是关键修复点
            scalers_dir = os.path.join(self.model_dir, 'scalers')
            if os.path.exists(scalers_dir):
                scaler_files = sorted(glob.glob(os.path.join(scalers_dir, 'scaler_*.joblib')))
                if scaler_files:
                    for scaler_file in scaler_files:
                        scaler = joblib.load(scaler_file)
                        self.scalers.append(scaler)
                        log(f"Loaded sub-model scaler: {os.path.basename(scaler_file)}")
                else:
                    log(f"WARNING: No sub-model scaler files found in {scalers_dir}")
            else:
                log(f"WARNING: Scaler directory not found: {scalers_dir}")
            
            # 5. 加载最终标准化器（作为备用）
            final_scaler_path = os.path.join(self.model_dir, 'final_scaler.joblib')
            if os.path.exists(final_scaler_path):
                self.final_scaler = joblib.load(final_scaler_path)
                log(f"Loaded final scaler: {final_scaler_path}")
            else:
                log(f"WARNING: Final scaler file not found: {final_scaler_path}")
            
            # 6. 加载权重
            weights_path = os.path.join(self.model_dir, 'model_weights.npy')
            if os.path.exists(weights_path):
                self.model_weights = np.load(weights_path)
                log(f"Loaded weights file: {weights_path}")
            else:
                # 如果没有权重文件，使用均等权重
                self.model_weights = np.ones(len(self.models)) / len(self.models)
                log("WARNING: Weights file not found, using equal weights")
            
            # 7. 提取训练数据范围 - 修改为使用真实范围
            self.extract_training_ranges()
            
            # 8. 加载特征重要性
            self.load_feature_importance()
            
            # 验证加载状态
            if len(self.models) > 0:
                log(f"Successfully loaded {len(self.models)} models and {len(self.scalers)} sub-model scalers")
                self.model_loaded = True  # 标记模型加载成功
            else:
                log(f"ERROR: No {self.target_name} models loaded")
                st.error(f"ERROR: No {self.target_name} models loaded. Please check the installation or contact the administrator.")
                return False
            
            # 特别标记标准化器问题
            if len(self.models) != len(self.scalers):
                log(f"WARNING: Model count ({len(self.models)}) does not match scaler count ({len(self.scalers)})")
                
            return True
            
        except Exception as e:
            log(f"Error loading model: {str(e)}")
            log(traceback.format_exc())
            st.error(f"Error loading {self.target_name} model: {str(e)}")
            return False
    
    def check_input_range(self, input_df):
        """检查输入值是否在训练数据范围内"""
        warnings = []
        
        if not self.training_ranges:
            log("WARNING: No training data range info, skipping range check")
            return warnings
        
        for feature, range_info in self.training_ranges.items():
            if feature in input_df.columns:
                value = input_df[feature].iloc[0]
                # 检查是否超出训练数据的真实范围
                if value < range_info['min'] or value > range_info['max']:
                    warning = f"{feature}: {value:.2f} (exceeds training range {range_info['min']:.2f} - {range_info['max']:.2f})"
                    warnings.append(warning)
                    log(f"WARNING: {warning}")
        
        return warnings
    
    def predict(self, input_features, return_individual=False):
        """使用每个子模型对应的标准化器进行预测"""
        try:
            # 验证模型组件
            if not self.model_loaded or not self.models or len(self.models) == 0:
                log(f"ERROR: No {self.target_name} model loaded or load failed")
                st.error(f"ERROR: {self.target_name} model not properly loaded. Please check the installation or contact the administrator.")
                if return_individual:
                    return np.array([0.0]), []
                else:
                    return np.array([0.0])
            
            # 确保输入特征包含所有必要特征
            missing_features = []
            if self.feature_names:
                for feature in self.feature_names:
                    if feature not in input_features.columns:
                        missing_features.append(feature)
            
            if missing_features:
                missing_str = ", ".join(missing_features)
                log(f"ERROR: Missing required features: {missing_str}")
                st.error(f"Input data is missing required features: {missing_str}")
                if return_individual:
                    return np.array([0.0]), []
                else:
                    return np.array([0.0])
            
            # 按照模型训练时的特征顺序重新排列
            if self.feature_names:
                input_ordered = input_features[self.feature_names].copy()
                log(f"{self.target_name} model: input features ordered to match training sequence")
            else:
                input_ordered = input_features
                log(f"WARNING: {self.target_name} model has no feature name list, using original input order")
            
            # 记录输入数据
            log(f"Prediction input data: {input_ordered.iloc[0].to_dict()}")
            
            # 使用每个子模型和对应的标准化器进行预测
            individual_predictions = []
            all_predictions = np.zeros((input_ordered.shape[0], len(self.models)))
            
            # 检查标准化器是否足够
            scalers_available = len(self.scalers) > 0
            
            for i, model in enumerate(self.models):
                try:
                    # 使用对应的标准化器（如果可用）
                    if scalers_available and i < len(self.scalers):
                        X_scaled = self.scalers[i].transform(input_ordered)
                        log(f"Model {i} using its corresponding scaler")
                    else:
                        # 如果没有对应的标准化器，使用最终标准化器
                        if self.final_scaler:
                            X_scaled = self.final_scaler.transform(input_ordered)
                            log(f"Model {i} using final scaler")
                        else:
                            # 如果没有任何标准化器可用，则使用原始特征
                            log(f"WARNING: Model {i} has no available scaler, using raw features")
                            X_scaled = input_ordered.values
                    
                    # 执行预测并确保返回的是标量值 (修复 invalid index to scalar variable 错误)
                    pred = model.predict(X_scaled)
                    # 确保预测值是标量，不是数组
                    pred_value = float(pred[0]) if isinstance(pred, (np.ndarray, list)) else float(pred)
                    all_predictions[:, i] = pred_value
                    individual_predictions.append(pred_value)
                    log(f"Model {i} prediction: {pred_value:.2f}")
                except Exception as e:
                    log(f"Model {i} prediction error: {str(e)}")
                    # 如果某个模型失败，使用其他模型的平均值
                    if i > 0:
                        avg_pred = np.mean(all_predictions[:, :i], axis=1)
                        avg_value = float(avg_pred[0]) if len(avg_pred) > 0 else 0.0
                        all_predictions[:, i] = avg_value
                        individual_predictions.append(avg_value)
                        log(f"Model {i} using average of previous models: {avg_value:.2f}")
            
            # 计算加权平均 - 修复：确保不会出现维度不匹配的问题
            if len(self.models) > 0:
                # 确保权重数组维度正确
                weights = self.model_weights
                if weights.ndim == 1:
                    weights = weights.reshape(1, -1)
                
                # 确保权重和预测维度匹配
                if weights.shape[1] != all_predictions.shape[1]:
                    log(f"WARNING: Weights shape {weights.shape} does not match predictions shape {all_predictions.shape}, using mean")
                    weighted_pred = np.mean(all_predictions, axis=1)
                else:
                    # 正确计算加权平均
                    weighted_pred = np.sum(all_predictions * weights, axis=1)
                
                log(f"{self.target_name} final weighted prediction: {weighted_pred[0]:.2f}")
            else:
                weighted_pred = np.array([0.0])
                log(f"WARNING: No models available, returning default value 0")
            
            # 计算评估指标 - 动态计算RMSE和R²
            std_dev = np.std(individual_predictions) if len(individual_predictions) > 0 else 0
            
            # 修复 - 确保有足够的数据进行计算
            if len(individual_predictions) > 1:
                # 创建一个正确的输入向量进行RMSE计算
                weighted_pred_reshaped = np.tile(weighted_pred.reshape(-1, 1), (1, all_predictions.shape[1]))
                rmse = np.sqrt(np.mean((all_predictions - weighted_pred_reshaped)**2))
                
                # 计算R² (避免除以零错误)
                total_variance = np.sum((all_predictions - np.mean(all_predictions))**2)
                explained_variance = total_variance - np.sum((all_predictions - weighted_pred_reshaped)**2)
                r2 = explained_variance / total_variance if total_variance > 0 else 0
                
                log(f"Prediction std dev: {std_dev:.4f}")
                log(f"RMSE: {float(rmse[0]) if isinstance(rmse, np.ndarray) else float(rmse):.4f}, R²: {r2:.4f}")
                
                # 存储评估指标到session_state - 确保性能指标动态更新
                st.session_state.current_rmse = float(rmse[0]) if isinstance(rmse, np.ndarray) else float(rmse)
                st.session_state.current_r2 = float(r2)
            else:
                log("WARNING: Not enough models for performance evaluation")
                # 设置默认值以避免后续显示错误
                st.session_state.current_rmse = 0.0
                st.session_state.current_r2 = 0.0
            
            if return_individual:
                return weighted_pred, individual_predictions
            else:
                return weighted_pred
            
        except Exception as e:
            log(f"Prediction error: {str(e)}")
            log(traceback.format_exc())
            st.error(f"Prediction error: {str(e)}")
            # 修复 - 返回默认值，确保类型一致
            if return_individual:
                return np.array([0.0]), []
            else:
                return np.array([0.0])
    
    def get_model_info(self):
        """获取模型信息摘要"""
        info = {
            "Model Type": "CatBoost Ensemble Model",
            "Model Count": len(self.models),
            "Feature Count": len(self.feature_names) if self.feature_names else 0,
            "Target Variable": self.target_name,
            "Model Load Status": "Success" if self.model_loaded else "Failed"
        }
        
        # 添加性能信息
        if self.metadata and 'performance' in self.metadata:
            performance = self.metadata['performance']
            info["Test R²"] = f"{performance.get('test_r2', 'N/A'):.4f}"
            info["Test RMSE"] = f"{performance.get('test_rmse', 'N/A'):.2f}"
        
        # 添加特征重要性信息
        if self.feature_importance is not None and len(self.feature_importance) > 0:
            top_features = self.feature_importance.head(3)
            info["Key Features"] = ", ".join(top_features['Feature'].tolist())
        
        # 添加标准化器信息
        info["Sub-model Scaler Count"] = len(self.scalers)
        
        return info

# 初始化预测器 - 使用当前选择的模型
predictor = CorrectedEnsemblePredictor(target_model=st.session_state.selected_model)

# 在侧边栏添加模型信息
model_info = predictor.get_model_info()
model_info_html = "<div class='sidebar-model-info'><h3>About Model</h3>"
for key, value in model_info.items():
    model_info_html += f"<p><b>{key}</b>: {value}</p>"

# 标准化器状态
model_info_html += "<h4>Scaler Status</h4>"
if len(predictor.scalers) == len(predictor.models):
    model_info_html += f"<p style='color:black'>✅ All {len(predictor.models)} sub-models are using their corresponding scalers</p>"
elif len(predictor.scalers) > 0:
    model_info_html += f"<p style='color:black'>⚠️ Found {len(predictor.scalers)}/{len(predictor.models)} sub-model scalers</p>"
else:
    model_info_html += "<p style='color:black'>❌ No sub-model scalers found, using final scaler</p>"

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

# 定义默认值 - 从用户截图中提取
default_values = {
    "C(%)": 46.00,  # 使用两位小数精度
    "H(%)": 5.50,
    "O(%)": 55.20,
    "N(%)": 0.60,
    "Ash(%)": 6.60,
    "VM(%)": 81.10,
    "FC(%)": 10.30,
    "PT(°C)": 500.00,  # 使用实际测试值
    "HR(℃/min)": 10.00,
    "RT(min)": 60.00
}

# 特征分类
feature_categories = {
    "Ultimate Analysis": ["C(%)", "H(%)", "O(%)", "N(%)"],
    "Proximate Analysis": ["Ash(%)", "VM(%)", "FC(%)"],
    "Pyrolysis Conditions": ["PT(°C)", "HR(℃/min)", "RT(min)"]
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

# Ultimate Analysis - 第一列
with col1:
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
            # 关键修改: 设置步长为0.01以支持两位小数
            features[feature] = st.number_input(
                "", 
                min_value=0.00, 
                max_value=100.00, 
                value=float(value), 
                step=0.01,  # 设置为0.01允许两位小数输入
                key=f"{category}_{feature}", 
                format="%.2f",  # 强制显示两位小数
                label_visibility="collapsed"
            )
            
            # 调试显示
            st.markdown(f"<span style='font-size:10px;color:black;'>Input value: {features[feature]:.2f}</span>", unsafe_allow_html=True)

# Proximate Analysis - 第二列
with col2:
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
            # 关键修改: 设置步长为0.01以支持两位小数
            features[feature] = st.number_input(
                "", 
                min_value=0.00, 
                max_value=100.00, 
                value=float(value), 
                step=0.01,  # 设置为0.01允许两位小数输入
                key=f"{category}_{feature}", 
                format="%.2f",  # 强制显示两位小数
                label_visibility="collapsed"
            )
            
            # 调试显示
            st.markdown(f"<span style='font-size:10px;color:black;'>Input value: {features[feature]:.2f}</span>", unsafe_allow_html=True)

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
        
        # 根据特征设置范围
        if feature == "PT(°C)":
            min_val, max_val = 200.00, 900.00
        elif feature == "HR(℃/min)":
            min_val, max_val = 1.00, 100.00
        elif feature == "RT(min)":
            min_val, max_val = 0.00, 120.00
        else:
            min_val, max_val = 0.00, 100.00
        
        col_a, col_b = st.columns([1, 0.5])
        with col_a:
            st.markdown(f"<div class='input-label' style='background-color: {color};'>{feature}</div>", unsafe_allow_html=True)
        with col_b:
            # 关键修改: 设置步长为0.01以支持两位小数
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
            st.markdown(f"<span style='font-size:10px;color:black;'>Input value: {features[feature]:.2f}</span>", unsafe_allow_html=True)

# 重置状态
if st.session_state.clear_pressed:
    st.session_state.clear_pressed = False

# 预测结果显示区域
result_container = st.container()

# 预测按钮区域
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🔮 Run Prediction", use_container_width=True, type="primary"):
        log(f"Starting {st.session_state.selected_model} prediction")
        st.session_state.predictions_running = True
        st.session_state.prediction_error = None  # 清除之前的错误

        # 记录输入
        log(f"Input features: {features}")
        
        # 创建输入数据框
        input_df = pd.DataFrame([features])
        
        # 检查输入范围
        warnings = predictor.check_input_range(input_df)
        st.session_state.warnings = warnings
        
        # 执行预测
        try:
            result, individual_preds = predictor.predict(input_df, return_individual=True)
            # 确保结果不为空，修复预测值不显示的问题
            if result is not None and len(result) > 0:
                st.session_state.prediction_result = float(result[0])
                st.session_state.individual_predictions = individual_preds
                log(f"Prediction successful: {st.session_state.prediction_result:.2f}")

                # 计算标准差作为不确定性指标
                std_dev = np.std(individual_preds) if individual_preds else 0
                log(f"Prediction std dev: {std_dev:.4f}")
            else:
                log("WARNING: Prediction result is empty")
                st.session_state.prediction_result = 0.0
                st.session_state.individual_predictions = []
            
        except Exception as e:
            st.session_state.prediction_error = str(e)
            log(f"Prediction error: {str(e)}")
            log(traceback.format_exc())
            st.error(f"Prediction error: {str(e)}")
        
        st.session_state.predictions_running = False
        st.rerun()

with col2:
    if st.button("🔄 Reset Inputs", use_container_width=True):
        log("Reset all input values")
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
        warnings_html = "<div class='warning-box'><b>⚠️ Warning: Some inputs exceed the training range</b><ul>"
        for warning in st.session_state.warnings:
            warnings_html += f"<li>{warning}</li>"
        warnings_html += "</ul><p>Prediction results may be unreliable.</p></div>"
        result_container.markdown(warnings_html, unsafe_allow_html=True)
    
    # 标准化器状态
    if len(predictor.scalers) < len(predictor.models):
        result_container.markdown(
            "<div class='warning-box'><b>⚠️ Note:</b> Some models used the final scaler instead of their corresponding sub-model scalers, which may affect prediction accuracy.</div>",
            unsafe_allow_html=True
        )
    
    # 技术说明部分 - 使用折叠式展示
    with st.expander("Technical Notes"):
        st.markdown("""
        <div class='tech-info'>
        <p>This model is built on an ensemble of multiple CatBoost models to predict biomass pyrolysis product distributions. It takes ultimate analysis, proximate analysis data, and pyrolysis conditions as inputs to calculate char yield, bio-oil yield, and gas yield.</p>

        <p><b>Important Notes:</b></p>
        <ul>
            <li>Input parameters should fall within the specified ranges, as this ensures they conform to the distribution of the model training data and guarantees prediction accuracy. A text warning will appear if any value exceeds the range.</li>
            <li>Since FC(%) was derived during model training using the formula 100 - Ash(%) - VM(%), users must also apply the same formula (100 - Ash(%) - VM(%) = FC(%)) when using this software to ensure accurate predictions.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# 添加页脚
st.markdown("---")
footer = """
<div style='text-align: center;'>
<p>© 2023 Biomass Nanomaterials & Smart Equipment Lab. Version: 2.3.0</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
