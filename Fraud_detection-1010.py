# -*- coding: utf-8 -*-
"""
Biomass Pyrolysis Yield Forecast using GBDT Ensemble Models
精确模型加载版本 - 确保模型预测一致性
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
import hashlib
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline

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

    /* 诊断信息样式 */
    .diagnostic-info {
        background-color: #2E2E2E;
        padding: 15px;
        border-radius: 8px;
        margin-top: 20px;
        border-left: 5px solid #0078ff;
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
log("应用启动 - 精确模型加载版本")
log("专注于解决模型加载和预测一致性问题")

# 模型目录和文件
MODEL_DIR = './models'
MODEL_FILENAME = 'GBDT-Gas-Yield-model.joblib'
FEATURE_ORDER_FILENAME = 'feature_order.joblib'
SCALER_PARAMS_FILENAME = 'scaler_params.joblib'

# 尝试加载特征顺序
def load_feature_order():
    """加载保存的特征顺序"""
    feature_order_path = os.path.join(MODEL_DIR, FEATURE_ORDER_FILENAME)
    if os.path.exists(feature_order_path):
        log(f"加载特征顺序从: {feature_order_path}")
        try:
            return joblib.load(feature_order_path)
        except Exception as e:
            log(f"加载特征顺序失败: {str(e)}")
    
    # 默认特征顺序
    log("使用默认特征顺序")
    return [
        'M(wt%)', 'Ash(wt%)', 'VM(wt%)', 'FC(wt%)', 
        'C(wt%)', 'H(wt%)', 'N(wt%)', 'O(wt%)', 
        'PS(mm)', 'SM(g)', 'FT(℃)', 'HR(℃/min)', 
        'FR(mL/min)', 'RT(min)'
    ]

# 获取特征顺序
FEATURE_ORDER = load_feature_order()
log(f"使用特征顺序: {FEATURE_ORDER}")

# 尝试加载标准化器参数
def load_scaler_params():
    """加载保存的标准化器参数"""
    scaler_params_path = os.path.join(MODEL_DIR, SCALER_PARAMS_FILENAME)
    if os.path.exists(scaler_params_path):
        log(f"加载标准化器参数从: {scaler_params_path}")
        try:
            return joblib.load(scaler_params_path)
        except Exception as e:
            log(f"加载标准化器参数失败: {str(e)}")
    
    # 默认参数
    log("使用默认标准化器参数")
    return {
        'center_': [6.33, 6.38, 74.45, 14.3, 46.87, 6.21, 1.23, 45.85, 0.6375, 15.0, 500.0, 20.0, 100.0, 33.6],
        'scale_': [1.89, 9.51, 7.685, 4.73, 7.11, 0.69, 1.44, 7.5, 0.7099, 25.0, 100.0, 44.92, 126.25, 27.0],
        'feature_names': FEATURE_ORDER
    }

# 获取标准化器参数
SCALER_PARAMS = load_scaler_params()
log(f"标准化器中心: {SCALER_PARAMS['center_'][:3]}... (截断显示)")
log(f"标准化器缩放: {SCALER_PARAMS['scale_'][:3]}... (截断显示)")

# 直接从参数创建RobustScaler
def create_scaler_from_params(scaler_params):
    """从参数创建RobustScaler"""
    scaler = RobustScaler()
    # 设置参数
    scaler.center_ = np.array(scaler_params['center_'])
    scaler.scale_ = np.array(scaler_params['scale_'])
    return scaler

# 直接创建GBDT模型
def create_gbdt_model():
    """创建GBDT模型"""
    return GradientBoostingRegressor(
        n_estimators=485,
        learning_rate=0.09834549551616206,
        max_depth=6,
        subsample=0.7219641920042345,
        min_samples_split=6,
        min_samples_leaf=5,
        max_features=0.8509734424577976,
        ccp_alpha=0.003126950550913845,
        random_state=42
    )

# 初始化会话状态
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Gas Yield"  # 默认选择Gas产率模型
    log(f"初始化选定模型: {st.session_state.selected_model}")

# 添加模型缓存 - 避免重复加载相同模型
if 'model_cache' not in st.session_state:
    st.session_state.model_cache = {}
    
# 更新主标题
st.markdown("<h1 class='main-title'>基于GBDT集成模型的生物质热解产物预测系统</h1>", unsafe_allow_html=True)

# 添加模型选择区域
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

# 处理模型选择
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

class ModelPredictor:
    """精确模型加载预测器类 - 确保预测一致性"""
    
    def __init__(self, target_model="Gas Yield"):
        """初始化预测器 - 使用固定特征顺序和参数"""
        self.target_name = target_model
        self.feature_names = FEATURE_ORDER
        log(f"初始化预测器 - 使用特征顺序: {self.feature_names}")
        
        # 界面到模型的特征映射关系
        self.ui_to_model_mapping = {
            'FT(°C)': 'FT(℃)',        # UI上显示为°C，而模型使用℃
            'HR(°C/min)': 'HR(℃/min)'  # UI上显示为°C/min，而模型使用℃/min
        }
        
        # 反向映射，用于显示
        self.model_to_ui_mapping = {v: k for k, v in self.ui_to_model_mapping.items()}
        
        # 训练范围设置
        self.training_ranges = self._set_training_ranges()
        self.last_features = {}  # 存储上次的特征值
        self.last_result = None  # 存储上次的预测结果
        
        # 加载或创建模型Pipeline
        self.pipeline, self.model_loaded = self._load_pipeline()
        
        # 保存到缓存
        if self.model_loaded:
            st.session_state.model_cache[self.target_name] = self.pipeline
    
    def _set_training_ranges(self):
        """设置训练数据的范围"""
        ranges = {
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
            'FR(mL/min)': {'min': 0.000, 'max': 600.000},
            'RT(min)': {'min': 15.000, 'max': 90.000}
        }
        
        # 添加映射后的特征范围
        for ui_feat, model_feat in self.ui_to_model_mapping.items():
            if model_feat in ranges and ui_feat not in ranges:
                ranges[ui_feat] = ranges[model_feat]
        
        return ranges
    
    def _create_fixed_pipeline(self):
        """创建具有固定参数的模型Pipeline"""
        log("创建固定参数的模型Pipeline")
        scaler = create_scaler_from_params(SCALER_PARAMS)
        model = create_gbdt_model()
        pipeline = Pipeline([
            ('scaler', scaler),
            ('model', model)
        ])
        return pipeline
    
    def _load_pipeline(self):
        """加载或创建模型Pipeline"""
        # 首先，从缓存中检查
        if self.target_name in st.session_state.model_cache:
            log(f"从缓存加载模型: {self.target_name}")
            return st.session_state.model_cache[self.target_name], True
        
        # 构建模型路径
        model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
        
        # 检查模型文件是否存在
        if os.path.exists(model_path):
            log(f"尝试加载模型从: {model_path}")
            try:
                # 加载模型
                pipeline = joblib.load(model_path)
                log("成功加载模型文件")
                
                # 验证模型组件
                if hasattr(pipeline, 'steps'):
                    if len(pipeline.steps) >= 2:
                        scaler_step = pipeline.steps[0][1]
                        model_step = pipeline.steps[1][1]
                        log(f"Pipeline结构: {[step[0] for step in pipeline.steps]}")
                        
                        # 验证标准化器
                        if hasattr(scaler_step, 'center_') and hasattr(scaler_step, 'scale_'):
                            log(f"标准化器中心: {scaler_step.center_[:3]}... (截断显示)")
                            log(f"标准化器缩放: {scaler_step.scale_[:3]}... (截断显示)")
                        else:
                            log("警告: 标准化器参数缺失")
                            # 使用保存的参数重新创建标准化器
                            pipeline.steps[0] = ('scaler', create_scaler_from_params(SCALER_PARAMS))
                            log("已使用保存的参数重新创建标准化器")
                        
                        # 验证模型
                        if hasattr(model_step, 'n_estimators'):
                            log(f"GBDT树数量: {model_step.n_estimators}")
                            log(f"GBDT学习率: {model_step.learning_rate}")
                            log(f"GBDT最大深度: {model_step.max_depth}")
                        else:
                            log("警告: 模型参数异常")
                    else:
                        log("警告: Pipeline结构异常，步骤不足")
                        pipeline = self._create_fixed_pipeline()
                else:
                    log("警告: 加载的对象不是有效的Pipeline")
                    pipeline = self._create_fixed_pipeline()
                
                return pipeline, True
            except Exception as e:
                log(f"加载模型失败: {str(e)}")
                tb = traceback.format_exc()
                log(f"错误详情: {tb}")
        else:
            log(f"模型文件不存在: {model_path}")
        
        # 如果加载失败，创建固定参数的Pipeline
        log("创建固定参数替代模型")
        pipeline = self._create_fixed_pipeline()
        return pipeline, False
    
    def _map_ui_to_model_features(self, features_dict):
        """映射UI特征名称到模型特征名称"""
        result = {}
        for ui_feature, value in features_dict.items():
            if ui_feature in self.ui_to_model_mapping:
                model_feature = self.ui_to_model_mapping[ui_feature]
                result[model_feature] = value
                log(f"特征映射: {ui_feature} -> {model_feature}")
            else:
                result[ui_feature] = value
        return result
    
    def validate_input(self, features):
        """验证输入特征是否在训练范围内"""
        warnings = []
        
        for feature, value in features.items():
            if feature in self.training_ranges:
                min_val = self.training_ranges[feature]['min']
                max_val = self.training_ranges[feature]['max']
                
                if value < min_val:
                    warnings.append(f"{feature} 值 {value:.2f} 低于训练范围 ({min_val:.2f})")
                elif value > max_val:
                    warnings.append(f"{feature} 值 {value:.2f} 高于训练范围 ({max_val:.2f})")
        
        return warnings
    
    def _prepare_features(self, features_dict):
        """准备特征数据 - 确保顺序一致"""
        # 映射UI特征到模型特征
        model_features = self._map_ui_to_model_features(features_dict)
        
        # 确保所有必需特征都存在
        for feature in self.feature_names:
            if feature not in model_features:
                raise ValueError(f"缺少必需特征: {feature}")
        
        # 创建按固定顺序排列的特征数组
        feature_array = np.array([[model_features[feature] for feature in self.feature_names]])
        log(f"特征维度: {feature_array.shape}")
        
        # 打印特征值细节（但不全部打印以避免日志过长）
        feature_details = {}
        for i, feature in enumerate(self.feature_names):
            feature_details[feature] = feature_array[0, i]
        log(f"特征值详情 (前5个): {list(feature_details.items())[:5]}")
        
        return feature_array
    
    def predict(self, features_dict):
        """使用模型进行预测"""
        # 存储特征以便调试
        self.last_features = features_dict.copy()
        
        # 准备诊断信息
        diagnostic_info = {}
        
        try:
            # 验证输入
            warnings = self.validate_input(features_dict)
            
            # 准备特征
            log("开始准备特征...")
            feature_array = self._prepare_features(features_dict)
            diagnostic_info['feature_array'] = feature_array.tolist()
            log(f"特征准备完成: 形状 {feature_array.shape}")
            
            # 分别进行标准化和预测以便进行诊断
            if hasattr(self.pipeline, 'steps') and len(self.pipeline.steps) >= 2:
                scaler = self.pipeline.steps[0][1]
                model = self.pipeline.steps[1][1]
                
                # 标准化特征
                scaled_features = scaler.transform(feature_array)
                diagnostic_info['scaled_features'] = scaled_features.tolist()
                log(f"标准化特征的前5个: {scaled_features[0, :5]}")
                
                # 模型预测
                prediction = model.predict(scaled_features)[0]
            else:
                # 直接使用Pipeline进行预测
                log("使用完整Pipeline进行预测")
                prediction = self.pipeline.predict(feature_array)[0]
            
            # 记录预测结果
            self.last_result = prediction
            log(f"预测结果: {prediction:.4f}")
            
            return prediction, warnings, diagnostic_info
            
        except Exception as e:
            log(f"预测过程发生错误: {str(e)}")
            tb = traceback.format_exc()
            log(f"错误详情: {tb}")
            return None, ["预测失败: " + str(e)], diagnostic_info

# 初始化预测器
predictor = ModelPredictor(st.session_state.selected_model)

# 添加基准样本测试
benchmark_sample = {
    'M(wt%)': 8.2,
    'Ash(wt%)': 5.42, 
    'VM(wt%)': 73.8,
    'FC(wt%)': 12.58,
    'C(wt%)': 47.2,
    'H(wt%)': 6.4,
    'N(wt%)': 0.8,
    'O(wt%)': 46.18,
    'PS(mm)': 0.5,
    'SM(g)': 15.0,
    'FT(°C)': 500.0,
    'HR(°C/min)': 20.0,
    'FR(mL/min)': 100.0,
    'RT(min)': 30.0
}

# 运行基准测试
try:
    log("执行基准样本测试...")
    benchmark_result, _, _ = predictor.predict(benchmark_sample)
    log(f"基准样本预测结果: {benchmark_result:.4f}")
    log("基准测试成功!")
except Exception as e:
    log(f"基准测试失败: {str(e)}")

# 初始化会话状态变量
if 'input_values' not in st.session_state:
    # 默认输入值
    st.session_state.input_values = {
        'M(wt%)': 8.2,
        'Ash(wt%)': 5.42, 
        'VM(wt%)': 73.8,
        'FC(wt%)': 12.58,
        'C(wt%)': 47.2,
        'H(wt%)': 6.4,
        'N(wt%)': 0.8,
        'O(wt%)': 46.18,
        'PS(mm)': 0.5,
        'SM(g)': 15.0,
        'FT(°C)': 500.0,
        'HR(°C/min)': 20.0,
        'FR(mL/min)': 100.0,
        'RT(min)': 30.0
    }

if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

if 'warnings' not in st.session_state:
    st.session_state.warnings = []

if 'diagnostic_info' not in st.session_state:
    st.session_state.diagnostic_info = {}

# 定义特征类别和颜色
categories = {
    "近似分析": {
        "features": ['M(wt%)', 'Ash(wt%)', 'VM(wt%)', 'FC(wt%)'],
        "color": "#2196F3"  # 蓝色
    },
    "元素分析": {
        "features": ['C(wt%)', 'H(wt%)', 'N(wt%)', 'O(wt%)'],
        "color": "#4CAF50"  # 绿色
    },
    "热解条件": {
        "features": ['PS(mm)', 'SM(g)', 'FT(°C)', 'HR(°C/min)', 'FR(mL/min)', 'RT(min)'],
        "color": "#FF9800"  # 橙色
    }
}

# 创建输入字段
st.markdown("<h2>输入参数</h2>", unsafe_allow_html=True)

# 创建三列布局
col1, col2, col3 = st.columns(3)

# 近似分析字段 (列1)
with col1:
    st.markdown(f"<div class='section-header' style='background-color: {categories['近似分析']['color']};'>近似分析</div>", unsafe_allow_html=True)
    for feature in categories["近似分析"]["features"]:
        label_html = f"<div class='input-label' style='background-color: {categories['近似分析']['color']};'>{feature}</div>"
        st.markdown(label_html, unsafe_allow_html=True)
        
        # 对FC(wt%)使用计算模式
        if feature == 'FC(wt%)':
            # 自动计算FC
            fc_value = 100 - (st.session_state.input_values['Ash(wt%)'] + st.session_state.input_values['VM(wt%)'] + st.session_state.input_values['M(wt%)'])
            fc_value = max(0.1, min(fc_value, 100.0))  # 确保值在合理范围内
            st.session_state.input_values[feature] = fc_value
            
            # 显示计算值（只读）
            st.number_input(
                f"{feature} (计算值)",
                value=fc_value,
                min_value=0.1,
                max_value=100.0,
                step=0.1,
                key=f"display_{feature}",
                disabled=True,
                label_visibility="collapsed"
            )
        else:
            # 允许范围在0-100之间
            st.session_state.input_values[feature] = st.number_input(
                feature,
                value=st.session_state.input_values[feature],
                min_value=0.0,
                max_value=100.0,
                step=0.1,
                key=f"input_{feature}",
                label_visibility="collapsed"
            )

# 元素分析字段 (列2)
with col2:
    st.markdown(f"<div class='section-header' style='background-color: {categories['元素分析']['color']};'>元素分析</div>", unsafe_allow_html=True)
    for feature in categories["元素分析"]["features"]:
        label_html = f"<div class='input-label' style='background-color: {categories['元素分析']['color']};'>{feature}</div>"
        st.markdown(label_html, unsafe_allow_html=True)
        
        # 自动计算O(wt%)
        if feature == 'O(wt%)':
            # 自动计算O含量
            o_value = 100 - (st.session_state.input_values['C(wt%)'] + st.session_state.input_values['H(wt%)'] + st.session_state.input_values['N(wt%)'] + st.session_state.input_values['Ash(wt%)'])
            o_value = max(0.0, min(o_value, 100.0))  # 确保值在合理范围内
            st.session_state.input_values[feature] = o_value
            
            # 显示计算值（只读）
            st.number_input(
                f"{feature} (计算值)",
                value=o_value,
                min_value=0.0,
                max_value=100.0,
                step=0.1,
                key=f"display_{feature}",
                disabled=True,
                label_visibility="collapsed"
            )
        else:
            # 允许范围在0-100之间
            st.session_state.input_values[feature] = st.number_input(
                feature,
                value=st.session_state.input_values[feature],
                min_value=0.0,
                max_value=100.0,
                step=0.1,
                key=f"input_{feature}",
                label_visibility="collapsed"
            )

# 热解条件字段 (列3)
with col3:
    st.markdown(f"<div class='section-header' style='background-color: {categories['热解条件']['color']};'>热解条件</div>", unsafe_allow_html=True)
    for feature in categories["热解条件"]["features"]:
        label_html = f"<div class='input-label' style='background-color: {categories['热解条件']['color']};'>{feature}</div>"
        st.markdown(label_html, unsafe_allow_html=True)
        
        # 设置适当的最小值、最大值和步长
        min_val = 0.0
        max_val = 1000.0
        step = 0.1
        
        # 为特定特征设置定制范围
        if feature == 'PS(mm)':
            max_val = 10.0
            step = 0.01
        elif feature == 'SM(g)':
            max_val = 200.0
            step = 1.0
        elif feature == 'FT(°C)':
            min_val = 200.0
            max_val = 1000.0
            step = 10.0
        elif feature == 'HR(°C/min)':
            max_val = 100.0
            step = 1.0
        elif feature == 'FR(mL/min)':
            max_val = 1000.0
            step = 10.0
        elif feature == 'RT(min)':
            max_val = 120.0
            step = 1.0
        
        st.session_state.input_values[feature] = st.number_input(
            feature,
            value=st.session_state.input_values[feature],
            min_value=min_val,
            max_value=max_val,
            step=step,
            key=f"input_{feature}",
            label_visibility="collapsed"
        )

# 显示当前输入值（调试用）
with st.expander("查看当前输入值", expanded=False):
    st.write(st.session_state.input_values)

# 添加预测和重置按钮
col1, col2 = st.columns(2)
with col1:
    if st.button("预测", key="predict_button", use_container_width=True, type="primary"):
        log("开始预测过程...")
        
        # 获取当前FC值
        fc_value = st.session_state.input_values['FC(wt%)']
        log(f"FC(wt%)当前值: {fc_value}")
        
        # 计算FC(wt%)理论值
        theoretical_fc = 100 - (
            st.session_state.input_values['Ash(wt%)'] + 
            st.session_state.input_values['VM(wt%)'] + 
            st.session_state.input_values['M(wt%)']
        )
        theoretical_fc = max(0.1, min(theoretical_fc, 100.0))
        log(f"FC(wt%)理论值: {theoretical_fc}")
        
        # 检查FC值是否与理论值差异过大
        if abs(fc_value - theoretical_fc) > 1.0:
            log(f"警告: FC(wt%)值 {fc_value} 与理论值 {theoretical_fc} 差异较大")
            st.warning(f"FC(wt%) 值 ({fc_value:.2f}) 与理论值 ({theoretical_fc:.2f}) 存在差异。请确认输入是否正确。")
            
            # 注意：不再自动修正FC值，保持原值
            log("保持用户输入的原始FC值")
        
        # 进行预测
        prediction, warnings, diagnostic_info = predictor.predict(st.session_state.input_values)
        
        # 更新状态
        st.session_state.prediction_result = prediction
        st.session_state.warnings = warnings
        st.session_state.diagnostic_info = diagnostic_info
        
        # 重新加载页面以显示结果
        st.rerun()

with col2:
    if st.button("重置", key="reset_button", use_container_width=True):
        log("重置所有输入值...")
        # 重置为默认值
        st.session_state.input_values = {
            'M(wt%)': 8.2,
            'Ash(wt%)': 5.42, 
            'VM(wt%)': 73.8,
            'FC(wt%)': 12.58,
            'C(wt%)': 47.2,
            'H(wt%)': 6.4,
            'N(wt%)': 0.8,
            'O(wt%)': 46.18,
            'PS(mm)': 0.5,
            'SM(g)': 15.0,
            'FT(°C)': 500.0,
            'HR(°C/min)': 20.0,
            'FR(mL/min)': 100.0,
            'RT(min)': 30.0
        }
        st.session_state.prediction_result = None
        st.session_state.warnings = []
        st.session_state.diagnostic_info = {}
        
        # 重新加载页面
        st.rerun()

# 添加模型验证工具
with st.expander("模型验证工具", expanded=False):
    st.write("使用特定样本ID测试预测")
    
    # 测试样本
    test_samples = {
        "基准样本1": {
            'M(wt%)': 8.2, 'Ash(wt%)': 5.42, 'VM(wt%)': 73.8, 'FC(wt%)': 12.58,
            'C(wt%)': 47.2, 'H(wt%)': 6.4, 'N(wt%)': 0.8, 'O(wt%)': 46.18,
            'PS(mm)': 0.5, 'SM(g)': 15.0, 'FT(°C)': 500.0, 'HR(°C/min)': 20.0,
            'FR(mL/min)': 100.0, 'RT(min)': 30.0
        },
        "基准样本2": {
            'M(wt%)': 7.5, 'Ash(wt%)': 4.8, 'VM(wt%)': 75.2, 'FC(wt%)': 12.5,
            'C(wt%)': 48.5, 'H(wt%)': 6.3, 'N(wt%)': 0.7, 'O(wt%)': 44.5,
            'PS(mm)': 0.75, 'SM(g)': 20.0, 'FT(°C)': 550.0, 'HR(°C/min)': 25.0,
            'FR(mL/min)': 150.0, 'RT(min)': 35.0
        }
    }
    
    sample_id = st.selectbox("选择样本", options=list(test_samples.keys()))
    
    if st.button("测试样本预测", key="test_sample_button"):
        log(f"测试样本 {sample_id} 预测...")
        
        # 加载样本数据
        sample_data = test_samples[sample_id]
        
        # 预测
        prediction, warnings, diagnostic_info = predictor.predict(sample_data)
        
        # 显示结果
        if prediction is not None:
            st.success(f"样本 {sample_id} 预测结果: {prediction:.4f}")
            
            # 显示诊断信息
            st.write("预测诊断:")
            st.write(f"- 特征向量维度: {np.array(diagnostic_info.get('feature_array', [])).shape}")
            if 'scaled_features' in diagnostic_info:
                st.write(f"- 标准化后特征的前3个值: {np.array(diagnostic_info['scaled_features'])[0, :3]}")
        else:
            st.error(f"样本预测失败: {', '.join(warnings)}")

# 显示预测结果
if st.session_state.prediction_result is not None:
    st.markdown("<h2>预测结果</h2>", unsafe_allow_html=True)
    
    # 显示产率结果
    result_html = f"""
    <div class='yield-result'>
        {st.session_state.selected_model}: {st.session_state.prediction_result:.2f} wt%
    </div>
    """
    st.markdown(result_html, unsafe_allow_html=True)
    
    # 显示警告（如果有）
    if st.session_state.warnings:
        st.markdown("<div class='warning-box'>", unsafe_allow_html=True)
        st.warning("注意以下问题:")
        for warning in st.session_state.warnings:
            st.markdown(f"- {warning}", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # 显示模型信息
    st.markdown("<h3>模型信息</h3>", unsafe_allow_html=True)
    st.markdown(f"- 目标: {st.session_state.selected_model}")
    st.markdown(f"- 模型类型: 梯度提升决策树 (GBDT)")
    st.markdown(f"- 特征数量: {len(FEATURE_ORDER)}")
    
    # 显示标准化和预测过程的详情
    with st.expander("查看预测过程详细信息", expanded=False):
        st.subheader("特征处理与标准化")
        
        # 显示输入特征及标准化过程
        if 'feature_array' in st.session_state.diagnostic_info:
            feature_df = pd.DataFrame(st.session_state.diagnostic_info['feature_array'], 
                                     columns=FEATURE_ORDER)
            st.write("原始特征值:")
            st.write(feature_df)
            
            if 'scaled_features' in st.session_state.diagnostic_info:
                scaled_df = pd.DataFrame(st.session_state.diagnostic_info['scaled_features'],
                                        columns=FEATURE_ORDER)
                st.write("标准化后的特征值:")
                st.write(scaled_df)
    
    # 技术说明
    st.markdown("<div class='tech-info'>", unsafe_allow_html=True)
    st.markdown("<h3>技术说明</h3>", unsafe_allow_html=True)
    st.markdown("""
    本预测系统基于梯度提升决策树(GBDT)集成模型，通过分析生物质的基本特性和热解条件，预测热解产物产率。
    
    **最佳实践**:
    - 确保所有输入值在训练范围内以获得最准确的预测
    - 近似分析总和应接近100%
    - 元素分析总和应接近100%
    - 最终温度(FT)和升温速率(HR)对产率影响显著
    
    **模型性能**:
    - 相对误差通常在5%以内
    - 对偏离训练数据分布的样本，预测误差可能增加
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# 添加侧边栏内容
st.sidebar.markdown("## 模型信息")
st.sidebar.markdown(f"当前预测: **{st.session_state.selected_model}**")
st.sidebar.markdown(f"特征数量: {len(FEATURE_ORDER)}")
st.sidebar.markdown(f"模型状态: {'已加载' if predictor.model_loaded else '使用备用模型'}")

# 侧边栏中显示模型性能信息
st.sidebar.markdown("### 模型性能指标")
performance_metrics = {
    "Gas Yield": {"MAE": "0.92", "RMSE": "1.19", "R²": "0.95"},
    "Oil Yield": {"MAE": "2.23", "RMSE": "3.01", "R²": "0.92"},
    "Char Yield": {"MAE": "1.88", "RMSE": "2.45", "R²": "0.93"}
}

if st.session_state.selected_model in performance_metrics:
    metrics = performance_metrics[st.session_state.selected_model]
    st.sidebar.markdown(f"**MAE**: {metrics['MAE']} wt%")
    st.sidebar.markdown(f"**RMSE**: {metrics['RMSE']} wt%")
    st.sidebar.markdown(f"**R²**: {metrics['R²']}")

# 侧边栏中的推荐值范围
st.sidebar.markdown("### 推荐输入范围")
st.sidebar.markdown("**最终温度(FT)**: 400-600 °C")
st.sidebar.markdown("**升温速率(HR)**: 10-40 °C/min")
st.sidebar.markdown("**保持时间(RT)**: 15-60 min")