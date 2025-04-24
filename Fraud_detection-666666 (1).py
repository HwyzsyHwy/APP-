# -*- coding: utf-8 -*-
"""
Biomass Pyrolysis Yield Forecast using GBDT Ensemble Models
修复版本 - 确保Pipeline正确预测
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
from sklearn.base import BaseEstimator, RegressorMixin

# 添加与训练代码相同的偏差校正类，确保模型加载时能够识别
class BiasCorrector(BaseEstimator, RegressorMixin):
    """基于平均预测偏差的校正器"""
    def __init__(self, base_model):
        self.base_model = base_model
        self.correction_factor = 1.0  # 默认为1，不校正
        
    def fit(self, X, y):
        # 训练基础模型
        self.base_model.fit(X, y.ravel() if hasattr(y, 'ravel') else y)
        
        # 计算乘法校正因子 (真实值/预测值的平均比率)
        base_predictions = self.base_model.predict(X)
        ratios = y.ravel() / base_predictions
        # 使用中位数避免异常值影响
        self.correction_factor = np.median(ratios)
        return self
        
    def predict(self, X):
        # 先用基础模型预测，然后乘以校正因子
        predictions = self.base_model.predict(X)
        return predictions * self.correction_factor

# 清除缓存，强制重新渲染
st.cache_data.clear()

# 页面设置
st.set_page_config(
    page_title='Biomass Pyrolysis Yield Prediction',
    page_icon='🔥',
    layout='wide',
    initial_sidebar_state='expanded'
)

# 自定义样式（保持原样）
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
log("应用启动 - 修改版本")
log("已修复特征名称和列顺序问题")
log("已移除O(wt%)特征")
log("已添加偏差校正器")

# 初始化会话状态 - 添加模型选择功能
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Char Yield"  # 默认选择Char产率模型
    log(f"初始化选定模型: {st.session_state.selected_model}")

# 添加模型缓存 - 避免重复加载相同模型
if 'model_cache' not in st.session_state:
    st.session_state.model_cache = {}
    
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

class ModelPredictor:
    """优化的预测器类 - 适配修改后的训练模型"""
    
    def __init__(self, target_model="Char Yield"):
        self.target_name = target_model
        self.model_path = None  # 初始化model_path属性为None
        
        # 定义正确的特征顺序（与训练时一致）- 移除O(wt%)
        self.feature_names = [
            'M(wt%)', 'Ash(wt%)', 'VM(wt%)', 'FC(wt%)', 
            'C(wt%)', 'H(wt%)', 'N(wt%)', 
            'PS(mm)', 'SM(g)', 'FT(℃)', 'HR(℃/min)', 
            'FR(mL/min)', 'RT(min)'
        ]
        
        # 定义UI到模型的特征映射关系
        self.ui_to_model_mapping = {
            'FT(°C)': 'FT(℃)',        # UI上显示为°C，而模型使用℃
            'HR(°C/min)': 'HR(℃/min)'  # UI上显示为°C/min，而模型使用℃/min
        }
        
        # 反向映射，用于显示
        self.model_to_ui_mapping = {v: k for k, v in self.ui_to_model_mapping.items()}
        
        # 训练范围不变
        self.training_ranges = self._set_training_ranges()
        self.last_features = {}  # 存储上次的特征值
        self.last_result = None  # 存储上次的预测结果
        
        # 使用缓存加载模型，避免重复加载相同模型
        self.pipeline = self._get_cached_model()
        self.model_loaded = self.pipeline is not None
        
        if not self.model_loaded:
            log(f"从缓存未找到模型，尝试加载{self.target_name}模型")
            # 查找并加载模型
            self.model_path = self._find_model_file()
            if self.model_path:
                self._load_pipeline()
    
    def _get_cached_model(self):
        """从缓存中获取模型"""
        if self.target_name in st.session_state.model_cache:
            log(f"从缓存加载{self.target_name}模型")
            return st.session_state.model_cache[self.target_name]
        return None
        
    def _find_model_file(self):
        """查找模型文件 - 更新后的版本"""
        # 为不同产率目标设置不同的模型文件和路径
        model_folders = {
            "Char Yield": ["炭产率", "char"],
            "Oil Yield": ["油产率", "oil"],
            "Gas Yield": ["气产率", "gas"] 
        }
        
        # 获取基本名称和文件夹
        model_id = self.target_name.split(" ")[0].lower()
        folders = model_folders.get(self.target_name, ["", model_id.lower()])
        
        # 尝试常见的模型文件名和路径
        search_dirs = [".", "./models", "../models", "/app/models", "/app"]
        for folder in folders:
            search_dirs.append(f"./{folder}")
            search_dirs.append(f"../{folder}")
        
        # 在所有可能的目录中搜索模型文件
        log(f"搜索{self.target_name}模型文件...")
        
        for directory in search_dirs:
            if not os.path.exists(directory):
                continue
                
            # 检查目录中的所有.joblib文件
            try:
                for file in os.listdir(directory):
                    if file.endswith('.joblib') and model_id in file.lower():
                        if 'scaler' not in file.lower():  # 排除单独保存的标准化器
                            model_path = os.path.join(directory, file)
                            log(f"找到模型文件: {model_path}")
                            return model_path
            except Exception as e:
                log(f"搜索目录{directory}时出错: {str(e)}")
        
        log(f"未找到{self.target_name}模型文件")
        return None
    
    def _load_pipeline(self):
        """加载Pipeline模型"""
        if not self.model_path:
            log("模型路径为空，无法加载")
            return False
        
        try:
            log(f"加载Pipeline模型: {self.model_path}")
            self.pipeline = joblib.load(self.model_path)
            
            # 验证是否能进行预测
            if hasattr(self.pipeline, 'predict'):
                log(f"模型加载成功，类型: {type(self.pipeline).__name__}")
                self.model_loaded = True
                
                # 将模型保存到缓存中
                st.session_state.model_cache[self.target_name] = self.pipeline
                
                # 尝试识别Pipeline的组件
                if hasattr(self.pipeline, 'named_steps'):
                    components = list(self.pipeline.named_steps.keys())
                    log(f"Pipeline组件: {', '.join(components)}")
                    
                    # 检查模型是否包含偏差校正
                    if 'model' in self.pipeline.named_steps and hasattr(self.pipeline.named_steps['model'], 'correction_factor'):
                        log(f"检测到偏差校正，校正因子: {self.pipeline.named_steps['model'].correction_factor:.4f}")
                    
                    # 检查模型是否包含feature_names_in_属性
                    if 'scaler' in self.pipeline.named_steps:
                        scaler = self.pipeline.named_steps['scaler']
                        if hasattr(scaler, 'feature_names_in_'):
                            # 更新我们的特征名列表，确保与模型匹配
                            self.feature_names = list(scaler.feature_names_in_)
                            log(f"从scaler中获取特征名: {self.feature_names}")
                return True
            else:
                log("加载的对象没有predict方法，不能用于预测")
                self.model_loaded = False
                return False
                
        except Exception as e:
            log(f"加载模型出错: {str(e)}")
            log(traceback.format_exc())
            self.model_loaded = False
            return False
    
    def _set_training_ranges(self):
        """设置训练数据的范围 - 使用正确的特征名称"""
        ranges = {
            'M(wt%)': {'min': 2.750, 'max': 12.640},
            'Ash(wt%)': {'min': 0.780, 'max': 29.510},
            'VM(wt%)': {'min': 51.640, 'max': 89.500},
            'FC(wt%)': {'min': 0.100, 'max': 23.900},
            'C(wt%)': {'min': 22.490, 'max': 53.300},
            'H(wt%)': {'min': 3.303, 'max': 8.200},
            'N(wt%)': {'min': 0.170, 'max': 4.870},
            'PS(mm)': {'min': 0.075, 'max': 10.000},
            'SM(g)': {'min': 3.000, 'max': 125.000},
            'FR(mL/min)': {'min': 0.000, 'max': 600.000},
            'RT(min)': {'min': 15.000, 'max': 90.000}
        }
        
        # 添加映射后的特征范围
        ranges['FT(℃)'] = {'min': 250.000, 'max': 900.000}
        ranges['HR(℃/min)'] = {'min': 1.000, 'max': 100.000}
        
        # 为UI特征也添加相同的范围
        ranges['FT(°C)'] = ranges['FT(℃)']
        ranges['HR(°C/min)'] = ranges['HR(℃/min)']
        
        return ranges
    
    def check_input_range(self, features):
        """检查输入值是否在训练数据范围内"""
        warnings = []
        
        for feature, value in features.items():
            # 获取映射后的特征名
            mapped_feature = self.ui_to_model_mapping.get(feature, feature)
            range_info = self.training_ranges.get(mapped_feature)
            
            if range_info:
                if value < range_info['min'] or value > range_info['max']:
                    warning = f"{feature}: {value:.2f} (超出训练范围 {range_info['min']:.2f} - {range_info['max']:.2f})"
                    warnings.append(warning)
                    log(f"警告: {warning}")
        
        return warnings
    
    def _prepare_features(self, features):
        """准备特征，处理特征名称映射和顺序"""
        # 创建一个空的DataFrame，所有特征初始化为0
        model_features = {feature: 0.0 for feature in self.feature_names}
        
        # 首先将UI特征映射到模型特征名称
        for ui_feature, value in features.items():
            # 跳过不在特征列表中的特征
            model_feature = self.ui_to_model_mapping.get(ui_feature, ui_feature)
            if model_feature in self.feature_names:
                model_features[model_feature] = value
                if ui_feature != model_feature:
                    log(f"特征映射: '{ui_feature}' -> '{model_feature}'")
        
        # 创建DataFrame并按照正确顺序排列列
        df = pd.DataFrame([model_features])
        
        # 确保列顺序与训练时一致
        df = df[self.feature_names]
        
        log(f"准备好的特征，列顺序: {list(df.columns)}")
        return df
    
    def predict(self, features):
        """预测方法 - 确保特征名称和顺序正确"""
        # 检查输入是否有变化
        features_changed = False
        if self.last_features:
            for feature, value in features.items():
                if feature in self.last_features and abs(self.last_features[feature] - value) > 0.001:
                    features_changed = True
                    break
        else:
            # 第一次预测
            features_changed = True
        
        # 如果输入没有变化且有上次结果，直接返回上次结果
        if not features_changed and self.last_result is not None:
            log("输入未变化，使用上次的预测结果")
            return self.last_result
        
        # 保存当前特征
        self.last_features = features.copy()
        
        # 准备特征数据
        log(f"开始准备{len(features)}个特征数据")
        features_df = self._prepare_features(features)
        
        # 尝试使用Pipeline进行预测
        if self.model_loaded and self.pipeline is not None:
            try:
                log("使用Pipeline模型预测")
                # 直接使用Pipeline进行预测，包含所有预处理步骤和偏差校正
                result = float(self.pipeline.predict(features_df)[0])
                log(f"Pipeline预测结果: {result:.2f}")
                self.last_result = result
                return result
            except Exception as e:
                log(f"Pipeline预测失败: {str(e)}")
                log(traceback.format_exc())
                # 如果加载失败，则尝试重新加载模型
                if self.model_path is None:
                    self.model_path = self._find_model_file()
                
                if self.model_path and self._load_pipeline():
                    try:
                        # 再次尝试预测
                        result = float(self.pipeline.predict(features_df)[0])
                        log(f"重新加载后预测结果: {result:.2f}")
                        self.last_result = result
                        return result
                    except Exception as new_e:
                        log(f"重新加载后预测仍然失败: {str(new_e)}")
        
        # 如果到这里，说明预测失败，返回错误提示
        log("所有预测尝试都失败，请检查模型文件和特征名称")
        raise ValueError("模型预测失败。请确保模型文件存在且特征格式正确。")
    
    def get_model_info(self):
        """获取模型信息摘要"""
        info = {
            "模型类型": "GBDT集成模型",
            "目标变量": self.target_name,
            "特征数量": len(self.feature_names),
            "模型状态": "已加载" if self.model_loaded else "未加载"
        }
        
        if self.model_loaded:
            if hasattr(self.pipeline, 'named_steps'):
                pipeline_steps = list(self.pipeline.named_steps.keys())
                info["Pipeline组件"] = ", ".join(pipeline_steps)
                
                # 如果有模型组件，显示其参数
                if 'model' in self.pipeline.named_steps:
                    model = self.pipeline.named_steps['model']
                    model_type = type(model).__name__
                    info["回归器类型"] = model_type
                    
                    # 显示是否使用了偏差校正
                    if hasattr(model, 'correction_factor'):
                        info["偏差校正因子"] = f"{model.correction_factor:.4f}"
                    
                    # 基础模型的超参数
                    if hasattr(model, 'base_model'):
                        base_model = model.base_model
                        if hasattr(base_model, 'n_estimators'):
                            info["树的数量"] = base_model.n_estimators
                        if hasattr(base_model, 'max_depth'):
                            info["最大深度"] = base_model.max_depth
                    elif hasattr(model, 'n_estimators'):  # 直接是基础模型的情况
                        info["树的数量"] = model.n_estimators
                        if hasattr(model, 'max_depth'):
                            info["最大深度"] = model.max_depth
                    
        return info

# 初始化预测器 - 使用当前选择的模型
predictor = ModelPredictor(target_model=st.session_state.selected_model)

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

# 定义默认值 - 从图表中提取均值作为默认值，移除O(wt%)
default_values = {
    "M(wt%)": 6.57,
    "Ash(wt%)": 5.87,
    "VM(wt%)": 74.22,
    "FC(wt%)": 13.32,
    "C(wt%)": 45.12,
    "H(wt%)": 5.95,
    "N(wt%)": 1.50,
    # O(wt%)已移除
    "PS(mm)": 1.23,
    "SM(g)": 27.03,
    "FT(°C)": 505.24,
    "HR(°C/min)": 27.81,
    "FR(mL/min)": 87.42,
    "RT(min)": 36.88
}

# 特征分类 - 从Ultimate Analysis中移除O(wt%)
feature_categories = {
    "Proximate Analysis": ["M(wt%)", "Ash(wt%)", "VM(wt%)", "FC(wt%)"],
    "Ultimate Analysis": ["C(wt%)", "H(wt%)", "N(wt%)"],  # 移除O(wt%)
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
            # 不再限制输入范围，但仍然使用默认值
            features[feature] = st.number_input(
                "", 
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
            # 不再限制输入范围
            features[feature] = st.number_input(
                "", 
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
        
        col_a, col_b = st.columns([1, 0.5])
        with col_a:
            st.markdown(f"<div class='input-label' style='background-color: {color};'>{feature}</div>", unsafe_allow_html=True)
        with col_b:
            # 不再限制输入范围
            features[feature] = st.number_input(
                "", 
                value=float(value), 
                step=0.01,
                key=f"{category}_{feature}",
                format="%.2f",
                label_visibility="collapsed"
            )

# 调试信息：显示所有当前输入值
with st.expander("显示当前输入值", expanded=False):
    debug_info = "<ul style='columns: 3;'>"
    for feature, value in features.items():
        debug_info += f"<li>{feature}: {value:.2f}</li>"
    debug_info += "</ul>"
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
        
        # 切换模型后需要重新初始化预测器
        if predictor.target_name != st.session_state.selected_model:
            log(f"检测到模型变更，重新初始化预测器: {st.session_state.selected_model}")
            predictor = ModelPredictor(target_model=st.session_state.selected_model)
        
        # 保存当前输入到会话状态
        st.session_state.feature_values = features.copy()
        
        log(f"开始{st.session_state.selected_model}预测，输入特征数: {len(features)}")
        
        # 检查输入范围
        warnings = predictor.check_input_range(features)
        st.session_state.warnings = warnings
        
        # 计算FC(wt%)是否满足FC(wt%) = 100 - Ash(wt%) - VM(wt%)的约束
        calculated_fc = 100 - features['Ash(wt%)'] - features['VM(wt%)']
        if abs(calculated_fc - features['FC(wt%)']) > 0.5:  # 允许0.5%的误差
            st.session_state.warnings.append(
                f"FC(wt%)值 ({features['FC(wt%)']:.2f}) 与计算值 (100 - Ash - VM = {calculated_fc:.2f}) 不符，这可能影响预测准确性。"
            )
            log(f"警告: FC(wt%)值与计算值不符: {features['FC(wt%)']:.2f} vs {calculated_fc:.2f}")
        
        # 执行预测
        try:
            # 确保预测器已正确初始化
            if not predictor.model_loaded:
                log("模型未加载，尝试重新加载")
                if predictor._find_model_file() and predictor._load_pipeline():
                    log("重新加载模型成功")
                else:
                    st.error("无法加载模型。请确保模型文件存在于正确位置。")
                    st.session_state.prediction_error = "模型加载失败"
                    st.rerun()
            
            result = predictor.predict(features)
            if result is not None:
                st.session_state.prediction_result = float(result)
                log(f"预测成功: {st.session_state.prediction_result:.2f}")
                st.session_state.prediction_error = None
            else:
                log("警告: 预测结果为空")
                st.session_state.prediction_error = "预测结果为空"
        except Exception as e:
            st.session_state.prediction_error = str(e)
            log(f"预测错误: {str(e)}")
            log(traceback.format_exc())
            st.error(f"预测失败: {str(e)}")
        
        # 添加重新运行以更新UI
        st.rerun()

with col2:
    clear_clicked = st.button("🔄 重置输入", use_container_width=True)
    if clear_clicked:
        log("重置所有输入值")
        st.session_state.clear_pressed = True
        st.session_state.prediction_result = None
        st.session_state.warnings = []
        st.session_state.prediction_error = None
        st.rerun()

# 显示警告
if st.session_state.warnings:
    warning_html = """
    <div class="warning-box">
        <h4 style="color: orange; margin-top: 0;">⚠️ 警告</h4>
        <p>以下输入值超出训练范围或可能存在逻辑错误，可能影响预测准确性:</p>
        <ul>
    """
    for warning in st.session_state.warnings:
        warning_html += f"<li>{warning}</li>"
    warning_html += """
        </ul>
    </div>
    """
    st.markdown(warning_html, unsafe_allow_html=True)

# 显示预测结果
if st.session_state.prediction_result is not None:
    # 格式化结果以及单位
    target_name = st.session_state.selected_model.split(" ")[0]
    target_unit = "wt%"
    formatted_result = f"{st.session_state.prediction_result:.2f}"
    
    # 使用expandable section
    with st.expander("📊 预测信息", expanded=True):
        # 基本结果显示
        st.markdown(f"""
        <div style="text-align: center; margin: 10px 0;">
            <h3>预测目标: {target_name} Yield ({target_unit})</h3>
            <div class="yield-result">{formatted_result} {target_unit}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # 显示详细预测信息
        info_cols = st.columns(2)
        with info_cols[0]:
            st.markdown("<h4>模型详情</h4>", unsafe_allow_html=True)
            if predictor.model_path:
                st.write(f"模型文件: {os.path.basename(predictor.model_path)}")
            else:
                st.write("模型文件: 从缓存加载")
            
            # 获取并显示模型类型和版本
            model_info = predictor.get_model_info()
            st.write(f"模型类型: {model_info.get('模型类型', 'GBDT')}")
            if "偏差校正因子" in model_info:
                st.write(f"偏差校正: 是 (因子 = {model_info['偏差校正因子']})")
            else:
                st.write("偏差校正: 否")
        
        with info_cols[1]:
            st.markdown("<h4>提示</h4>", unsafe_allow_html=True)
            st.write("• 结果单位为重量百分比 (wt%)")
            if st.session_state.warnings:
                st.write("• ⚠️ 存在可能影响准确性的警告")
            else:
                st.write("• ✅ 所有输入值在模型训练范围内")
            st.write("• 结果不考虑实验效率和损失")

# 在预测失败时显示错误信息
if st.session_state.prediction_error:
    error_html = """
    <div class="error-box">
        <h4 style="color: red; margin-top: 0;">❌ 预测失败</h4>
        <p><b>错误信息:</b> {}</p>
        <p>请检查:</p>
        <ul>
            <li>模型文件是否存在于正确位置</li>
            <li>输入数据是否合理</li>
            <li>FC(wt%) + Ash(wt%) + VM(wt%) 是否约等于 100%</li>
        </ul>
    </div>
    """.format(st.session_state.prediction_error)
    st.markdown(error_html, unsafe_allow_html=True)

# 技术说明部分
with st.expander("📘 技术说明", expanded=False):
    st.markdown("""
    <div class="tech-info">
        <h3>模型说明</h3>
        <p>本模型基于梯度提升决策树 (GBDT) 算法构建，用于生物质热解产物产率预测。</p>
        
        <h4>输入要求</h4>
        <ul>
            <li><b>近似分析 (Proximate Analysis):</b> 水分、灰分、挥发分和固定碳含量 (wt%)</li>
            <li><b>元素分析 (Ultimate Analysis):</b> 碳、氢、氮元素含量 (wt%)</li>
            <li><b>热解条件 (Pyrolysis Conditions):</b> 粒径、样品质量、最终温度、升温速率、载气流速和停留时间</li>
        </ul>
        
        <h4>重要提示</h4>
        <ul>
            <li>输入值最好在模型的训练范围内，超出范围可能导致预测准确性下降</li>
            <li>注意固定碳 (FC)、灰分 (Ash) 和挥发分 (VM) 应满足: FC + Ash + VM ≈ 100%</li>
            <li>模型预测值为理论产率，实际生产中需考虑工艺效率和产物收集效率</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# 添加FC自动计算器
with st.expander("🧮 FC(wt%) 计算器", expanded=False):
    st.markdown("""
    <p>固定碳含量可通过以下公式计算: FC(wt%) = 100% - Ash(wt%) - VM(wt%)</p>
    <p>使用此工具自动计算并更新FC值:</p>
    """, unsafe_allow_html=True)
    
    # 创建两列布局用于FC计算器
    fc_col1, fc_col2 = st.columns([3, 1])
    
    with fc_col1:
        # 显示当前值
        st.markdown(f"""
        <div style="margin-bottom: 10px;">
            <p><b>当前值:</b> Ash = {features['Ash(wt%)']:.2f}%, VM = {features['VM(wt%)']:.2f}%, FC = {features['FC(wt%)']:.2f}%</p>
            <p><b>计算值:</b> FC = 100% - {features['Ash(wt%)']:.2f}% - {features['VM(wt%)']:.2f}% = {100-features['Ash(wt%)']-features['VM(wt%)']:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with fc_col2:
        # 添加计算按钮
        calculate_fc = st.button("更新FC值", key="calculate_fc")
    
    if calculate_fc:
        # 计算FC值
        log("自动计算FC(wt%)值")
        new_fc = 100 - features['Ash(wt%)'] - features['VM(wt%)']
        
        # 更新会话状态中的FC值
        st.session_state.feature_values['FC(wt%)'] = new_fc
        
        # 显示更新消息
        st.success(f"已更新FC(wt%)值为: {new_fc:.2f}%")
        
        # 用于自动重新渲染页面
        st.rerun()

# 添加页脚
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 10px; border-top: 1px solid #555;">
    <p style="color: #888; font-size: 14px;">
        © 2023-2024 Biomass Pyrolysis Product Yield Prediction System 
        <br>版本 2.0.1-修正版 (April 2024)
    </p>
</div>
""", unsafe_allow_html=True)
