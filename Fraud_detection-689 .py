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

# 自定义样式（添加背景图片）
st.markdown(
    """
    <style>
    /* 全局字体设置和背景图片 */
    html, body, [class*="css"] {
        font-size: 16px !important;
    }

    /* 主应用背景 */
    .stApp {
        background-image: url('https://raw.githubusercontent.com/HwyzsyHwy/APP-/main/背景.png');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    /* 标题 */
    .main-title {
        text-align: center;
        font-size: 32px !important;
        font-weight: bold;
        margin-bottom: 20px;
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
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
log("应用启动 - 根据图片特征统计信息正确修复版本")
log("特征顺序：M, Ash, VM, O/C, H/C, N/C, FT, HR, FR")

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
    char_button = st.button("🔥 Char Yield", 
                           key="char_button", 
                           help="预测焦炭产率 (wt%)", 
                           use_container_width=True,
                           type="primary" if st.session_state.selected_model == "Char Yield" else "secondary")
with col2:
    oil_button = st.button("🛢️ Oil Yield", 
                          key="oil_button", 
                          help="预测生物油产率 (wt%)", 
                          use_container_width=True,
                          type="primary" if st.session_state.selected_model == "Oil Yield" else "secondary")
with col3:
    gas_button = st.button("💨 Gas Yield", 
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
    """根据图片特征统计信息正确调整的预测器类"""
    
    def __init__(self, target_model="Char Yield"):
        self.target_name = target_model
        
        # 根据图片中的特征统计信息，按照正确顺序定义特征名称
        self.feature_names = [
            'M(wt%)',           # 水分
            'Ash(wt%)',         # 灰分  
            'VM(wt%)',          # 挥发分
            'O/C',              # 氧碳比
            'H/C',              # 氢碳比
            'N/C',              # 氮碳比
            'FT(℃)',           # 热解温度
            'HR(℃/min)',       # 升温速率
            'FR(mL/min)'        # 流量
        ]
        
        # 根据图片中的统计信息设置训练范围
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
        
        # UI显示的特征映射（处理温度符号）
        self.ui_to_model_mapping = {
            'FT(°C)': 'FT(℃)',
            'HR(°C/min)': 'HR(℃/min)'
        }
        
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
        """查找模型文件"""
        # 根据训练代码的模型保存路径
        model_file_patterns = {
            "Char Yield": [
                "GBDT-Char Yield-improved.joblib",
                "GBDT-Char-improved.joblib",
                "*char*.joblib",
                "*炭产率*.joblib"
            ],
            "Oil Yield": [
                "GBDT-Oil Yield-improved.joblib", 
                "GBDT-Oil-improved.joblib",
                "*oil*.joblib",
                "*油产率*.joblib"
            ],
            "Gas Yield": [
                "GBDT-Gas Yield-improved.joblib",
                "GBDT-Gas-improved.joblib", 
                "*gas*.joblib",
                "*气产率*.joblib"
            ]
        }
        
        # 搜索目录
        search_dirs = [
            ".", "./models", "../models", "/app/models", "/app",
            "./炭产率", "./油产率", "./气产率",
            "../炭产率", "../油产率", "../气产率"
        ]
        
        patterns = model_file_patterns.get(self.target_name, [])
        log(f"搜索{self.target_name}模型文件，模式: {patterns}")
        
        for directory in search_dirs:
            if not os.path.exists(directory):
                continue
                
            try:
                for pattern in patterns:
                    # 使用glob匹配文件
                    matches = glob.glob(os.path.join(directory, pattern))
                    for match in matches:
                        if os.path.isfile(match):
                            log(f"找到模型文件: {match}")
                            return match
                            
                # 也检查目录中的所有.joblib文件
                for file in os.listdir(directory):
                    if file.endswith('.joblib'):
                        model_id = self.target_name.split(" ")[0].lower()
                        if model_id in file.lower():
                            model_path = os.path.join(directory, file)
                            log(f"找到匹配的模型文件: {model_path}")
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
            
            # 验证Pipeline结构
            if hasattr(self.pipeline, 'predict') and hasattr(self.pipeline, 'named_steps'):
                log(f"Pipeline加载成功，组件: {list(self.pipeline.named_steps.keys())}")
                
                # 验证Pipeline包含scaler和model
                if 'scaler' in self.pipeline.named_steps and 'model' in self.pipeline.named_steps:
                    scaler_type = type(self.pipeline.named_steps['scaler']).__name__
                    model_type = type(self.pipeline.named_steps['model']).__name__
                    log(f"Scaler类型: {scaler_type}, Model类型: {model_type}")
                    
                    self.model_loaded = True
                    # 将模型保存到缓存中
                    st.session_state.model_cache[self.target_name] = self.pipeline
                    return True
                else:
                    log("Pipeline结构不符合预期，缺少scaler或model组件")
                    return False
            else:
                log("加载的对象不是有效的Pipeline")
                return False
                
        except Exception as e:
            log(f"加载模型出错: {str(e)}")
            log(traceback.format_exc())
            self.model_loaded = False
            return False
    
    def check_input_range(self, features):
        """检查输入值是否在训练数据范围内"""
        warnings = []
        
        for feature, value in features.items():
            # 获取映射后的特征名
            mapped_feature = self.ui_to_model_mapping.get(feature, feature)
            range_info = self.training_ranges.get(mapped_feature)
            
            if range_info:
                if value < range_info['min'] or value > range_info['max']:
                    warning = f"{feature}: {value:.3f} (超出训练范围 {range_info['min']:.3f} - {range_info['max']:.3f})"
                    warnings.append(warning)
                    log(f"警告: {warning}")
        
        return warnings
    
    def _prepare_features(self, features):
        """准备特征，确保顺序与训练时一致"""
        # 创建特征字典，按训练时的顺序
        model_features = {}
        
        # 首先将UI特征映射到模型特征名称
        for ui_feature, value in features.items():
            model_feature = self.ui_to_model_mapping.get(ui_feature, ui_feature)
            if model_feature in self.feature_names:
                model_features[model_feature] = value
                if ui_feature != model_feature:
                    log(f"特征映射: '{ui_feature}' -> '{model_feature}'")
        
        # 确保所有特征都存在，缺失的设为均值（根据图片统计信息）
        feature_defaults = {
            'M(wt%)': 6.430226,
            'Ash(wt%)': 4.498340,
            'VM(wt%)': 75.375509,
            'O/C': 0.715385,
            'H/C': 1.534106,
            'N/C': 0.034083,
            'FT(℃)': 505.811321,
            'HR(℃/min)': 29.011321,
            'FR(mL/min)': 93.962264
        }
        
        for feature in self.feature_names:
            if feature not in model_features:
                default_value = feature_defaults.get(feature, 0.0)
                model_features[feature] = default_value
                log(f"警告: 特征 '{feature}' 缺失，设为默认值: {default_value}")
        
        # 创建DataFrame并按照正确顺序排列列
        df = pd.DataFrame([model_features])
        df = df[self.feature_names]  # 确保列顺序与训练时一致
        
        log(f"准备好的特征DataFrame形状: {df.shape}, 列: {list(df.columns)}")
        return df
    
    def predict(self, features):
        """预测方法 - 使用Pipeline进行预测"""
        # 检查输入是否有变化
        features_changed = False
        if self.last_features:
            for feature, value in features.items():
                if feature not in self.last_features or abs(self.last_features[feature] - value) > 0.001:
                    features_changed = True
                    break
        else:
            features_changed = True
        
        # 如果输入没有变化且有上次结果，直接返回上次结果
        if not features_changed and self.last_result is not None:
            log("输入未变化，使用上次的预测结果")
            return self.last_result
        
        # 保存当前特征
        self.last_features = features.copy()
        
        # 准备特征数据
        log(f"开始准备{len(features)}个特征数据进行预测")
        features_df = self._prepare_features(features)
        
        # 使用Pipeline进行预测
        if self.model_loaded and self.pipeline is not None:
            try:
                log("使用Pipeline进行预测（包含RobustScaler预处理）")
                # Pipeline会自动进行预处理（RobustScaler）然后预测
                result = float(self.pipeline.predict(features_df)[0])
                log(f"预测成功: {result:.4f}")
                self.last_result = result
                return result
            except Exception as e:
                log(f"Pipeline预测失败: {str(e)}")
                log(traceback.format_exc())
                
                # 尝试重新加载模型
                if self._find_model_file() and self._load_pipeline():
                    try:
                        result = float(self.pipeline.predict(features_df)[0])
                        log(f"重新加载后预测成功: {result:.4f}")
                        self.last_result = result
                        return result
                    except Exception as new_e:
                        log(f"重新加载后预测仍然失败: {str(new_e)}")
        
        # 如果到这里，说明预测失败
        log("所有预测尝试都失败")
        raise ValueError(f"模型预测失败。请确保模型文件存在且格式正确。当前模型: {self.target_name}")
    
    def get_model_info(self):
        """获取模型信息摘要"""
        info = {
            "模型类型": "GBDT Pipeline (RobustScaler + GradientBoostingRegressor)",
            "目标变量": self.target_name,
            "特征数量": len(self.feature_names),
            "模型状态": "已加载" if self.model_loaded else "未加载"
        }
        
        if self.model_loaded and hasattr(self.pipeline, 'named_steps'):
            pipeline_steps = list(self.pipeline.named_steps.keys())
            info["Pipeline组件"] = " → ".join(pipeline_steps)
            
            # 如果有模型组件，显示其参数
            if 'model' in self.pipeline.named_steps:
                model = self.pipeline.named_steps['model']
                model_type = type(model).__name__
                info["回归器类型"] = model_type
                
                # 显示部分关键超参数
                if hasattr(model, 'n_estimators'):
                    info["树的数量"] = model.n_estimators
                if hasattr(model, 'max_depth'):
                    info["最大深度"] = model.max_depth
                if hasattr(model, 'learning_rate'):
                    info["学习率"] = f"{model.learning_rate:.3f}"
                    
        return info

# 初始化预测器 - 使用当前选择的模型
predictor = ModelPredictor(target_model=st.session_state.selected_model)

# 在侧边栏添加模型信息
model_info = predictor.get_model_info()
model_info_html = "<div class='sidebar-model-info'><h3>模型信息</h3>"
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
    st.session_state.feature_values = {}

# 根据图片特征统计信息定义默认值（使用均值）
default_values = {
    "M(wt%)": 6.430,
    "Ash(wt%)": 4.498,
    "VM(wt%)": 75.376,
    "O/C": 0.715,
    "H/C": 1.534,
    "N/C": 0.034,
    "FT(°C)": 505.811,
    "HR(°C/min)": 29.011,
    "FR(mL/min)": 93.962
}

# 保持原有的特征分类名称
feature_categories = {
    "Proximate Analysis": ["M(wt%)", "Ash(wt%)", "VM(wt%)"],
    "Ultimate Analysis": ["O/C", "H/C", "N/C"],
    "Pyrolysis Conditions": ["FT(°C)", "HR(°C/min)", "FR(mL/min)"]
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
            value = st.session_state.feature_values.get(feature, default_values[feature])
        
        col_a, col_b = st.columns([1, 0.5])
        with col_a:
            st.markdown(f"<div class='input-label' style='background-color: {color};'>{feature}</div>", unsafe_allow_html=True)
        with col_b:
            features[feature] = st.number_input(
                "", 
                value=float(value), 
                step=0.01,
                key=f"{category}_{feature}",
                format="%.3f",
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
            features[feature] = st.number_input(
                "", 
                value=float(value), 
                step=0.001,
                key=f"{category}_{feature}",
                format="%.3f",
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
            # 不同特征使用不同的步长
            if feature == "FT(°C)":
                step = 1.0
                format_str = "%.1f"
            elif feature == "FR(mL/min)":
                step = 1.0
                format_str = "%.1f"
            else:  # HR(°C/min)
                step = 0.1
                format_str = "%.2f"
            
            features[feature] = st.number_input(
                "", 
                value=float(value), 
                step=step,
                key=f"{category}_{feature}",
                format=format_str,
                label_visibility="collapsed"
            )

# 调试信息：显示所有当前输入值
with st.expander("📊 显示当前输入值", expanded=False):
    debug_info = "<div style='columns: 3; column-gap: 20px;'>"
    for feature, value in features.items():
        debug_info += f"<p><b>{feature}</b>: {value:.3f}</p>"
    debug_info += "</div>"
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
        log("开始预测流程...")
        
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
        
        # 执行预测
        try:
            # 确保预测器已正确加载
            if not predictor.model_loaded:
                log("模型未加载，尝试重新加载")
                if predictor._find_model_file() and predictor._load_pipeline():
                    log("重新加载模型成功")
                else:
                    error_msg = f"无法加载{st.session_state.selected_model}模型。请确保模型文件存在于正确位置。"
                    st.error(error_msg)
                    st.session_state.prediction_error = error_msg
                    st.rerun()
            
            # 执行预测
            result = predictor.predict(features)
            if result is not None:
                st.session_state.prediction_result = float(result)
                log(f"预测成功: {st.session_state.prediction_result:.4f}")
                st.session_state.prediction_error = None
            else:
                log("警告: 预测结果为空")
                st.session_state.prediction_error = "预测结果为空"
                
        except Exception as e:
            error_msg = f"预测过程中发生错误: {str(e)}"
            st.session_state.prediction_error = error_msg
            log(f"预测错误: {str(e)}")
            log(traceback.format_exc())
            st.error(error_msg)

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
    result_container.markdown(
        f"<div class='yield-result'>{st.session_state.selected_model}: {st.session_state.prediction_result:.2f} wt%</div>", 
        unsafe_allow_html=True
    )
    
    # 显示模型状态
    if not predictor.model_loaded:
        result_container.markdown(
            "<div class='error-box'><b>⚠️ 错误：</b> 模型未成功加载，无法执行预测。请检查模型文件是否存在。</div>", 
            unsafe_allow_html=True
        )
    
    # 显示警告
    if st.session_state.warnings:
        warnings_html = "<div class='warning-box'><b>⚠️ 输入警告</b><ul>"
        for warning in st.session_state.warnings:
            warnings_html += f"<li>{warning}</li>"
        warnings_html += "</ul><p><i>建议调整输入值以获得更准确的预测结果。</i></p></div>"
        result_container.markdown(warnings_html, unsafe_allow_html=True)
    
    # 显示预测详情
    with st.expander("📈 预测详情", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **预测信息:**
            - 目标变量: {st.session_state.selected_model}
            - 预测结果: {st.session_state.prediction_result:.4f} wt%
            - 模型类型: GBDT Pipeline
            - 预处理: RobustScaler
            """)
        with col2:
            st.markdown(f"""
            **模型状态:**
            - 加载状态: {'✅ 正常' if predictor.model_loaded else '❌ 失败'}
            - 特征数量: {len(predictor.feature_names)}
            - 警告数量: {len(st.session_state.warnings)}
            """)

elif st.session_state.prediction_error is not None:
    st.markdown("---")
    error_html = f"""
    <div class='error-box'>
        <h3>❌ 预测失败</h3>
        <p><b>错误信息:</b> {st.session_state.prediction_error}</p>
        <p><b>可能的解决方案:</b></p>
        <ul>
            <li>确保模型文件 (.joblib) 存在于应用目录中</li>
            <li>检查模型文件名是否包含对应的关键词 (char/oil/gas)</li>
            <li>验证输入数据格式是否正确</li>
            <li>确认特征顺序：M, Ash, VM, O/C, H/C, N/C, FT, HR, FR</li>
        </ul>
    </div>
    """
    st.markdown(error_html, unsafe_allow_html=True)

# 技术说明部分
with st.expander("📚 技术说明与使用指南", expanded=False):
    st.markdown("""
    <div class='tech-info'>
    <h4>🔬 模型技术说明</h4>
    <p>本系统基于<b>梯度提升决策树(GBDT)</b>算法构建，采用Pipeline架构集成数据预处理和模型预测：</p>
    <ul>
        <li><b>预处理:</b> RobustScaler标准化，对异常值具有较强的鲁棒性</li>
        <li><b>模型:</b> GradientBoostingRegressor，通过集成多个弱学习器提高预测精度</li>
        <li><b>特征:</b> 9个输入特征，包括近似分析、元素比例和热解工艺条件</li>
    </ul>
    
    <h4>📋 特征说明</h4>
    <ul>
        <li><b>Proximate Analysis:</b> M(wt%) - 水分含量, Ash(wt%) - 灰分含量, VM(wt%) - 挥发分含量</li>
        <li><b>Ultimate Analysis:</b> O/C - 氧碳比, H/C - 氢碳比, N/C - 氮碳比</li>
        <li><b>Pyrolysis Conditions:</b> FT(°C) - 热解温度, HR(°C/min) - 升温速率, FR(mL/min) - 载气流量</li>
    </ul>
    
    <h4>📋 使用建议</h4>
    <ul>
        <li><b>数据质量:</b> 输入参数建议在训练数据分布范围内，以保证预测精度</li>
        <li><b>单位统一:</b> 确保所有输入参数的单位与标签一致</li>
        <li><b>合理性检查:</b> 系统会自动检查输入范围并给出警告提示</li>
    </ul>
    
    <h4>⚠️ 重要提醒</h4>
    <p>模型基于特定的训练数据集开发，预测结果仅供参考。实际应用时请结合专业知识和实验验证。</p>
    </div>
    """, unsafe_allow_html=True)

# 添加页脚
st.markdown("---")
footer = """
<div style='text-align: center; color: #666;'>
<p>© 2024 生物质纳米材料与智能装备实验室 | 基于GBDT的生物质热解产物预测系统 | 版本: 6.2.0</p>
<p>🔥 支持Char、Oil、Gas三种产率预测 | 🚀 Pipeline架构 | 📊 实时范围检查</p>
<p>特征顺序: M(wt%) → Ash(wt%) → VM(wt%) → O/C → H/C → N/C → FT(℃) → HR(℃/min) → FR(mL/min)</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)