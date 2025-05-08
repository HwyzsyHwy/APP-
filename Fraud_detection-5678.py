# -*- coding: utf-8 -*-
"""
吸附能力预测系统 基于XGBoost机器学习模型
专注于Cd2+和TC的吸附容量预测
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
    page_title='吸附容量预测系统',
    page_icon='🧪',
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
log("吸附预测应用启动 - XGBoost模型版本")
log("已加载Cd2+和TC的吸附容量预测模型")

# 初始化会话状态 - 添加模型选择功能
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Cd2+—AC"  # 默认选择Cd2+吸附模型
    log(f"初始化选定模型: {st.session_state.selected_model}")

# 添加模型缓存 - 避免重复加载相同模型
if 'model_cache' not in st.session_state:
    st.session_state.model_cache = {}
    
# 更新主标题以显示当前选定的模型
st.markdown("<h1 class='main-title'>基于XGBoost集成模型的吸附容量预测系统</h1>", unsafe_allow_html=True)

# 添加模型选择区域 - 修改为两个按钮一排
st.markdown("<div class='model-selector'>", unsafe_allow_html=True)
st.markdown("<h3>选择预测目标</h3>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    cd_button = st.button(" Cd2+—AC", 
                           key="cd_button", 
                           help="预测Cd2+吸附容量 (mg/g)", 
                           use_container_width=True,
                           type="primary" if st.session_state.selected_model == "Cd2+—AC" else "secondary")
with col2:
    tc_button = st.button(" TC—AC", 
                          key="tc_button", 
                          help="预测TC吸附容量 (mg/g)", 
                          use_container_width=True,
                          type="primary" if st.session_state.selected_model == "TC—AC" else "secondary")

# 处理模型选择
if cd_button and st.session_state.selected_model != "Cd2+—AC":
    st.session_state.selected_model = "Cd2+—AC"
    st.session_state.prediction_result = None
    st.session_state.warnings = []
    log(f"切换到模型: {st.session_state.selected_model}")
    st.rerun()

if tc_button and st.session_state.selected_model != "TC—AC":
    st.session_state.selected_model = "TC—AC"
    st.session_state.prediction_result = None
    st.session_state.warnings = []
    log(f"切换到模型: {st.session_state.selected_model}")
    st.rerun()

st.markdown(f"<p style='text-align:center;'>当前模型: <b>{st.session_state.selected_model}</b></p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

class ModelPredictor:
    """优化的预测器类 - 适用于吸附模型"""
    
    def __init__(self, target_model="Cd2+—AC"):
        self.target_name = target_model
        
        # 定义正确的特征顺序（与训练时一致）
        self.feature_names = [
            'FT/℃', 'RT/min', 'T/℃', 'TIME/min', 'pH', 'C0/mg/L', 'CAR/g/L'
        ]
        
        # 定义UI到模型的特征映射关系 - 暂无需映射
        self.ui_to_model_mapping = {}
        
        # 反向映射，用于显示
        self.model_to_ui_mapping = {v: k for k, v in self.ui_to_model_mapping.items()}
        
        # 训练范围估计值
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
        # 为不同目标设置不同的模型文件和路径
        model_files = {
            "Cd2+—AC": ["XGBoost-Cd2+-model.joblib"],
            "TC—AC": ["XGBoost-TC-model.joblib"]
        }
        
        # 获取当前模型的文件名列表
        filenames = model_files.get(self.target_name, [])
        
        # 尝试常见的模型文件名和路径
        search_dirs = [".", "./models", "../models", "/app/models", "/app"]
        
        # 在所有可能的目录中搜索模型文件
        log(f"搜索{self.target_name}模型文件...")
        
        for directory in search_dirs:
            if not os.path.exists(directory):
                continue
                
            # 直接检查是否存在指定文件名
            for filename in filenames:
                model_path = os.path.join(directory, filename)
                if os.path.isfile(model_path):
                    log(f"找到模型文件: {model_path}")
                    return model_path
        
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
        """设置训练数据的范围估计值"""
        ranges = {
            'FT/℃': {'min': 250.0, 'max': 900.0},
            'RT/min': {'min': 15.0, 'max': 120.0},
            'T/℃': {'min': 20.0, 'max': 50.0},
            'TIME/min': {'min': 15.0, 'max': 180.0},
            'pH': {'min': 2.0, 'max': 10.0},
            'C0/mg/L': {'min': 10.0, 'max': 1000.0},
            'CAR/g/L': {'min': 0.1, 'max': 10.0}
        }
        
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
                    warning = f"{feature}: {value:.2f} (超出估计训练范围 {range_info['min']:.2f} - {range_info['max']:.2f})"
                    warnings.append(warning)
                    log(f"警告: {warning}")
        
        return warnings
    
    def _prepare_features(self, features):
        """准备特征，处理特征名称映射和顺序"""
        # 创建一个空的DataFrame，所有特征初始化为0
        model_features = {feature: 0.0 for feature in self.feature_names}
        
        # 首先将UI特征映射到模型特征名称
        for ui_feature, value in features.items():
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
                # 直接使用Pipeline进行预测，包含所有预处理步骤
                result = float(self.pipeline.predict(features_df)[0])
                log(f"Pipeline预测结果: {result:.2f}")
                self.last_result = result
                return result
            except Exception as e:
                log(f"Pipeline预测失败: {str(e)}")
                log(traceback.format_exc())
                # 如果加载失败，则尝试重新加载模型
                if self._load_pipeline():
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
            "模型类型": "XGBoost集成模型",
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
                    
                    # 显示部分关键超参数
                    if hasattr(model, 'n_estimators'):
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

# 定义默认值
default_values = {
    "FT/℃": 500.0,
    "RT/min": 60.0,
    "T/℃": 30.0,
    "TIME/min": 60.0,
    "pH": 6.0,
    "C0/mg/L": 100.0,
    "CAR/g/L": 1.0
}

# 特征分类 - 分为三组但不显示标签
feature_categories = {
    "Group1": ["FT/℃", "RT/min", "T/℃"],
    "Group2": ["TIME/min", "pH"],
    "Group3": ["C0/mg/L", "CAR/g/L"]
}

# 颜色配置
category_colors = {
    "Group1": "#501d8a",  
    "Group2": "#1c8041",  
    "Group3": "#e55709" 
}

# 创建三列布局
col1, col2, col3 = st.columns(3)

# 使用字典存储所有输入值
features = {}

# Group1 - 第一列
with col1:
    category = "Group1"
    color = category_colors[category]
    # 不显示分类标签
    # st.markdown(f"<div class='section-header' style='background-color: {color};'>{category}</div>", unsafe_allow_html=True)
    
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

# Group2 - 第二列
with col2:
    category = "Group2"
    color = category_colors[category]
    # 不显示分类标签
    # st.markdown(f"<div class='section-header' style='background-color: {color};'>{category}</div>", unsafe_allow_html=True)
    
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

# Group3 - 第三列
with col3:
    category = "Group3"
    color = category_colors[category]
    # 不显示分类标签
    # st.markdown(f"<div class='section-header' style='background-color: {color};'>{category}</div>", unsafe_allow_html=True)
    
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
    result_container.markdown(f"<div class='yield-result'>{st.session_state.selected_model}: {st.session_state.prediction_result:.2f} mg/g</div>", unsafe_allow_html=True)
    
    # 显示模型信息
    if not predictor.model_loaded:
        result_container.markdown(
            "<div class='error-box'><b>⚠️ 错误：</b> 模型未成功加载，无法执行预测。请检查模型文件是否存在。</div>", 
            unsafe_allow_html=True
        )
    
    # 显示警告
    if st.session_state.warnings:
        warnings_html = "<div class='warning-box'><b>⚠️ 警告：部分输入可能影响预测精度</b><ul>"
        for warning in st.session_state.warnings:
            warnings_html += f"<li>{warning}</li>"
        warnings_html += "</ul><p>请根据提示调整输入值以获得更准确的预测。</p></div>"
        result_container.markdown(warnings_html, unsafe_allow_html=True)
    
    # 显示预测信息
    with st.expander("预测信息", expanded=False):
        st.markdown(f"""
        - **目标变量:** {st.session_state.selected_model}
        - **预测结果:** {st.session_state.prediction_result:.2f} mg/g
        - **使用模型:** {"Pipeline模型" if predictor.model_loaded else "未能加载模型"}
        """)
    
    # 技术说明部分 - 使用折叠式展示
    with st.expander("技术说明", expanded=False):
        st.markdown("""
        <div class='tech-info'>
        <p>本模型基于XGBoost（极限梯度提升）算法创建，预测吸附剂对Cd2+和TC的吸附容量。模型使用吸附试验条件作为输入，计算最终吸附容量。</p>
        
        <p><b>特别提醒：</b></p>
        <ul>
            <li>输入参数建议在训练数据的分布范围内，以保证软件的预测精度</li>
            <li>当输入超出模型训练范围时，预测精度可能会降低</li>
            <li>pH值对吸附过程影响显著，请确保输入合理的pH值</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

elif st.session_state.prediction_error is not None:
    st.markdown("---")
    error_html = f"""
    <div class='error-box'>
        <h3>预测失败</h3>
        <p>{st.session_state.prediction_error}</p>
        <p>请检查：</p>
        <ul>
            <li>确保模型文件 (.joblib) 存在于正确位置</li>
            <li>确保输入数据符合模型要求</li>
            <li>检查输入值是否在合理范围内</li>
        </ul>
    </div>
    """
    st.markdown(error_html, unsafe_allow_html=True)

# 添加页脚
st.markdown("---")
footer = """
<div style='text-align: center;'>
<p>© 2024 吸附预测系统. 版本: 1.0.0</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)