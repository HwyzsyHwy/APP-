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
    log_entry = "[" + timestamp + "] " + message
    st.session_state.log_messages.append(log_entry)
    # 只保留最近的100条日志
    if len(st.session_state.log_messages) > 100:
        st.session_state.log_messages = st.session_state.log_messages[-100:]
    
    # 更新日志显示
    log_text.markdown(
        "<div class='log-container'>" + '<br>'.join(st.session_state.log_messages) + "</div>", 
        unsafe_allow_html=True
    )

# 记录启动日志
log("应用启动 - 根据图片特征统计信息正确修复版本")
log("特征顺序：M, Ash, VM, O/C, H/C, N/C, FT, HR, FR")

# 初始化会话状态 - 添加模型选择功能
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Char Yield"  # 默认选择Char产率模型
    log("初始化选定模型: " + st.session_state.selected_model)

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
    log("切换到模型: " + st.session_state.selected_model)
    st.rerun()

if oil_button and st.session_state.selected_model != "Oil Yield":
    st.session_state.selected_model = "Oil Yield"
    st.session_state.prediction_result = None
    st.session_state.warnings = []
    log("切换到模型: " + st.session_state.selected_model)
    st.rerun()

if gas_button and st.session_state.selected_model != "Gas Yield":
    st.session_state.selected_model = "Gas Yield"
    st.session_state.prediction_result = None
    st.session_state.warnings = []
    log("切换到模型: " + st.session_state.selected_model)
    st.rerun()

st.markdown("<p style='text-align:center;'>当前模型: <b>" + st.session_state.selected_model + "</b></p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

class ModelPredictor:
    """根据图片特征统计信息正确调整的预测器类"""
    
    def __init__(self, target_model="Char Yield"):
        self.target_name = target_model
        
        # 定义特征名称（按照训练时的顺序）
        self.feature_names = [
            'M(wt%)', 'Ash(wt%)', 'VM(wt%)', 'O/C', 'H/C', 'N/C',
            'FT(℃)', 'HR(℃/min)', 'FR(mL/min)'
        ]
        
        # UI到模型特征的映射
        self.ui_to_model_mapping = {
            'M(wt%)': 'M(wt%)',
            'Ash(wt%)': 'Ash(wt%)', 
            'VM(wt%)': 'VM(wt%)',
            'O/C': 'O/C',
            'H/C': 'H/C',
            'N/C': 'N/C',
            'FT(°C)': 'FT(℃)',
            'HR(°C/min)': 'HR(℃/min)',
            'FR(mL/min)': 'FR(mL/min)'
        }
        
        # 根据图片特征统计信息定义训练数据范围
        self.training_ranges = {
            'M(wt%)': {'min': 2.750, 'max': 11.630},
            'Ash(wt%)': {'min': 0.410, 'max': 11.600},
            'VM(wt%)': {'min': 65.700, 'max': 89.500},
            'O/C': {'min': 0.301, 'max': 0.988},
            'H/C': {'min': 1.212, 'max': 1.895},
            'N/C': {'min': 0.003, 'max': 0.129},
            'FT(°C)': {'min': 300.000, 'max': 900.000},
            'HR(°C/min)': {'min': 5.000, 'max': 100.000},
            'FR(mL/min)': {'min': 0.000, 'max': 600.000}
        }
        
        self.last_features = {}  # 存储上次的特征值
        self.last_result = None  # 存储上次的预测结果
        
        # 使用缓存加载模型，避免重复加载相同模型
        self.pipeline = self._get_cached_model()
        self.model_loaded = self.pipeline is not None
        
        if not self.model_loaded:
            log("从缓存未找到模型，尝试加载" + self.target_name + "模型")
            # 查找并加载模型
            self.model_path = self._find_model_file()
            if self.model_path:
                self._load_pipeline()
    
    def _get_cached_model(self):
        """从缓存中获取模型"""
        if self.target_name in st.session_state.model_cache:
            log("从缓存加载" + self.target_name + "模型")
            return st.session_state.model_cache[self.target_name]
        return None
    
    def _find_model_file(self):
        """查找对应的模型文件"""
        # 获取当前目录下的所有.joblib文件
        current_dir = os.getcwd()
        joblib_files = glob.glob(os.path.join(current_dir, "*.joblib"))
        
        log("当前目录: " + current_dir)
        log("找到的.joblib文件: " + str(joblib_files))
        
        # 根据目标模型查找对应文件
        target_keywords = {
            "Char Yield": ["char", "Char"],
            "Oil Yield": ["oil", "Oil", "bio"],
            "Gas Yield": ["gas", "Gas"]
        }
        
        keywords = target_keywords.get(self.target_name, [])
        
        for file_path in joblib_files:
            filename = os.path.basename(file_path).lower()
            for keyword in keywords:
                if keyword.lower() in filename:
                    log("找到匹配的模型文件: " + file_path)
                    return file_path
        
        log("警告: 未找到匹配的" + self.target_name + "模型文件")
        return None
    
    def _load_pipeline(self):
        """加载Pipeline模型"""
        try:
            if not self.model_path or not os.path.exists(self.model_path):
                log("错误: 模型文件不存在: " + str(self.model_path))
                return False
            
            log("开始加载Pipeline: " + self.model_path)
            self.pipeline = joblib.load(self.model_path)
            
            # 缓存模型
            st.session_state.model_cache[self.target_name] = self.pipeline
            
            log("Pipeline加载成功，类型: " + str(type(self.pipeline)))
            
            # 验证Pipeline结构
            if hasattr(self.pipeline, 'steps'):
                log("Pipeline步骤: " + str([step[0] for step in self.pipeline.steps]))
            
            self.model_loaded = True
            return True
                
        except Exception as e:
            log("加载模型出错: " + str(e))
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
                    warning = feature + ": " + str(round(value, 3)) + " (超出训练范围 " + str(round(range_info['min'], 3)) + " - " + str(round(range_info['max'], 3)) + ")"
                    warnings.append(warning)
                    log("警告: " + warning)
        
        return warnings
    
    def _prepare_features(self, features):
        """准备特征数据用于预测"""
        # 按照模型训练时的特征顺序排列
        feature_values = []
        for feature_name in self.feature_names:
            # 从UI特征映射到模型特征
            ui_feature = None
            for ui_key, model_key in self.ui_to_model_mapping.items():
                if model_key == feature_name:
                    ui_feature = ui_key
                    break
            
            if ui_feature and ui_feature in features:
                feature_values.append(features[ui_feature])
                log("特征 " + feature_name + ": " + str(features[ui_feature]))
            else:
                log("警告: 未找到特征 " + feature_name)
                feature_values.append(0.0)  # 默认值
        
        # 创建DataFrame
        features_df = pd.DataFrame([feature_values], columns=self.feature_names)
        log("准备的特征DataFrame形状: " + str(features_df.shape))
        log("特征值: " + str(feature_values))
        
        return features_df
    
    def predict(self, features):
        """执行预测"""
        log("=" * 50)
        log("开始预测流程 - 目标: " + self.target_name)
        
        # 检查特征是否有变化
        if features == self.last_features and self.last_result is not None:
            log("特征未变化，返回缓存结果: " + str(self.last_result))
            return self.last_result
        
        # 保存当前特征
        self.last_features = features.copy()
        
        # 准备特征数据
        log("开始准备" + str(len(features)) + "个特征数据进行预测")
        features_df = self._prepare_features(features)
        
        # 使用Pipeline进行预测
        if self.model_loaded and self.pipeline is not None:
            try:
                log("使用Pipeline进行预测（包含RobustScaler预处理）")
                # Pipeline会自动进行预处理（RobustScaler）然后预测
                result = float(self.pipeline.predict(features_df)[0])
                log("预测成功: " + str(round(result, 4)))
                self.last_result = result
                return result
            except Exception as e:
                log("Pipeline预测失败: " + str(e))
                log(traceback.format_exc())
                
                # 尝试重新加载模型
                if self._find_model_file() and self._load_pipeline():
                    try:
                        result = float(self.pipeline.predict(features_df)[0])
                        log("重新加载后预测成功: " + str(round(result, 4)))
                        self.last_result = result
                        return result
                    except Exception as new_e:
                        log("重新加载后预测仍然失败: " + str(new_e))
        
        log("预测失败，返回None")
        return None
    
    def get_model_info(self):
        """获取模型信息"""
        info = {
            "目标变量": self.target_name,
            "模型状态": "✅ 已加载" if self.model_loaded else "❌ 未加载",
            "特征数量": len(self.feature_names),
            "模型类型": "GBDT Pipeline",
            "预处理": "RobustScaler"
        }
        
        if hasattr(self, 'model_path') and self.model_path:
            info["模型文件"] = os.path.basename(self.model_path)
        
        return info

# 初始化预测器 - 使用当前选择的模型
predictor = ModelPredictor(target_model=st.session_state.selected_model)

# 在侧边栏添加模型信息
model_info = predictor.get_model_info()
model_info_html = "<div class='sidebar-model-info'><h3>模型信息</h3>"
for key, value in model_info.items():
    model_info_html += "<p><b>" + key + "</b>: " + str(value) + "</p>"

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
    st.markdown("<div class='section-header' style='background-color: " + color + ";'>" + category + "</div>", unsafe_allow_html=True)
    
    for feature in feature_categories[category]:
        if st.session_state.clear_pressed:
            value = default_values[feature]
        else:
            value = st.session_state.feature_values.get(feature, default_values[feature])
        
        col_a, col_b = st.columns([1, 0.5])
        with col_a:
            st.markdown("<div class='input-label' style='background-color: " + color + ";'>" + feature + "</div>", unsafe_allow_html=True)
        with col_b:
            features[feature] = st.number_input(
                "", 
                value=float(value), 
                step=0.01,
                key=category + "_" + feature,
                format="%.3f",
                label_visibility="collapsed"
            )

# Ultimate Analysis - 第二列
with col2:
    category = "Ultimate Analysis"
    color = category_colors[category]
    st.markdown("<div class='section-header' style='background-color: " + color + ";'>" + category + "</div>", unsafe_allow_html=True)
    
    for feature in feature_categories[category]:
        if st.session_state.clear_pressed:
            value = default_values[feature]
        else:
            value = st.session_state.feature_values.get(feature, default_values[feature])
        
        col_a, col_b = st.columns([1, 0.5])
        with col_a:
            st.markdown("<div class='input-label' style='background-color: " + color + ";'>" + feature + "</div>", unsafe_allow_html=True)
        with col_b:
            features[feature] = st.number_input(
                "", 
                value=float(value), 
                step=0.001,
                key=category + "_" + feature,
                format="%.3f",
                label_visibility="collapsed"
            )

# Pyrolysis Conditions - 第三列
with col3:
    category = "Pyrolysis Conditions"
    color = category_colors[category]
    st.markdown("<div class='section-header' style='background-color: " + color + ";'>" + category + "</div>", unsafe_allow_html=True)
    
    for feature in feature_categories[category]:
        if st.session_state.clear_pressed:
            value = default_values[feature]
        else:
            value = st.session_state.feature_values.get(feature, default_values[feature])
        
        col_a, col_b = st.columns([1, 0.5])
        with col_a:
            st.markdown("<div class='input-label' style='background-color: " + color + ";'>" + feature + "</div>", unsafe_allow_html=True)
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
                key=category + "_" + feature,
                format=format_str,
                label_visibility="collapsed"
            )

# 调试信息：显示所有当前输入值
with st.expander("📊 显示当前输入值", expanded=False):
    debug_info = "<div style='columns: 3; column-gap: 20px;'>"
    for feature, value in features.items():
        debug_info += "<p><b>" + feature + "</b>: " + str(round(value, 3)) + "</p>"
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
            log("检测到模型变更，重新初始化预测器: " + st.session_state.selected_model)
            predictor = ModelPredictor(target_model=st.session_state.selected_model)
        
        # 保存当前输入到会话状态
        st.session_state.feature_values = features.copy()
        
        log("开始" + st.session_state.selected_model + "预测，输入特征数: " + str(len(features)))
        
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
                    error_msg = "无法加载" + st.session_state.selected_model + "模型。请确保模型文件存在于正确位置。"
                    st.error(error_msg)
                    st.session_state.prediction_error = error_msg
                    st.rerun()
            
            # 执行预测
            result = predictor.predict(features)
            if result is not None:
                st.session_state.prediction_result = float(result)
                log("预测成功: " + str(round(st.session_state.prediction_result, 4)))
                st.session_state.prediction_error = None
            else:
                log("警告: 预测结果为空")
                st.session_state.prediction_error = "预测结果为空"
                
        except Exception as e:
            error_msg = "预测过程中发生错误: " + str(e)
            st.session_state.prediction_error = error_msg
            log("预测错误: " + str(e))
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
        "<div class='yield-result'>" + st.session_state.selected_model + ": " + str(round(st.session_state.prediction_result, 2)) + " wt%</div>", 
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
            warnings_html += "<li>" + warning + "</li>"
        warnings_html += "</ul><p><i>建议调整输入值以获得更准确的预测结果。</i></p></div>"
        result_container.markdown(warnings_html, unsafe_allow_html=True)
    
    # 显示预测详情
    with st.expander("📈 预测详情", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **预测信息:**
            - 目标变量: """ + st.session_state.selected_model + """
            - 预测结果: """ + str(round(st.session_state.prediction_result, 4)) + """ wt%
            - 模型类型: GBDT Pipeline
            - 预处理: RobustScaler
            """)
        with col2:
            st.markdown("""
            **模型状态:**
            - 加载状态: """ + ('✅ 正常' if predictor.model_loaded else '❌ 失败') + """
            - 特征数量: """ + str(len(predictor.feature_names)) + """
            - 警告数量: """ + str(len(st.session_state.warnings)) + """
            """)

elif st.session_state.prediction_error is not None:
    st.markdown("---")
    error_html = """
    <div class='error-box'>
        <h3>❌ 预测失败</h3>
        <p><b>错误信息:</b> """ + st.session_state.prediction_error + """</p>
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