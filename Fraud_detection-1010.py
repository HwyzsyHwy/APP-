# -*- coding: utf-8 -*-
"""
Biomass Pyrolysis Yield Forecast using GBDT Ensemble Models
优化版本 - 修复特征顺序和处理问题
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
log("应用启动 - 优化版本")
log("已修复特征顺序和处理问题")

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
    """优化的预测器类 - 修复特征顺序和处理问题"""
    
    def __init__(self, target_model="Char Yield"):
        """初始化预测器 - 修改为动态获取特征顺序"""
        self.target_name = target_model
        
        # 尝试从训练数据中获取正确的特征顺序
        training_data_paths = [
            r'C:\Users\HWY\Desktop\最终版-代码\rf_imputed_data-11.csv',
            './rf_imputed_data-11.csv',
            '../rf_imputed_data-11.csv',
            './models/rf_imputed_data-11.csv',
            '/app/rf_imputed_data-11.csv'
        ]
        
        # 尝试从训练数据获取特征顺序
        self.feature_names = None
        for path in training_data_paths:
            try:
                if os.path.exists(path):
                    log(f"尝试从训练数据获取特征顺序: {path}")
                    df = pd.read_csv(path)
                    # 排除目标列
                    self.feature_names = list(df.drop(['Char Yield(wt%)', 'Oil Yield(wt%)', 'Gas Yield(wt%)'], axis=1).columns)
                    log(f"从训练数据获取到的特征顺序: {self.feature_names}")
                    break
            except Exception as e:
                log(f"从{path}读取特征顺序失败: {str(e)}")
        
        # 如果无法获取，则使用备用的硬编码特征列表
        if not self.feature_names:
            log("未能从训练数据获取特征顺序，使用备用特征列表")
            self.feature_names = [
                'M(wt%)', 'Ash(wt%)', 'VM(wt%)', 'FC(wt%)', 
                'C(wt%)', 'H(wt%)', 'N(wt%)', 'O(wt%)', 
                'PS(mm)', 'SM(g)', 'FT(℃)', 'HR(℃/min)', 
                'FR(mL/min)', 'RT(min)'
            ]
            log(f"使用备用特征顺序: {self.feature_names}")
        
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
        """查找模型文件"""
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
            'O(wt%)': {'min': 34.000, 'max': 73.697},
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
        """准备特征，处理特征名称映射和顺序 - 优化版本"""
        # 详细记录输入的特征
        log(f"准备特征，当前特征名称: {list(features.keys())}")
        log(f"目标特征名称: {self.feature_names}")
        
        # 创建一个空的特征字典，确保所有特征初始化为0
        model_features = {feature: 0.0 for feature in self.feature_names}
        
        # 映射并填充特征值
        for ui_feature, value in features.items():
            model_feature = self.ui_to_model_mapping.get(ui_feature, ui_feature)
            if model_feature in self.feature_names:
                model_features[model_feature] = value
                if ui_feature != model_feature:
                    log(f"特征映射: '{ui_feature}' -> '{model_feature}'")
            else:
                log(f"警告: 特征 '{ui_feature}' 在模型特征中不存在")
        
        # 创建DataFrame并确保列顺序
        df = pd.DataFrame([model_features])
        
        # 检查是否所有特征都存在
        missing_cols = set(self.feature_names) - set(df.columns)
        if missing_cols:
            log(f"警告: 缺少特征: {missing_cols}")
        
        # 确保列顺序与训练时一致 - 这是核心修复
        df = df[self.feature_names]
        
        # 详细记录特征值
        log(f"最终特征列顺序: {list(df.columns)}")
        for col in self.feature_names:
            log(f"  {col}: {df[col].values[0]}")
        
        return df
    
    def validate_input(self, features_df):
        """验证输入数据的合理性"""
        # 检查是否所有特征都存在
        missing_cols = set(self.feature_names) - set(features_df.columns)
        if missing_cols:
            raise ValueError(f"缺少必要特征: {missing_cols}")
        
        # 确保没有NaN值
        if features_df.isnull().any().any():
            raise ValueError("输入数据包含NaN值")
        
        # 检查特征值范围
        for feature in self.feature_names:
            value = features_df[feature].values[0]
            if np.isinf(value) or np.isnan(value):
                raise ValueError(f"特征 {feature} 值无效: {value}")
        
        return True
    
    def predict(self, features):
        """预测方法 - 添加详细的调试信息"""
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
        
        # 详细输出输入特征
        log(f"收到的原始特征输入:")
        for f, v in features.items():
            log(f"  {f}: {v}")
        
        # 准备特征数据
        log(f"开始准备特征数据")
        features_df = self._prepare_features(features)
        
        # 验证特征数据
        try:
            self.validate_input(features_df)
        except ValueError as e:
            log(f"输入验证失败: {str(e)}")
            raise
        
        # 尝试使用Pipeline进行预测
        if self.model_loaded and self.pipeline is not None:
            try:
                # 输出Pipeline各组件信息
                if hasattr(self.pipeline, 'named_steps'):
                    log(f"Pipeline组件: {list(self.pipeline.named_steps.keys())}")
                    
                    # 如果有scaler组件，输出缩放前后的数据
                    if 'scaler' in self.pipeline.named_steps:
                        scaler = self.pipeline.named_steps['scaler']
                        scaler_type = type(scaler).__name__
                        log(f"使用缩放器: {scaler_type}")
                        
                        try:
                            # 尝试单独应用缩放器看结果
                            scaled_data = scaler.transform(features_df)
                            log(f"缩放后的特征值示例: {scaled_data[0][:3]}")
                        except Exception as e:
                            log(f"应用缩放器时出错: {str(e)}")
                
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
            "模型类型": "GBDT集成模型",
            "目标变量": self.target_name,
            "特征数量": len(self.feature_names),
            "特征顺序来源": "训练数据" if self.feature_names else "默认配置",
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
            st.error(f"预测过程中发生错误: {str(e)}")

with col2:
    if st.button("🔄 重置输入", use_container_width=True):
        log("重置所有输入值")
        st.session_state.clear_pressed = True
        st.session_state.prediction_result = None
        st.session_state.warnings = []
        st.session_state.prediction_error = None
        st.rerun()

# 添加验证数据测试功能
with st.expander("验证模型", expanded=False):
    st.markdown("### 模型验证")
    st.markdown("使用此功能可以验证模型对特定样本的预测能力。")
    
    # 创建简单的验证工具
    validate_col1, validate_col2 = st.columns([3, 1])
    
    with validate_col1:
        sample_id = st.number_input("选择样本ID", min_value=1, max_value=100, value=1, step=1)
    
    with validate_col2:
        validate_button = st.button("验证样本", use_container_width=True)
    
    if validate_button:
        log(f"尝试验证样本ID: {sample_id}")
        try:
            # 尝试从训练数据集加载样本
            training_data_paths = [
                r'C:\Users\HWY\Desktop\最终版-代码\rf_imputed_data-11.csv',
                './rf_imputed_data-11.csv',
                '../rf_imputed_data-11.csv',
                './models/rf_imputed_data-11.csv',
                '/app/rf_imputed_data-11.csv'
            ]
            
            sample_loaded = False
            for path in training_data_paths:
                if os.path.exists(path):
                    try:
                        df = pd.read_csv(path)
                        if len(df) >= sample_id:
                            # 获取样本
                            sample = df.iloc[sample_id-1:sample_id]
                            
                            # 获取特征和真实值
                            X_sample = sample.drop(['Char Yield(wt%)', 'Oil Yield(wt%)', 'Gas Yield(wt%)'], axis=1)
                            true_values = {
                                "Char Yield": float(sample['Char Yield(wt%)'].values[0]),
                                "Oil Yield": float(sample['Oil Yield(wt%)'].values[0]),
                                "Gas Yield": float(sample['Gas Yield(wt%)'].values[0])
                            }
                            
                            # 准备UI特征字典
                            ui_features = {}
                            for col in X_sample.columns:
                                # 对温度特征进行映射
                                if col == 'FT(℃)':
                                    ui_features['FT(°C)'] = float(X_sample[col].values[0])
                                elif col == 'HR(℃/min)':
                                    ui_features['HR(°C/min)'] = float(X_sample[col].values[0])
                                else:
                                    ui_features[col] = float(X_sample[col].values[0])
                            
                            log(f"从{path}加载样本ID {sample_id}成功")
                            
                            # 使用同样的预测器进行预测
                            predicted = predictor.predict(ui_features)
                            
                            # 显示结果比较
                            st.markdown(f"#### 样本 {sample_id} 验证结果")
                            
                            cols = st.columns(2)
                            with cols[0]:
                                st.markdown("##### 样本特征")
                                feature_html = "<ul style='columns: 2;'>"
                                for f, v in ui_features.items():
                                    feature_html += f"<li><b>{f}:</b> {v:.2f}</li>"
                                feature_html += "</ul>"
                                st.markdown(feature_html, unsafe_allow_html=True)
                            
                            with cols[1]:
                                st.markdown("##### 预测结果比较")
                                true_val = true_values[st.session_state.selected_model]
                                rel_error = abs(true_val - predicted) / true_val * 100 if true_val != 0 else float('inf')
                                
                                st.markdown(f"**真实值:** {true_val:.2f} wt%")
                                st.markdown(f"**预测值:** {predicted:.2f} wt%")
                                
                                if rel_error < 5:
                                    st.markdown(f"**相对误差:** <span style='color:green'>{rel_error:.2f}%</span>", unsafe_allow_html=True)
                                elif rel_error < 10:
                                    st.markdown(f"**相对误差:** <span style='color:orange'>{rel_error:.2f}%</span>", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"**相对误差:** <span style='color:red'>{rel_error:.2f}%</span>", unsafe_allow_html=True)
                            
                            sample_loaded = True
                            break
                    except Exception as e:
                        log(f"从{path}加载样本时出错: {str(e)}")
                        continue
            
            if not sample_loaded:
                st.warning(f"未能找到样本ID {sample_id}，请确保训练数据文件存在且包含足够多的样本。")
                log(f"未能找到样本ID {sample_id}")
        except Exception as e:
            st.error(f"验证过程中出错: {str(e)}")
            log(f"验证错误: {str(e)}")

# 显示预测结果
if st.session_state.prediction_result is not None:
    st.markdown("---")
    
    # 显示主预测结果
    result_container.markdown(f"<div class='yield-result'>{st.session_state.selected_model}: {st.session_state.prediction_result:.2f} wt%</div>", unsafe_allow_html=True)
    
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
        - **预测结果:** {st.session_state.prediction_result:.2f} wt%
        - **使用模型:** {"Pipeline模型" if predictor.model_loaded else "未能加载模型"}
        """)
        
        # 添加模型详细信息
        if predictor.model_loaded and hasattr(predictor.pipeline, 'named_steps'):
            if 'model' in predictor.pipeline.named_steps:
                model = predictor.pipeline.named_steps['model']
                st.markdown("##### 模型参数")
                model_params = model.get_params()
                params_html = "<ul style='columns: 2;'>"
                for param, value in model_params.items():
                    if param in ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'max_features']:
                        params_html += f"<li><b>{param}:</b> {value}</li>"
                params_html += "</ul>"
                st.markdown(params_html, unsafe_allow_html=True)
    
    # 技术说明部分 - 使用折叠式展示
    with st.expander("技术说明", expanded=False):
        st.markdown("""
        <div class='tech-info'>
        <p>本模型基于GBDT（梯度提升决策树）算法创建，预测生物质热解产物分布。模型使用生物质的元素分析、近似分析数据和热解条件作为输入，计算热解炭、热解油和热解气体产量。</p>
        
        <p><b>特别提醒：</b></p>
        <ul>
            <li>输入参数建议在训练数据的分布范围内，以保证软件的预测精度</li>
            <li>由于模型训练时FC(wt%)通过100-Ash(wt%)-VM(wt%)公式转换得出，所以用户使用此软件进行预测时也建议使用此公式对FC(wt%)进行计算</li>
            <li>所有特征的训练范围都基于真实训练数据的统计信息，如输入超出范围将会收到提示</li>
        </ul>
        
        <p><b>特征顺序问题：</b></p>
        <p>模型训练和预测时特征顺序必须完全一致，否则会导致预测结果严重偏差。此版本已修复特征顺序问题，确保与训练时完全一致。</p>
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
            <li>检查FC(wt%)是否满足 100-Ash(wt%)-VM(wt%) 约束</li>
        </ul>
    </div>
    """
    st.markdown(error_html, unsafe_allow_html=True)

# 添加调试工具
with st.expander("调试工具", expanded=False):
    st.markdown("### 模型调试")
    
    debug_cols = st.columns(3)
    
    with debug_cols[0]:
        if st.button("检查特征顺序", use_container_width=True):
            if predictor.feature_names:
                st.code("\n".join([f"{i+1}. {f}" for i, f in enumerate(predictor.feature_names)]))
                log(f"显示特征顺序: {len(predictor.feature_names)}个特征")
            else:
                st.warning("特征顺序未定义")
    
    with debug_cols[1]:
        if st.button("检查模型组件", use_container_width=True):
            if predictor.model_loaded and hasattr(predictor.pipeline, 'named_steps'):
                components = predictor.pipeline.named_steps
                st.code("\n".join([f"{k}: {type(v).__name__}" for k, v in components.items()]))
                log(f"显示模型组件: {', '.join(components.keys())}")
            else:
                st.warning("模型未加载或不是Pipeline")
    
    with debug_cols[2]:
        if st.button("测试预测流程", use_container_width=True):
            try:
                # 使用当前输入进行测试预测
                log("开始测试预测流程...")
                
                # 准备特征
                features_df = predictor._prepare_features(features)
                st.subheader("1. 准备好的特征")
                st.dataframe(features_df)
                
                # 如果有scaler，单独应用
                if predictor.model_loaded and hasattr(predictor.pipeline, 'named_steps'):
                    if 'scaler' in predictor.pipeline.named_steps:
                        scaler = predictor.pipeline.named_steps['scaler']
                        scaled_data = scaler.transform(features_df)
                        
                        st.subheader("2. 缩放后的特征")
                        st.write(f"缩放器类型: {type(scaler).__name__}")
                        
                        # 创建带有原始特征名的DataFrame来显示缩放后的数据
                        scaled_df = pd.DataFrame(scaled_data, columns=features_df.columns)
                        st.dataframe(scaled_df)
                        
                        log(f"缩放器应用成功: {type(scaler).__name__}")
                
                # 执行完整Pipeline预测
                result = predictor.predict(features)
                st.subheader("3. 预测结果")
                st.success(f"预测值: {result:.4f}")
                log(f"测试预测流程完成: {result:.4f}")
                
            except Exception as e:
                st.error(f"测试预测流程失败: {str(e)}")
                log(f"测试预测流程错误: {str(e)}")

# 添加页脚
st.markdown("---")
footer = """
<div style='text-align: center;'>
<p>© 2024 生物质纳米材料与智能装备实验室. 版本: 5.2.0</p>
<p><small>优化版本：已修复特征顺序问题，提高预测精度</small></p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)