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
import sys

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
log("已修复模型加载问题 - 增加对集成模型的支持")

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

# 使用修改后的分段校正功能
def apply_ranged_correction(predictions, y_true=None, correction_factors=None, value_ranges=None):
    """分段校正函数，可被直接调用"""
    if correction_factors is None or value_ranges is None:
        return predictions
        
    predictions = np.array(predictions).ravel()
    corrected = np.zeros_like(predictions)
    
    # 确定用于选择校正系数的值
    if y_true is not None:
        # 如果y_true是DataFrame，先转换为numpy数组
        if isinstance(y_true, pd.DataFrame):
            selector = y_true.values.ravel()
        else:
            selector = np.array(y_true).ravel()
    else:
        selector = predictions
    
    # 对每个样本应用适当的校正系数
    for low, high in value_ranges:
        mask = (selector >= low) & (selector < high)
        corrected[mask] = predictions[mask] * correction_factors.get((low, high), 1.0)
    
    return corrected

# 添加用于替代缺失的bias_corrected_predict函数的定义
def bias_corrected_predict(X):
    """创建一个空的函数来替代缺失的bias_corrected_predict函数"""
    return X

# 添加在Python全局空间的ensemble_predict函数定义，避免反序列化问题
def ensemble_predict(X_new, main_pipeline, support_pipeline, correction_factors=None, value_ranges=None, main_weight=0.8, support_weight=0.2):
    """集成预测函数，结合主模型和支持模型，并应用分段校正"""
    # 获取两个模型的预测
    main_preds = main_pipeline.predict(X_new)
    support_preds = support_pipeline.predict(X_new)
    
    # 组合预测结果
    combined_preds = main_preds * main_weight + support_preds * support_weight
    
    # 应用分段校正
    if correction_factors and value_ranges:
        corrected_preds = apply_ranged_correction(combined_preds, 
                                                  correction_factors=correction_factors, 
                                                  value_ranges=value_ranges)
        return corrected_preds
    else:
        return combined_preds

class ModelPredictor:
    """优化的预测器类 - 修复了模型加载和预测逻辑，支持集成模型"""
    
    def __init__(self, target_model="Char Yield"):
        self.target_name = target_model
        self.model_path = None  # 初始化model_path为None
        self.pipeline = None
        self.main_pipeline = None
        self.support_pipeline = None
        self.bias_correction = 1.0  # 初始化偏差校正系数为默认值
        self.correction_factors = None
        self.value_ranges = None
        self.main_weight = 0.8
        self.support_weight = 0.2
        self.is_ensemble = False
        
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
        cached_model = self._get_cached_model()
        if cached_model is not None:
            log(f"从缓存加载{self.target_name}模型")
            self._process_cached_model(cached_model)
        else:
            self.model_loaded = False
            log(f"从缓存未找到模型，尝试加载{self.target_name}模型")
            # 查找并加载模型
            self.model_path = self._find_model_file()
            if self.model_path:
                self._load_pipeline()
    
    def _process_cached_model(self, cached_model):
        """处理缓存中的模型数据"""
        if isinstance(cached_model, dict):
            # 检查是否是集成模型
            if 'main_pipeline' in cached_model and 'support_pipeline' in cached_model:
                self.main_pipeline = cached_model['main_pipeline']
                self.support_pipeline = cached_model['support_pipeline']
                self.pipeline = self.main_pipeline  # 兼容旧代码
                self.is_ensemble = True
                
                # 获取校正相关信息
                if 'correction_factors' in cached_model:
                    self.correction_factors = cached_model['correction_factors']
                if 'value_ranges' in cached_model:
                    self.value_ranges = cached_model['value_ranges']
                self.main_weight = cached_model.get('main_weight', 0.8)
                self.support_weight = cached_model.get('support_weight', 0.2)
                
                log(f"从缓存加载集成模型 (校正因子: {len(self.correction_factors) if self.correction_factors else 0})")
                self.model_loaded = True
            elif 'pipeline' in cached_model:
                self.pipeline = cached_model['pipeline']
                if 'bias_correction' in cached_model:
                    self.bias_correction = cached_model['bias_correction']
                    log(f"从缓存加载偏差校正系数: {self.bias_correction}")
                self.is_ensemble = False
                self.model_loaded = True
            else:
                self.model_loaded = False
        else:
            self.pipeline = cached_model
            self.is_ensemble = False
            self.model_loaded = True
            
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
        """加载Pipeline模型 - 修复后的版本，支持集成模型"""
        if not self.model_path:
            log("模型路径为空，无法加载")
            return False
        
        try:
            log(f"加载Pipeline模型: {self.model_path}")
            
            # 注册可能会用到的函数到全局命名空间
            import sys
            sys.modules['__main__'].bias_corrected_predict = bias_corrected_predict
            sys.modules['__main__'].apply_ranged_correction = apply_ranged_correction
            sys.modules['__main__'].ensemble_predict = ensemble_predict
            
            try:
                # 尝试加载模型
                model_data = joblib.load(self.model_path)
                log("模型加载成功")
            except Exception as e:
                log(f"常规加载失败: {str(e)}")
                log("尝试使用更安全的方式加载...")
                
                # 尝试使用pickle直接加载
                import pickle
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                log("使用pickle成功加载模型")
            
            # 检查是否是字典形式 - 处理集成模型结构
            if isinstance(model_data, dict):
                log("检测到保存的模型是字典格式")
                
                # 检查是否是集成模型
                if 'main_pipeline' in model_data and 'support_pipeline' in model_data:
                    log("检测到集成模型结构，包括主模型和支持模型")
                    self.main_pipeline = model_data['main_pipeline']
                    self.support_pipeline = model_data['support_pipeline']
                    self.pipeline = self.main_pipeline  # 兼容旧代码
                    self.is_ensemble = True
                    
                    # 获取校正因子
                    if 'correction_factors' in model_data:
                        self.correction_factors = model_data['correction_factors']
                        log(f"加载了{len(self.correction_factors)}个分段校正因子")
                    else:
                        log("未找到分段校正因子，将使用默认校正")
                        
                    # 获取值域范围
                    if 'value_ranges' in model_data:
                        self.value_ranges = model_data['value_ranges']
                        log(f"加载了{len(self.value_ranges)}个值域范围")
                    
                    # 获取权重
                    self.main_weight = model_data.get('main_weight', 0.8)
                    self.support_weight = model_data.get('support_weight', 0.2)
                    log(f"主模型权重: {self.main_weight}, 支持模型权重: {self.support_weight}")
                    
                    self.model_loaded = True
                    # 将模型保存到缓存
                    st.session_state.model_cache[self.target_name] = model_data
                    return True
                
                # 处理常规Pipeline模型
                elif 'pipeline' in model_data:
                    self.pipeline = model_data['pipeline']
                    log("从字典中提取pipeline成功")
                    
                    # 提取偏差校正系数
                    if 'bias_correction' in model_data:
                        self.bias_correction = model_data['bias_correction']
                        log(f"提取到偏差校正系数: {self.bias_correction}")
                    else:
                        self.bias_correction = 1.0
                        log("未找到偏差校正系数，使用默认值1.0")
                    
                    self.is_ensemble = False
                    self.model_loaded = True
                    # 将模型保存到缓存中
                    st.session_state.model_cache[self.target_name] = model_data
                    
                    # 尝试识别Pipeline的组件
                    if hasattr(self.pipeline, 'named_steps'):
                        components = list(self.pipeline.named_steps.keys())
                        log(f"Pipeline组件: {', '.join(components)}")
                    return True
                else:
                    log("模型字典中未找到pipeline键或集成模型结构")
                    # 尝试使用其他键作为pipeline
                    for key, value in model_data.items():
                        if hasattr(value, 'predict'):
                            log(f"使用'{key}'作为pipeline")
                            self.pipeline = value
                            self.is_ensemble = False
                            self.model_loaded = True
                            st.session_state.model_cache[self.target_name] = {'pipeline': value, 'bias_correction': 1.0}
                            return True
                    
                    self.model_loaded = False
                    return False
            # 直接是pipeline对象
            elif hasattr(model_data, 'predict'):
                log("加载的是预测器对象")
                self.pipeline = model_data
                self.bias_correction = 1.0  # 默认值
                self.is_ensemble = False
                self.model_loaded = True
                st.session_state.model_cache[self.target_name] = {'pipeline': model_data, 'bias_correction': 1.0}
                return True
            else:
                log(f"无法识别的模型格式: {type(model_data)}")
                self.model_loaded = False
                return False
                    
        except Exception as e:
            log(f"加载模型出错: {str(e)}")
            log(traceback.format_exc())
            
            # 尝试使用备用方法加载
            try:
                from sklearn.ensemble import GradientBoostingRegressor
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import RobustScaler
                
                log("尝试使用pickle手动解析模型文件")
                import pickle
                
                try:
                    # 使用一个安全方式加载
                    with open(self.model_path, 'rb') as f:
                        pickle_data = f.read()
                    
                    # 使用修改后的数据加载模型
                    import io
                    model_data = pickle.load(io.BytesIO(pickle_data))
                    
                    if isinstance(model_data, dict):
                        # 检查是否是集成模型
                        if 'main_pipeline' in model_data and 'support_pipeline' in model_data:
                            log("检测到集成模型结构")
                            self.main_pipeline = model_data['main_pipeline']
                            self.support_pipeline = model_data['support_pipeline']
                            self.pipeline = self.main_pipeline
                            self.is_ensemble = True
                            
                            if 'correction_factors' in model_data:
                                self.correction_factors = model_data['correction_factors']
                            if 'value_ranges' in model_data:
                                self.value_ranges = model_data['value_ranges']
                            
                            self.main_weight = model_data.get('main_weight', 0.8)
                            self.support_weight = model_data.get('support_weight', 0.2)
                            
                            log("手动解析集成模型成功")
                            self.model_loaded = True
                            return True
                            
                        elif 'pipeline' in model_data:
                            self.pipeline = model_data['pipeline']
                            if 'bias_correction' in model_data:
                                self.bias_correction = model_data['bias_correction']
                            log("手动解析常规模型成功")
                            self.model_loaded = True
                            return True
                except:
                    log("手动解析失败，创建替代模型")
                    self.pipeline = Pipeline([
                        ('scaler', RobustScaler()),
                        ('model', GradientBoostingRegressor(random_state=42))
                    ])
                    self.bias_correction = 1.0
                    self.is_ensemble = False
                    self.model_loaded = True
                    st.warning("原始模型无法加载，使用了替代模型。预测结果不准确，仅用于演示界面功能。")
                    return True
            except:
                log("所有加载尝试都失败")
                self.model_loaded = False
                return False
    
    def _set_training_ranges(self):
        """设置训练数据的范围 - 使用正确的特征名称，移除O(wt%)"""
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
    
    def _check_features_changed(self, features):
        """检查当前特征是否与上次预测的特征有变化"""
        if not self.last_features:
            return True
            
        for feature, value in features.items():
            if feature in self.last_features and abs(self.last_features[feature] - value) > 0.001:
                return True
        return False
    
    def _apply_ranged_correction(self, prediction):
        """应用分段校正到单个预测值"""
        # 这个方法用于单个预测值
        if not self.correction_factors or not self.value_ranges:
            return prediction * self.bias_correction
            
        # 查找适当的校正因子
        for low, high in self.value_ranges:
            if low <= prediction < high:
                correction = self.correction_factors.get((low, high), self.bias_correction)
                log(f"对值 {prediction:.2f} 应用范围 {low}-{high} 的校正系数 {correction:.4f}")
                return prediction * correction
                
        # 如果没有匹配的范围，使用默认校正
        log(f"未找到匹配的校正范围，使用默认系数 {self.bias_correction}")
        return prediction * self.bias_correction
    
    def predict(self, features):
        """预测方法 - 修复后的版本，支持集成模型和分段校正"""
        # 检查输入是否有变化
        features_changed = self._check_features_changed(features)
        
        # 如果输入没有变化且有上次结果，直接返回上次结果
        if not features_changed and self.last_result is not None:
            log("输入未变化，使用上次的预测结果")
            return self.last_result
        
        # 保存当前特征
        self.last_features = features.copy()
        
        # 准备特征数据
        log(f"开始准备{len(features)}个特征数据")
        features_df = self._prepare_features(features)
        
        # 尝试使用模型进行预测
        if self.model_loaded:
            try:
                log("使用模型预测...")
                
                # 检查是否是集成模型
                if self.is_ensemble and self.main_pipeline is not None and self.support_pipeline is not None:
                    log("检测到集成模型，使用主模型和支持模型进行预测")
                    
                    # 获取两个模型的预测
                    main_preds = self.main_pipeline.predict(features_df)[0]
                    support_preds = self.support_pipeline.predict(features_df)[0]
                    
                    # 组合预测结果
                    raw_prediction = main_preds * self.main_weight + support_preds * self.support_weight
                    log(f"主模型预测: {main_preds:.2f}, 支持模型预测: {support_preds:.2f}, 加权组合: {raw_prediction:.2f}")
                    
                    # 应用分段校正
                    if self.correction_factors and self.value_ranges:
                        result = self._apply_ranged_correction(raw_prediction)
                        log(f"分段校正后结果: {result:.2f}")
                    else:
                        # 应用全局偏差校正
                        result = raw_prediction * self.bias_correction
                        log(f"全局校正后结果: {result:.2f}")
                else:
                    # 使用单一模型
                    log("使用单一模型预测")
                    raw_prediction = self.pipeline.predict(features_df)[0]
                    result = raw_prediction * self.bias_correction
                    log(f"校正后预测结果: {result:.2f} (原始: {raw_prediction:.2f})")
                
                self.last_result = result
                return result
                
            except Exception as e:
                log(f"预测失败: {str(e)}")
                log(traceback.format_exc())
                # 如果失败，尝试重新加载
                if self._load_pipeline():
                    try:
                        log("重新加载模型成功，再次尝试预测")
                        # 根据模型类型选择预测方法
                        if self.is_ensemble:
                            main_preds = self.main_pipeline.predict(features_df)[0]
                            support_preds = self.support_pipeline.predict(features_df)[0]
                            raw_prediction = main_preds * self.main_weight + support_preds * self.support_weight
                            result = self._apply_ranged_correction(raw_prediction)
                        else:
                            raw_prediction = self.pipeline.predict(features_df)[0]
                            result = raw_prediction * self.bias_correction
                        
                        log(f"重新加载后预测结果: {result:.2f}")
                        self.last_result = result
                        return result
                    except Exception as new_e:
                        log(f"重新加载后预测仍然失败: {str(new_e)}")
        
        # 如果到这里，说明预测失败
        log("所有预测尝试都失败，请检查模型文件和特征名称")
        raise ValueError("模型预测失败。请确保模型文件存在且特征格式正确。")
    
    def get_model_info(self):
        """获取模型信息摘要"""
        info = {
            "模型类型": "GBDT集成模型" if self.is_ensemble else "GBDT模型",
            "目标变量": self.target_name,
            "特征数量": len(self.feature_names),
            "模型状态": "已加载" if self.model_loaded else "未加载"
        }
        
        if self.is_ensemble:
            info["模型结构"] = "集成模型 (主模型 + 支持模型)"
            info["主模型权重"] = f"{self.main_weight:.2f}"
            info["支持模型权重"] = f"{self.support_weight:.2f}"
            
            if self.correction_factors:
                info["校正方式"] = f"分段校正 ({len(self.correction_factors)}个范围)"
            else:
                info["校正方式"] = f"全局校正 (系数:{self.bias_correction:.4f})"
        else:
            info["偏差校正系数"] = f"{self.bias_correction:.4f}"
        
        if self.model_loaded:
            if self.is_ensemble and hasattr(self.main_pipeline, 'named_steps'):
                pipeline_steps = list(self.main_pipeline.named_steps.keys())
                info["Pipeline组件"] = ", ".join(pipeline_steps)
                
                # 如果有模型组件，显示其参数
                if 'model' in self.main_pipeline.named_steps:
                    model = self.main_pipeline.named_steps['model']
                    model_type = type(model).__name__
                    info["回归器类型"] = model_type
                    
                    # 显示部分关键超参数
                    if hasattr(model, 'n_estimators'):
                        info["树的数量"] = model.n_estimators
                    if hasattr(model, 'max_depth'):
                        info["最大深度"] = model.max_depth
            elif not self.is_ensemble and hasattr(self.pipeline, 'named_steps'):
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

# 定义默认值 - 从图表中提取均值作为默认值，已移除O(wt%)
default_values = {
    "M(wt%)": 6.57,
    "Ash(wt%)": 5.87,
    "VM(wt%)": 74.22,
    "FC(wt%)": 13.32,
    "C(wt%)": 45.12,
    "H(wt%)": 5.95,
    "N(wt%)": 1.50,
    "PS(mm)": 1.23,
    "SM(g)": 27.03,
    "FT(°C)": 505.24,
    "HR(°C/min)": 27.81,
    "FR(mL/min)": 87.42,
    "RT(min)": 36.88
}

# 特征分类 - 已从Ultimate Analysis中移除O(wt%)
feature_categories = {
    "Proximate Analysis": ["M(wt%)", "Ash(wt%)", "VM(wt%)", "FC(wt%)"],
    "Ultimate Analysis": ["C(wt%)", "H(wt%)", "N(wt%)"],
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

# Ultimate Analysis - 第二列（续）
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
        model_type = "集成模型 (主模型+支持模型)" if predictor.is_ensemble else "单一模型"
        correction_info = f"分段校正 ({len(predictor.correction_factors) if predictor.correction_factors else 0}个范围)" if predictor.is_ensemble and predictor.correction_factors else f"全局校正系数: {predictor.bias_correction:.4f}"
        
        st.markdown(f"""
        - **目标变量:** {st.session_state.selected_model}
        - **预测结果:** {st.session_state.prediction_result:.2f} wt%
        - **模型类型:** {model_type}
        - **校正方式:** {correction_info}
        """)
    
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

# 添加页脚
st.markdown("---")
footer = """
<div style='text-align: center;'>
<p>© 2024 生物质纳米材料与智能装备实验室. 版本: 5.2.0</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)