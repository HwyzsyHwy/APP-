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

# 初始化会话状态 - 必须在最开始
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = "预测"
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Char Yield"
if 'model_cache' not in st.session_state:
    st.session_state.model_cache = {}

# 定义日志函数
def add_log(message):
    """添加日志消息到会话状态"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    st.session_state.log_messages.append(log_entry)
    if len(st.session_state.log_messages) > 100:
        st.session_state.log_messages = st.session_state.log_messages[-100:]

def display_logs():
    """显示日志"""
    if st.session_state.log_messages:
        log_content = '<br>'.join(st.session_state.log_messages)
        st.markdown(
            f"<div class='log-container'>{log_content}</div>", 
            unsafe_allow_html=True
        )

# 自定义样式
st.markdown("""
<style>
html, body, [class*="css"] {
    font-size: 16px !important;
}

.main-title {
    text-align: center;
    font-size: 32px !important;
    font-weight: bold;
    margin-bottom: 20px;
    color: white !important;
}

.section-header {
    color: white;
    font-weight: bold;
    font-size: 22px;
    text-align: center;
    padding: 10px;
    border-radius: 8px;
    margin-bottom: 15px;
}

.input-label {
    padding: 5px;
    border-radius: 5px;
    margin-bottom: 5px;
    font-size: 18px;
    color: white;
}

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

[data-testid="stNumberInput"] input {
    background-color: white !important;
    color: black !important;
}

.stButton button {
    font-size: 18px !important;
}

.warning-box {
    background-color: rgba(255, 165, 0, 0.2);
    border-left: 5px solid orange;
    padding: 10px;
    margin: 10px 0;
    border-radius: 5px;
}

.error-box {
    background-color: rgba(255, 0, 0, 0.2);
    border-left: 5px solid red;
    padding: 10px;
    margin: 10px 0;
    border-radius: 5px;
}

.log-container {
    background-color: #1E1E1E;
    color: #00FF00;
    font-family: 'Courier New', monospace;
    font-size: 12px;
    padding: 10px;
    border-radius: 5px;
    max-height: 300px;
    overflow-y: auto;
    white-space: pre-wrap;
}

.sidebar-model-info {
    background-color: #2E2E2E;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
}

.sidebar-model-info h3 {
    color: #4CAF50;
    margin-bottom: 10px;
}

.sidebar-model-info p {
    color: white;
    margin: 5px 0;
    font-size: 14px;
}

.tech-info {
    background-color: #2E2E2E;
    padding: 15px;
    border-radius: 8px;
    color: white;
}

.tech-info h4 {
    color: #4CAF50;
    margin-top: 15px;
    margin-bottom: 10px;
}

.model-selector {
    background-color: #2E2E2E;
    padding: 15px;
    border-radius: 8px;
    margin-bottom: 20px;
}

.model-selector h3 {
    color: white;
    text-align: center;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# 记录启动日志
add_log("应用启动 - 根据图片特征统计信息正确修复版本")
add_log("特征顺序：M, Ash, VM, O/C, H/C, N/C, FT, HR, FR")
add_log(f"初始化选定模型: {st.session_state.selected_model}")

# 侧边栏导航
with st.sidebar:
    st.markdown("### 🧭 导航菜单")
    
    if st.button("🔮 预测", use_container_width=True, type="primary" if st.session_state.current_page == "预测" else "secondary"):
        st.session_state.current_page = "预测"
        st.rerun()
    
    if st.button("🤖 模型信息", use_container_width=True, type="primary" if st.session_state.current_page == "模型信息" else "secondary"):
        st.session_state.current_page = "模型信息"
        st.rerun()
    
    if st.button("🔬 技术说明", use_container_width=True, type="primary" if st.session_state.current_page == "技术说明" else "secondary"):
        st.session_state.current_page = "技术说明"
        st.rerun()
    
    if st.button("📋 使用指南", use_container_width=True, type="primary" if st.session_state.current_page == "使用指南" else "secondary"):
        st.session_state.current_page = "使用指南"
        st.rerun()
    
    st.markdown("### 📋 执行日志")
    display_logs()

# 根据当前页面显示不同内容
if st.session_state.current_page == "预测":
    st.markdown("<h1 class='main-title'>基于GBDT集成模型的生物质热解产物预测系统</h1>", unsafe_allow_html=True)

    # 模型选择区域
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

    # 处理模型选择
    if char_button and st.session_state.selected_model != "Char Yield":
        st.session_state.selected_model = "Char Yield"
        st.session_state.prediction_result = None
        st.session_state.warnings = []
        add_log(f"切换到模型: {st.session_state.selected_model}")
        st.rerun()

    if oil_button and st.session_state.selected_model != "Oil Yield":
        st.session_state.selected_model = "Oil Yield"
        st.session_state.prediction_result = None
        st.session_state.warnings = []
        add_log(f"切换到模型: {st.session_state.selected_model}")
        st.rerun()

    if gas_button and st.session_state.selected_model != "Gas Yield":
        st.session_state.selected_model = "Gas Yield"
        st.session_state.prediction_result = None
        st.session_state.warnings = []
        add_log(f"切换到模型: {st.session_state.selected_model}")
        st.rerun()

    st.markdown(f"<p style='text-align:center;'>当前模型: <b>{st.session_state.selected_model}</b></p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    class ModelPredictor:
        """根据图片特征统计信息正确调整的预测器类"""
        
        def __init__(self, target_model="Char Yield"):
            self.target_name = target_model
            
            self.feature_names = [
                'M(wt%)', 'Ash(wt%)', 'VM(wt%)', 'O/C', 'H/C', 'N/C',
                'FT(℃)', 'HR(℃/min)', 'FR(mL/min)'
            ]
            
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
            
            self.ui_to_model_mapping = {
                'FT(°C)': 'FT(℃)',
                'HR(°C/min)': 'HR(℃/min)'
            }
            
            self.last_features = {}
            self.last_result = None
            
            self.pipeline = self._get_cached_model()
            self.model_loaded = self.pipeline is not None
            
            if not self.model_loaded:
                add_log(f"从缓存未找到模型，尝试加载{self.target_name}模型")
                self.model_path = self._find_model_file()
                if self.model_path:
                    self._load_pipeline()
        
        def _get_cached_model(self):
            if self.target_name in st.session_state.model_cache:
                add_log(f"从缓存加载{self.target_name}模型")
                return st.session_state.model_cache[self.target_name]
            return None
            
        def _find_model_file(self):
            model_file_patterns = {
                "Char Yield": ["GBDT-Char Yield-improved.joblib", "GBDT-Char-improved.joblib", "*char*.joblib"],
                "Oil Yield": ["GBDT-Oil Yield-improved.joblib", "GBDT-Oil-improved.joblib", "*oil*.joblib"],
                "Gas Yield": ["GBDT-Gas Yield-improved.joblib", "GBDT-Gas-improved.joblib", "*gas*.joblib"]
            }
            
            search_dirs = [".", "./models", "../models", "/app/models", "/app"]
            patterns = model_file_patterns.get(self.target_name, [])
            add_log(f"搜索{self.target_name}模型文件，模式: {patterns}")
            
            for directory in search_dirs:
                if not os.path.exists(directory):
                    continue
                    
                try:
                    for pattern in patterns:
                        matches = glob.glob(os.path.join(directory, pattern))
                        for match in matches:
                            if os.path.isfile(match):
                                add_log(f"找到模型文件: {match}")
                                return match
                                
                    for file in os.listdir(directory):
                        if file.endswith('.joblib'):
                            model_id = self.target_name.split(" ")[0].lower()
                            if model_id in file.lower():
                                model_path = os.path.join(directory, file)
                                add_log(f"找到匹配的模型文件: {model_path}")
                                return model_path
                except Exception as e:
                    add_log(f"搜索目录{directory}时出错: {str(e)}")
            
            add_log(f"未找到{self.target_name}模型文件")
            return None
        
        def _load_pipeline(self):
            if not self.model_path:
                add_log("模型路径为空，无法加载")
                return False
            
            try:
                add_log(f"加载Pipeline模型: {self.model_path}")
                self.pipeline = joblib.load(self.model_path)
                
                if hasattr(self.pipeline, 'predict') and hasattr(self.pipeline, 'named_steps'):
                    add_log(f"Pipeline加载成功，组件: {list(self.pipeline.named_steps.keys())}")
                    
                    if 'scaler' in self.pipeline.named_steps and 'model' in self.pipeline.named_steps:
                        scaler_type = type(self.pipeline.named_steps['scaler']).__name__
                        model_type = type(self.pipeline.named_steps['model']).__name__
                        add_log(f"Scaler类型: {scaler_type}, Model类型: {model_type}")
                        
                        self.model_loaded = True
                        st.session_state.model_cache[self.target_name] = self.pipeline
                        return True
                    else:
                        add_log("Pipeline结构不符合预期，缺少scaler或model组件")
                        return False
                else:
                    add_log("加载的对象不是有效的Pipeline")
                    return False
                    
            except Exception as e:
                add_log(f"加载模型出错: {str(e)}")
                self.model_loaded = False
                return False
        
        def check_input_range(self, features):
            warnings = []
            
            for feature, value in features.items():
                mapped_feature = self.ui_to_model_mapping.get(feature, feature)
                range_info = self.training_ranges.get(mapped_feature)
                
                if range_info:
                    if value < range_info['min'] or value > range_info['max']:
                        warning = f"{feature}: {value:.3f} (超出训练范围 {range_info['min']:.3f} - {range_info['max']:.3f})"
                        warnings.append(warning)
                        add_log(f"警告: {warning}")
            
            return warnings
        
        def _prepare_features(self, features):
            model_features = {}
            
            for ui_feature, value in features.items():
                model_feature = self.ui_to_model_mapping.get(ui_feature, ui_feature)
                if model_feature in self.feature_names:
                    model_features[model_feature] = value
                    if ui_feature != model_feature:
                        add_log(f"特征映射: '{ui_feature}' -> '{model_feature}'")
            
            feature_defaults = {
                'M(wt%)': 6.430226, 'Ash(wt%)': 4.498340, 'VM(wt%)': 75.375509,
                'O/C': 0.715385, 'H/C': 1.534106, 'N/C': 0.034083,
                'FT(℃)': 505.811321, 'HR(℃/min)': 29.011321, 'FR(mL/min)': 93.962264
            }
            
            for feature in self.feature_names:
                if feature not in model_features:
                    default_value = feature_defaults.get(feature, 0.0)
                    model_features[feature] = default_value
                    add_log(f"警告: 特征 '{feature}' 缺失，设为默认值: {default_value}")
            
            df = pd.DataFrame([model_features])
            df = df[self.feature_names]
            
            add_log(f"准备好的特征DataFrame形状: {df.shape}, 列: {list(df.columns)}")
            return df
        
        def predict(self, features):
            features_changed = False
            if self.last_features:
                for feature, value in features.items():
                    if feature not in self.last_features or abs(self.last_features[feature] - value) > 0.001:
                        features_changed = True
                        break
            else:
                features_changed = True
            
            if not features_changed and self.last_result is not None:
                add_log("输入未变化，使用上次的预测结果")
                return self.last_result
            
            self.last_features = features.copy()
            
            add_log(f"开始准备{len(features)}个特征数据进行预测")
            features_df = self._prepare_features(features)
            
            if self.model_loaded and self.pipeline is not None:
                try:
                    add_log("使用Pipeline进行预测（包含RobustScaler预处理）")
                    result = float(self.pipeline.predict(features_df)[0])
                    add_log(f"预测成功: {result:.4f}")
                    self.last_result = result
                    return result
                except Exception as e:
                    add_log(f"Pipeline预测失败: {str(e)}")
                    
                    if self._find_model_file() and self._load_pipeline():
                        try:
                            result = float(self.pipeline.predict(features_df)[0])
                            add_log(f"重新加载后预测成功: {result:.4f}")
                            self.last_result = result
                            return result
                        except Exception as new_e:
                            add_log(f"重新加载后预测仍然失败: {str(new_e)}")
            
            add_log("所有预测尝试都失败")
            raise ValueError(f"模型预测失败。请确保模型文件存在且格式正确。当前模型: {self.target_name}")
        
        def get_model_info(self):
            return {
                "模型类型": "GBDT Pipeline (RobustScaler + GradientBoostingRegressor)",
                "目标变量": self.target_name,
                "特征数量": len(self.feature_names),
                "模型状态": "已加载" if self.model_loaded else "未加载"
            }

    # 初始化预测器
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

    # 默认值
    default_values = {
        "M(wt%)": 6.430, "Ash(wt%)": 4.498, "VM(wt%)": 75.376,
        "O/C": 0.715, "H/C": 1.534, "N/C": 0.034,
        "FT(°C)": 505.811, "HR(°C/min)": 29.011, "FR(mL/min)": 93.962
    }

    feature_categories = {
        "Proximate Analysis": ["M(wt%)", "Ash(wt%)", "VM(wt%)"],
        "Ultimate Analysis": ["O/C", "H/C", "N/C"],
        "Pyrolysis Conditions": ["FT(°C)", "HR(°C/min)", "FR(mL/min)"]
    }

    category_colors = {
        "Ultimate Analysis": "#501d8a",  
        "Proximate Analysis": "#1c8041",  
        "Pyrolysis Conditions": "#e55709" 
    }

    # 创建三列布局
    col1, col2, col3 = st.columns(3)
    features = {}

    # Proximate Analysis
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
                    "", value=float(value), step=0.01,
                    key=f"{category}_{feature}", format="%.3f",
                    label_visibility="collapsed"
                )

    # Ultimate Analysis
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
                    "", value=float(value), step=0.001,
                    key=f"{category}_{feature}", format="%.3f",
                    label_visibility="collapsed"
                )

    # Pyrolysis Conditions
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
                if feature == "FT(°C)":
                    step, format_str = 1.0, "%.1f"
                elif feature == "FR(mL/min)":
                    step, format_str = 1.0, "%.1f"
                else:
                    step, format_str = 0.1, "%.2f"
                
                features[feature] = st.number_input(
                    "", value=float(value), step=step,
                    key=f"{category}_{feature}", format=format_str,
                    label_visibility="collapsed"
                )

    # 调试信息
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

    # 预测按钮区域
    col1, col2 = st.columns([1, 1])

    with col1:
        predict_clicked = st.button("🔮 运行预测", use_container_width=True, type="primary")
        if predict_clicked:
            add_log("开始预测流程...")
            
            if predictor.target_name != st.session_state.selected_model:
                add_log(f"检测到模型变更，重新初始化预测器: {st.session_state.selected_model}")
                predictor = ModelPredictor(target_model=st.session_state.selected_model)
            
            st.session_state.feature_values = features.copy()
            add_log(f"开始{st.session_state.selected_model}预测，输入特征数: {len(features)}")
            
            warnings = predictor.check_input_range(features)
            st.session_state.warnings = warnings
            
            try:
                if not predictor.model_loaded:
                    add_log("模型未加载，尝试重新加载")
                    if predictor._find_model_file() and predictor._load_pipeline():
                        add_log("重新加载模型成功")
                    else:
                        error_msg = f"无法加载{st.session_state.selected_model}模型。请确保模型文件存在于正确位置。"
                        st.error(error_msg)
                        st.session_state.prediction_error = error_msg
                        st.rerun()
                
                result = predictor.predict(features)
                if result is not None:
                    st.session_state.prediction_result = float(result)
                    add_log(f"预测成功: {st.session_state.prediction_result:.4f}")
                    st.session_state.prediction_error = None
                else:
                    add_log("警告: 预测结果为空")
                    st.session_state.prediction_error = "预测结果为空"
                    
            except Exception as e:
                error_msg = f"预测过程中发生错误: {str(e)}"
                st.session_state.prediction_error = error_msg
                add_log(f"预测错误: {str(e)}")
                st.error(error_msg)

    with col2:
        if st.button("🔄 重置输入", use_container_width=True):
            add_log("重置所有输入值")
            st.session_state.clear_pressed = True
            st.session_state.prediction_result = None
            st.session_state.warnings = []
            st.session_state.prediction_error = None
            st.rerun()

    # 显示预测结果
    if st.session_state.prediction_result is not None:
        st.markdown("---")
        
        st.markdown(
            f"<div class='yield-result'>{st.session_state.selected_model}: {st.session_state.prediction_result:.2f} wt%</div>", 
            unsafe_allow_html=True
        )
        
        if not predictor.model_loaded:
            st.markdown(
                "<div class='error-box'><b>⚠️ 错误：</b> 模型未成功加载，无法执行预测。请检查模型文件是否存在。</div>", 
                unsafe_allow_html=True
            )
        
        if st.session_state.warnings:
            warnings_html = "<div class='warning-box'><b>⚠️ 输入警告</b><ul>"
            for warning in st.session_state.warnings:
                warnings_html += f"<li>{warning}</li>"
            warnings_html += "</ul><p><i>建议调整输入值以获得更准确的预测结果。</i></p></div>"
            st.markdown(warnings_html, unsafe_allow_html=True)
        
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

elif st.session_state.current_page == "模型信息":
    st.markdown('<div class="main-title">模型信息</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.subheader(f"🤖 当前模型: {st.session_state.selected_model}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**基本信息:**")
        st.write("• 模型类型: GBDT Pipeline")
        st.write("• 预处理: RobustScaler + GradientBoostingRegressor")
        if 'prediction_result' in st.session_state and st.session_state.prediction_result is not None:
            st.write(f"• 预测结果: {st.session_state.prediction_result:.4f} wt%")
        st.write("• 特征数量: 9个输入特征")
        st.write("• 模型状态: 🟢 正常运行")
    
    with col2:
        st.write("**支持的预测目标:**")
        st.write("• 🔥 **Char Yield:** 焦炭产率预测")
        st.write("• 🛢️ **Oil Yield:** 生物油产率预测")
        st.write("• 💨 **Gas Yield:** 气体产率预测")

elif st.session_state.current_page == "技术说明":
    st.markdown('<div class="main-title">技术说明</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.subheader("🔬 算法原理")
    st.write("本系统基于**梯度提升决策树(GBDT)**算法构建，采用Pipeline架构集成数据预处理和模型预测。")
    
    st.subheader("🏗️ 系统架构")
    st.write("• **数据预处理:** RobustScaler标准化，对异常值具有较强的鲁棒性")
    st.write("• **机器学习模型:** GradientBoostingRegressor，通过集成多个弱学习器提高预测精度")
    st.write("• **Pipeline集成:** 自动化的数据流处理，确保预测的一致性和可靠性")

elif st.session_state.current_page == "使用指南":
    st.markdown('<div class="main-title">使用指南</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.subheader("📋 操作步骤")
    st.write("1. **选择预测目标:** 点击Char Yield、Oil Yield或Gas Yield按钮选择要预测的产物")
    st.write("2. **输入特征参数:** 在三个特征组中输入相应的数值")
    st.write("3. **执行预测:** 点击"运行预测"按钮获得预测结果")
    st.write("4. **查看结果:** 在右侧面板查看详细的预测信息")

# 添加页脚
st.markdown("---")
footer = """
<div style='text-align: center; color: #666;'>
<p>© 2024 生物质纳米材料与智能装备实验室 | 基于GBDT的生物质热解产物预测系统 | 版本: 6.3.0</p>
<p>🔥 支持Char、Oil、Gas三种产率预测 | 🚀 Pipeline架构 | 📊 实时范围检查</p>
<p>特征顺序: M(wt%) → Ash(wt%) → VM(wt%) → O/C → H/C → N/C → FT(℃) → HR(℃/min) → FR(mL/min)</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)