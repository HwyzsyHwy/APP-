# -*- coding: utf-8 -*-
"""
电化学模型在线预测系统
基于GBDT模型预测I(uA)
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import joblib
import traceback
from datetime import datetime

# 页面设置
st.set_page_config(
    page_title='电化学模型预测系统',
    page_icon='⚡',
    layout='wide',
    initial_sidebar_state='expanded'
)

# 自定义样式
st.markdown(
    """
    <style>
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
    </style>
    """,
    unsafe_allow_html=True
)

# 初始化日志
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []

def log(message):
    """记录日志"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    st.session_state.log_messages.append(log_entry)
    if len(st.session_state.log_messages) > 50:
        st.session_state.log_messages = st.session_state.log_messages[-50:]

# 主标题
st.markdown("<h1 class='main-title'>基于GBDT模型的电化学响应预测系统</h1>", unsafe_allow_html=True)

class ModelPredictor:
    """电化学模型预测器类"""
    
    def __init__(self):
        self.target_name = "I(uA)"
        self.feature_names = [
            'DT(ml)', 'PH', 'SS(mV/s)', 'P(V)', 'TM(min)', 'C0(uM)'
        ]
        
        self.training_ranges = {
            'DT(ml)': {'min': 0.0, 'max': 10.0},
            'PH': {'min': 3.0, 'max': 9.0},
            'SS(mV/s)': {'min': 10.0, 'max': 200.0},
            'P(V)': {'min': -1.0, 'max': 1.0},
            'TM(min)': {'min': 0.0, 'max': 60.0},
            'C0(uM)': {'min': 1.0, 'max': 100.0}
        }
        
        self.model_loaded = False
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        model_paths = [
            "GBDT.joblib",
            "./GBDT.joblib",
            "../GBDT.joblib",
            r"C:\Users\HWY\Desktop\开题-7.2\GBDT.joblib"
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                try:
                    self.pipeline = joblib.load(path)
                    self.model_loaded = True
                    log(f"模型加载成功: {path}")
                    break
                except Exception as e:
                    log(f"加载模型失败: {path}, 错误: {str(e)}")
        
        if not self.model_loaded:
            log("未找到模型文件")
    
    def check_input_range(self, features):
        """检查输入范围"""
        warnings = []
        for feature, value in features.items():
            range_info = self.training_ranges.get(feature)
            if range_info:
                if value < range_info['min'] or value > range_info['max']:
                    warning = f"{feature}: {value:.3f} (建议范围 {range_info['min']:.3f} - {range_info['max']:.3f})"
                    warnings.append(warning)
        return warnings
    
    def predict(self, features):
        """预测"""
        if not self.model_loaded:
            raise ValueError("模型未加载")
        
        # 准备数据
        data = []
        for feature in self.feature_names:
            data.append(features.get(feature, 0.0))
        
        df = pd.DataFrame([data], columns=self.feature_names)
        
        try:
            result = self.pipeline.predict(df)[0]
            return float(result)
        except Exception as e:
            raise ValueError(f"预测失败: {str(e)}")

# 初始化预测器
predictor = ModelPredictor()

# 侧边栏 - 简化版本，移除有问题的模型信息
st.sidebar.markdown("### 模型状态")
if predictor.model_loaded:
    st.sidebar.success("✅ 模型已加载")
else:
    st.sidebar.error("❌ 模型未加载")

st.sidebar.markdown("### 执行日志")
if st.session_state.log_messages:
    for msg in st.session_state.log_messages[-10:]:  # 只显示最近10条
        st.sidebar.text(msg)

# 初始化会话状态
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'warnings' not in st.session_state:
    st.session_state.warnings = []
if 'prediction_error' not in st.session_state:
    st.session_state.prediction_error = None

# 默认值
default_values = {
    "DT(ml)": 5.0,
    "PH": 7.0,
    "SS(mV/s)": 100.0,
    "P(V)": 0.0,
    "TM(min)": 30.0,
    "C0(uM)": 50.0
}

# 特征分类
feature_categories = {
    "电化学参数": ["DT(ml)", "PH"],
    "测量条件": ["SS(mV/s)", "P(V)"],
    "实验参数": ["TM(min)", "C0(uM)"]
}

category_colors = {
    "电化学参数": "#501d8a",  
    "测量条件": "#1c8041",  
    "实验参数": "#e55709" 
}

# 创建三列布局
col1, col2, col3 = st.columns(3)
features = {}

# 第一列
with col1:
    category = "电化学参数"
    color = category_colors[category]
    st.markdown(f"<div class='section-header' style='background-color: {color};'>{category}</div>", unsafe_allow_html=True)
    
    for feature in feature_categories[category]:
        col_a, col_b = st.columns([1, 0.5])
        with col_a:
            st.markdown(f"<div class='input-label' style='background-color: {color};'>{feature}</div>", unsafe_allow_html=True)
        with col_b:
            features[feature] = st.number_input(
                "", 
                value=default_values[feature], 
                step=0.1,
                key=f"input_{feature}",
                format="%.2f",
                label_visibility="collapsed"
            )

# 第二列
with col2:
    category = "测量条件"
    color = category_colors[category]
    st.markdown(f"<div class='section-header' style='background-color: {color};'>{category}</div>", unsafe_allow_html=True)
    
    for feature in feature_categories[category]:
        col_a, col_b = st.columns([1, 0.5])
        with col_a:
            st.markdown(f"<div class='input-label' style='background-color: {color};'>{feature}</div>", unsafe_allow_html=True)
        with col_b:
            if feature == "SS(mV/s)":
                step = 1.0
                format_str = "%.1f"
            else:
                step = 0.01
                format_str = "%.3f"
            
            features[feature] = st.number_input(
                "", 
                value=default_values[feature], 
                step=step,
                key=f"input_{feature}",
                format=format_str,
                label_visibility="collapsed"
            )

# 第三列
with col3:
    category = "实验参数"
    color = category_colors[category]
    st.markdown(f"<div class='section-header' style='background-color: {color};'>{category}</div>", unsafe_allow_html=True)
    
    for feature in feature_categories[category]:
        col_a, col_b = st.columns([1, 0.5])
        with col_a:
            st.markdown(f"<div class='input-label' style='background-color: {color};'>{feature}</div>", unsafe_allow_html=True)
        with col_b:
            features[feature] = st.number_input(
                "", 
                value=default_values[feature], 
                step=1.0,
                key=f"input_{feature}",
                format="%.1f",
                label_visibility="collapsed"
            )

# 预测按钮
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("⚡ 运行预测", use_container_width=True, type="primary"):
        log("开始预测流程")
        
        warnings = predictor.check_input_range(features)
        st.session_state.warnings = warnings
        
        try:
            result = predictor.predict(features)
            st.session_state.prediction_result = result
            st.session_state.prediction_error = None
            log(f"预测成功: {result:.4f}")
            
        except Exception as e:
            error_msg = f"预测失败: {str(e)}"
            st.session_state.prediction_error = error_msg
            st.session_state.prediction_result = None
            log(error_msg)

with col2:
    if st.button("🔄 重置输入", use_container_width=True):
        st.rerun()

# 显示结果
if st.session_state.prediction_result is not None:
    st.markdown("---")
    st.markdown(
        f"<div class='yield-result'>电流响应 I(uA): {st.session_state.prediction_result:.4f}</div>", 
        unsafe_allow_html=True
    )
    
    if st.session_state.warnings:
        warnings_html = "<div class='warning-box'><b>⚠️ 输入警告</b><ul>"
        for warning in st.session_state.warnings:
            warnings_html += f"<li>{warning}</li>"
        warnings_html += "</ul></div>"
        st.markdown(warnings_html, unsafe_allow_html=True)

elif st.session_state.prediction_error is not None:
    st.markdown("---")
    error_html = f"""
    <div class='error-box'>
        <h3>❌ 预测失败</h3>
        <p><b>错误信息:</b> {st.session_state.prediction_error}</p>
    </div>
    """
    st.markdown(error_html, unsafe_allow_html=True)

# 页脚
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
<p>© 2024 电化学分析实验室 | 基于GBDT的电化学响应预测系统</p>
<p>特征顺序: DT(ml) → PH → SS(mV/s) → P(V) → TM(min) → C0(uM)</p>
</div>
""", unsafe_allow_html=True)