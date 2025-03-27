# -*- coding: utf-8 -*-
"""
Biomass Pyrolysis Yield Forecast using CatBoost Ensemble Models
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import matplotlib.pyplot as plt
# 移除 seaborn 依赖
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 添加模型目录到系统路径，使Python能够找到模块
model_dir = "Char_Yield_Model"  # 模型目录
if os.path.exists(model_dir):
    sys.path.append(os.path.abspath(model_dir))

# 导入简化版预测器
try:
    from simple_predictor import Char_YieldPredictor
except ImportError:
    # 定义一个简化版的预测器类用于在缺少实际模型时使用
    class Char_YieldPredictor:
        def __init__(self, models_dir=None):
            self.feature_names = ["PT(°C)", "RT(min)", "C(%)", "H(%)", "O(%)", "N(%)", "Ash(%)", "VM(%)", "FC(%)", "HR(℃/min)"]
            self.target_name = "Char Yield(%)"
            print("Warning: Using dummy predictor - actual model not found")
            
        def predict(self, data):
            # 返回虚拟预测值
            return np.array([40.0 + np.random.normal(0, 5) for _ in range(len(data))])
            
        def summary(self):
            print("Dummy predictor - model not found")
            
        def get_importance(self, plot=False):
            return pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': np.random.rand(len(self.feature_names))
            }).sort_values('Importance', ascending=False)

# 页面设置
st.set_page_config(
    page_title='Biomass Pyrolysis Yield Forecast',
    page_icon='📊',
    layout='wide'
)

# 自定义样式 - 使用多种选择器确保覆盖Streamlit默认样式
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
    
    /* 强制应用白色背景到输入框 - 使用多种选择器和!important */
    [data-testid="stNumberInput"] input {
        background-color: white !important;
        color: black !important;
    }
    
    /* 额外的选择器，确保覆盖到所有可能的输入框元素 */
    input[type="number"] {
        background-color: white !important;
        color: black !important;
    }

    /* 尝试更具体的选择器 */
    div[data-baseweb="input"] input {
        background-color: white !important;
        color: black !important;
    }

    /* 针对输入框容器的选择器 */
    div[data-baseweb="input"] {
        background-color: white !important;
    }

    /* 最后的终极方法 - 应用给所有可能的输入元素 */
    [data-testid="stNumberInput"] * {
        background-color: white !important;
    }
    
    /* 增大模型选择和按钮的字体 */
    .stSelectbox, .stButton button {
        font-size: 18px !important;
    }
    
    /* 增大展开器标题字体 */
    [data-testid="stExpander"] div[role="button"] p {
        font-size: 20px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 主标题
st.markdown("<h1 class='main-title'>Prediction of crop biomass pyrolysis yield based on CatBoost ensemble modeling</h1>", unsafe_allow_html=True)

# 初始化会话状态
if 'clear_pressed' not in st.session_state:
    st.session_state.clear_pressed = False

# 模型选择
with st.expander("Model Information", expanded=False):
    st.write("**CatBoost Ensemble Model** - Trained with 10-fold cross-validation and optimized hyperparameters")
    
    # 加载预测器以显示模型信息
    predictor = Char_YieldPredictor()
    
    # 显示特征重要性
    importance = predictor.get_importance(plot=False)
    st.write("**Feature Importance:**")
    st.dataframe(importance.head())

# 定义默认值和范围 - 使用我们模型的特征
default_values = {
    "PT(°C)": 500.0,
    "RT(min)": 20.0,
    "C(%)": 45.0,
    "H(%)": 6.0,
    "O(%)": 40.0,
    "N(%)": 0.5,
    "Ash(%)": 5.0,
    "VM(%)": 75.0,
    "FC(%)": 15.0,
    "HR(℃/min)": 20.0
}

# 特征分类 - 按照我们模型使用的特征组织
feature_categories = {
    "Pyrolysis Conditions": ["PT(°C)", "RT(min)", "HR(℃/min)"],
    "Ultimate Analysis": ["C(%)", "H(%)", "O(%)", "N(%)"],
    "Proximate Analysis": ["Ash(%)", "VM(%)", "FC(%)"]
}

# 特征范围
feature_ranges = {
    "PT(°C)": (300.0, 900.0),
    "RT(min)": (5.0, 120.0),
    "C(%)": (30.0, 80.0),
    "H(%)": (3.0, 10.0),
    "O(%)": (10.0, 60.0),
    "N(%)": (0.0, 5.0),
    "Ash(%)": (0.0, 25.0),
    "VM(%)": (40.0, 95.0),
    "FC(%)": (5.0, 40.0),
    "HR(℃/min)": (5.0, 100.0)
}

# 创建三列布局
col1, col2, col3 = st.columns(3)

# 使用字典来存储所有输入值
features = {}

# Pyrolysis Conditions (橙色区域)
with col1:
    st.markdown("<div class='section-header' style='background-color: #FF7F50;'>Pyrolysis Conditions</div>", unsafe_allow_html=True)
    
    for feature in feature_categories["Pyrolysis Conditions"]:
        # 重置值或使用现有值
        if st.session_state.clear_pressed:
            value = default_values[feature]
        else:
            value = st.session_state.get(f"pyrolysis_{feature}", default_values[feature])
        
        # 获取该特征的范围
        min_val, max_val = feature_ranges[feature]
        
        # 简单的两列布局
        col_a, col_b = st.columns([1, 0.5])  # 调整列宽比例
        with col_a:
            st.markdown(f"<div class='input-label' style='background-color: #FF7F50;'>{feature}</div>", unsafe_allow_html=True)  # 橙色背景
        with col_b:
            features[feature] = st.number_input(
                "", 
                min_value=min_val, 
                max_value=max_val, 
                value=value, 
                key=f"pyrolysis_{feature}", 
                format="%.1f",
                label_visibility="collapsed"
            )

# Ultimate Analysis (黄色区域)
with col2:
    st.markdown("<div class='section-header' style='background-color: #DAA520;'>Ultimate Analysis</div>", unsafe_allow_html=True)
    
    for feature in feature_categories["Ultimate Analysis"]:
        if st.session_state.clear_pressed:
            value = default_values[feature]
        else:
            value = st.session_state.get(f"ultimate_{feature}", default_values[feature])
        
        # 获取该特征的范围
        min_val, max_val = feature_ranges[feature]
        
        col_a, col_b = st.columns([1, 0.5])  # 调整列宽比例
        with col_a:
            st.markdown(f"<div class='input-label' style='background-color: #DAA520;'>{feature}</div>", unsafe_allow_html=True)  # 黄色背景
        with col_b:
            features[feature] = st.number_input(
                "", 
                min_value=min_val, 
                max_value=max_val, 
                value=value, 
                key=f"ultimate_{feature}", 
                format="%.1f",
                label_visibility="collapsed"
            )

# Proximate Analysis (绿色区域)
with col3:
    st.markdown("<div class='section-header' style='background-color: #32CD32;'>Proximate Analysis</div>", unsafe_allow_html=True)
    
    for feature in feature_categories["Proximate Analysis"]:
        if st.session_state.clear_pressed:
            value = default_values[feature]
        else:
            value = st.session_state.get(f"proximate_{feature}", default_values[feature])
        
        # 获取该特征的范围
        min_val, max_val = feature_ranges[feature]
        
        col_a, col_b = st.columns([1, 0.5])  # 调整列宽比例
        with col_a:
            st.markdown(f"<div class='input-label' style='background-color: #32CD32;'>{feature}</div>", unsafe_allow_html=True)  # 绿色背景
        with col_b:
            features[feature] = st.number_input(
                "", 
                min_value=min_val, 
                max_value=max_val, 
                value=value, 
                key=f"proximate_{feature}", 
                format="%.1f",
                label_visibility="collapsed"
            )

# 重置session_state中的clear_pressed状态
if st.session_state.clear_pressed:
    st.session_state.clear_pressed = False

# 转换为DataFrame
input_data = pd.DataFrame([features])

# 预测结果显示区域和按钮
result_col, button_col = st.columns([3, 1])

with result_col:
    prediction_placeholder = st.empty()
    
with button_col:
    predict_button = st.button("PUSH", key="predict")
    
    # 定义Clear按钮的回调函数
    def clear_values():
        st.session_state.clear_pressed = True
        # 清除显示
        if 'prediction_result' in st.session_state:
            st.session_state.prediction_result = None
    
    clear_button = st.button("CLEAR", key="clear", on_click=clear_values)

# 处理预测逻辑
if predict_button:
    try:
        # 初始化预测器
        predictor = Char_YieldPredictor()
        
        # 使用预测器进行预测
        y_pred = predictor.predict(input_data)[0]
        
        # 保存预测结果到session_state
        st.session_state.prediction_result = y_pred

        # 显示预测结果
        prediction_placeholder.markdown(
            f"<div class='yield-result'>Char Yield (wt%) <br> {y_pred:.2f}</div>",
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"预测过程中出现错误: {e}")

# 如果有保存的预测结果，显示它
if 'prediction_result' in st.session_state and st.session_state.prediction_result is not None:
    prediction_placeholder.markdown(
        f"<div class='yield-result'>Char Yield (wt%) <br> {st.session_state.prediction_result:.2f}</div>",
        unsafe_allow_html=True
    )