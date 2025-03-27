# -*- coding: utf-8 -*-
"""
Biomass Pyrolysis Yield Forecast using CatBoost Ensemble Models
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# 页面设置
st.set_page_config(
    page_title='Biomass Pyrolysis Yield Forecast',
    page_icon='📊',
    layout='wide'
)

# 添加模型目录到系统路径，确保能找到simple_predictor模块
model_dir = "Char_Yield_Model"  # 模型目录
if os.path.exists(model_dir):
    if model_dir not in sys.path:
        sys.path.append(os.path.abspath(model_dir))

# 尝试导入预测器类
try:
    from simple_predictor import Char_YieldPredictor
    predictor = Char_YieldPredictor()
    model_loaded = True
    st.sidebar.success("🟢 模型加载成功")
    # 打印特征列表，用于调试
    st.sidebar.write("模型特征列表:")
    st.sidebar.write(predictor.feature_names)
except Exception as e:
    model_loaded = False
    st.sidebar.error(f"❌ 模型加载失败: {str(e)}")
    # 定义一个虚拟预测器以避免程序崩溃
    class DummyPredictor:
        def __init__(self):
            self.feature_names = ["PT(°C)", "RT(min)", "C(%)", "H(%)", "O(%)", "N(%)", "Ash(%)", "VM(%)", "FC(%)", "HR(℃/min)"]
        
        def predict(self, data):
            return np.array([30.0])  # 返回一个固定值
    
    predictor = DummyPredictor()

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

# 定义默认值和范围
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

# 特征分类
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
        col_a, col_b = st.columns([1, 0.5])
        with col_a:
            st.markdown(f"<div class='input-label' style='background-color: #FF7F50;'>{feature}</div>", unsafe_allow_html=True)
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
        
        col_a, col_b = st.columns([1, 0.5])
        with col_a:
            st.markdown(f"<div class='input-label' style='background-color: #DAA520;'>{feature}</div>", unsafe_allow_html=True)
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
        
        col_a, col_b = st.columns([1, 0.5])
        with col_a:
            st.markdown(f"<div class='input-label' style='background-color: #32CD32;'>{feature}</div>", unsafe_allow_html=True)
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

# 转换为DataFrame - 确保按照模型需要的特征顺序
feature_df = pd.DataFrame([features])

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

# 调试信息
debug_expander = st.expander("Debug Information", expanded=False)
with debug_expander:
    st.write("Input Features:")
    st.write(feature_df)
    
    if model_loaded:
        st.write("Model Features:")
        st.write(predictor.feature_names)
    else:
        st.write("No model loaded")

# 处理预测逻辑
if predict_button:
    try:
        # 使用预测器进行预测
        if model_loaded:
            # 确保特征顺序正确
            ordered_data = feature_df.copy()
            
            # 如果特征名称格式不同，尝试进行映射
            model_features = predictor.feature_names
            feature_mapping = {}
            
            # 检查是否需要特征名称映射（例如，C(%) 到 C(wt%)）
            for app_feature in feature_df.columns:
                for model_feature in model_features:
                    # 尝试匹配去掉单位等标记后的基本名称
                    app_base = app_feature.split('(')[0]
                    model_base = model_feature.split('(')[0]
                    if app_base == model_base:
                        feature_mapping[app_feature] = model_feature
                        break
            
            # 如果找到映射关系，应用它
            if feature_mapping and len(feature_mapping) == len(feature_df.columns):
                ordered_data = feature_df.rename(columns=feature_mapping)
                st.sidebar.write("应用特征映射:")
                st.sidebar.write(feature_mapping)
            
            # 进行预测
            y_pred = predictor.predict(ordered_data)[0]
            
            # 记录调试信息
            st.session_state.debug_info = {
                'input_features': ordered_data.to_dict('records')[0],
                'prediction': float(y_pred)
            }
        else:
            # 使用简单模拟进行预测
            pt = features["PT(°C)"]
            rt = features["RT(min)"]
            
            # 模拟预测计算
            y_pred = 33.0 - 0.04 * (pt - 400) + 0.2 * rt
            
            # 记录为模拟预测
            st.session_state.debug_info = {
                'note': 'Using simulation prediction (model not loaded)',
                'input_features': features,
                'prediction': float(y_pred)
            }
        
        # 保存预测结果到session_state
        st.session_state.prediction_result = y_pred

        # 显示预测结果
        prediction_placeholder.markdown(
            f"<div class='yield-result'>Char Yield (wt%) <br> {y_pred:.2f}</div>",
            unsafe_allow_html=True
        )
        
        # 在调试区显示详细信息
        with debug_expander:
            st.write("Prediction Details:")
            st.write(st.session_state.debug_info)
            
    except Exception as e:
        st.error(f"预测过程中出现错误: {str(e)}")
        st.exception(e)  # 显示详细错误信息

# 如果有保存的预测结果，显示它
if 'prediction_result' in st.session_state and st.session_state.prediction_result is not None:
    prediction_placeholder.markdown(
        f"<div class='yield-result'>Char Yield (wt%) <br> {st.session_state.prediction_result:.2f}</div>",
        unsafe_allow_html=True
    )

# 添加关于模型的信息
st.markdown("""
### About the Model
This application uses a CatBoost ensemble model to predict char yield in biomass pyrolysis.

#### Key Factors Affecting Char Yield:
- **Pyrolysis Temperature**: Higher temperature generally decreases char yield
- **Residence Time**: Longer residence time generally increases char yield
- **Biomass Composition**: Carbon content and ash content significantly affect the final yield

The model was trained using 10-fold cross-validation with optimized hyperparameters, achieving high prediction accuracy.
""")

# 显示调试信息
if 'debug_info' in st.session_state:
    with debug_expander:
        st.write("Last Prediction Details:")
        st.json(st.session_state.debug_info)