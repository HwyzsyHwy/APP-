# -*- coding: utf-8 -*-
"""
Biomass Pyrolysis Yield Forecast using CatBoost Ensemble Models
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import glob
import joblib
import json
from io import StringIO

# 页面设置
st.set_page_config(
    page_title='Biomass Pyrolysis Yield Forecast',
    page_icon='📊',
    layout='wide'
)

# 增加搜索模型文件的功能
def find_model_files():
    """
    搜索目录中的模型文件和simple_predictor.py
    """
    # 搜索当前目录及子目录中的模型文件
    model_files = glob.glob("**/model_*.joblib", recursive=True)
    scaler_files = glob.glob("**/final_scaler.joblib", recursive=True)
    predictor_files = glob.glob("**/simple_predictor.py", recursive=True)
    
    return {
        "model_files": model_files,
        "scaler_files": scaler_files,
        "predictor_files": predictor_files
    }

# 定义内嵌的简单预测器类
class EmbeddedPredictor:
    """
    内嵌的简单预测器类，实现CatBoost集成模型的基本功能
    """
    def __init__(self):
        # 查找模型文件
        model_info = find_model_files()
        st.sidebar.write("模型文件搜索结果:", model_info)
        
        # 模型和缩放器路径
        self.model_paths = model_info["model_files"]
        self.scaler_paths = model_info["scaler_files"]
        
        # 初始化
        self.models = []
        self.final_scaler = None
        self.model_weights = None
        self.feature_names = ["PT(°C)", "RT(min)", "HR(℃/min)", "C(%)", "H(%)", "O(%)", "N(%)", "Ash(%)", "VM(%)", "FC(%)"]
        
        # 尝试加载模型
        self._load_components()
    
    def _load_components(self):
        """加载模型组件"""
        try:
            # 尝试加载模型文件
            if self.model_paths:
                models_dir = os.path.dirname(self.model_paths[0])
                st.sidebar.success(f"找到模型文件夹: {models_dir}")
                
                # 加载模型
                for model_path in sorted(self.model_paths):
                    st.sidebar.write(f"加载模型: {model_path}")
                    self.models.append(joblib.load(model_path))
                
                # 加载缩放器
                if self.scaler_paths:
                    st.sidebar.write(f"加载缩放器: {self.scaler_paths[0]}")
                    self.final_scaler = joblib.load(self.scaler_paths[0])
                
                # 加载权重
                weights_path = os.path.join(models_dir, "model_weights.npy")
                if os.path.exists(weights_path):
                    st.sidebar.write(f"加载权重: {weights_path}")
                    self.model_weights = np.load(weights_path)
                else:
                    # 如果没有权重文件，使用均等权重
                    self.model_weights = np.ones(len(self.models)) / len(self.models)
                
                # 加载元数据
                metadata_path = os.path.join(models_dir, "metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        if 'feature_names' in metadata:
                            self.feature_names = metadata['feature_names']
                            st.sidebar.write("从元数据加载特征名称")
                
                st.sidebar.success(f"成功加载 {len(self.models)} 个模型")
                return True
            else:
                st.sidebar.warning("未找到模型文件")
                return False
        except Exception as e:
            st.sidebar.error(f"加载模型组件时出错: {str(e)}")
            return False
    
    def predict(self, X):
        """
        使用集成模型进行预测
        """
        try:
            # 确保输入是DataFrame
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            
            # 调试信息
            st.sidebar.write("输入特征:", X.columns.tolist())
            st.sidebar.write("模型特征:", self.feature_names)
            
            # 检查特征是否需要重命名
            if not set(self.feature_names).issubset(set(X.columns)):
                # 尝试映射特征名
                mapped_features = {}
                for model_feat in self.feature_names:
                    for input_feat in X.columns:
                        # 移除单位部分进行比较
                        model_base = model_feat.split('(')[0] if '(' in model_feat else model_feat
                        input_base = input_feat.split('(')[0] if '(' in input_feat else input_feat
                        
                        if model_base == input_base:
                            mapped_features[input_feat] = model_feat
                            break
                
                if len(mapped_features) == len(X.columns):
                    X = X.rename(columns=mapped_features)
                    st.sidebar.success("特征名称已重映射")
                    st.sidebar.write("映射关系:", mapped_features)
            
            # 确保特征顺序正确
            if not all(feat in X.columns for feat in self.feature_names):
                missing = set(self.feature_names) - set(X.columns)
                st.sidebar.error(f"缺少特征: {missing}")
                return np.array([33.0])  # 返回默认值
            
            # 按模型需要的顺序提取特征
            X = X[self.feature_names]
            
            # 如果有缩放器，应用标准化
            if self.final_scaler:
                X_scaled = self.final_scaler.transform(X)
                st.sidebar.success("数据已标准化")
            else:
                X_scaled = X.values
                st.sidebar.warning("没有标准化器，使用原始数据")
            
            # 使用所有模型进行预测
            if self.models:
                all_predictions = np.zeros((X.shape[0], len(self.models)))
                for i, model in enumerate(self.models):
                    pred = model.predict(X_scaled)
                    all_predictions[:, i] = pred
                    st.sidebar.write(f"模型 {i} 预测值: {pred[0]:.2f}")
                
                # 计算加权平均
                weighted_pred = np.sum(all_predictions * self.model_weights.reshape(1, -1), axis=1)
                st.sidebar.success(f"最终加权预测值: {weighted_pred[0]:.2f}")
                return weighted_pred
            else:
                # 如果没有模型，返回基于规则的估计
                st.sidebar.warning("无可用模型，使用简单估计")
                return self._simple_estimate(X)
        except Exception as e:
            st.sidebar.error(f"预测过程中出错: {str(e)}")
            import traceback
            st.sidebar.error(traceback.format_exc())
            return np.array([33.0])  # 返回默认值
    
    def _simple_estimate(self, X):
        """简单估计，在没有模型时使用"""
        # 提取关键特征
        pt = X["PT(°C)"].values[0] if "PT(°C)" in X.columns else 500
        rt = X["RT(min)"].values[0] if "RT(min)" in X.columns else 20
        
        # 基于温度和停留时间的简单估计
        base_yield = 40.0
        temp_effect = -0.03 * (pt - 500)  # 高温降低产率
        time_effect = 0.1 * (rt - 20)     # 更长时间增加产率
        
        estimated_yield = base_yield + temp_effect + time_effect
        estimated_yield = max(20, min(80, estimated_yield))  # 限制在合理范围内
        
        return np.array([estimated_yield])

# 尝试查找simple_predictor模块
found_predictor = False
predictor_files = glob.glob("**/simple_predictor.py", recursive=True)

if predictor_files:
    # 找到了predictor文件
    predictor_path = predictor_files[0]
    predictor_dir = os.path.dirname(os.path.abspath(predictor_path))
    
    # 添加目录到sys.path
    if predictor_dir not in sys.path:
        sys.path.append(predictor_dir)
    
    st.sidebar.success(f"找到predictor文件: {predictor_path}")
    st.sidebar.write(f"添加目录到sys.path: {predictor_dir}")
    
    # 尝试导入
    try:
        import simple_predictor
        from simple_predictor import Char_YieldPredictor
        predictor = Char_YieldPredictor()
        found_predictor = True
        st.sidebar.success("成功导入并实例化Char_YieldPredictor")
    except Exception as e:
        st.sidebar.error(f"导入simple_predictor失败: {str(e)}")
        # 失败后尝试使用内嵌预测器
        predictor = EmbeddedPredictor()
else:
    st.sidebar.warning("未找到simple_predictor.py文件")
    # 使用内嵌预测器
    predictor = EmbeddedPredictor()

# 自定义样式
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
        # 使用predictor进行预测
        y_pred = predictor.predict(input_data)[0]
        
        # 保存预测结果到session_state
        st.session_state.prediction_result = y_pred

        # 显示预测结果
        prediction_placeholder.markdown(
            f"<div class='yield-result'>Char Yield (wt%) <br> {y_pred:.2f}</div>",
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"预测过程中出现错误: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

# 如果有保存的预测结果，显示它
if 'prediction_result' in st.session_state and st.session_state.prediction_result is not None:
    prediction_placeholder.markdown(
        f"<div class='yield-result'>Char Yield (wt%) <br> {st.session_state.prediction_result:.2f}</div>",
        unsafe_allow_html=True
    )

# 添加调试信息
with st.expander("Debug Information", expanded=False):
    st.write("Input Features:")
    st.write(input_data)
    
    if hasattr(predictor, 'feature_names'):
        st.write("Model Features:")
        st.write(predictor.feature_names)

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