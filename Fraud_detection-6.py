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
import traceback
import logging

# 配置日志系统
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Pyrolysis_App")

# 页面设置
st.set_page_config(
    page_title='Biomass Pyrolysis Yield Forecast',
    page_icon='🔥',
    layout='wide'
)

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

# 创建侧边栏日志区域
log_container = st.sidebar.container()
log_container.write("### 调试日志")

def log(message):
    """记录调试信息到侧边栏和日志系统"""
    log_container.write(message)
    logger.info(message)

class EnsembleModelPredictor:
    """
    集成模型预测器 - 负责加载和使用CatBoost集成模型进行预测
    """
    def __init__(self):
        # 初始化时不设置特征列表，而是从元数据中获取
        self.models = []
        self.model_weights = None
        self.final_scaler = None
        self.feature_names = None
        self.metadata = None
        self.model_dir = None
        
        # 加载模型组件
        self.load_model_components()
    
    def find_model_directory(self):
        """
        查找模型目录的多种方法
        """
        # 可能的模型目录路径
        possible_dirs = [
            "Char_Yield_Model",
            "models/Char_Yield_Model",
            "../Char_Yield_Model",
            "../../Char_Yield_Model",
            "./Char_Yield_Model",
            os.path.join(os.getcwd(), "Char_Yield_Model"),
            "C:/Users/HWY/Desktop/方-3/Char_Yield_Model"
        ]
        
        # 首先尝试直接定位模型目录
        for dir_path in possible_dirs:
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                log(f"找到模型目录: {dir_path}")
                return os.path.abspath(dir_path)
        
        # 如果找不到，尝试基于模型文件推断
        model_files = glob.glob("**/model_*.joblib", recursive=True)
        if model_files:
            # 取第一个模型文件所在的上两级目录
            model_dir = os.path.dirname(os.path.dirname(model_files[0]))
            log(f"基于模型文件推断模型目录: {model_dir}")
            return model_dir
        
        # 如果找不到模型目录，记录错误并返回当前目录
        log("警告: 无法找到模型目录，将使用当前目录")
        return os.getcwd()
    
    def load_model_components(self):
        """加载模型的所有组件"""
        try:
            # 1. 查找模型目录
            self.model_dir = self.find_model_directory()
            log(f"使用模型目录: {self.model_dir}")
            
            # 2. 加载元数据 - 这一步非常重要，包含了特征名称
            metadata_path = os.path.join(self.model_dir, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                
                # 从元数据中获取特征名称 - 这是确保特征顺序正确的关键
                self.feature_names = self.metadata.get('feature_names', None)
                log(f"从元数据加载特征列表: {self.feature_names}")
                
                if not self.feature_names:
                    log("警告: 元数据中没有特征列表")
            else:
                log(f"警告: 未找到元数据文件: {metadata_path}")
                # 使用默认特征列表 - 这必须与训练时完全一致
                self.feature_names = [
                    'PT(°C)', 'RT(min)', 'HT(°C/min)', 
                    'C(%)', 'H(%)', 'O(%)', 'N(%)',
                    'Ash(%)', 'VM(%)', 'FC(%)'
                ]
                log(f"使用默认特征列表: {self.feature_names}")
            
            # 3. 加载模型文件
            models_dir = os.path.join(self.model_dir, 'models')
            if os.path.exists(models_dir):
                model_files = sorted(glob.glob(os.path.join(models_dir, 'model_*.joblib')))
                if not model_files:
                    log(f"未在 {models_dir} 中找到模型文件")
                    return False
                
                # 按顺序加载所有模型
                for model_file in model_files:
                    try:
                        model = joblib.load(model_file)
                        self.models.append(model)
                        log(f"加载模型: {os.path.basename(model_file)}")
                    except Exception as e:
                        log(f"加载模型 {model_file} 时出错: {str(e)}")
            else:
                log(f"模型目录不存在: {models_dir}")
                return False
            
            # 4. 加载权重文件
            weights_path = os.path.join(self.model_dir, 'model_weights.npy')
            if os.path.exists(weights_path):
                self.model_weights = np.load(weights_path)
                log(f"加载权重文件: {weights_path}")
            else:
                log(f"警告: 未找到权重文件: {weights_path}")
                # 如果找不到权重，使用均等权重
                self.model_weights = np.ones(len(self.models)) / len(self.models)
                log("使用均等权重")
            
            # 5. 加载标准化器（这一步非常重要）
            scaler_path = os.path.join(self.model_dir, 'final_scaler.joblib')
            if os.path.exists(scaler_path):
                self.final_scaler = joblib.load(scaler_path)
                log(f"加载标准化器: {scaler_path}")
                
                # 打印标准化器的均值和标准差，用于验证
                if hasattr(self.final_scaler, 'mean_'):
                    log(f"特征均值: {self.final_scaler.mean_}")
                if hasattr(self.final_scaler, 'scale_'):
                    log(f"特征标准差: {self.final_scaler.scale_}")
            else:
                log(f"错误: 未找到标准化器文件: {scaler_path}")
                return False
            
            log(f"成功加载 {len(self.models)} 个模型")
            return True
            
        except Exception as e:
            log(f"加载模型组件时出错: {str(e)}")
            log(traceback.format_exc())
            return False
    
    def check_feature_order(self, input_df):
        """
        确保输入特征的顺序与训练时一致
        """
        # 检查特征名称是否存在
        if self.feature_names is None:
            log("错误: 特征名称列表为空")
            return input_df
        
        # 记录原始输入
        log(f"原始输入特征: {input_df.columns.tolist()}")
        log(f"模型需要的特征顺序: {self.feature_names}")
        
        # 创建新的DataFrame，严格按照模型特征顺序
        ordered_df = pd.DataFrame(index=input_df.index)
        
        for feature in self.feature_names:
            # 精确匹配
            if feature in input_df.columns:
                ordered_df[feature] = input_df[feature]
            # 基于前缀匹配
            else:
                feature_base = feature.split('(')[0].strip()
                for col in input_df.columns:
                    col_base = col.split('(')[0].strip()
                    if col_base == feature_base:
                        log(f"映射特征: {col} -> {feature}")
                        ordered_df[feature] = input_df[col]
                        break
                else:
                    # 未找到匹配，使用默认值
                    log(f"警告: 未找到特征 {feature} 的对应输入，使用默认值0")
                    ordered_df[feature] = 0.0
        
        log(f"重排后的特征顺序: {ordered_df.columns.tolist()}")
        return ordered_df
    
    def predict(self, input_features):
        """
        使用加载的模型进行预测
        
        参数:
            input_features: 包含特征数据的DataFrame
        """
        try:
            # 检查模型是否已加载
            if not self.models or not self.final_scaler:
                log("错误: 模型或标准化器未加载")
                return np.array([0.0])
            
            # 确保特征顺序与训练时一致
            input_ordered = self.check_feature_order(input_features)
            
            # 记录详细的输入数据
            log(f"预测输入数据: {input_ordered.to_dict('records')}")
            
            # 应用标准化
            X_scaled = self.final_scaler.transform(input_ordered)
            log(f"标准化后的数据形状: {X_scaled.shape}")
            
            # 使用每个模型进行预测
            all_predictions = np.zeros((input_ordered.shape[0], len(self.models)))
            for i, model in enumerate(self.models):
                pred = model.predict(X_scaled)
                all_predictions[:, i] = pred
                log(f"模型 {i} 预测结果: {pred[0]:.2f}")
            
            # 应用模型权重计算最终预测
            weighted_pred = np.sum(all_predictions * self.model_weights.reshape(1, -1), axis=1)
            log(f"最终加权预测结果: {weighted_pred[0]:.2f}")
            
            return weighted_pred
            
        except Exception as e:
            log(f"预测过程中出错: {str(e)}")
            log(traceback.format_exc())
            # 返回默认值
            return np.array([0.0])

# 初始化预测器
predictor = EnsembleModelPredictor()

# 初始化会话状态
if 'clear_pressed' not in st.session_state:
    st.session_state.clear_pressed = False

# 定义默认值
default_values = {
    "C(%)": 38.3,  # 使用截图中显示的值
    "H(%)": 5.5,
    "O(%)": 55.2,
    "N(%)": 0.6,
    "Ash(%)": 6.6,
    "VM(%)": 81.1,
    "FC(%)": 10.3,
    "PT(°C)": 500.0,
    "HR(℃/min)": 10.0,
    "RT(min)": 60.0
}

# 特征分类
feature_categories = {
    "Ultimate Analysis": ["C(%)", "H(%)", "O(%)", "N(%)"],
    "Proximate Analysis": ["Ash(%)", "VM(%)", "FC(%)"],
    "Pyrolysis Conditions": ["PT(°C)", "HR(℃/min)", "RT(min)"]
}

# 创建三列布局
col1, col2, col3 = st.columns(3)

# 使用字典来存储所有输入值
features = {}

# Ultimate Analysis (黄色区域) - 第一列
with col1:
    st.markdown("<div class='section-header' style='background-color: #DAA520;'>Ultimate Analysis</div>", unsafe_allow_html=True)
    
    for feature in feature_categories["Ultimate Analysis"]:
        if st.session_state.clear_pressed:
            value = default_values[feature]
        else:
            value = st.session_state.get(f"ultimate_{feature}", default_values[feature])
        
        col_a, col_b = st.columns([1, 0.5])
        with col_a:
            st.markdown(f"<div class='input-label' style='background-color: #DAA520;'>{feature}</div>", unsafe_allow_html=True)
        with col_b:
            features[feature] = st.number_input(
                "", 
                min_value=0.0, 
                max_value=100.0, 
                value=value, 
                key=f"ultimate_{feature}", 
                format="%.1f",
                label_visibility="collapsed"
            )

# Proximate Analysis (绿色区域) - 第二列
with col2:
    st.markdown("<div class='section-header' style='background-color: #32CD32;'>Proximate Analysis</div>", unsafe_allow_html=True)
    
    for feature in feature_categories["Proximate Analysis"]:
        if st.session_state.clear_pressed:
            value = default_values[feature]
        else:
            value = st.session_state.get(f"proximate_{feature}", default_values[feature])
        
        col_a, col_b = st.columns([1, 0.5])
        with col_a:
            st.markdown(f"<div class='input-label' style='background-color: #32CD32;'>{feature}</div>", unsafe_allow_html=True)
        with col_b:
            features[feature] = st.number_input(
                "", 
                min_value=0.0, 
                max_value=100.0, 
                value=value, 
                key=f"proximate_{feature}", 
                format="%.1f",
                label_visibility="collapsed"
            )

# Pyrolysis Conditions (橙色区域) - 第三列
with col3:
    st.markdown("<div class='section-header' style='background-color: #FF7F50;'>Pyrolysis Conditions</div>", unsafe_allow_html=True)
    
    for feature in feature_categories["Pyrolysis Conditions"]:
        # 重置值或使用现有值
        if st.session_state.clear_pressed:
            value = default_values[feature]
        else:
            value = st.session_state.get(f"pyrolysis_{feature}", default_values[feature])
        
        # 对于温度和其他参数使用不同的范围
        if feature == "PT(°C)":
            min_val, max_val = 200.0, 1000.0
        elif feature == "HR(℃/min)":
            min_val, max_val = 1.0, 100.0
        elif feature == "RT(min)":
            min_val, max_val = 0.0, 500.0
        else:
            min_val, max_val = 0.0, 100.0
        
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

# 重置session_state中的clear_pressed状态
if st.session_state.clear_pressed:
    st.session_state.clear_pressed = False

# 预测结果显示区域和按钮
result_col, button_col = st.columns([5, 1])

with result_col:
    st.subheader("Char Yield (wt%)")
    prediction_placeholder = st.empty()

with button_col:
    # 预测按钮
    predict_button = st.button("PUSH", type="primary")
    
    # 定义Clear按钮的回调函数
    def clear_values():
        st.session_state.clear_pressed = True
        # 清除预测结果
        if 'prediction_result' in st.session_state:
            del st.session_state.prediction_result
    
    clear_button = st.button("CLEAR", on_click=clear_values)

# 转换为DataFrame
input_data = pd.DataFrame([features])

# 当点击预测按钮时
if predict_button:
    try:
        # 记录预测开始
        log("开始进行预测...")
        
        # 使用预测器预测
        result = predictor.predict(input_data)[0]
        
        # 保存预测结果
        st.session_state.prediction_result = result
        
        # 显示预测结果
        prediction_placeholder.markdown(
            f"<div class='yield-result'>{result:.2f}</div>",
            unsafe_allow_html=True
        )
        
        # 记录预测完成
        log(f"预测完成: Char Yield(%) = {result:.2f}")
        
    except Exception as e:
        log(f"预测过程中出现错误: {str(e)}")
        st.error(f"预测失败: {str(e)}")

# 如果有保存的预测结果，显示它
if 'prediction_result' in st.session_state:
    prediction_placeholder.markdown(
        f"<div class='yield-result'>{st.session_state.prediction_result:.2f}</div>",
        unsafe_allow_html=True
    )

# 添加调试信息折叠区域
with st.expander("Debug Information", expanded=False):
    st.write("### 输入特征")
    st.write(input_data)
    
    st.write("### 模型信息")
    if predictor.feature_names:
        st.write(f"特征列表: {predictor.feature_names}")
    if predictor.metadata and 'performance' in predictor.metadata:
        st.write(f"模型性能: {predictor.metadata['performance']}")
    
    st.write(f"使用目录: {predictor.model_dir}")
    st.write(f"加载的模型数量: {len(predictor.models)}")

# 添加模型描述
st.markdown("""
### About the Model

This application uses a CatBoost ensemble model to predict char yield in biomass pyrolysis.

#### Key Factors Affecting Char Yield:
* **Pyrolysis Temperature**: Higher temperature generally decreases char yield
* **Residence Time**: Longer residence time generally increases char yield
* **Biomass Composition**: Carbon content and ash content significantly affect the final yield

The model was trained using 10-fold cross-validation with optimized hyperparameters, achieving high prediction accuracy (R² = 0.93, RMSE = 3.39 on test set).
""")

# 页脚
st.markdown("---")
st.caption("© 2023 Biomass Pyrolysis Research Team. All rights reserved.")