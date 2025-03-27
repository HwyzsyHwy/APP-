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

# 页面设置
st.set_page_config(
    page_title='Biomass Pyrolysis Yield Forecast',
    page_icon='📊',
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
    """记录调试信息到侧边栏"""
    log_container.write(message)

# 增加搜索模型文件的功能
def find_model_files():
    """
    搜索目录中的模型文件和标准化器文件
    """
    # 搜索当前目录及子目录中的模型文件
    model_files = glob.glob("**/model_*.joblib", recursive=True)
    model_files = sorted(model_files, key=lambda x: int(x.split('model_')[1].split('.')[0]))
    scaler_files = glob.glob("**/final_scaler.joblib", recursive=True)
    metadata_files = glob.glob("**/metadata.json", recursive=True)
    
    log(f"找到 {len(model_files)} 个模型文件")
    for f in model_files:
        log(f"- {f}")
    log(f"找到 {len(scaler_files)} 个标准化器文件: {scaler_files}")
    log(f"找到 {len(metadata_files)} 个元数据文件: {metadata_files}")
    
    return model_files, scaler_files, metadata_files

# 使用直接加载方式的预测器
class DirectPredictor:
    """直接加载模型文件进行预测的预测器"""
    
    def __init__(self):
        self.models = []
        self.model_weights = None
        self.scaler = None
        self.metadata = None
        self.feature_names = None
        
        # 查找并加载模型
        self.load_model_components()
    
    def load_model_components(self):
        """加载模型组件"""
        try:
            # 查找模型文件
            model_files, scaler_files, metadata_files = find_model_files()
            
            # 先加载元数据，获取特征名称
            if metadata_files:
                metadata_path = metadata_files[0]
                log(f"加载元数据: {metadata_path}")
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                self.feature_names = self.metadata.get('feature_names', None)
                log(f"元数据中的特征名称: {self.feature_names}")
            
            # 加载模型
            if model_files:
                models_dir = os.path.dirname(model_files[0])
                log(f"模型目录: {models_dir}")
                
                for model_path in model_files:
                    log(f"加载模型: {model_path}")
                    try:
                        model = joblib.load(model_path)
                        self.models.append(model)
                        log(f"成功加载模型 {len(self.models)}")
                    except Exception as e:
                        log(f"加载模型 {model_path} 失败: {str(e)}")
                
                # 加载模型权重
                weights_path = os.path.join(models_dir, 'model_weights.npy')
                if os.path.exists(weights_path):
                    log(f"加载权重: {weights_path}")
                    try:
                        self.model_weights = np.load(weights_path)
                        log(f"权重形状: {self.model_weights.shape}")
                    except Exception as e:
                        log(f"加载权重失败: {str(e)}")
                        self.model_weights = np.ones(len(self.models)) / len(self.models)
                else:
                    log("未找到权重文件，使用均等权重")
                    self.model_weights = np.ones(len(self.models)) / len(self.models)
            else:
                log("未找到模型文件")
            
            # 加载标准化器
            if scaler_files:
                scaler_path = scaler_files[0]
                log(f"加载标准化器: {scaler_path}")
                try:
                    self.scaler = joblib.load(scaler_path)
                    log("标准化器加载成功")
                    
                    # 检查标准化器的特征名称
                    if hasattr(self.scaler, 'feature_names_in_'):
                        log(f"标准化器特征名称: {self.scaler.feature_names_in_}")
                        # 如果元数据中没有特征名称，使用标准化器中的
                        if self.feature_names is None:
                            self.feature_names = self.scaler.feature_names_in_.tolist()
                            log(f"使用标准化器中的特征名称: {self.feature_names}")
                except Exception as e:
                    log(f"加载标准化器失败: {str(e)}")
            else:
                log("未找到标准化器文件")
            
            # 如果仍然没有特征名称，使用默认值
            if self.feature_names is None:
                self.feature_names = ["PT(°C)", "RT(min)", "C(%)", "H(%)", "O(%)", "N(%)", "Ash(%)", "VM(%)", "FC(%)", "HR(℃/min)"]
                log(f"使用默认特征名称: {self.feature_names}")
            
            # 检查模型是否成功加载
            if self.models:
                log(f"成功加载 {len(self.models)} 个模型")
                return True
            else:
                log("未能加载任何模型")
                return False
                
        except Exception as e:
            log(f"加载模型组件时出错: {str(e)}")
            log(traceback.format_exc())
            return False
    
    def predict(self, X):
        """
        使用加载的模型进行预测
        
        参数:
            X: 特征数据，DataFram格式
        
        返回:
            预测结果数组
        """
        try:
            if not self.models:
                log("没有加载模型，无法预测")
                return np.array([33.0])  # 返回默认值
            
            # 提取特征顺序
            if isinstance(X, pd.DataFrame):
                log(f"输入特征顺序: {X.columns.tolist()}")
                log(f"模型特征顺序: {self.feature_names}")
                
                # 检查是否需要重新排序
                if sorted(X.columns.tolist()) == sorted(self.feature_names):
                    log("特征集合匹配，确保顺序一致")
                    X_ordered = X[self.feature_names].copy()
                    log(f"重排后的特征顺序: {X_ordered.columns.tolist()}")
                else:
                    log(f"特征不匹配! 输入: {X.columns.tolist()}, 模型需要: {self.feature_names}")
                    
                    # 尝试映射特征
                    matching_features = {}
                    for model_feat in self.feature_names:
                        model_base = model_feat.split('(')[0]
                        for input_feat in X.columns:
                            input_base = input_feat.split('(')[0]
                            if model_base == input_base:
                                matching_features[model_feat] = input_feat
                                break
                    
                    log(f"特征映射: {matching_features}")
                    
                    if len(matching_features) == len(self.feature_names):
                        # 创建一个新的DataFrame，按照模型需要的顺序和名称
                        X_ordered = pd.DataFrame(index=X.index)
                        for model_feat in self.feature_names:
                            if model_feat in matching_features:
                                input_feat = matching_features[model_feat]
                                X_ordered[model_feat] = X[input_feat].values
                            else:
                                log(f"无法映射特征: {model_feat}")
                                return np.array([33.0])
                        
                        log(f"映射后特征顺序: {X_ordered.columns.tolist()}")
                    else:
                        log("无法完全映射特征名称")
                        return np.array([33.0])
            else:
                log("输入不是DataFrame格式")
                return np.array([33.0])
            
            # 应用标准化
            if self.scaler:
                log("应用标准化器")
                # 转换为numpy数组，去除特征名以避免顺序问题
                X_values = X_ordered.values
                X_scaled = self.scaler.transform(X_values)
                log(f"标准化后数据形状: {X_scaled.shape}")
            else:
                log("没有标准化器，使用原始数据")
                X_scaled = X_ordered.values
            
            # 使用每个模型进行预测
            all_predictions = np.zeros((X_scaled.shape[0], len(self.models)))
            
            for i, model in enumerate(self.models):
                try:
                    pred = model.predict(X_scaled)
                    all_predictions[:, i] = pred
                    log(f"模型 {i} 预测结果: {pred[0]:.2f}")
                except Exception as e:
                    log(f"模型 {i} 预测失败: {str(e)}")
                    # 使用平均值填充
                    if i > 0:
                        all_predictions[:, i] = np.mean(all_predictions[:, :i], axis=1)
            
            # 计算加权平均预测
            weighted_pred = np.sum(all_predictions * self.model_weights.reshape(1, -1), axis=1)
            log(f"最终加权预测结果: {weighted_pred[0]:.2f}")
            
            return weighted_pred
            
        except Exception as e:
            log(f"预测过程中出错: {str(e)}")
            log(traceback.format_exc())
            return np.array([33.0])  # 返回默认值
    

# 初始化预测器
predictor = DirectPredictor()

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
        # 记录输入数据
        log("进行预测:")
        log(f"输入数据: {input_data.to_dict('records')}")
        
        # 使用predictor进行预测
        y_pred = predictor.predict(input_data)[0]
        log(f"预测完成: {y_pred:.2f}")
        
        # 保存预测结果到session_state
        st.session_state.prediction_result = y_pred

        # 显示预测结果
        prediction_placeholder.markdown(
            f"<div class='yield-result'>Char Yield (wt%) <br> {y_pred:.2f}</div>",
            unsafe_allow_html=True
        )
    except Exception as e:
        log(f"预测过程中出错: {str(e)}")
        log(traceback.format_exc())
        st.error(f"预测过程中出现错误: {str(e)}")

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