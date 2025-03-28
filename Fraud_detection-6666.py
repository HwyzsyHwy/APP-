# -*- coding: utf-8 -*-
"""
Biomass Pyrolysis Yield Forecast using CatBoost Ensemble Models
修复版本 - 解决小数精度问题和子模型标准化器问题
添加多模型切换功能 - 支持Char、Oil和Gas产率预测
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import joblib
import json
import traceback
import matplotlib.pyplot as plt
from datetime import datetime
import io
from PIL import Image

# 清除缓存，强制重新渲染
if "debug" not in st.session_state:
    st.cache_data.clear()
    st.session_state.debug = True
    st.session_state.decimal_test = 46.12  # 测试两位小数

# 页面设置
st.set_page_config(
    page_title='Biomass Pyrolysis Yield Prediction',
    page_icon='🔥',
    layout='wide',
    initial_sidebar_state='expanded'
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
log("应用启动 - 支持两位小数和多模型切换功能")

# 初始化会话状态 - 添加模型选择功能
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Char Yield(%)"  # 默认选择Char产率模型
    log(f"初始化选定模型: {st.session_state.selected_model}")

# 更新主标题以显示当前选定的模型
st.markdown("<h1 class='main-title'>基于Catboost集成模型的生物质热解产物预测系统</h1>", unsafe_allow_html=True)

# 添加模型选择区域 - 修改为三个按钮一排
st.markdown("<div class='model-selector'>", unsafe_allow_html=True)
st.markdown("<h3>选择预测目标</h3>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    char_button = st.button("🔥 Char Yield", 
                           key="char_button", 
                           help="预测焦炭产率 (%)", 
                           use_container_width=True,
                           type="primary" if st.session_state.selected_model == "Char Yield(%)" else "secondary")
with col2:
    oil_button = st.button("💧 Oil Yield", 
                          key="oil_button", 
                          help="预测生物油产率 (%)", 
                          use_container_width=True,
                          type="primary" if st.session_state.selected_model == "Oil Yield(%)" else "secondary")
with col3:
    gas_button = st.button("💨 Gas Yield", 
                          key="gas_button", 
                          help="预测气体产率 (%)", 
                          use_container_width=True,
                          type="primary" if st.session_state.selected_model == "Gas Yield(%)" else "secondary")

# 处理模型选择
if char_button:
    st.session_state.selected_model = "Char Yield(%)"
    st.session_state.prediction_result = None
    st.session_state.warnings = []
    st.session_state.individual_predictions = []
    log(f"切换到模型: {st.session_state.selected_model}")
    st.rerun()

if oil_button:
    st.session_state.selected_model = "Oil Yield(%)"
    st.session_state.prediction_result = None
    st.session_state.warnings = []
    st.session_state.individual_predictions = []
    log(f"切换到模型: {st.session_state.selected_model}")
    st.rerun()

if gas_button:
    st.session_state.selected_model = "Gas Yield(%)"
    st.session_state.prediction_result = None
    st.session_state.warnings = []
    st.session_state.individual_predictions = []
    log(f"切换到模型: {st.session_state.selected_model}")
    st.rerun()

st.markdown(f"<p style='text-align:center;'>当前模型: <b>{st.session_state.selected_model}</b></p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

class CorrectedEnsemblePredictor:
    """修复版集成模型预测器 - 解决子模型标准化器问题，支持多模型切换"""
    
    def __init__(self, target_model="Char Yield(%)"):
        self.models = []
        self.scalers = []  # 每个子模型的标准化器
        self.final_scaler = None  # 最终标准化器（备用）
        self.model_weights = None
        self.feature_names = None
        self.target_name = target_model  # 设置目标变量名称
        self.metadata = None
        self.model_dir = None
        self.feature_importance = None
        self.training_ranges = {}
        self.model_loaded = False  # 新增：标记模型加载状态
        
        # 加载模型
        self.load_model()
    
    def find_model_directory(self):
        """查找模型目录的多种方法，支持不同模型类型"""
        # 根据目标变量确定模型目录名称
        model_name = self.target_name.replace(' ', '_').replace('(', '').replace(')', '')
        log(f"尝试查找模型目录: {model_name}_Model")
        
        # 模型目录可能的路径 - 添加更多可能的路径以提高查找成功率
        possible_dirs = [
            # 当前目录和父目录
            f"./{model_name}_Model",
            f"../{model_name}_Model",
            # 应用根目录
            f"{model_name}_Model",
            # 更多可能的位置
            f"./models/{model_name}_Model",
            f"../models/{model_name}_Model",
            # 系统路径
            f"C:/Users/HWY/Desktop/方-3/{model_name}_Model",
            # 如果是在云服务上运行
            f"/app/{model_name}_Model",
            f"/app/models/{model_name}_Model",
            f"/mount/src/{model_name}_Model",
            # 应用当前工作目录
            os.path.join(os.getcwd(), f"{model_name}_Model"),
            # 特定路径 (从截图中看到的)
            f"/source/src/app/{model_name}_Model"
        ]
        
        # 尝试所有可能路径
        for dir_path in possible_dirs:
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                log(f"找到模型目录: {dir_path}")
                return os.path.abspath(dir_path)
        
        # 如果找不到，尝试全局模糊搜索 (先搜索当前目录和子目录)
        try:
            log("在当前目录及子目录搜索模型文件...")
            # 使用 ** 通配符进行递归搜索
            for pattern in [
                f"**/{model_name}_Model",
                f"**/models/{model_name}_Model",
                f"**/{model_name}_Model/**",
                f"**/models/**/{model_name}_Model"
            ]:
                matches = glob.glob(pattern, recursive=True)
                if matches:
                    for match in matches:
                        if os.path.isdir(match):
                            log(f"通过全局搜索找到模型目录: {match}")
                            return os.path.abspath(match)
            
            # 如果上面的搜索失败，尝试根据模型文件反向查找目录
            model_files = glob.glob(f"**/{model_name}_Model/**/model_*.joblib", recursive=True)
            if model_files:
                model_dir = os.path.dirname(os.path.dirname(model_files[0]))
                log(f"基于模型文件推断模型目录: {model_dir}")
                return model_dir
        except Exception as e:
            log(f"搜索模型目录时出错: {str(e)}")
        
        # 返回当前目录作为最后的退路，同时记录警告
        log(f"严重警告: 无法找到{self.target_name}模型目录，将使用当前目录。预测将返回默认值!")
        return os.getcwd()
    
    def load_feature_importance(self):
        """加载特征重要性数据"""
        try:
            # 尝试从CSV文件加载特征重要性
            importance_csv = os.path.join(self.model_dir, "feature_importance.csv")
            if os.path.exists(importance_csv):
                importance_df = pd.read_csv(importance_csv)
                self.feature_importance = importance_df
                log(f"已加载特征重要性数据，共 {len(importance_df)} 个特征")
                return True
            
            # 如果CSV不存在，尝试从元数据中加载
            if self.metadata and 'feature_importance' in self.metadata:
                importance_data = self.metadata['feature_importance']
                self.feature_importance = pd.DataFrame(importance_data)
                log(f"从元数据加载特征重要性数据")
                return True
            
            # 尝试通过加载的模型计算特征重要性
            if self.models and self.model_weights is not None and self.feature_names:
                log("通过模型计算特征重要性")
                importance = np.zeros(len(self.feature_names))
                for i, model in enumerate(self.models):
                    try:
                        model_importance = model.get_feature_importance()
                        importance += model_importance * self.model_weights[i]
                    except Exception as e:
                        log(f"获取模型 {i} 特征重要性时出错: {str(e)}")
                
                self.feature_importance = pd.DataFrame({
                    'Feature': self.feature_names,
                    'Importance': importance
                }).sort_values('Importance', ascending=False)
                
                log(f"计算得到特征重要性数据，最重要特征: {self.feature_importance['Feature'].iloc[0]}")
                return True
                
            log("警告: 无法加载或计算特征重要性")
            return False
        except Exception as e:
            log(f"加载特征重要性时出错: {str(e)}")
            return False
    
    def extract_training_ranges(self):
        """从标准化器中提取训练数据范围"""
        if not hasattr(self.final_scaler, 'mean_') or not hasattr(self.final_scaler, 'scale_'):
            log("警告: 标准化器没有均值或标准差信息")
            return
        
        if not self.feature_names:
            log("警告: 无法获取特征名称")
            return
        
        # 提取特征的均值和标准差
        means = self.final_scaler.mean_
        stds = self.final_scaler.scale_
        
        # 计算每个特征的95%置信区间 (均值±2标准差)
        for i, feature in enumerate(self.feature_names):
            if i < len(means) and i < len(stds):
                mean_val = means[i]
                std_val = stds[i]
                self.training_ranges[feature] = {
                    'mean': mean_val,
                    'std': std_val,
                    'min': mean_val - 2 * std_val,  # 近似95%置信区间下限
                    'max': mean_val + 2 * std_val,  # 近似95%置信区间上限
                }
        
        if self.training_ranges:
            log(f"已提取 {len(self.training_ranges)} 个特征的训练范围")
    
    def load_model(self):
        """加载所有模型组件，包括每个子模型的标准化器"""
        try:
            # 清空之前的模型数据
            self.models = []
            self.scalers = []
            self.feature_importance = None
            self.training_ranges = {}
            self.model_loaded = False  # 重置加载状态
            
            # 1. 查找模型目录
            self.model_dir = self.find_model_directory()
            log(f"使用{self.target_name}模型目录: {self.model_dir}")
            
            # 2. 加载元数据
            metadata_path = os.path.join(self.model_dir, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                
                # 获取特征名称和目标变量
                self.feature_names = self.metadata.get('feature_names', None)
                if self.metadata.get('target_name'):
                    self.target_name = self.metadata['target_name']
                
                log(f"从元数据加载特征列表: {self.feature_names}")
                log(f"目标变量: {self.target_name}")
            else:
                log(f"警告: 未找到元数据文件 {metadata_path}")
                # 使用默认特征列表 - 必须与模型训练时完全一致
                self.feature_names = [
                    'C(%)', 'H(%)', 'O(%)', 'N(%)', 'Ash(%)', 'VM(%)', 'FC(%)', 
                    'PT(°C)', 'HR(℃/min)', 'RT(min)'
                ]
                log(f"使用默认特征列表: {self.feature_names}")
            
            # 3. 加载模型
            models_dir = os.path.join(self.model_dir, 'models')
            if os.path.exists(models_dir):
                model_files = sorted(glob.glob(os.path.join(models_dir, 'model_*.joblib')))
                if model_files:
                    for model_file in model_files:
                        model = joblib.load(model_file)
                        self.models.append(model)
                        log(f"加载模型: {os.path.basename(model_file)}")
                else:
                    log(f"错误: 未找到模型文件在 {models_dir}")
                    st.error(f"错误: 未找到{self.target_name}模型文件。请检查应用安装或联系管理员。")
                    return False
            else:
                log(f"错误: 模型目录不存在: {models_dir}")
                st.error(f"错误: {self.target_name}模型目录不存在。请检查应用安装或联系管理员。")
                return False
            
            # 4. 加载每个子模型的标准化器 - 这是关键修复点
            scalers_dir = os.path.join(self.model_dir, 'scalers')
            if os.path.exists(scalers_dir):
                scaler_files = sorted(glob.glob(os.path.join(scalers_dir, 'scaler_*.joblib')))
                if scaler_files:
                    for scaler_file in scaler_files:
                        scaler = joblib.load(scaler_file)
                        self.scalers.append(scaler)
                        log(f"加载子模型标准化器: {os.path.basename(scaler_file)}")
                else:
                    log(f"警告: 未找到子模型标准化器文件在 {scalers_dir}")
            else:
                log(f"警告: 未找到标准化器目录: {scalers_dir}")
            
            # 5. 加载最终标准化器（作为备用）
            final_scaler_path = os.path.join(self.model_dir, 'final_scaler.joblib')
            if os.path.exists(final_scaler_path):
                self.final_scaler = joblib.load(final_scaler_path)
                log(f"加载最终标准化器: {final_scaler_path}")
                
                # 打印标准化器信息
                if hasattr(self.final_scaler, 'mean_'):
                    log(f"特征均值: {self.final_scaler.mean_}")
                if hasattr(self.final_scaler, 'scale_'):
                    log(f"特征标准差: {self.final_scaler.scale_}")
                
                # 提取训练数据范围
                self.extract_training_ranges()
            else:
                log(f"警告: 未找到最终标准化器文件 {final_scaler_path}")
            
            # 6. 加载权重
            weights_path = os.path.join(self.model_dir, 'model_weights.npy')
            if os.path.exists(weights_path):
                self.model_weights = np.load(weights_path)
                log(f"加载权重文件: {weights_path}")
            else:
                # 如果没有权重文件，使用均等权重
                self.model_weights = np.ones(len(self.models)) / len(self.models)
                log("警告: 未找到权重文件，使用均等权重")
            
            # 7. 加载特征重要性
            self.load_feature_importance()
            
            # 验证加载状态
            if len(self.models) > 0:
                log(f"成功加载 {len(self.models)} 个模型和 {len(self.scalers)} 个子模型标准化器")
                self.model_loaded = True  # 标记模型加载成功
            else:
                log(f"错误: 没有加载到任何{self.target_name}模型")
                st.error(f"错误: 没有加载到任何{self.target_name}模型。请检查应用安装或联系管理员。")
                return False
            
            # 特别标记标准化器问题
            if len(self.models) != len(self.scalers):
                log(f"警告: 模型数量 ({len(self.models)}) 与标准化器数量 ({len(self.scalers)}) 不匹配")
                
            return True
            
        except Exception as e:
            log(f"加载模型时出错: {str(e)}")
            log(traceback.format_exc())
            st.error(f"加载{self.target_name}模型时发生错误: {str(e)}")
            return False
    
    def check_input_range(self, input_df):
        """检查输入值是否在训练数据范围内"""
        warnings = []
        
        if not self.training_ranges:
            log("警告: 没有训练数据范围信息，跳过范围检查")
            return warnings
        
        for feature, range_info in self.training_ranges.items():
            if feature in input_df.columns:
                value = input_df[feature].iloc[0]
                # 检查是否超出训练数据的95%置信区间
                if value < range_info['min'] or value > range_info['max']:
                    warning = f"{feature}: {value:.2f} (超出训练范围 {range_info['min']:.2f} - {range_info['max']:.2f})"
                    warnings.append(warning)
                    log(f"警告: {warning}")
        
        return warnings
    
    def predict(self, input_features, return_individual=False):
        """使用每个子模型对应的标准化器进行预测"""
        try:
            # 验证模型组件
            if not self.model_loaded or not self.models or len(self.models) == 0:
                log(f"错误: 没有加载{self.target_name}模型或模型加载失败")
                st.error(f"错误: {self.target_name}模型未正确加载。请检查应用安装或联系管理员。")
                if return_individual:
                    return np.array([0.0]), []
                else:
                    return np.array([0.0])
            
            # 确保输入特征包含所有必要特征
            missing_features = []
            if self.feature_names:
                for feature in self.feature_names:
                    if feature not in input_features.columns:
                        missing_features.append(feature)
            
            if missing_features:
                missing_str = ", ".join(missing_features)
                log(f"错误: 输入缺少以下特征: {missing_str}")
                st.error(f"输入数据缺少以下必要特征: {missing_str}")
                if return_individual:
                    return np.array([0.0]), []
                else:
                    return np.array([0.0])
            
            # 按照模型训练时的特征顺序重新排列
            if self.feature_names:
                input_ordered = input_features[self.feature_names].copy()
                log(f"{self.target_name}模型: 输入特征已按照训练时的顺序排列")
            else:
                input_ordered = input_features
                log(f"警告: {self.target_name}模型没有特征名称列表，使用原始输入顺序")
            
            # 记录输入数据
            log(f"预测输入数据: {input_ordered.iloc[0].to_dict()}")
            
            # 使用每个子模型和对应的标准化器进行预测
            individual_predictions = []
            all_predictions = np.zeros((input_ordered.shape[0], len(self.models)))
            
            # 检查标准化器是否足够
            scalers_available = len(self.scalers) > 0
            
            for i, model in enumerate(self.models):
                try:
                    # 使用对应的标准化器（如果可用）
                    if scalers_available and i < len(self.scalers):
                        X_scaled = self.scalers[i].transform(input_ordered)
                        log(f"模型 {i} 使用对应的标准化器")
                    else:
                        # 如果没有对应的标准化器，使用最终标准化器
                        if self.final_scaler:
                            X_scaled = self.final_scaler.transform(input_ordered)
                            log(f"模型 {i} 使用最终标准化器")
                        else:
                            # 如果没有任何标准化器可用，则使用原始特征
                            log(f"警告: 模型 {i} 没有可用的标准化器，使用原始特征")
                            X_scaled = input_ordered.values
                    
                    # 执行预测并确保返回的是标量值 (修复 invalid index to scalar variable 错误)
                    pred = model.predict(X_scaled)
                    # 确保预测值是标量，不是数组
                    pred_value = float(pred[0]) if isinstance(pred, (np.ndarray, list)) else float(pred)
                    all_predictions[:, i] = pred_value
                    individual_predictions.append(pred_value)
                    log(f"模型 {i} 预测结果: {pred_value:.2f}")
                except Exception as e:
                    log(f"模型 {i} 预测时出错: {str(e)}")
                    # 如果某个模型失败，使用其他模型的平均值
                    if i > 0:
                        avg_pred = np.mean(all_predictions[:, :i], axis=1)
                        avg_value = float(avg_pred[0]) if len(avg_pred) > 0 else 0.0
                        all_predictions[:, i] = avg_value
                        individual_predictions.append(avg_value)
                        log(f"模型 {i} 使用之前模型的平均值: {avg_value:.2f}")
            
            # 计算加权平均 - 修复：确保不会出现维度不匹配的问题
            if len(self.models) > 0:
                # 确保权重数组维度正确
                weights = self.model_weights
                if weights.ndim == 1:
                    weights = weights.reshape(1, -1)
                
                # 确保权重和预测维度匹配
                if weights.shape[1] != all_predictions.shape[1]:
                    log(f"警告: 权重维度 {weights.shape} 与预测维度 {all_predictions.shape} 不匹配，使用平均值")
                    weighted_pred = np.mean(all_predictions, axis=1)
                else:
                    # 正确计算加权平均
                    weighted_pred = np.sum(all_predictions * weights, axis=1)
                
                log(f"{self.target_name}最终加权预测结果: {weighted_pred[0]:.2f}")
            else:
                weighted_pred = np.array([0.0])
                log(f"警告: 没有可用模型，返回默认值0")
            
            # 计算评估指标 - 动态计算RMSE和R²
            std_dev = np.std(individual_predictions) if len(individual_predictions) > 0 else 0
            
            # 修复 - 确保有足够的数据进行计算
            if len(individual_predictions) > 1:
                # 创建一个正确的输入向量进行RMSE计算
                weighted_pred_reshaped = np.tile(weighted_pred.reshape(-1, 1), (1, all_predictions.shape[1]))
                rmse = np.sqrt(np.mean((all_predictions - weighted_pred_reshaped)**2))
                
                # 计算R² (避免除以零错误)
                total_variance = np.sum((all_predictions - np.mean(all_predictions))**2)
                explained_variance = total_variance - np.sum((all_predictions - weighted_pred_reshaped)**2)
                r2 = explained_variance / total_variance if total_variance > 0 else 0
                
                log(f"预测标准差: {std_dev:.4f}")
                log(f"计算得到RMSE: {float(rmse[0]) if isinstance(rmse, np.ndarray) else float(rmse):.4f}, R²: {r2:.4f}")
                
                # 存储评估指标到session_state - 确保性能指标动态更新
                st.session_state.current_rmse = float(rmse[0]) if isinstance(rmse, np.ndarray) else float(rmse)
                st.session_state.current_r2 = float(r2)
            else:
                log("警告: 没有足够的模型进行性能评估")
                # 设置默认值以避免后续显示错误
                st.session_state.current_rmse = 0.0
                st.session_state.current_r2 = 0.0
            
            if return_individual:
                return weighted_pred, individual_predictions
            else:
                return weighted_pred
            
        except Exception as e:
            log(f"预测过程中出错: {str(e)}")
            log(traceback.format_exc())
            st.error(f"预测过程中发生错误: {str(e)}")
            # 修复 - 返回默认值，确保类型一致
            if return_individual:
                return np.array([0.0]), []
            else:
                return np.array([0.0])
    
    def get_model_info(self):
        """获取模型信息摘要"""
        info = {
            "模型类型": "CatBoost集成模型",
            "模型数量": len(self.models),
            "特征数量": len(self.feature_names) if self.feature_names else 0,
            "目标变量": self.target_name,
            "模型加载状态": "成功" if self.model_loaded else "失败"
        }
        
        # 添加性能信息
        if self.metadata and 'performance' in self.metadata:
            performance = self.metadata['performance']
            info["测试集R²"] = f"{performance.get('test_r2', 'N/A'):.4f}"
            info["测试集RMSE"] = f"{performance.get('test_rmse', 'N/A'):.2f}"
        
        # 添加特征重要性信息
        if self.feature_importance is not None and len(self.feature_importance) > 0:
            top_features = self.feature_importance.head(3)
            info["重要特征"] = ", ".join(top_features['Feature'].tolist())
        
        # 添加标准化器信息
        info["子模型标准化器数量"] = len(self.scalers)
        
        return info

# 初始化预测器 - 使用当前选择的模型
predictor = CorrectedEnsemblePredictor(target_model=st.session_state.selected_model)

# 在侧边栏添加模型信息
model_info = predictor.get_model_info()
model_info_html = "<div class='sidebar-model-info'><h3>关于模型</h3>"
for key, value in model_info.items():
    model_info_html += f"<p><b>{key}</b>: {value}</p>"

# 标准化器状态
model_info_html += "<h4>标准化器状态</h4>"
if len(predictor.scalers) == len(predictor.models):
    model_info_html += f"<p style='color:green'>✅ 所有 {len(predictor.models)} 个子模型都使用了对应的标准化器</p>"
elif len(predictor.scalers) > 0:
    model_info_html += f"<p style='color:orange'>⚠️ 找到 {len(predictor.scalers)}/{len(predictor.models)} 个子模型标准化器</p>"
else:
    model_info_html += "<p style='color:red'>❌ 未找到子模型标准化器，使用最终标准化器</p>"

model_info_html += "</div>"
st.sidebar.markdown(model_info_html, unsafe_allow_html=True)

# 性能指标显示区域（在预测后动态更新）
performance_container = st.sidebar.container()

# 初始化会话状态
if 'clear_pressed' not in st.session_state:
    st.session_state.clear_pressed = False
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'warnings' not in st.session_state:
    st.session_state.warnings = []
if 'individual_predictions' not in st.session_state:
    st.session_state.individual_predictions = []
if 'current_rmse' not in st.session_state:
    st.session_state.current_rmse = None
if 'current_r2' not in st.session_state:
    st.session_state.current_r2 = None
if 'prediction_error' not in st.session_state:
    st.session_state.prediction_error = None

# 定义默认值 - 从用户截图中提取
default_values = {
    "C(%)": 46.00,  # 使用两位小数精度
    "H(%)": 5.50,
    "O(%)": 55.20,
    "N(%)": 0.60,
    "Ash(%)": 6.60,
    "VM(%)": 81.10,
    "FC(%)": 10.30,
    "PT(°C)": 500.00,  # 使用实际测试值
    "HR(℃/min)": 10.00,
    "RT(min)": 60.00
}

# 特征分类
feature_categories = {
    "Ultimate Analysis": ["C(%)", "H(%)", "O(%)", "N(%)"],
    "Proximate Analysis": ["Ash(%)", "VM(%)", "FC(%)"],
    "Pyrolysis Conditions": ["PT(°C)", "HR(℃/min)", "RT(min)"]
}

# 颜色配置
category_colors = {
    "Ultimate Analysis": "#DAA520",  # 黄色
    "Proximate Analysis": "#32CD32",  # 绿色
    "Pyrolysis Conditions": "#FF7F50"  # 橙色
}

# 创建三列布局
col1, col2, col3 = st.columns(3)

# 使用字典存储所有输入值
features = {}

# Ultimate Analysis - 第一列
with col1:
    category = "Ultimate Analysis"
    color = category_colors[category]
    st.markdown(f"<div class='section-header' style='background-color: {color};'>{category}</div>", unsafe_allow_html=True)
    
    for feature in feature_categories[category]:
        if st.session_state.clear_pressed:
            value = default_values[feature]
        else:
            value = st.session_state.get(f"{category}_{feature}", default_values[feature])
        
        col_a, col_b = st.columns([1, 0.5])
        with col_a:
            st.markdown(f"<div class='input-label' style='background-color: {color};'>{feature}</div>", unsafe_allow_html=True)
        with col_b:
            # 关键修改: 设置步长为0.01以支持两位小数
            features[feature] = st.number_input(
                "", 
                min_value=0.00, 
                max_value=100.00, 
                value=float(value), 
                step=0.01,  # 设置为0.01允许两位小数输入
                key=f"{category}_{feature}", 
                format="%.2f",  # 强制显示两位小数
                label_visibility="collapsed"
            )
            
            # 调试显示
            st.markdown(f"<span style='font-size:10px;color:gray;'>输入值: {features[feature]:.2f}</span>", unsafe_allow_html=True)

# Proximate Analysis - 第二列
with col2:
    category = "Proximate Analysis"
    color = category_colors[category]
    st.markdown(f"<div class='section-header' style='background-color: {color};'>{category}</div>", unsafe_allow_html=True)
    
    for feature in feature_categories[category]:
        if st.session_state.clear_pressed:
            value = default_values[feature]
        else:
            value = st.session_state.get(f"{category}_{feature}", default_values[feature])
        
        col_a, col_b = st.columns([1, 0.5])
        with col_a:
            st.markdown(f"<div class='input-label' style='background-color: {color};'>{feature}</div>", unsafe_allow_html=True)
        with col_b:
            # 关键修改: 设置步长为0.01以支持两位小数
            features[feature] = st.number_input(
                "", 
                min_value=0.00, 
                max_value=100.00, 
                value=float(value), 
                step=0.01,  # 设置为0.01允许两位小数输入
                key=f"{category}_{feature}", 
                format="%.2f",  # 强制显示两位小数
                label_visibility="collapsed"
            )
            
            # 调试显示
            st.markdown(f"<span style='font-size:10px;color:gray;'>输入值: {features[feature]:.2f}</span>", unsafe_allow_html=True)

# Pyrolysis Conditions - 第三列
with col3:
    category = "Pyrolysis Conditions"
    color = category_colors[category]
    st.markdown(f"<div class='section-header' style='background-color: {color};'>{category}</div>", unsafe_allow_html=True)
    
    for feature in feature_categories[category]:
        if st.session_state.clear_pressed:
            value = default_values[feature]
        else:
            value = st.session_state.get(f"{category}_{feature}", default_values[feature])
        
        # 根据特征设置范围
        if feature == "PT(°C)":
            min_val, max_val = 200.00, 900.00
        elif feature == "HR(℃/min)":
            min_val, max_val = 1.00, 100.00
        elif feature == "RT(min)":
            min_val, max_val = 0.00, 120.00
        else:
            min_val, max_val = 0.00, 100.00
        
        col_a, col_b = st.columns([1, 0.5])
        with col_a:
            st.markdown(f"<div class='input-label' style='background-color: {color};'>{feature}</div>", unsafe_allow_html=True)
        with col_b:
            # 关键修改: 设置步长为0.01以支持两位小数
            features[feature] = st.number_input(
                "", 
                min_value=float(min_val), 
                max_value=float(max_val), 
                value=float(value), 
                step=0.01,  # 设置为0.01允许两位小数输入
                key=f"{category}_{feature}", 
                format="%.2f",  # 强制显示两位小数
                label_visibility="collapsed"
            )
            
            # 调试显示
            st.markdown(f"<span style='font-size:10px;color:gray;'>输入值: {features[feature]:.2f}</span>", unsafe_allow_html=True)

# 重置状态
if st.session_state.clear_pressed:
    st.session_state.clear_pressed = False

# 预测结果显示区域
result_container = st.container()

# 预测按钮区域
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🔮 运行预测", use_container_width=True, type="primary"):
        log(f"开始{st.session_state.selected_model}预测")
        st.session_state.predictions_running = True
        st.session_state.prediction_error = None  # 清除之前的错误
        
        # 记录输入
        log(f"输入特征: {features}")
        
        # 创建输入数据框
        input_df = pd.DataFrame([features])
        
        # 检查输入范围
        warnings = predictor.check_input_range(input_df)
        st.session_state.warnings = warnings
        
        # 执行预测
        try:
            result, individual_preds = predictor.predict(input_df, return_individual=True)
            # 确保结果不为空，修复预测值不显示的问题
            if result is not None and len(result) > 0:
                st.session_state.prediction_result = float(result[0])
                st.session_state.individual_predictions = individual_preds
                log(f"预测成功: {st.session_state.prediction_result:.2f}")
                
                # 计算标准差作为不确定性指标
                std_dev = np.std(individual_preds) if individual_preds else 0
                log(f"预测标准差: {std_dev:.4f}")
            else:
                log("警告: 预测结果为空")
                st.session_state.prediction_result = 0.0
                st.session_state.individual_predictions = []
            
        except Exception as e:
            st.session_state.prediction_error = str(e)
            log(f"预测错误: {str(e)}")
            log(traceback.format_exc())
            st.error(f"预测过程中发生错误: {str(e)}")
        
        st.session_state.predictions_running = False
        st.rerun()

with col2:
    if st.button("🔄 重置输入", use_container_width=True):
        log("重置所有输入值")
        st.session_state.clear_pressed = True
        st.session_state.prediction_result = None
        st.session_state.warnings = []
        st.session_state.individual_predictions = []
        st.session_state.prediction_error = None
        st.rerun()

# 显示预测结果
if st.session_state.prediction_result is not None:
    st.markdown("---")
    
    # 显示主预测结果
    result_container.markdown(f"<div class='yield-result'>{st.session_state.selected_model}: {st.session_state.prediction_result:.2f}%</div>", unsafe_allow_html=True)
    
    # 显示警告
    if st.session_state.warnings:
        warnings_html = "<div class='warning-box'><b>⚠️ 警告：部分输入超出训练范围</b><ul>"
        for warning in st.session_state.warnings:
            warnings_html += f"<li>{warning}</li>"
        warnings_html += "</ul><p>预测结果可能不太可靠。</p></div>"
        result_container.markdown(warnings_html, unsafe_allow_html=True)
    
    # 标准化器状态
    if len(predictor.scalers) < len(predictor.models):
        result_container.markdown(
            "<div class='warning-box'><b>⚠️ 注意：</b> 部分模型使用了最终标准化器而非其对应的子模型标准化器，这可能影响预测精度。</div>", 
            unsafe_allow_html=True
        )
    
    # 技术说明部分 - 使用折叠式展示
    with st.expander("技术说明"):
        st.markdown("""
        <div class='tech-info'>
        <p>本模型基于多个CatBoost模型集成创建，预测生物质热解产物分布。模型使用生物质的元素分析、近似分析数据和热解条件作为输入，计算热解炭、热解油和热解气体产量。</p>
        
        <p><b>特别提醒：</b></p>
        <ul>
            <li>输入参数应该满足设定好的范围内，因为这样符合模型训练数据的分布范围，可以保证软件的预测精度，如果超过范围，会有文字提醒</li>
            <li>由于模型训练时FC(%)通过100-Ash(%)-VM(%)公式转换得出，所以用户使用此软件进行预测时也需要使用100-Ash(%)-VM(%)公式对FC(%)进行转换，以保证预测的准确性。</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# 添加页脚
st.markdown("---")
footer = """
<div style='text-align: center;'>
<p>© 2023 生物质纳米材料与智能装备实验室团队. 版本: 2.3.0</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
