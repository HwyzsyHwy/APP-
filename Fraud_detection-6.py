# -*- coding: utf-8 -*-
"""
Biomass Pyrolysis Yield Forecast using CatBoost Ensemble Models
修复版本 - 解决小数精度问题和子模型标准化器问题
添加多模型切换功能 - 支持Char和Oil产率预测
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

# 添加调试信息
st.sidebar.write(f"调试信息: 支持两位小数测试值 = {st.session_state.decimal_test:.2f}")

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
    
    /* 居中显示内容 */
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
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
log("应用启动 - 支持两位小数和模型切换功能")

# 初始化会话状态 - 添加模型选择功能
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Char Yield(%)"
    log(f"初始化选定模型: {st.session_state.selected_model}")

# 更新主标题以显示当前选定的模型
st.markdown("<h1 class='main-title'>Prediction of biomass pyrolysis yield based on CatBoost ensemble modeling</h1>", unsafe_allow_html=True)

# 添加模型选择区域
st.markdown("<div class='model-selector'>", unsafe_allow_html=True)
st.markdown("<h3>选择预测目标</h3>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    char_button = st.button("🔥 Char Yield", 
                           key="char_button", 
                           help="预测焦炭产率 (wt%)", 
                           use_container_width=True,
                           type="primary" if st.session_state.selected_model == "Char Yield(%)" else "secondary")
with col2:
    oil_button = st.button("💧 Oil Yield", 
                          key="oil_button", 
                          help="预测生物油产率 (wt%)", 
                          use_container_width=True,
                          type="primary" if st.session_state.selected_model == "Oil Yield(%)" else "secondary")

# 处理模型选择
if char_button:
    st.session_state.selected_model = "Char Yield(%)"
    st.session_state.prediction_result = None
    st.session_state.warnings = []
    st.session_state.individual_predictions = []
    log(f"切换到模型: {st.session_state.selected_model}")
    st.rerun()  # 使用st.rerun()代替st.experimental_rerun()

if oil_button:
    st.session_state.selected_model = "Oil Yield(%)"
    st.session_state.prediction_result = None
    st.session_state.warnings = []
    st.session_state.individual_predictions = []
    log(f"切换到模型: {st.session_state.selected_model}")
    st.rerun()  # 使用st.rerun()代替st.experimental_rerun()

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
        
        # 加载模型
        self.load_model()
    
    def find_model_directory(self):
        """查找模型目录的多种方法，支持不同模型类型"""
        # 根据目标变量确定模型目录名称
        model_name = self.target_name.replace(' ', '_').replace('(', '').replace(')', '')
        
        # 模型目录可能的路径
        possible_dirs = [
            # 直接路径
            f"{model_name}_Model",
            # 相对路径
            f"./{model_name}_Model",
            f"../{model_name}_Model",
            # 绝对路径示例
            f"C:/Users/HWY/Desktop/方-3/{model_name}_Model"
        ]
        
        # 尝试所有可能路径
        for dir_path in possible_dirs:
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                log(f"找到模型目录: {dir_path}")
                return os.path.abspath(dir_path)
        
        # 如果找不到，尝试通过模型文件推断
        try:
            model_files = glob.glob(f"**/{model_name}_Model/models/model_*.joblib", recursive=True)
            if model_files:
                model_dir = os.path.dirname(os.path.dirname(model_files[0]))
                log(f"基于模型文件推断模型目录: {model_dir}")
                return model_dir
        except Exception as e:
            log(f"通过模型文件推断目录时出错: {str(e)}")
        
        # 当前目录作为最后的退路
        log(f"警告: 无法找到{self.target_name}模型目录，将使用当前目录")
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
                    log(f"未找到模型文件在 {models_dir}")
                    return False
            else:
                log(f"模型目录不存在: {models_dir}")
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
                self.model_weights = np.ones(len(self.models)) / len(self.models)
                log("警告: 未找到权重文件，使用均等权重")
            
            # 7. 加载特征重要性
            self.load_feature_importance()
            
            # 验证加载状态
            log(f"成功加载 {len(self.models)} 个模型和 {len(self.scalers)} 个子模型标准化器")
            
            # 特别标记标准化器问题
            if len(self.models) != len(self.scalers):
                log(f"警告: 模型数量 ({len(self.models)}) 与标准化器数量 ({len(self.scalers)}) 不匹配")
                
            return True
            
        except Exception as e:
            log(f"加载模型时出错: {str(e)}")
            log(traceback.format_exc())
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
            if not self.models or len(self.models) == 0:
                log(f"错误: 没有加载{self.target_name}模型")
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
                            log(f"错误: 模型 {i} 没有可用的标准化器")
                            continue
                    
                    pred = model.predict(X_scaled)
                    all_predictions[:, i] = pred
                    individual_predictions.append(float(pred[0]))
                    log(f"模型 {i} 预测结果: {pred[0]:.2f}")
                except Exception as e:
                    log(f"模型 {i} 预测时出错: {str(e)}")
                    # 如果某个模型失败，使用其他模型的平均值
                    if i > 0:
                        avg_pred = np.mean(all_predictions[:, :i], axis=1)
                        all_predictions[:, i] = avg_pred
                        individual_predictions.append(float(avg_pred[0]))
                        log(f"模型 {i} 使用之前模型的平均值: {avg_pred[0]:.2f}")
            
            # 计算加权平均
            weighted_pred = np.sum(all_predictions * self.model_weights.reshape(1, -1), axis=1)
            log(f"{self.target_name}最终加权预测结果: {weighted_pred[0]:.2f}")
            
            if return_individual:
                return weighted_pred, individual_predictions
            else:
                return weighted_pred
            
        except Exception as e:
            log(f"预测过程中出错: {str(e)}")
            log(traceback.format_exc())
            return np.array([0.0])
    
    def get_model_info(self):
        """获取模型信息摘要"""
        info = {
            "模型类型": "CatBoost集成模型",
            "模型数量": len(self.models),
            "特征数量": len(self.feature_names) if self.feature_names else 0,
            "目标变量": self.target_name
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

# 初始化会话状态
if 'clear_pressed' not in st.session_state:
    st.session_state.clear_pressed = False
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'warnings' not in st.session_state:
    st.session_state.warnings = []
if 'individual_predictions' not in st.session_state:
    st.session_state.individual_predictions = []

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
col1, col2 = st.columns([5, 1])

with col2:
    # 预测按钮
    predict_button = st.button("PUSH", type="primary")
    
    # Clear按钮
    def clear_values():
        st.session_state.clear_pressed = True
        st.session_state.prediction_result = None
        st.session_state.warnings = []
        st.session_state.individual_predictions = []
        log("清除所有输入和预测结果")
    
    clear_button = st.button("CLEAR", on_click=clear_values)

# 创建输入数据DataFrame
input_data = pd.DataFrame([features])

# 预测流程
if predict_button:
    log("="*40)
    log(f"开始新的{st.session_state.selected_model}预测")
    
    try:
        # 确保使用正确的模型
        if predictor.target_name != st.session_state.selected_model:
            log(f"重新加载{st.session_state.selected_model}模型")
            predictor = CorrectedEnsemblePredictor(target_model=st.session_state.selected_model)
        
        # 检查输入范围
        warnings = predictor.check_input_range(input_data)
        st.session_state.warnings = warnings
        
        # 执行预测 - 现在使用每个子模型对应的标准化器
        result, individual_preds = predictor.predict(input_data, return_individual=True)
        
        # 保存结果
        st.session_state.prediction_result = float(result[0])
        st.session_state.individual_predictions = individual_preds
        
        log(f"{st.session_state.selected_model}预测成功完成: {st.session_state.prediction_result:.2f}")
    except Exception as e:
        log(f"预测过程中出错: {str(e)}")
        st.error(f"预测失败: {str(e)}")
        st.error(traceback.format_exc())

# 显示结果
with result_container:
    # 主预测结果 - 显示当前选择的模型结果
    if st.session_state.selected_model == "Char Yield(%)":
        result_label = "Char Yield (wt%)"
    else:
        result_label = "Oil Yield (%)"
    
    st.subheader(result_label)
    
    if st.session_state.prediction_result is not None:
        # 显示预测结果
        st.markdown(
            f"<div class='yield-result'>{st.session_state.prediction_result:.2f}</div>",
            unsafe_allow_html=True
        )
        
        # 显示警告
        if st.session_state.warnings:
            warning_html = "<div class='warning-box'><b>⚠️ 警告:</b> 以下输入值超出训练范围，可能影响预测准确性:<ul>"
            for warning in st.session_state.warnings:
                warning_html += f"<li>{warning}</li>"
            warning_html += "</ul></div>"
            st.markdown(warning_html, unsafe_allow_html=True)
        
        # 标准化器状态提示
        if len(predictor.scalers) == len(predictor.models):
            st.markdown(
                "<div class='success-box'>✅ 每个子模型都使用了对应的标准化器，预测结果可靠度高。</div>",
                unsafe_allow_html=True
            )
        elif len(predictor.scalers) > 0:
            st.markdown(
                "<div class='warning-box'>⚠️ 部分子模型使用了对应的标准化器，部分使用了最终标准化器，预测结果可能存在轻微偏差。</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='error-box'>❌ 没有找到子模型对应的标准化器，所有模型使用最终标准化器，预测结果可能存在较大偏差。</div>",
                unsafe_allow_html=True
            )
        
        # 模型详细信息区域
        with st.expander("预测详情", expanded=False):
            # 显示各个模型的预测结果
            if st.session_state.individual_predictions:
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.write("### 各子模型预测值")
                    pred_df = pd.DataFrame({
                        '模型': [f"模型 {i+1}" for i in range(len(st.session_state.individual_predictions))],
                        '预测值': st.session_state.individual_predictions,
                        '偏差': [p - st.session_state.prediction_result for p in st.session_state.individual_predictions]
                    })
                    st.dataframe(pred_df.style.format({
                        '预测值': '{:.2f}',
                        '偏差': '{:.2f}'
                    }))
                    
                    # 计算标准差
                    std_dev = np.std(st.session_state.individual_predictions)
                    st.write(f"模型间预测标准差: {std_dev:.2f}")
                    if std_dev > 3.0:
                        st.warning("⚠️ 标准差较大，表示模型预测一致性较低")
                    elif std_dev < 1.0:
                        st.success("✅ 标准差较小，表示模型预测一致性高")
                with col2:
                    # 显示子模型预测分布图
                    st.write("### 预测分布")
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.hist(st.session_state.individual_predictions, bins=5, alpha=0.7, color='skyblue')
                    ax.axvline(st.session_state.prediction_result, color='red', linestyle='--', linewidth=2, label='最终预测')
                    ax.set_xlabel('预测值')
                    ax.set_ylabel('频率')
                    ax.legend()
                    st.pyplot(fig)
            
            # 显示输入特征表
            st.write("### 输入特征")
            input_df = pd.DataFrame([features])
            
            # 格式化为两位小数显示
            display_df = input_df.applymap(lambda x: f"{x:.2f}")
            st.dataframe(display_df)

# 关于模型部分 - 移除特征重要性部分，仅保留关于模型的基本信息，并居中显示
st.subheader("关于模型")

# 居中显示模型信息
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # 获取模型信息
    model_info = predictor.get_model_info()
    
    # 创建信息表
    for key, value in model_info.items():
        st.markdown(f"**{key}**: {value}")
    
    # 标准化器状态
    st.markdown("#### 标准化器状态")
    if len(predictor.scalers) == len(predictor.models):
        st.success(f"✅ 所有 {len(predictor.models)} 个子模型都使用了对应的标准化器")
    elif len(predictor.scalers) > 0:
        st.warning(f"⚠️ 找到 {len(predictor.scalers)}/{len(predictor.models)} 个子模型标准化器")
    else:
        st.error("❌ 未找到子模型标准化器，使用最终标准化器")
    
    # 子模型与标准差可视化
    st.markdown("#### 预测标准差")
    if st.session_state.individual_predictions:
        std_dev = np.std(st.session_state.individual_predictions)
        
        # 创建进度条表示标准差
        st.progress(min(std_dev / 5.0, 1.0))  # 标准化到0-1范围
        
        # 根据标准差大小显示不同消息
        if std_dev < 1.0:
            st.success(f"预测一致性高 (标准差 = {std_dev:.2f})")
        elif std_dev < 3.0:
            st.info(f"预测一致性中等 (标准差 = {std_dev:.2f})")
        else:
            st.warning(f"预测一致性低 (标准差 = {std_dev:.2f})")

# 技术说明区域
with st.expander("技术说明", expanded=False):
    st.markdown(f"""
    ### {st.session_state.selected_model}预测模型精度说明
    
    本模型是基于CatBoost的集成学习模型，通过10个子模型共同预测以提高准确性和稳定性。
    
    #### 模型性能指标
    
    {("模型在测试集上达到了约0.93的R²和3.39的RMSE。" if st.session_state.selected_model == "Char Yield(%)" else "模型在测试集上的性能根据元数据显示的指标而定。")}
    
    #### 已修复的问题
    
    1. **子模型标准化器问题**: 应用正确加载并应用每个子模型的标准化器，确保特征的标准化与训练时一致。
    2. **输入精度问题**: 允许输入两位小数而不是一位，减少舍入误差。
    3. **多模型切换功能**: 现在支持在Char和Oil产率预测之间自由切换。
    
    #### 使用建议
    
    1. 尽量使用在训练范围内的输入值，超出范围的预测会通过警告提示，但可能不准确。
    2. **预测前需要将FC(%)的值使用1-Ash(%)-VM(%)公式进行转换后再进行输入，以提高模型预测精度，因为训练模型时，就按照此公式进行了数值的转换。**
    3. 如果多个子模型的预测差异较大(标准差>3)，表明当前输入条件下的预测可能不稳定。
    """)

# 数据验证建议
with st.expander("数据验证与精度建议", expanded=False):
    st.markdown(f"""
    ### 提高{st.session_state.selected_model.replace('(%)', '')}预测精度的建议
    
    1. **确保数据质量**:
       - 使用两位小数输入可以减少舍入误差
       - 通过实验验证输入的分析数据
    
    2. **优先关注重要特征**:
       - 热解温度(PT)是最关键的参数，确保其准确性
       - 停留时间(RT)是第二重要的参数，需要精确控制
    
    3. **注意特征之间的相关性**:
       - C、H、O含量通常相关，确保它们的总和合理
       - VM和FC含量也应与元素分析结果相符
    
    4. **模型的局限性**:
       - 模型主要在训练数据范围内有效
       - 超出范围的预测会通过警告提示，但可能不准确
       
    5. **多模型比较**:
       - 考虑同时预测Char、Oil和Gas产率，验证三者总和是否接近100%
       - 显著偏离可能表明输入数据或预测结果有问题
    """)

# 页脚
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: gray; font-size: 14px;">
    © 2023 生物质热解产率预测系统 | 支持预测目标: Char Yield, Oil Yield | 集成 10 个 CatBoost 子模型<br>
    版本更新: 支持两位小数输入 & 修复子模型标准化器问题 & 添加多模型切换功能
</div>
""", unsafe_allow_html=True)
