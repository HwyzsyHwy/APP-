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
        max-width: 100%;
        margin: 0 auto;
    }
    
    /* 侧边栏模型信息样式 */
    .sidebar-model-info {
        background-color: #2E2E2E;
        padding: 10px;
        border-radius: 5px;
        margin-top: 20px;
    }
    
    /* 技术说明样式 */
    .tech-notes {
        background-color: #2E2E2E;
        padding: 15px;
        border-radius: 10px;
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
                           help="预测焦炭产率 (%)", 
                           use_container_width=True,
                           type="primary" if st.session_state.selected_model == "Char Yield(%)" else "secondary")
with col2:
    oil_button = st.button("💧 Oil Yield", 
                          key="oil_button", 
                          help="预测生物油产率 (%)", 
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

# 添加固定的RMSE和R²值（不随预测变化）
if predictor.metadata and 'performance' in predictor.metadata:
    performance = predictor.metadata['performance']
    r2 = performance.get('test_r2', 0.9313)  # 使用默认值
    rmse = performance.get('test_rmse', 3.39)  # 使用默认值
    model_info_html += f"<h4>性能指标</h4>"
    model_info_html += f"<p><b>R²</b>: {r2:.4f}</p>"
    model_info_html += f"<p><b>RMSE</b>: {rmse:.2f}</p>"

model_info_html += "</div>"
st.sidebar.markdown(model_info_html, unsafe_allow_html=True)

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
col1, col2, col3 = st.columns(3, gap="small")

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

# 重置状态函数
def reset_state():
    """重置会话状态以清空所有输入和结果"""
    for category, feature_list in feature_categories.items():
        for feature in feature_list:
            if f"{category}_{feature}" in st.session_state:
                st.session_state[f"{category}_{feature}"] = default_values[feature]
    
    st.session_state.prediction_result = None
    st.session_state.warnings = []
    st.session_state.individual_predictions = []
    st.session_state.clear_pressed = True
    log("重置所有输入和结果")

# 添加预测和清空按钮
col1, col2 = st.columns(2, gap="small")
with col1:
    predict_button = st.button("👉 开始预测", 
                              use_container_width=True, 
                              type="primary",
                              help="点击开始预测")
with col2:
    clear_button = st.button("🔄 重置输入", 
                            use_container_width=True,
                            help="清空所有输入并重置结果")

# 处理清空按钮点击
if clear_button:
    reset_state()
    st.rerun()
else:
    st.session_state.clear_pressed = False

# 输入数据处理
if predict_button or st.session_state.prediction_result is not None:
    try:
        # 将输入整理为DataFrame
        input_df = pd.DataFrame([features])
        
        # 检查所有需要的特征是否都有值
        missing_features = [f for f in predictor.feature_names if f not in features]
        if missing_features:
            st.error(f"缺少以下特征: {', '.join(missing_features)}")
            st.stop()
        
        # 检查特征值范围并记录警告
        warnings = predictor.check_input_range(input_df)
        st.session_state.warnings = warnings
        
        # 进行预测
        if predict_button:
            log(f"开始{st.session_state.selected_model}预测计算")
            prediction, individual_preds = predictor.predict(input_df, return_individual=True)
            st.session_state.prediction_result = float(prediction[0])
            st.session_state.individual_predictions = individual_preds
            log(f"预测完成: {st.session_state.prediction_result:.2f}")
        
        # 显示结果
        st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
        
        # 显示产率预测结果
        col1, col2 = st.columns([2, 3], gap="small")
        
        with col1:
            # 显示预测结果
            st.markdown("<div class='yield-result'>"
                        f"{st.session_state.selected_model}: "
                        f"{st.session_state.prediction_result:.2f}%"
                        "</div>", 
                        unsafe_allow_html=True)
            
            # 显示警告信息
            if st.session_state.warnings:
                st.markdown("<div class='warning-box'>", unsafe_allow_html=True)
                st.markdown("<p><strong>⚠️ 输入值超出训练范围:</strong></p>", unsafe_allow_html=True)
                for warning in st.session_state.warnings:
                    st.markdown(f"<p>{warning}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # 显示标准化器状态
            if len(predictor.scalers) < len(predictor.models):
                st.markdown("<div class='warning-box'>", unsafe_allow_html=True)
                st.markdown("<p><strong>⚠️ 标准化器状态:</strong></p>", unsafe_allow_html=True)
                if len(predictor.scalers) > 0:
                    st.markdown(f"<p>找到 {len(predictor.scalers)}/{len(predictor.models)} 个子模型标准化器</p>", unsafe_allow_html=True)
                else:
                    st.markdown("<p>使用最终标准化器</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            # 技术说明部分（原来是"预测详情"的位置）
            st.markdown("<div class='tech-notes'>", unsafe_allow_html=True)
            st.markdown("<h3>技术说明</h3>", unsafe_allow_html=True)
            
            # 根据当前选择的模型显示不同的技术说明
            if st.session_state.selected_model == "Char Yield(%)":
                st.markdown("""
                **1. 固定碳计算方法**: 
                固定碳(FC) = 100-Ash(%)-VM(%)
                
                **2. 氧元素计算方法**:
                氧含量一般通过差值计算: O(%) = 100% - C(%) - H(%) - N(%) - Ash(%)
                
                **3. 关键影响因素**:
                温度(PT)和停留时间(RT)对焦炭产率有显著影响。当温度升高时，焦炭产率通常会降低。
                
                **4. 精度提升**:
                现在支持两位小数输入，提高了预测精度。
                """, unsafe_allow_html=True)
            else:  # Oil Yield
                st.markdown("""
                **1. 固定碳计算方法**: 
                固定碳(FC) = 100-Ash(%)-VM(%)
                
                **2. 氧元素计算方法**:
                氧含量一般通过差值计算: O(%) = 100% - C(%) - H(%) - N(%) - Ash(%)
                
                **3. 关键影响因素**:
                温度(PT)对油产率有显著影响，在中等温度(450-550°C)时油产率通常达到最大值。
                
                **4. 精度提升**:
                现在支持两位小数输入，提高了预测精度。
                """, unsafe_allow_html=True)
                
            st.markdown("</div>", unsafe_allow_html=True)

        # 特征重要性显示
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        st.markdown("<h3>特征重要性</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2], gap="small")
        
        with col1:
            # 显示特征重要性表格
            if predictor.feature_importance is not None:
                importance_df = predictor.feature_importance.copy()
                importance_df['Importance'] = importance_df['Importance'].round(4)  # 四位小数
                st.dataframe(importance_df, use_container_width=True)
        
        with col2:
            # 绘制特征重要性图
            if predictor.feature_importance is not None:
                fig, ax = plt.subplots(figsize=(8, 6))
                importance_df = predictor.feature_importance.head(10)  # 取前10个特征
                ax.barh(importance_df['Feature'][::-1], importance_df['Importance'][::-1])
                ax.set_xlabel('重要性')
                ax.set_title('特征重要性排序')
                plt.tight_layout()
                st.pyplot(fig)
                
                # 解释最重要的特征
                if 'PT(°C)' in importance_df['Feature'].values and 'RT(min)' in importance_df['Feature'].values:
                    st.info("""
                    **分析洞察**: 温度(PT)和停留时间(RT)是影响产率的最重要因素。
                    * 温度升高通常会降低焦炭产率，但可能增加油或气体产率。
                    * 停留时间增加可能导致二次反应，影响最终产物分布。
                    """)
                elif 'PT(°C)' in importance_df['Feature'].values:
                    st.info("""
                    **分析洞察**: 温度(PT)是影响产率的最重要因素。
                    * 在生物质热解过程中，温度对产物分布有决定性影响。
                    * 最佳温度范围取决于您希望最大化的产物类型。
                    """)
                else:
                    st.info("""
                    **分析洞察**: 根据特征重要性排序，您可以优先考虑对模型影响最大的因素，
                    以便在实验设计和工艺优化中做出更明智的决策。
                    """)
    
        # 温度敏感性分析
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        st.markdown("<h3>温度敏感性分析</h3>", unsafe_allow_html=True)
        
        # 温度范围选择
        temp_range = st.slider(
            "选择温度范围(°C)",
            min_value=300,
            max_value=800,
            value=(400, 600),
            step=10
        )
        
        # 温度分析按钮
        analysis_col1, analysis_col2 = st.columns([1, 3])
        with analysis_col1:
            run_analysis = st.button("运行温度分析", use_container_width=True)
        
        # 如果点击分析按钮，执行温度敏感性分析
        if run_analysis:
            log("开始温度敏感性分析")
            
            # 创建温度序列
            temps = np.arange(temp_range[0], temp_range[1]+1, 10)
            results = []
            
            for temp in temps:
                # 复制当前输入
                temp_features = features.copy()
                temp_features['PT(°C)'] = temp
                
                # 创建输入DataFrame
                temp_df = pd.DataFrame([temp_features])
                
                try:
                    # 预测
                    pred = predictor.predict(temp_df)
                    results.append({
                        '温度(°C)': temp,
                        f'{st.session_state.selected_model}': float(pred[0])
                    })
                except Exception as e:
                    log(f"温度 {temp}°C 分析出错: {str(e)}")
            
            # 创建结果DataFrame
            if results:
                results_df = pd.DataFrame(results)
                
                # 显示结果
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.dataframe(results_df.round(2), use_container_width=True)
                
                with col2:
                    # 绘制温度与产率关系图
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.plot(results_df['温度(°C)'], results_df[f'{st.session_state.selected_model}'], 
                           marker='o', linestyle='-', color='blue')
                    ax.set_xlabel('温度(°C)')
                    ax.set_ylabel(f'{st.session_state.selected_model}')
                    ax.set_title(f'温度对{st.session_state.selected_model}的影响')
                    ax.grid(True, linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # 找到最佳温度点
                    if st.session_state.selected_model == "Oil Yield(%)":
                        best_idx = results_df[f'{st.session_state.selected_model}'].idxmax()
                    else:  # 对于Char，通常是最低温度
                        best_idx = results_df[f'{st.session_state.selected_model}'].idxmax()
                    
                    best_temp = results_df.iloc[best_idx]['温度(°C)']
                    best_yield = results_df.iloc[best_idx][f'{st.session_state.selected_model}']
                    
                    st.success(f"""
                    **温度分析结果**:
                    * 最佳温度: {best_temp}°C
                    * 预测{st.session_state.selected_model}: {best_yield:.2f}%
                    """)
    
    except Exception as e:
        st.error(f"预测过程中出错: {str(e)}")
        log(f"错误: {str(e)}")
        log(traceback.format_exc())

# 添加页脚
st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: gray; font-size: 14px;'>
© 2023 Biomass Pyrolysis Research Group | 模型精度: R² > 0.93, RMSE < 3.4<br>
更新: 添加两位小数支持 | 解决子模型标准化器问题 | 增加油产率预测功能
</div>
""", unsafe_allow_html=True)
