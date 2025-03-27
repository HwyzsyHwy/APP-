# -*- coding: utf-8 -*-
"""
Biomass Pyrolysis Yield Forecast using CatBoost Ensemble Models
修复版本 - 解决小数精度问题和子模型标准化器问题
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

# 主标题
st.markdown("<h1 class='main-title'>Prediction of crop biomass pyrolysis yield based on CatBoost ensemble modeling</h1>", unsafe_allow_html=True)

class CorrectedEnsemblePredictor:
    """修复版集成模型预测器 - 解决子模型标准化器问题"""
    
    def __init__(self):
        self.models = []
        self.scalers = []  # 每个子模型的标准化器
        self.final_scaler = None  # 最终标准化器（备用）
        self.model_weights = None
        self.feature_names = None
        self.target_name = "Char Yield(%)"
        self.metadata = None
        self.model_dir = None
        self.feature_importance = None
        self.training_ranges = {}
        
        # 加载模型
        self.load_model()
    
    def find_model_directory(self):
        """查找模型目录的多种方法"""
        # 模型目录可能的路径
        possible_dirs = [
            # 直接路径
            "Char_Yield_Model",
            "Char_Yield%_Model",
            # 相对路径
            "./Char_Yield_Model",
            "./Char_Yield%_Model",
            "../Char_Yield_Model",
            "../Char_Yield%_Model",
            # 绝对路径示例
            "C:/Users/HWY/Desktop/方-3/Char_Yield_Model",
            "C:/Users/HWY/Desktop/方-3/Char_Yield%_Model"
        ]
        
        # 尝试所有可能路径
        for dir_path in possible_dirs:
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                log(f"找到模型目录: {dir_path}")
                return os.path.abspath(dir_path)
        
        # 如果找不到，尝试通过模型文件推断
        try:
            model_files = glob.glob("**/model_*.joblib", recursive=True)
            if model_files:
                model_dir = os.path.dirname(os.path.dirname(model_files[0]))
                log(f"基于模型文件推断模型目录: {model_dir}")
                return model_dir
        except Exception as e:
            log(f"通过模型文件推断目录时出错: {str(e)}")
        
        # 当前目录作为最后的退路
        log("警告: 无法找到模型目录，将使用当前目录")
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
            # 1. 查找模型目录
            self.model_dir = self.find_model_directory()
            log(f"使用模型目录: {self.model_dir}")
            
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
                log("错误: 没有加载模型")
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
                log("输入特征已按照训练时的顺序排列")
            else:
                input_ordered = input_features
                log("警告: 没有特征名称列表，使用原始输入顺序")
            
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
            log(f"最终加权预测结果: {weighted_pred[0]:.2f}")
            
            if return_individual:
                return weighted_pred, individual_predictions
            else:
                return weighted_pred
            
        except Exception as e:
            log(f"预测过程中出错: {str(e)}")
            log(traceback.format_exc())
            return np.array([0.0])
    
    def get_feature_importance_plot(self):
        """生成特征重要性图"""
        if self.feature_importance is None or len(self.feature_importance) == 0:
            return None
        
        try:
            # 创建图表
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # 提取数据
            importance_df = self.feature_importance.sort_values('Importance', ascending=True)
            features = importance_df['Feature'].tolist()
            importance = importance_df['Importance'].tolist()
            
            # 创建水平条形图
            ax.barh(features, importance, color='skyblue')
            
            # 添加标题和标签
            ax.set_title('Feature Importance', fontsize=14)
            ax.set_xlabel('Importance Score', fontsize=12)
            ax.set_ylabel('Feature', fontsize=12)
            
            # 调整布局
            plt.tight_layout()
            
            # 将图表转换为图像
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            
            # 使用PIL打开图像并返回
            img = Image.open(buf)
            return img
            
        except Exception as e:
            log(f"创建特征重要性图时出错: {str(e)}")
            return None
    
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

# 初始化预测器
predictor = CorrectedEnsemblePredictor()

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
    
    clear_button = st.button("CLEAR", on_click=clear_values)

# 创建输入数据DataFrame
input_data = pd.DataFrame([features])

# 预测流程
if predict_button:
    log("="*40)
    log("开始新预测")
    
    try:
        # 检查输入范围
        warnings = predictor.check_input_range(input_data)
        st.session_state.warnings = warnings
        
        # 执行预测 - 现在使用每个子模型对应的标准化器
        result, individual_preds = predictor.predict(input_data, return_individual=True)
        
        # 保存结果
        st.session_state.prediction_result = float(result[0])
        st.session_state.individual_predictions = individual_preds
        
        log(f"预测成功完成: {st.session_state.prediction_result:.2f}")
        
    except Exception as e:
        log(f"预测过程中出错: {str(e)}")
        st.error(f"预测失败: {str(e)}")

# 显示结果
with result_container:
    # 主预测结果
    st.subheader("Char Yield (wt%)")
    
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

# 特征重要性和模型信息部分
col1, col2 = st.columns([1, 1])

with col1:
    # 特征重要性部分
    st.subheader("特征重要性")
    
    if predictor.feature_importance is not None:
        # 显示特征重要性表格
        importance_df = predictor.feature_importance.copy()
        
        # 格式化重要性分数，使用4位小数
        formatted_df = importance_df.copy()
        formatted_df['Importance'] = formatted_df['Importance'].apply(lambda x: f"{x:.4f}")
        
        st.dataframe(formatted_df, use_container_width=True)
        
        # 显示特征重要性图
        importance_img = predictor.get_feature_importance_plot()
        if importance_img:
            st.image(importance_img, use_column_width=True)
        
        # 提供特征重要性的洞察
        st.markdown("#### 重要特征洞察")
        
        # 获取前两个最重要的特征
        top_features = importance_df['Feature'].tolist()[:2]
        
        if 'PT(°C)' in top_features:
            st.info("""
            📌 **温度(PT)** 是影响产率的最重要因素，这与热解理论一致：
            - 较低温度下，生物质降解不完全，导致焦炭产率较高
            - 随着温度升高，热解反应更彻底，气体产物增加，焦炭产率下降
            """)
        
        if 'RT(min)' in top_features:
            st.info("""
            📌 **停留时间(RT)** 显著影响热解程度：
            - 较短的停留时间可能导致热解不完全
            - 较长的停留时间允许更多的挥发分释放，减少焦炭产率
            """)
    else:
        st.warning("无法加载特征重要性数据")

with col2:
    # 关于模型部分
    st.subheader("关于模型")
    
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

# 调试信息区域
with st.expander("调试信息", expanded=False):
    st.markdown("### 输入特征详情")
    # 显示带两位小数格式的输入特征
    formatted_features = {k: f"{v:.2f}" for k, v in features.items()}
    st.json(formatted_features)
    
    st.markdown("### 模型信息")
    st.json({
        "模型数量": len(predictor.models),
        "标准化器数量": len(predictor.scalers),
        "特征数量": len(predictor.feature_names) if predictor.feature_names else 0,
        "特征列表": predictor.feature_names,
        "模型目录": predictor.model_dir
    })
    
    st.markdown("### 标准化器信息")
    if predictor.final_scaler and hasattr(predictor.final_scaler, 'mean_'):
        scaler_info = {
            "均值": predictor.final_scaler.mean_.tolist(),
            "标准差": predictor.final_scaler.scale_.tolist() if hasattr(predictor.final_scaler, 'scale_') else None
        }
        st.json(scaler_info)
    else:
        st.warning("最终标准化器信息不可用")

# 技术说明区域
with st.expander("技术说明", expanded=False):
    st.markdown("""
    ### 预测精度说明
    
    本模型是基于CatBoost的集成学习模型，通过10个子模型共同预测以提高准确性和稳定性。模型在测试集上达到了约0.93的R²和3.39的RMSE。
    
    #### 已修复的问题
    
    1. **子模型标准化器问题**: 应用正确加载并应用每个子模型的标准化器，确保特征的标准化与训练时一致。
    2. **输入精度问题**: 允许输入两位小数而不是一位，减少舍入误差。
    
    #### 使用建议
    
    1. 尽量使用在训练范围内的输入值，超出范围的预测可能不准确。
    2. 对于生物质热解，温度(PT)和停留时间(RT)是最关键的参数，建议重点关注这些参数的设置。
    3. 如果多个子模型的预测差异较大(标准差>3)，表明当前输入条件下的预测可能不稳定。
    """)

# 温度敏感性分析
with st.expander("温度敏感性分析", expanded=False):
    st.markdown("### 分析温度对产率的影响")
    
    # 温度范围滑块
    temp_range = st.slider("温度范围(°C)", 
                          min_value=200, 
                          max_value=900, 
                          value=(300, 700),
                          step=50)
    
    # 温度步长
    temp_step = st.selectbox("温度步长", options=[10, 25, 50, 100], index=1)
    
    # 执行分析按钮
    if st.button("运行温度敏感性分析"):
        # 创建温度序列
        temps = np.arange(temp_range[0], temp_range[1] + 1, temp_step)
        
        # 创建保存当前输入特征的副本
        base_features = features.copy()
        
        # 结果容器
        results = []
        
        # 执行预测
        for temp in temps:
            temp_features = base_features.copy()
            temp_features['PT(°C)'] = temp
            
            # 创建输入DataFrame
            temp_input = pd.DataFrame([temp_features])
            
            # 预测
            try:
                pred = predictor.predict(temp_input)
                results.append((temp, float(pred[0])))
            except Exception as e:
                st.error(f"温度 {temp}°C 预测失败: {str(e)}")
        
        # 显示结果
        if results:
            # 创建DataFrame
            result_df = pd.DataFrame(results, columns=['温度(°C)', '预测产率(%)'])
            
            # 显示表格
            st.dataframe(result_df.style.format({
                '温度(°C)': '{:.0f}',
                '预测产率(%)': '{:.2f}'
            }))
            
            # 绘制曲线
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(result_df['温度(°C)'], result_df['预测产率(%)'], marker='o', linewidth=2)
            ax.set_xlabel('温度(°C)', fontsize=12)
            ax.set_ylabel('预测产率(%)', fontsize=12)
            ax.set_title('温度对产率的影响', fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # 添加当前温度标记
            current_temp = base_features['PT(°C)']
            if temp_range[0] <= current_temp <= temp_range[1]:
                # 找到最接近的预测点
                closest_idx = np.abs(result_df['温度(°C)'] - current_temp).argmin()
                closest_temp = result_df.iloc[closest_idx]['温度(°C)']
                closest_yield = result_df.iloc[closest_idx]['预测产率(%)']
                
                # 标记当前温度点
                ax.scatter([closest_temp], [closest_yield], color='red', s=100, zorder=5, 
                           label=f'当前温度: {current_temp:.0f}°C')
                ax.legend()
            
            st.pyplot(fig)
            
            # 找出最大和最小产率点
            max_idx = result_df['预测产率(%)'].idxmax()
            min_idx = result_df['预测产率(%)'].idxmin()
            
            max_temp = result_df.iloc[max_idx]['温度(°C)']
            max_yield = result_df.iloc[max_idx]['预测产率(%)']
            
            min_temp = result_df.iloc[min_idx]['温度(°C)']
            min_yield = result_df.iloc[min_idx]['预测产率(%)']
            
            # 显示分析结果
            st.markdown(f"""
            ### 分析结果
            
            - 在分析范围内，产率最高点为: **{max_yield:.2f}%** (温度 = {max_temp:.0f}°C)
            - 在分析范围内，产率最低点为: **{min_yield:.2f}%** (温度 = {min_temp:.0f}°C)
            - 温度变化 1°C 平均导致产率变化约 {abs(max_yield - min_yield) / abs(max_temp - min_temp):.4f}%
            """)

# 数据验证建议
with st.expander("数据验证与精度建议", expanded=False):
    st.markdown("""
    ### 提高预测精度的建议
    
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
    """)

# 页脚
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 14px;">
    © 2023 生物质热解产率预测系统 | 模型精度: R² = 0.93, RMSE = 3.39 | 集成 10 个 CatBoost 子模型<br>
    版本更新: 支持两位小数输入 & 修复子模型标准化器问题
</div>
""", unsafe_allow_html=True)





