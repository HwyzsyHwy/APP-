# -*- coding: utf-8 -*-
"""
Biomass Pyrolysis Yield Forecast using CatBoost Ensemble Models
完全优化版本 - 解决预测精度问题
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
import base64
import io
from PIL import Image
from datetime import datetime

# 页面设置
st.set_page_config(
    page_title='Biomass Pyrolysis Yield Forecast',
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

class CatBoostEnsemblePredictor:
    """增强版集成模型预测器 - 解决预测精度问题"""
    
    def __init__(self):
        self.models = []
        self.model_weights = None
        self.final_scaler = None
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
                    model_importance = model.get_feature_importance()
                    importance += model_importance * self.model_weights[i]
                
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
        """加载所有模型组件"""
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
            
            # 4. 加载权重
            weights_path = os.path.join(self.model_dir, 'model_weights.npy')
            if os.path.exists(weights_path):
                self.model_weights = np.load(weights_path)
                log(f"加载权重文件: {weights_path}")
            else:
                log(f"警告: 未找到权重文件 {weights_path}")
                # 使用均等权重
                self.model_weights = np.ones(len(self.models)) / len(self.models)
                log("使用均等权重")
            
            # 5. 加载标准化器
            scaler_path = os.path.join(self.model_dir, 'final_scaler.joblib')
            if os.path.exists(scaler_path):
                self.final_scaler = joblib.load(scaler_path)
                log(f"加载标准化器: {scaler_path}")
                
                # 打印标准化器信息
                if hasattr(self.final_scaler, 'mean_'):
                    log(f"特征均值: {self.final_scaler.mean_}")
                if hasattr(self.final_scaler, 'scale_'):
                    log(f"特征标准差: {self.final_scaler.scale_}")
                
                # 提取训练数据范围
                self.extract_training_ranges()
            else:
                log(f"错误: 未找到标准化器文件 {scaler_path}")
                return False
            
            # 6. 加载特征重要性
            self.load_feature_importance()
            
            log(f"成功加载 {len(self.models)} 个模型")
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
                    warning = f"{feature}: {value:.1f} (超出训练范围 {range_info['min']:.1f} - {range_info['max']:.1f})"
                    warnings.append(warning)
                    log(f"警告: {warning}")
        
        return warnings
    
    def predict(self, input_features, return_individual=False):
        """使用模型进行预测"""
        try:
            # 验证模型
            if not self.models or len(self.models) == 0:
                log("错误: 没有加载模型")
                return np.array([0.0])
            
            if not self.final_scaler:
                log("错误: 没有加载标准化器")
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
            
            # 按照模型定义的特征顺序重新排列
            if self.feature_names:
                input_ordered = input_features[self.feature_names].copy()
                log("输入特征已按照训练时的顺序排列")
            else:
                input_ordered = input_features
                log("警告: 没有特征名称列表，使用原始输入顺序")
            
            # 记录输入数据
            log(f"预测输入数据: {input_ordered.iloc[0].to_dict()}")
            
            # 应用标准化
            X_scaled = self.final_scaler.transform(input_ordered)
            log(f"数据已标准化，形状: {X_scaled.shape}")
            
            # 使用每个模型进行预测
            individual_predictions = []
            all_predictions = np.zeros((input_ordered.shape[0], len(self.models)))
            
            for i, model in enumerate(self.models):
                pred = model.predict(X_scaled)
                all_predictions[:, i] = pred
                individual_predictions.append(float(pred[0]))
                log(f"模型 {i} 预测结果: {pred[0]:.2f}")
            
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
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 提取数据
            features = self.feature_importance['Feature'].tolist()
            importance = self.feature_importance['Importance'].tolist()
            
            # 反转顺序，使最重要的特征显示在顶部
            features.reverse()
            importance.reverse()
            
            # 创建水平条形图
            ax.barh(features, importance, color='skyblue')
            
            # 添加标题和标签
            ax.set_title('特征重要性分析', fontsize=14)
            ax.set_xlabel('重要性得分', fontsize=12)
            ax.set_ylabel('特征', fontsize=12)
            
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
        
        return info

# 初始化预测器
predictor = CatBoostEnsemblePredictor()

# 初始化会话状态
if 'clear_pressed' not in st.session_state:
    st.session_state.clear_pressed = False
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'warnings' not in st.session_state:
    st.session_state.warnings = []
if 'individual_predictions' not in st.session_state:
    st.session_state.individual_predictions = []

# 定义默认值 - 从用户日志中提取
default_values = {
    "C(%)": 38.3,
    "H(%)": 5.5,
    "O(%)": 55.2,
    "N(%)": 0.6,
    "Ash(%)": 6.6,
    "VM(%)": 81.1,
    "FC(%)": 10.3,
    "PT(°C)": 500.0,  # 改为更合理的温度
    "HR(℃/min)": 10.0,
    "RT(min)": 60.0
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
            features[feature] = st.number_input(
                "", 
                min_value=0.0, 
                max_value=100.0, 
                value=value, 
                key=f"{category}_{feature}", 
                format="%.1f",
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
            features[feature] = st.number_input(
                "", 
                min_value=0.0, 
                max_value=100.0, 
                value=value, 
                key=f"{category}_{feature}", 
                format="%.1f",
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
            min_val, max_val = 200.0, 900.0
        elif feature == "HR(℃/min)":
            min_val, max_val = 1.0, 100.0
        elif feature == "RT(min)":
            min_val, max_val = 0.0, 120.0
        else:
            min_val, max_val = 0.0, 100.0
        
        col_a, col_b = st.columns([1, 0.5])
        with col_a:
            st.markdown(f"<div class='input-label' style='background-color: {color};'>{feature}</div>", unsafe_allow_html=True)
        with col_b:
            features[feature] = st.number_input(
                "", 
                min_value=min_val, 
                max_value=max_val, 
                value=value, 
                key=f"{category}_{feature}", 
                format="%.1f",
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
        
        # 执行预测
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
        
        # 模型详细信息区域
        with st.expander("预测详情", expanded=False):
            # 显示各个模型的预测结果
            if st.session_state.individual_predictions:
                st.write("各模型预测值:")
                pred_df = pd.DataFrame({
                    '模型': [f"模型 {i+1}" for i in range(len(st.session_state.individual_predictions))],
                    '预测值': st.session_state.individual_predictions
                })
                st.dataframe(pred_df)
                
                # 计算标准差
                std_dev = np.std(st.session_state.individual_predictions)
                st.write(f"模型间预测标准差: {std_dev:.2f} (较大的标准差表示模型意见不一致)")
                
                # 简单柱状图
                st.bar_chart(pred_df.set_index('模型'))
            
            # 显示输入特征及其重要性
            if predictor.feature_importance is not None:
                st.write("输入特征值及其重要性排名:")
                
                # 合并特征重要性和输入值
                input_values = input_data.iloc[0].to_dict()
                importance_data = predictor.feature_importance.copy()
                
                # 计算排名
                importance_data['排名'] = importance_data.index + 1
                
                # 添加输入值列
                importance_data['输入值'] = importance_data['Feature'].map(input_values)
                
                # 调整显示列
                display_df = importance_data[['排名', 'Feature', '输入值', 'Importance']]
                display_df.columns = ['排名', '特征', '输入值', '重要性得分']
                
                st.dataframe(display_df)
                
                # 显示特征重要性图
                importance_img = predictor.get_feature_importance_plot()
                if importance_img:
                    st.image(importance_img, caption="特征重要性分析", use_column_width=True)

# 添加模型信息区域
with st.expander("About the Model", expanded=False):
    # 左右两列布局
    info_col, chart_col = st.columns([1, 1])
    
    with info_col:
        st.write("### 模型信息")
        model_info = predictor.get_model_info()
        for key, value in model_info.items():
            st.write(f"**{key}:** {value}")
        
        st.write("### 关键影响因素")
        st.markdown("""
        * **热解温度(PT)**: 更高的温度通常会降低焦炭产率
        * **停留时间(RT)**: 更长的停留时间通常会增加焦炭产率
        * **生物质成分**: 碳含量和灰分含量显著影响最终产率
        """)
        
    with chart_col:
        if predictor.feature_importance is not None:
            importance_img = predictor.get_feature_importance_plot()
            if importance_img:
                st.image(importance_img, caption="特征重要性分析", use_column_width=True)

# 添加页脚
st.markdown("---")
st.caption("© 2023 Biomass Pyrolysis Research Team. All rights reserved.")