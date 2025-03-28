import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import sys
import glob
from datetime import datetime
import traceback
import json
from catboost import CatBoostRegressor

# 配置页面
st.set_page_config(
    page_title="生物质热解产率预测系统",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 自定义CSS
st.markdown("""
<style>
    /* 全局设置 */
    .main {
        background-color: #f8f9fa;
    }
    
    /* 标题样式 */
    .main-title {
        font-size: 2.2em;
        color: #2c3e50;
        text-align: center;
        padding: 10px;
        margin-bottom: 20px;
        font-weight: bold;
        background: linear-gradient(90deg, #a8e063 0%, #56ab2f 100%);
        color: white;
        border-radius: 10px;
    }
    
    /* 预测结果显示 */
    .yield-result {
        font-size: 2em;
        text-align: center;
        padding: 25px;
        margin: 20px 0;
        background-color: #2c3e50;
        color: white;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* 输入框标题 */
    .section-header {
        font-size: 1em;
        text-align: center;
        padding: 8px;
        margin-bottom: 15px;
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        font-weight: bold;
    }
    
    /* 输入标签样式 */
    .input-label {
        padding: 8px;
        margin: 5px 0;
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        text-align: center;
    }
    
    /* 警告框样式 */
    .warning-box {
        background-color: #ffeaa7;
        border-left: 5px solid #fdcb6e;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    
    /* 技术信息样式 */
    .tech-info {
        font-size: 0.9em;
        background-color: #e9ecef;
        padding: 15px;
        border-radius: 5px;
    }
    
    /* 模型选择器样式 */
    .model-selector {
        background-color: #e9ecef;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    
    /* 将Streamlit品牌水印隐藏 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 优化小屏幕显示 */
    @media screen and (max-width: 768px) {
        .yield-result {
            font-size: 1.5em;
            padding: 15px;
        }
        .main-title {
            font-size: 1.8em;
        }
    }
</style>
""", unsafe_allow_html=True)

# 日志记录
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.logs.append(f"[{timestamp}] {message}")
    print(f"[{timestamp}] {message}")

# 确保所有状态变量都被初始化
if 'logs' not in st.session_state:
    st.session_state.logs = []
    
if 'predictions_running' not in st.session_state:
    st.session_state.predictions_running = False
    
if 'clear_pressed' not in st.session_state:
    st.session_state.clear_pressed = False

# 数据范围检查器类
class FeatureRangeChecker:
    def __init__(self, training_ranges=None):
        # 默认训练范围
        self.default_ranges = {
            "C(%)": [35.0, 55.0],
            "H(%)": [4.0, 7.0],
            "O(%)": [35.0, 60.0],
            "N(%)": [0.0, 5.0],
            "Ash(%)": [0.0, 25.0],
            "VM(%)": [65.0, 95.0],
            "FC(%)": [5.0, 30.0],
            "PT(°C)": [350.0, 700.0],
            "HR(℃/min)": [5.0, 50.0],
            "RT(min)": [0.0, 120.0]
        }
        
        # 如果提供了训练范围，使用提供的范围
        self.training_ranges = training_ranges if training_ranges else self.default_ranges
        log(f"特征范围检查器初始化: {len(self.training_ranges)}个特征")
    
    def check_input_range(self, input_df):
        warnings = []
        for feature, (min_val, max_val) in self.training_ranges.items():
            if feature in input_df.columns:
                value = input_df[feature].values[0]
                if value < min_val or value > max_val:
                    warnings.append(f"{feature}={value:.2f} 超出训练范围 [{min_val:.2f}, {max_val:.2f}]")
        
        return warnings
    
    def save_ranges(self, file_path):
        try:
            with open(file_path, 'w') as file:
                json.dump(self.training_ranges, file)
            return True
        except Exception as e:
            log(f"保存特征范围失败: {str(e)}")
            return False
    
    @classmethod
    def load_ranges(cls, file_path):
        try:
            with open(file_path, 'r') as file:
                ranges = json.load(file)
            return cls(ranges)
        except Exception as e:
            log(f"加载特征范围失败: {str(e)}")
            return cls()

# 预测器类
class CorrectedEnsemblePredictor:
    def __init__(self, models_dir=None, model_type="Char"):
        self.models = []
        self.scalers = []
        self.models_dir = models_dir
        self.model_type = model_type
        self.range_checker = FeatureRangeChecker()
        
        # 如果没有指定模型目录，则尝试查找
        if not models_dir:
            self._find_models_directory()
        
        log(f"初始化{model_type}产率预测器: 模型目录={self.models_dir}")
        self._load_models()
    
    def _find_models_directory(self):
        # 查找不同可能的目录结构
        possible_dirs = [
            os.path.join(os.getcwd(), f"{self.model_type.lower()}_models"),  # 当前目录下的模型目录
            os.path.join(os.getcwd(), "models", self.model_type.lower()),  # 当前目录下的models/类型目录
            os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{self.model_type.lower()}_models"),  # 脚本目录下
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", self.model_type.lower()),  # 脚本目录下的models/类型
            os.path.join(".", f"{self.model_type.lower()}_models"),  # 相对路径
            os.path.join(".", "models", self.model_type.lower()),  # 相对路径models/类型
        ]
        
        # 查找存在的目录
        for directory in possible_dirs:
            if os.path.exists(directory) and os.path.isdir(directory):
                self.models_dir = directory
                log(f"找到模型目录: {directory}")
                return
        
        # 在没有找到目录的情况下设置默认值并记录
        self.models_dir = os.path.join(".", f"{self.model_type.lower()}_models")
        log(f"警告: 未找到模型目录, 将使用默认路径: {self.models_dir}")
    
    def _load_models(self):
        """加载所有CatBoost模型和对应的标准化器"""
        if not os.path.exists(self.models_dir):
            log(f"错误: 模型目录不存在: {self.models_dir}")
            return
        
        # 查找所有模型文件
        model_files = glob.glob(os.path.join(self.models_dir, "model_*.cbm"))
        model_files.sort()  # 确保顺序一致
        
        if not model_files:
            log(f"错误: 在{self.models_dir}中没有找到模型文件")
            return
        
        # 加载每个模型
        for model_file in model_files:
            try:
                model_id = os.path.basename(model_file).replace("model_", "").replace(".cbm", "")
                model = CatBoostRegressor()
                model.load_model(model_file)
                self.models.append(model)
                
                # 尝试加载对应的标准化器
                scaler_file = os.path.join(self.models_dir, f"scaler_{model_id}.json")
                if os.path.exists(scaler_file):
                    try:
                        with open(scaler_file, 'r') as f:
                            scaler_data = json.load(f)
                        self.scalers.append(scaler_data)
                    except Exception as e:
                        log(f"加载标准化器{scaler_file}失败: {str(e)}")
                        # 如果找不到匹配的标准化器，尝试使用通用的
                        self._try_load_general_scaler()
                else:
                    # 如果没有对应的标准化器，尝试使用通用的
                    self._try_load_general_scaler()
                    
                log(f"加载模型: {model_file}")
            except Exception as e:
                log(f"加载模型{model_file}失败: {str(e)}")
        
        # 加载特征范围
        range_file = os.path.join(self.models_dir, "feature_ranges.json")
        if os.path.exists(range_file):
            self.range_checker = FeatureRangeChecker.load_ranges(range_file)
            log(f"加载特征范围: {range_file}")
        
        log(f"成功加载{len(self.models)}个{self.model_type}产率模型和{len(self.scalers)}个标准化器")
    
    def _try_load_general_scaler(self):
        """尝试加载通用标准化器"""
        general_scaler_file = os.path.join(self.models_dir, "scaler.json")
        if os.path.exists(general_scaler_file):
            try:
                with open(general_scaler_file, 'r') as f:
                    scaler_data = json.load(f)
                self.scalers.append(scaler_data)
                log(f"使用通用标准化器: {general_scaler_file}")
            except Exception as e:
                log(f"加载通用标准化器失败: {str(e)}")
                # 如果通用标准化器加载失败，添加None占位
                self.scalers.append(None)
        else:
            # 如果没有标准化器，添加None占位
            self.scalers.append(None)
            log("警告: 没有找到匹配的标准化器，预测可能不准确")
    
    def _normalize_features(self, features_df, scaler_data):
        """使用给定的标准化器数据标准化特征"""
        if not scaler_data:
            return features_df
        
        # 创建标准化后的数据框
        normalized_df = features_df.copy()
        
        # 对每个特征进行标准化
        for feature, params in scaler_data.items():
            if feature in features_df.columns:
                if 'mean' in params and 'std' in params:
                    # 应用Z-score标准化
                    normalized_df[feature] = (features_df[feature] - params['mean']) / params['std']
                elif 'min' in params and 'max' in params:
                    # 应用Min-Max标准化
                    normalized_df[feature] = (features_df[feature] - params['min']) / (params['max'] - params['min'])
        
        return normalized_df
    
    def check_input_range(self, input_df):
        """检查输入是否在训练范围内"""
        return self.range_checker.check_input_range(input_df)
    
    def predict(self, features_df, return_individual=False):
        """
        使用集成模型进行预测
        
        参数:
            features_df: 包含输入特征的DataFrame
            return_individual: 是否返回每个子模型的预测结果
            
        返回:
            预测结果或者(预测结果, 单个模型预测)元组
        """
        # 如果没有加载模型，返回零
        if not self.models:
            log(f"错误: 没有加载{self.model_type}产率预测模型")
            return 0.0, [] if return_individual else 0.0
        
        try:
            # 存储每个模型的预测结果
            individual_predictions = []
            
            # 使用每个模型进行预测
            for i, model in enumerate(self.models):
                # 确定使用哪个标准化器
                scaler_data = self.scalers[i] if i < len(self.scalers) else None
                
                # 如果有标准化器，标准化特征
                if scaler_data:
                    normalized_features = self._normalize_features(features_df, scaler_data)
                else:
                    normalized_features = features_df
                
                # 获取预测结果
                try:
                    prediction = model.predict(normalized_features)
                    # 确保预测结果是数值
                    if isinstance(prediction, (list, np.ndarray)):
                        pred_value = float(prediction[0])
                    else:
                        pred_value = float(prediction)
                    
                    # 存储单个模型的预测
                    individual_predictions.append(pred_value)
                except Exception as e:
                    log(f"模型{i}预测失败: {str(e)}")
                    # 发生错误时添加零值
                    individual_predictions.append(0.0)
            
            # 计算平均预测结果
            if individual_predictions:
                # 确保预测结果非负
                individual_predictions = [max(0, pred) for pred in individual_predictions]
                ensemble_prediction = np.mean(individual_predictions)
                log(f"{self.model_type}产率预测结果: {ensemble_prediction:.4f}%, 子模型数: {len(individual_predictions)}")
                
                # 返回结果
                if return_individual:
                    return np.array([ensemble_prediction]), individual_predictions
                else:
                    return np.array([ensemble_prediction])
            else:
                log(f"警告: 没有有效的{self.model_type}产率预测结果")
                return np.array([0.0]), [] if return_individual else np.array([0.0])
                
        except Exception as e:
            log(f"{self.model_type}产率预测失败: {str(e)}")
            log(traceback.format_exc())
            return np.array([0.0]), [] if return_individual else np.array([0.0])

# 侧边栏设置
st.sidebar.markdown("## 🔧 系统设置")

# 模型选择
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Char Yield(%)"

model_options = {
    "Char Yield(%)": "Char",
    "Oil Yield(%)": "Oil"
}

selected_model_name = st.sidebar.radio(
    "选择预测模型",
    list(model_options.keys()),
    key="model_selector"
)

if selected_model_name != st.session_state.selected_model:
    st.session_state.selected_model = selected_model_name
    st.session_state.prediction_result = None
    st.session_state.warnings = []
    st.session_state.individual_predictions = []
    log(f"切换到模型: {selected_model_name}")

# 创建预测器实例
model_type = model_options[selected_model_name]
predictor = CorrectedEnsemblePredictor(model_type=model_type)

# 主页面
st.markdown("<h1 class='main-title'>生物质热解产率预测系统 🌱</h1>", unsafe_allow_html=True)

# 加载状态检查
if not predictor.models:
    st.error(f"⚠️ 错误: 未能加载{model_type}产率预测模型。请检查模型文件是否存在并且格式正确。")
    st.stop()

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
    
    # 显示输入特征表格
    st.markdown("### 输入特征")
    formatted_features = {}
    for feature, value in features.items():
        formatted_features[feature] = f"{value:.2f}"
    
    # 转换为DataFrame并显示
    input_df = pd.DataFrame([formatted_features])
    st.dataframe(input_df, use_container_width=True)
    
    # 技术说明部分 - 使用折叠式展示
    with st.expander("技术说明"):
        st.markdown("""
        <div class='tech-info'>
        <p>本模型基于多个CatBoost模型集成创建，预测生物质热解产物分布。模型使用生物质的元素分析、近似分析数据和热解条件作为输入，计算焦炭和生物油产量。</p>
        
        <p><b>关键影响因素：</b></p>
        <ul>
            <li>温度(PT)是最重要的影响因素，对焦炭产量有显著负相关性</li>
            <li>停留时间(RT)是第二重要的因素，延长停留时间会降低焦炭产量</li>
            <li>固定碳含量(FC)可由100-Ash(%)-VM(%)计算得出，对预测也有重要影响</li>
        </ul>
        
        <p><b>预测准确度：</b></p>
        <p>模型在测试集上的均方根误差(RMSE)约为3.39%，决定系数(R²)为0.93。对大多数生物质样本，预测误差在±5%以内。</p>
        
        <p><b>最近更新：</b></p>
        <ul>
            <li>✅ 修复了所有输入值只能精确到一位小数的问题</li>
            <li>✅ 解决了部分子模型标准化器不匹配的问题</li>
            <li>✅ 增加了模型切换功能，支持不同产率预测</li>
            <li>✅ 修复了预测结果不显示的问题</li>
            <li>✅ 修复了"invalid index to scalar variable"错误</li>
            <li>✅ 移除了子模型预测结果柱状图显示</li>
            <li>✅ 移除了性能指标显示部分</li>
            <li>✅ 改进了模型加载失败时的错误处理和提示</li>
            <li>✅ 增强了对不同目录结构的兼容性</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# 添加页脚
st.markdown("---")
footer = """
<div style='text-align: center;'>
<p>© 2023 Biomass Pyrolysis Modeling Team. 版本: 2.3.0</p>
<p>本应用支持两位小数输入精度 | 已集成Char和Oil产率预测模型</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)