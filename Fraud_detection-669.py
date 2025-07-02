# -*- coding: utf-8 -*-
"""
电化学模型在线预测系统
基于GBDT模型预测I(uA)
修复版本 - 根据实际训练特征调整
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import joblib
import traceback
import matplotlib.pyplot as plt
from datetime import datetime

# 清除缓存，强制重新渲染
st.cache_data.clear()

# 页面设置
st.set_page_config(
    page_title='电化学模型预测系统',
    page_icon='⚡',
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
    
    /* 侧边栏模型信息样式 */
    .sidebar-model-info {
        background-color: #2E2E2E;
        padding: 10px;
        border-radius: 5px;
        margin-top: 20px;
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
log("应用启动 - 电化学模型预测系统")
log("特征顺序：DT(ml), PH, SS(mV/s), P(V), TM(min), C0(uM)")

# 更新主标题
st.markdown("<h1 class='main-title'>基于GBDT模型的电化学响应预测系统</h1>", unsafe_allow_html=True)

class ModelPredictor:
    """电化学模型预测器类"""
    
    def __init__(self):
        self.target_name = "I(uA)"
        
        # 根据训练代码，特征顺序为除I(uA)外的所有列
        self.feature_names = [
            'DT(ml)',           # 滴定体积
            'PH',               # pH值
            'SS(mV/s)',         # 扫描速率
            'P(V)',             # 电位
            'TM(min)',          # 时间
            'C0(uM)'            # 初始浓度
        ]
        
        # 根据之前的数据统计信息设置训练范围
        self.training_ranges = {
            'DT(ml)': {'min': 0.0, 'max': 10.0},
            'PH': {'min': 3.0, 'max': 9.0},
            'SS(mV/s)': {'min': 10.0, 'max': 200.0},
            'P(V)': {'min': -1.0, 'max': 1.0},
            'TM(min)': {'min': 0.0, 'max': 60.0},
            'C0(uM)': {'min': 1.0, 'max': 100.0}
        }
        
        self.last_features = {}
        self.last_result = None
        
        # 查找并加载模型
        self.model_path = self._find_model_file()
        if self.model_path:
            self._load_pipeline()
        else:
            self.model_loaded = False
            self.pipeline = None
    
    def _find_model_file(self):
        """查找模型文件"""
        model_file_patterns = [
            "GBDT.joblib",
            "*GBDT*.joblib",
            "*gbdt*.joblib"
        ]
        
        search_dirs = [
            ".", "./models", "../models", "/app/models", "/app",
            r"C:\Users\HWY\Desktop\开题-7.2"
        ]
        
        log(f"搜索GBDT模型文件，模式: {model_file_patterns}")
        
        for directory in search_dirs:
            if not os.path.exists(directory):
                continue
                
            try:
                for pattern in model_file_patterns:
                    matches = glob.glob(os.path.join(directory, pattern))
                    for match in matches:
                        if os.path.isfile(match):
                            log(f"找到模型文件: {match}")
                            return match
                            
                for file in os.listdir(directory):
                    if file.endswith('.joblib') and 'gbdt' in file.lower():
                        model_path = os.path.join(directory, file)
                        log(f"找到匹配的模型文件: {model_path}")
                        return model_path
            except Exception as e:
                log(f"搜索目录{directory}时出错: {str(e)}")
        
        log(f"未找到GBDT模型文件")
        return None
    
    def _load_pipeline(self):
        """加载Pipeline模型"""
        if not self.model_path:
            log("模型路径为空，无法加载")
            return False
        
        try:
            log(f"加载Pipeline模型: {self.model_path}")
            self.pipeline = joblib.load(self.model_path)
            
            if hasattr(self.pipeline, 'predict') and hasattr(self.pipeline, 'named_steps'):
                log(f"Pipeline加载成功，组件: {list(self.pipeline.named_steps.keys())}")
                
                if 'scaler' in self.pipeline.named_steps and 'model' in self.pipeline.named_steps:
                    scaler_type = type(self.pipeline.named_steps['scaler']).__name__
                    model_type = type(self.pipeline.named_steps['model']).__name__
                    log(f"Scaler类型: {scaler_type}, Model类型: {model_type}")
                    
                    self.model_loaded = True
                    return True
                else:
                    log("Pipeline结构不符合预期，缺少scaler或model组件")
                    return False
            else:
                log("加载的对象不是有效的Pipeline")
                return False
                
        except Exception as e:
            log(f"加载模型出错: {str(e)}")
            log(traceback.format_exc())
            self.model_loaded = False
            return False
    
    def check_input_range(self, features):
        """检查输入值是否在训练数据范围内"""
        warnings = []
        
        for feature, value in features.items():
            range_info = self.training_ranges.get(feature)
            
            if range_info:
                if value < range_info['min'] or value > range_info['max']:
                    warning = f"{feature}: {value:.3f} (建议范围 {range_info['min']:.3f} - {range_info['max']:.3f})"
                    warnings.append(warning)
                    log(f"警告: {warning}")
        
        return warnings
    
    def _prepare_features(self, features):
        """准备特征，确保顺序与训练时一致"""
        model_features = {}
        
        for feature in self.feature_names:
            if feature in features:
                model_features[feature] = features[feature]
            else:
                default_values = {
                    'DT(ml)': 5.0,
                    'PH': 7.0,
                    'SS(mV/s)': 100.0,
                    'P(V)': 0.0,
                    'TM(min)': 30.0,
                    'C0(uM)': 50.0
                }
                default_value = default_values.get(feature, 0.0)
                model_features[feature] = default_value
                log(f"警告: 特征 '{feature}' 缺失，设为默认值: {default_value}")
        
        df = pd.DataFrame([model_features])
        df = df[self.feature_names]
        
        log(f"准备好的特征DataFrame形状: {df.shape}, 列: {list(df.columns)}")
        return df
    
    def predict(self, features):
        """预测方法 - 使用Pipeline进行预测"""
        features_changed = False
        if self.last_features:
            for feature, value in features.items():
                if feature not in self.last_features or abs(self.last_features[feature] - value) > 0.001:
                    features_changed = True
                    break
        else:
            features_changed = True
        
        if not features_changed and self.last_result is not None:
            log("输入未变化，使用上次的预测结果")
            return self.last_result
        
        self.last_features = features.copy()
        
        log(f"开始准备{len(features)}个特征数据进行预测")
        features_df = self._prepare_features(features)
        
        if self.model_loaded and self.pipeline is not None:
            try:
                log("使用Pipeline进行预测（包含RobustScaler预处理）")
                result = float(self.pipeline.predict(features_df)[0])
                log(f"预测成功: {result:.4f}")
                self.last_result = result
                return result
            except Exception as e:
                log(f"Pipeline预测失败: {str(e)}")
                log(traceback.format_exc())
                
                if self._find_model_file() and self._load_pipeline():
                    try:
                        result = float(self.pipeline.predict(features_df)[0])
                        log(f"重新加载后预测成功: {result:.4f}")
                        self.last_result = result
                        return result
                    except Exception as new_e:
                        log(f"重新加载后预测仍然失败: {str(new_e)}")
        
        log("所有预测尝试都失败")
        raise ValueError(f"模型预测失败。请确保模型文件存在且格式正确。")
    
    def get_model_info(self):
        """获取模型信息摘要"""
        info = {
            "模型类型": "GBDT Pipeline (RobustScaler + GradientBoostingRegressor)",
            "目标变量": self.target_name,
            "特征数量": len(self.feature_names),
            "模型状态": "已加载" if self.model_loaded else "未加载"
        }
        
        if self.model_loaded and hasattr(self.pipeline, 'named_steps'):
            pipeline_steps = list(self.pipeline.named_steps.keys())
            info["Pipeline组件"] = " → ".join(pipeline_steps)
            
            if 'model' in self.pipeline.named_steps:
                model = self.pipeline.named_steps['model']
                model_type = type(model).__name__
                info["回归器类型"] = model_type
                
                if hasattr(model, 'n_estimators'):
                    info["树的数量"] = model.n_estimators
                if hasattr(model, 'max_depth'):
                    info["最大深度"] = model.max_depth
                if hasattr(model, 'learning_rate'):
                    info["学习率"] = f"{model.learning_rate:.3f}"
                    
        return info

# 初始化预测器
predictor = ModelPredictor()

# 在侧边栏添加模型信息 - 这里是关键修复点！
model_info = predictor.get_model_info()
model_info_html = "<div class='sidebar-model-info'><h3>模型信息</h3>"
for key, value in model_info.items():  # 确保这里是 .items() 而不是 _items()
    model_info_html += f"<p><b>{key}</b>: {value}</p>"
model_info_html += "</div>"
st.sidebar.markdown(model_info_html, unsafe_allow_html=True)

# 初始化会话状态
if 'clear_pressed' not in st.session_state:
    st.session_state.clear_pressed = False
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'warnings' not in st.session_state:
    st.session_state.warnings = []
if 'prediction_error' not in st.session_state:
    st.session_state.prediction_error = None
if 'feature_values' not in st.session_state:
    st.session_state.feature_values = {}

# 默认值
default_values = {
    "DT(ml)": 5.0,
    "PH": 7.0,
    "SS(mV/s)": 100.0,
    "P(V)": 0.0,
    "TM(min)": 30.0,
    "C0(uM)": 50.0
}

# 将6个特征平均分成三列
feature_categories = {
    "电化学参数": ["DT(ml)", "PH"],
    "测量条件": ["SS(mV/s)", "P(V)"],
    "实验参数": ["TM(min)", "C0(uM)"]
}

# 颜色配置
category_colors = {
    "电化学参数": "#501d8a",  
    "测量条件": "#1c8041",  
    "实验参数": "#e55709" 
}

# 创建三列布局
col1, col2, col3 = st.columns(3)

# 使用字典存储所有输入值
features = {}

# 电化学参数 - 第一列
with col1:
    category = "电化学参数"
    color = category_colors[category]
    st.markdown(f"<div class='section-header' style='background-color: {color};'>{category}</div>", unsafe_allow_html=True)
    
    for feature in feature_categories[category]:
        if st.session_state.clear_pressed:
            value = default_values[feature]
        else:
            value = st.session_state.feature_values.get(feature, default_values[feature])
        
        col_a, col_b = st.columns([1, 0.5])
        with col_a:
            st.markdown(f"<div class='input-label' style='background-color: {color};'>{feature}</div>", unsafe_allow_html=True)
        with col_b:
            if feature == "DT(ml)":
                step = 0.1
                format_str = "%.2f"
            elif feature == "PH":
                step = 0.1
                format_str = "%.2f"
            else:
                step = 0.01
                format_str = "%.3f"
            
            features[feature] = st.number_input(
                "", 
                value=float(value), 
                step=step,
                key=f"{category}_{feature}",
                format=format_str,
                label_visibility="collapsed"
            )

# 测量条件 - 第二列
with col2:
    category = "测量条件"
    color = category_colors[category]
    st.markdown(f"<div class='section-header' style='background-color: {color};'>{category}</div>", unsafe_allow_html=True)
    
    for feature in feature_categories[category]:
        if st.session_state.clear_pressed:
            value = default_values[feature]
        else:
            value = st.session_state.feature_values.get(feature, default_values[feature])
        
        col_a, col_b = st.columns([1, 0.5])
        with col_a:
            st.markdown(f"<div class='input-label' style='background-color: {color};'>{feature}</div>", unsafe_allow_html=True)
        with col_b:
            if feature == "SS(mV/s)":
                step = 1.0
                format_str = "%.1f"
            elif feature == "P(V)":
                step = 0.01
                format_str = "%.3f"
            else:
                step = 0.01
                format_str = "%.3f"
            
            features[feature] = st.number_input(
                "", 
                value=float(value), 
                step=step,
                key=f"{category}_{feature}",
                format=format_str,
                label_visibility="collapsed"
            )

# 实验参数 - 第三列
with col3:
    category = "实验参数"
    color = category_colors[category]
    st.markdown(f"<div class='section-header' style='background-color: {color};'>{category}</div>", unsafe_allow_html=True)
    
    for feature in feature_categories[category]:
        if st.session_state.clear_pressed:
            value = default_values[feature]
        else:
            value = st.session_state.feature_values.get(feature, default_values[feature])
        
        col_a, col_b = st.columns([1, 0.5])
        with col_a:
            st.markdown(f"<div class='input-label' style='background-color: {color};'>{feature}</div>", unsafe_allow_html=True)
        with col_b:
            if feature == "TM(min)":
                step = 1.0
                format_str = "%.1f"
            elif feature == "C0(uM)":
                step = 1.0
                format_str = "%.1f"
            else:
                step = 0.01
                format_str = "%.3f"
            
            features[feature] = st.number_input(
                "", 
                value=float(value), 
                step=step,
                key=f"{category}_{feature}",
                format=format_str,
                label_visibility="collapsed"
            )

# 调试信息
with st.expander("📊 显示当前输入值", expanded=False):
    debug_info = "<div style='columns: 3; column-gap: 20px;'>"
    for feature, value in features.items():
        debug_info += f"<p><b>{feature}</b>: {value:.3f}</p>"
    debug_info += "</div>"
    st.markdown(debug_info, unsafe_allow_html=True)

# 重置状态
if st.session_state.clear_pressed:
    st.session_state.feature_values = {}
    st.session_state.clear_pressed = False

# 预测结果显示区域
result_container = st.container()

# 预测按钮区域
col1, col2 = st.columns([1, 1])

with col1:
    predict_clicked = st.button("⚡ 运行预测", use_container_width=True, type="primary")
    if predict_clicked:
        log("开始预测流程...")
        
        st.session_state.feature_values = features.copy()
        
        log(f"开始电化学响应预测，输入特征数: {len(features)}")
        
        warnings = predictor.check_input_range(features)
        st.session_state.warnings = warnings
        
        try:
            if not predictor.model_loaded:
                log("模型未加载，尝试重新加载")
                if predictor._find_model_file() and predictor._load_pipeline():
                    log("重新加载模型成功")
                else:
                    error_msg = f"无法加载GBDT模型。请确保模型文件存在于正确位置。"
                    st.error(error_msg)
                    st.session_state.prediction_error = error_msg
                    st.rerun()
            
            result = predictor.predict(features)
            if result is not None:
                st.session_state.prediction_result = float(result)
                log(f"预测成功: {st.session_state.prediction_result:.4f}")
                st.session_state.prediction_error = None
            else:
                log("警告: 预测结果为空")
                st.session_state.prediction_error = "预测结果为空"
                
        except Exception as e:
            error_msg = f"预测过程中发生错误: {str(e)}"
            st.session_state.prediction_error = error_msg
            log(f"预测错误: {str(e)}")
            log(traceback.format_exc())
            st.error(error_msg)

with col2:
    if st.button("🔄 重置输入", use_container_width=True):
        log("重置所有输入值")
        st.session_state.clear_pressed = True
        st.session_state.prediction_result = None
        st.session_state.warnings = []
        st.session_state.prediction_error = None
        st.rerun()

# 显示预测结果
if st.session_state.prediction_result is not None:
    st.markdown("---")
    
    result_container.markdown(
        f"<div class='yield-result'>电流响应 I(uA): {st.session_state.prediction_result:.4f}</div>", 
        unsafe_allow_html=True
    )
    
    if not predictor.model_loaded:
        result_container.markdown(
            "<div class='error-box'><b>⚠️ 错误：</b> 模型未成功加载，无法执行预测。请检查模型文件是否存在。</div>", 
            unsafe_allow_html=True
        )
    
    if st.session_state.warnings:
        warnings_html = "<div class='warning-box'><b>⚠️ 输入警告</b><ul>"
        for warning in st.session_state.warnings:
            warnings_html += f"<li>{warning}</li>"
        warnings_html += "</ul><p><i>建议调整输入值以获得更准确的预测结果。</i></p></div>"
        result_container.markdown(warnings_html, unsafe_allow_html=True)
    
    with st.expander("📈 预测详情", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **预测信息:**
            - 目标变量: I(uA)
            - 预测结果: {st.session_state.prediction_result:.4f} uA
            - 模型类型: GBDT Pipeline
            - 预处理: RobustScaler
            """)
        with col2:
            st.markdown(f"""
            **模型状态:**
            - 加载状态: {'✅ 正常' if predictor.model_loaded else '❌ 失败'}
            - 特征数量: {len(predictor.feature_names)}
            - 警告数量: {len(st.session_state.warnings)}
            """)

elif st.session_state.prediction_error is not None:
    st.markdown("---")
    error_html = f"""
    <div class='error-box'>
        <h3>❌ 预测失败</h3>
        <p><b>错误信息:</b> {st.session_state.prediction_error}</p>
        <p><b>可能的解决方案:</b></p>
        <ul>
            <li>确保模型文件 GBDT.joblib 存在于应用目录中</li>
            <li>检查模型文件格式是否正确</li>
            <li>验证输入数据格式是否正确</li>
            <li>确认特征顺序：DT(ml), PH, SS(mV/s), P(V), TM(min), C0(uM)</li>
        </ul>
    </div>
    """
    st.markdown(error_html, unsafe_allow_html=True)

# 技术说明部分
with st.expander("📚 技术说明与使用指南", expanded=False):
    st.markdown("""
    <div class='tech-info'>
    <h4>🔬 模型技术说明</h4>
    <p>本系统基于<b>梯度提升决策树(GBDT)</b>算法构建，采用Pipeline架构集成数据预处理和模型预测：</p>
    <ul>
        <li><b>预处理:</b> RobustScaler标准化，对异常值具有较强的鲁棒性</li>
        <li><b>模型:</b> GradientBoostingRegressor，通过集成多个弱学习器提高预测精度</li>
        <li><b>特征:</b> 6个输入特征，包括电化学参数、测量条件和实验参数</li>
    </ul>
    
    <h4>📋 特征说明</h4>
    <ul>
        <li><b>电化学参数:</b> DT(ml) - 滴定体积, PH - 溶液pH值</li>
        <li><b>测量条件:</b> SS(mV/s) - 扫描速率, P(V) - 电位</li>
        <li><b>实验参数:</b> TM(min) - 测量时间, C0(uM) - 初始浓度</li>
    </ul>
    
    <h4>📋 使用建议</h4>
    <ul>
        <li><b>数据质量:</b> 输入参数建议在合理的物理范围内</li>
        <li><b>单位统一:</b> 确保所有输入参数的单位与标签一致</li>
        <li><b>合理性检查:</b> 系统会自动检查输入范围并给出警告提示</li>
    </ul>
    
    <h4>⚠️ 重要提醒</h4>
    <p>模型基于特定的训练数据集开发，预测结果仅供参考。实际应用时请结合专业知识和实验验证。</p>
    </div>
    """, unsafe_allow_html=True)

# 添加页脚
st.markdown("---")
footer = """
<div style='text-align: center; color: #666;'>
<p>© 2024 电化学分析实验室 | 基于GBDT的电化学响应预测系统 | 版本: 1.0.0</p>
<p>⚡ 电流响应预测 | 🚀 Pipeline架构 | 📊 实时范围检查</p>
<p>特征顺序: DT(ml) → PH → SS(mV/s) → P(V) → TM(min) → C0(uM)</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)