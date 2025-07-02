# -*- coding: utf-8 -*-
"""
电化学传感器检测新烟碱农药预测系统
基于GBDT模型预测电流响应I(uA)
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime

# 页面设置
st.set_page_config(
    page_title='电化学传感器检测系统',
    page_icon='⚡',
    layout='wide',
    initial_sidebar_state='expanded'
)

# 自定义样式 - 使用更强力的CSS选择器
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 32px !important;
        font-weight: bold;
        margin-bottom: 20px;
        color: white !important;
    }
    
    /* 强制覆盖所有number input的样式 */
    .stNumberInput > div > div > input,
    .stNumberInput input,
    input[type="number"] {
        font-size: 16px !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        border-width: 2px !important;
        transition: all 0.3s ease !important;
    }
    
    /* 第一列和第三列（奇数位置）- 蓝色 */
    .stNumberInput:nth-of-type(odd) > div > div > input,
    .stNumberInput:nth-of-type(odd) input {
        background: linear-gradient(135deg, #E3F2FD, #BBDEFB) !important;
        color: #1565C0 !important;
        border: 2px solid #2196F3 !important;
    }
    
    /* 第二列和第四列（偶数位置）- 橙色 */
    .stNumberInput:nth-of-type(even) > div > div > input,
    .stNumberInput:nth-of-type(even) input {
        background: linear-gradient(135deg, #FFF3E0, #FFE0B2) !important;
        color: #E65100 !important;
        border: 2px solid #FF9800 !important;
    }
    
    /* 第五列和第六列 - 绿色 */
    .stNumberInput:nth-of-type(5) > div > div > input,
    .stNumberInput:nth-of-type(5) input,
    .stNumberInput:nth-of-type(6) > div > div > input,
    .stNumberInput:nth-of-type(6) input {
        background: linear-gradient(135deg, #E8F5E8, #C8E6C9) !important;
        color: #2E7D32 !important;
        border: 2px solid #4CAF50 !important;
    }
    
    /* 使用更通用的方法 - 所有输入框默认样式 */
    [data-baseweb="input"] input {
        background: linear-gradient(135deg, #E3F2FD, #BBDEFB) !important;
        color: #1565C0 !important;
        border: 2px solid #2196F3 !important;
        border-radius: 8px !important;
        font-weight: bold !important;
    }
    
    /* 聚焦状态 */
    .stNumberInput input:focus,
    input[type="number"]:focus {
        box-shadow: 0 0 15px rgba(33, 150, 243, 0.6) !important;
        outline: none !important;
        transform: scale(1.02) !important;
    }
    
    .result-display {
        background-color: #1E1E1E;
        color: white;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        border: 2px solid #2E86AB;
    }
    
    .stButton button {
        font-size: 18px !important;
        font-weight: bold !important;
    }
    
    .warning-box {
        background-color: rgba(255, 165, 0, 0.2);
        border-left: 5px solid orange;
        padding: 15px;
        margin: 15px 0;
        border-radius: 5px;
    }
    
    .error-box {
        background-color: rgba(255, 0, 0, 0.2);
        border-left: 5px solid red;
        padding: 15px;
        margin: 15px 0;
        border-radius: 5px;
    }
    
    .info-box {
        background-color: rgba(0, 123, 255, 0.2);
        border-left: 5px solid #007bff;
        padding: 15px;
        margin: 15px 0;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 初始化日志
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []

def log(message):
    """记录日志"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    st.session_state.log_messages.append(log_entry)
    if len(st.session_state.log_messages) > 50:
        st.session_state.log_messages = st.session_state.log_messages[-50:]

# 主标题
st.markdown("<h1 class='main-title'>电化学传感器检测新烟碱农药预测系统</h1>", unsafe_allow_html=True)

class NeonicotinoidPredictor:
    """新烟碱农药电化学检测预测器"""
    
    def __init__(self):
        self.target_name = "I(uA)"
        # 按照你提供的特征顺序：DT, PH, SS, P, TM, C0
        self.feature_names = [
            'DT(ml)',     # 滴涂量
            'PH',         # pH值  
            'SS(mV/s)',   # 扫描速率
            'P(V)',       # 电压
            'TM(min)',    # 孵化时间
            'C0(uM)'      # 底液初始浓度
        ]
        
        # 根据电化学检测实验的合理范围设置
        self.parameter_ranges = {
            'DT(ml)': {'min': 0.1, 'max': 20.0},     # 滴涂量通常几微升到几十微升
            'PH': {'min': 3.0, 'max': 10.0},          # pH范围
            'SS(mV/s)': {'min': 10.0, 'max': 500.0},  # 扫描速率
            'P(V)': {'min': -2.0, 'max': 2.0},        # 电压范围
            'TM(min)': {'min': 1.0, 'max': 120.0},    # 孵化时间
            'C0(uM)': {'min': 0.1, 'max': 1000.0}     # 浓度范围
        }
        
        self.model_loaded = False
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """加载GBDT模型"""
        model_paths = [
            "GBDT.joblib",
            "./GBDT.joblib", 
            "../GBDT.joblib",
            r"C:\Users\HWY\Desktop\开题-7.2\GBDT.joblib",
            "./models/GBDT.joblib",
            "../models/GBDT.joblib"
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                try:
                    self.pipeline = joblib.load(path)
                    self.model_loaded = True
                    log(f"模型加载成功: {path}")
                    break
                except Exception as e:
                    log(f"加载模型失败: {path}, 错误: {str(e)}")
        
        if not self.model_loaded:
            log("警告: 未找到GBDT模型文件")
    
    def check_parameter_ranges(self, parameters):
        """检查参数是否在合理范围内"""
        warnings = []
        for param, value in parameters.items():
            range_info = self.parameter_ranges.get(param)
            if range_info:
                if value < range_info['min'] or value > range_info['max']:
                    warning = f"{param}: {value:.3f} (建议范围: {range_info['min']:.1f} - {range_info['max']:.1f})"
                    warnings.append(warning)
                    log(f"参数警告: {warning}")
        return warnings
    
    def predict(self, parameters):
        """执行预测"""
        if not self.model_loaded:
            raise ValueError("GBDT模型未加载，无法进行预测")
        
        # 按照特征顺序准备数据
        data = []
        for feature in self.feature_names:
            data.append(parameters.get(feature, 0.0))
        
        # 创建DataFrame
        df = pd.DataFrame([data], columns=self.feature_names)
        log(f"输入数据: {dict(zip(self.feature_names, data))}")
        
        try:
            # 使用Pipeline进行预测（包含预处理）
            result = self.pipeline.predict(df)[0]
            log(f"预测成功，电流响应: {result:.4f} uA")
            return float(result)
        except Exception as e:
            error_msg = f"预测过程出错: {str(e)}"
            log(error_msg)
            raise ValueError(error_msg)

# 初始化预测器
predictor = NeonicotinoidPredictor()

# 侧边栏状态显示
st.sidebar.markdown("### 📊 系统状态")
if predictor.model_loaded:
    st.sidebar.success("✅ GBDT模型已加载")
    st.sidebar.info(f"📈 特征数量: {len(predictor.feature_names)}")
    st.sidebar.info("🎯 目标: 电流响应 I(uA)")
else:
    st.sidebar.error("❌ 模型未加载")
    st.sidebar.warning("请确保GBDT.joblib文件在正确位置")

# 显示最近日志
st.sidebar.markdown("### 📝 执行日志")
if st.session_state.log_messages:
    for msg in st.session_state.log_messages[-8:]:
        st.sidebar.text(msg)

# 初始化会话状态
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'warnings' not in st.session_state:
    st.session_state.warnings = []
if 'prediction_error' not in st.session_state:
    st.session_state.prediction_error = None

# 参数输入区域
st.markdown("### 🔬 实验参数输入")

# 根据电化学检测的实际参数设置默认值
default_values = {
    "DT(ml)": 5.0,      # 滴涂量
    "PH": 7.0,          # pH值
    "SS(mV/s)": 100.0,  # 扫描速率
    "P(V)": 0.0,        # 电压
    "TM(min)": 30.0,    # 孵化时间
    "C0(uM)": 50.0      # 底液初始浓度
}

# 强制添加自定义CSS到每个输入框
st.markdown("""
<script>
setTimeout(function() {
    const inputs = document.querySelectorAll('input[type="number"]');
    inputs.forEach((input, index) => {
        if (index < 2) {
            // 第一列 - 蓝色
            input.style.background = 'linear-gradient(135deg, #E3F2FD, #BBDEFB)';
            input.style.color = '#1565C0';
            input.style.border = '2px solid #2196F3';
        } else if (index < 4) {
            // 第二列 - 橙色
            input.style.background = 'linear-gradient(135deg, #FFF3E0, #FFE0B2)';
            input.style.color = '#E65100';
            input.style.border = '2px solid #FF9800';
        } else {
            // 第三列 - 绿色
            input.style.background = 'linear-gradient(135deg, #E8F5E8, #C8E6C9)';
            input.style.color = '#2E7D32';
            input.style.border = '2px solid #4CAF50';
        }
        input.style.borderRadius = '8px';
        input.style.fontWeight = 'bold';
    });
}, 1000);
</script>
""", unsafe_allow_html=True)

# 创建三列布局，每列2个参数
col1, col2, col3 = st.columns(3)

parameters = {}

with col1:
    # DT(ml) - 滴涂量
    parameters['DT(ml)'] = st.number_input(
        "DT(ml) - 滴涂量", 
        value=default_values['DT(ml)'], 
        step=0.1,
        key="dt_input",
        format="%.2f",
        help="电极表面的样品滴涂体积"
    )
    
    # SS(mV/s) - 扫描速率
    parameters['SS(mV/s)'] = st.number_input(
        "SS(mV/s) - 扫描速率", 
        value=default_values['SS(mV/s)'], 
        step=10.0,
        key="ss_input",
        format="%.1f",
        help="差分脉冲伏安法的扫描速率"
    )

with col2:
    # PH - pH值
    parameters['PH'] = st.number_input(
        "PH - 溶液pH值", 
        value=default_values['PH'], 
        step=0.1,
        key="ph_input",
        format="%.2f",
        help="检测溶液的pH值"
    )
    
    # P(V) - 电压
    parameters['P(V)'] = st.number_input(
        "P(V) - 检测电压", 
        value=default_values['P(V)'], 
        step=0.01,
        key="p_input",
        format="%.3f",
        help="差分脉冲伏安法的检测电压"
    )

with col3:
    # TM(min) - 孵化时间
    parameters['TM(min)'] = st.number_input(
        "TM(min) - 孵化时间", 
        value=default_values['TM(min)'], 
        step=5.0,
        key="tm_input",
        format="%.1f",
        help="样品与电极的反应孵化时间"
    )
    
    # C0(uM) - 底液初始浓度
    parameters['C0(uM)'] = st.number_input(
        "C0(uM) - 底液初始浓度", 
        value=default_values['C0(uM)'], 
        step=1.0,
        key="c0_input",
        format="%.1f",
        help="电解质底液中目标物的初始浓度"
    )

# 显示当前参数值
with st.expander("📋 查看当前参数设置", expanded=False):
    params_display = ""
    for param, value in parameters.items():
        params_display += f"**{param}**: {value} | "
    st.markdown(params_display[:-3])

# 预测控制按钮
st.markdown("### 🚀 执行预测")
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    predict_clicked = st.button(
        "⚡ 开始预测", 
        use_container_width=True, 
        type="primary",
        help="使用GBDT模型预测电流响应"
    )

with col2:
    if st.button("🔄 重置参数", use_container_width=True):
        st.rerun()

with col3:
    show_details = st.checkbox("显示详细信息", value=False)

# 执行预测
if predict_clicked:
    log("=" * 50)
    log("开始新烟碱农药电化学检测预测")
    
    # 检查参数范围
    warnings = predictor.check_parameter_ranges(parameters)
    st.session_state.warnings = warnings
    
    try:
        # 执行预测
        result = predictor.predict(parameters)
        st.session_state.prediction_result = result
        st.session_state.prediction_error = None
        log(f"预测完成，结果: {result:.4f} uA")
        
    except Exception as e:
        error_msg = str(e)
        st.session_state.prediction_error = error_msg
        st.session_state.prediction_result = None
        log(f"预测失败: {error_msg}")

# 结果显示
if st.session_state.prediction_result is not None:
    st.markdown("---")
    
    # 主要结果显示
    st.markdown(
        f"<div class='result-display'>🎯 预测电流响应: {st.session_state.prediction_result:.4f} μA</div>", 
        unsafe_allow_html=True
    )
    
    # 警告显示
    if st.session_state.warnings:
        warnings_html = "<div class='warning-box'><h4⚠️ 参数范围警告</h4><ul>"
        for warning in st.session_state.warnings:
            warnings_html += f"<li>{warning}</li>"
        warnings_html += "</ul><p><em>建议检查参数设置，确保在实验合理范围内。</em></p></div>"
        st.markdown(warnings_html, unsafe_allow_html=True)
    
    # 详细信息显示
    if show_details:
        with st.expander("📊 预测详细信息", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                **预测信息:**
                - 目标变量: {predictor.target_name}
                - 预测值: {st.session_state.prediction_result:.6f} μA
                - 模型类型: GBDT Pipeline
                - 预处理: RobustScaler标准化
                """)
            with col2:
                st.markdown(f"""
                **模型状态:**
                - 加载状态: {'✅ 正常' if predictor.model_loaded else '❌ 失败'}
                - 特征数量: {len(predictor.feature_names)}
                - 参数警告: {len(st.session_state.warnings)}个
                - 应用领域: 新烟碱农药检测
                """)

elif st.session_state.prediction_error is not None:
    st.markdown("---")
    error_html = f"""
    <div class='error-box'>
        <h3>❌ 预测失败</h3>
        <p><strong>错误信息:</strong> {st.session_state.prediction_error}</p>
        <p><strong>可能的解决方案:</strong></p>
        <ul>
            <li>确保GBDT.joblib模型文件存在</li>
            <li>检查参数数值是否合理</li>
            <li>验证模型文件格式是否正确</li>
            <li>确认特征顺序: DT(ml) → PH → SS(mV/s) → P(V) → TM(min) → C0(uM)</li>
        </ul>
    </div>
    """
    st.markdown(error_html, unsafe_allow_html=True)

# 技术说明
with st.expander("📚 电化学检测技术说明", expanded=False):
    st.markdown("""
    <div class='info-box'>
    <h4>🔬 新烟碱农药电化学检测原理</h4>
    <p>本系统基于<strong>差分脉冲伏安法(DPV)</strong>进行新烟碱农药的电化学检测，使用GBDT机器学习模型预测电流响应。</p>
    
    <h4>📋 参数说明</h4>
    <ul>
        <li><strong>DT(ml)</strong>: 滴涂量 - 电极表面样品的滴涂体积</li>
        <li><strong>PH</strong>: pH值 - 检测溶液的酸碱度</li>
        <li><strong>SS(mV/s)</strong>: 扫描速率 - 电压扫描的速度</li>
        <li><strong>P(V)</strong>: 检测电压 - 目标化合物的氧化还原电位</li>
        <li><strong>TM(min)</strong>: 孵化时间 - 样品与电极的反应时间</li>
        <li><strong>C0(uM)</strong>: 底液初始浓度 - 电解质中目标物浓度</li>
    </ul>
    
    <h4>🎯 应用场景</h4>
    <p>适用于吡虫啉、噻虫嗪、噻虫胺等新烟碱类农药的快速定量检测，为食品安全和环境监测提供技术支持。</p>
    </div>
    """, unsafe_allow_html=True)

# 页脚信息
st.markdown("---")
footer_info = """
<div style='text-align: center; color: #666; padding: 20px;'>
<p><strong>© 2024 电化学传感器实验室</strong> | 新烟碱农药检测预测系统 | 版本: 2.0.0</p>
<p>🔬 基于GBDT算法 | ⚡ 差分脉冲伏安法 | 🎯 高精度预测</p>
<p><em>特征顺序: DT(ml) → PH → SS(mV/s) → P(V) → TM(min) → C0(uM)</em></p>
</div>
"""
st.markdown(footer_info, unsafe_allow_html=True)