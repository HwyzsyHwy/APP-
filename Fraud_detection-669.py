# -*- coding: utf-8 -*-
"""
电化学传感器检测新烟碱农药传感参数优化系统
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
    page_title='电化学传感器传感参数优化系统',
    page_icon='⚡',
    layout='wide',
    initial_sidebar_state='expanded'
)

# 强制颜色填充的CSS - 使用更激进的方法
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
    
    /* 覆盖Streamlit主题的所有输入框样式 */
    .stNumberInput input {
        font-size: 16px !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        border-width: 2px !important;
        padding: 12px !important;
        transition: all 0.3s ease !important;
    }
    
    /* 蓝色输入框样式 */
    .blue-input input {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%) !important;
        color: #1565C0 !important;
        border: 2px solid #2196F3 !important;
    }
    
    /* 橙色输入框样式 */
    .orange-input input {
        background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%) !important;
        color: #E65100 !important;
        border: 2px solid #FF9800 !important;
    }
    
    /* 绿色输入框样式 */
    .green-input input {
        background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C9 100%) !important;
        color: #2E7D32 !important;
        border: 2px solid #4CAF50 !important;
    }
    
    /* 聚焦效果 */
    .blue-input input:focus {
        box-shadow: 0 0 15px rgba(33, 150, 243, 0.6) !important;
        transform: scale(1.02) !important;
    }
    
    .orange-input input:focus {
        box-shadow: 0 0 15px rgba(255, 152, 0, 0.6) !important;
        transform: scale(1.02) !important;
    }
    
    .green-input input:focus {
        box-shadow: 0 0 15px rgba(76, 175, 80, 0.6) !important;
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
st.markdown("<h1 class='main-title'>电化学传感器检测新烟碱农药传感参数优化系统</h1>", unsafe_allow_html=True)

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
st.markdown("### 🔬 传感参数输入")

# 根据电化学检测的实际参数设置默认值
default_values = {
    "DT(ml)": 5.0,      # 滴涂量
    "PH": 7.0,          # pH值
    "SS(mV/s)": 100.0,  # 扫描速率
    "P(V)": 0.0,        # 电压
    "TM(min)": 30.0,    # 孵化时间
    "C0(uM)": 50.0      # 底液初始浓度
}

# 创建三列布局，每列2个参数，使用自定义CSS类
col1, col2, col3 = st.columns(3)

parameters = {}

with col1:
    # 使用HTML容器为输入框添加自定义类
    st.markdown('<div class="blue-input">', unsafe_allow_html=True)
    
    # DT(ml) - 滴涂量
    parameters['DT(ml)'] = st.number_input(
        "DT(ml) - 滴涂量", 
        value=default_values['DT(ml)'], 
        step=0.1,
        key="dt_input",
        format="%.2f",
        help="电极表面的样品滴涂体积"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="blue-input">', unsafe_allow_html=True)
    
    # SS(mV/s) - 扫描速率
    parameters['SS(mV/s)'] = st.number_input(
        "SS(mV/s) - 扫描速率", 
        value=default_values['SS(mV/s)'], 
        step=10.0,
        key="ss_input",
        format="%.1f",
        help="差分脉冲伏安法的扫描速率"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="orange-input">', unsafe_allow_html=True)
    
    # PH - pH值
    parameters['PH'] = st.number_input(
        "PH - 溶液pH值", 
        value=default_values['PH'], 
        step=0.1,
        key="ph_input",
        format="%.2f",
        help="检测溶液的pH值"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="orange-input">', unsafe_allow_html=True)
    
    # P(V) - 电压
    parameters['P(V)'] = st.number_input(
        "P(V) - 检测电压", 
        value=default_values['P(V)'], 
        step=0.01,
        key="p_input",
        format="%.3f",
        help="差分脉冲伏安法的检测电压"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="green-input">', unsafe_allow_html=True)
    
    # TM(min) - 孵化时间
    parameters['TM(min)'] = st.number_input(
        "TM(min) - 孵化时间", 
        value=default_values['TM(min)'], 
        step=5.0,
        key="tm_input",
        format="%.1f",
        help="样品与电极的反应孵化时间"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="green-input">', unsafe_allow_html=True)
    
    # C0(uM) - 底液初始浓度
    parameters['C0(uM)'] = st.number_input(
        "C0(uM) - 底液初始浓度", 
        value=default_values['C0(uM)'], 
        step=1.0,
        key="c0_input",
        format="%.1f",
        help="电解质底液中目标物的初始浓度"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

# 强制JavaScript应用颜色
st.markdown("""
<script>
setTimeout(function() {
    // 为第一列的输入框添加蓝色样式
    const col1Inputs = document.querySelectorAll('[data-testid="column"]:nth-child(1) input[type="number"]');
    col1Inputs.forEach(input => {
        input.style.background = 'linear-gradient(135deg, #E3F2FD, #BBDEFB)';
        input.style.color = '#1565C0';
        input.style.border = '2px solid #2196F3';
        input.style.borderRadius = '8px';
        input.style.fontWeight = 'bold';
        input.style.fontSize = '16px';
        input.style.padding = '12px';
    });
    
    // 为第二列的输入框添加橙色样式
    const col2Inputs = document.querySelectorAll('[data-testid="column"]:nth-child(2) input[type="number"]');
    col2Inputs.forEach(input => {
        input.style.background = 'linear-gradient(135deg, #FFF3E0, #FFE0B2)';
        input.style.color = '#E65100';
        input.style.border = '2px solid #FF9800';
        input.style.borderRadius = '8px';
        input.style.fontWeight = 'bold';
        input.style.fontSize = '16px';
        input.style.padding = '12px';
    });
    
    // 为第三列的输入框添加绿色样式
    const col3Inputs = document.querySelectorAll('[data-testid="column"]:nth-child(3) input[type="number"]');
    col3Inputs.forEach(input => {
        input.style.background = 'linear-gradient(135deg, #E8F5E8, #C8E6C9)';
        input.style.color = '#2E7D32';
        input.style.border = '2px solid #4CAF50';
        input.style.borderRadius = '8px';
        input.style.fontWeight = 'bold';
        input.style.fontSize = '16px';
        input.style.padding = '12px';
    });
}, 2000);

// 定期重新应用样式，确保颜色持续存在
setInterval(function() {
    const allInputs = document.querySelectorAll('input[type="number"]');
    allInputs.forEach((input, index) => {
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
        input.style.fontSize = '16px';
        input.style.padding = '12px';
    });
}, 5000);
</script>
""", unsafe_allow_html=True)

# 显示当前参数值
with st.expander("📋 查看当前参数设置", expanded=False):
    params_display = ""
    for param, value in parameters.items():
        params_display += f"**{param}**: {value} | "
    st.markdown(params_display[:-3])

# 预测控制按钮
st.markdown("### 🚀 执行优化预测")
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    predict_clicked = st.button(
        "⚡ 开始优化", 
        use_container_width=True, 
        type="primary",
        help="使用GBDT模型优化传感参数并预测电流响应"
    )

with col2:
    if st.button("🔄 重置参数", use_container_width=True):
        st.rerun()

with col3:
    show_details = st.checkbox("显示详细信息", value=False)

# 执行预测
if predict_clicked:
    log("=" * 50)
    log("开始新烟碱农药传感参数优化预测")
    
    # 检查参数范围
    warnings = predictor.check_parameter_ranges(parameters)
    st.session_state.warnings = warnings
    
    try:
        # 执行预测
        result = predictor.predict(parameters)
        st.session_state.prediction_result = result
        st.session_state.prediction_error = None
        log(f"参数优化完成，预测电流响应: {result:.4f} uA")
        
    except Exception as e:
        error_msg = str(e)
        st.session_state.prediction_error = error_msg
        st.session_state.prediction_result = None
        log(f"优化预测失败: {error_msg}")

# 结果显示
if st.session_state.prediction_result is not None:
    st.markdown("---")
    
    # 主要结果显示
    st.markdown(
        f"<div class='result-display'>🎯 优化后电流响应: {st.session_state.prediction_result:.4f} μA</div>", 
        unsafe_allow_html=True
    )
    
    # 警告显示
    if st.session_state.warnings:
        warnings_html = "<div class='warning-box'><h4>⚠️ 参数范围警告</h4><ul>"
        for warning in st.session_state.warnings:
            warnings_html += f"<li>{warning}</li>"
        warnings_html += "</ul><p><em>建议检查参数设置，确保在实验合理范围内。</em></p></div>"
        st.markdown(warnings_html, unsafe_allow_html=True)
    
    # 详细信息显示
    if show_details:
        with st.expander("📊 优化详细信息", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                **优化信息:**
                - 目标变量: {predictor.target_name}
                - 优化预测值: {st.session_state.prediction_result:.6f} μA
                - 模型类型: GBDT Pipeline
                - 预处理: RobustScaler标准化
                """)
            with col2:
                st.markdown(f"""
                **系统状态:**
                - 加载状态: {'✅ 正常' if predictor.model_loaded else '❌ 失败'}
                - 特征数量: {len(predictor.feature_names)}
                - 参数警告: {len(st.session_state.warnings)}个
                - 应用领域: 新烟碱农药检测
                """)

elif st.session_state.prediction_error is not None:
    st.markdown("---")
    error_html = f"""
    <div class='error-box'>
        <h3>❌ 优化失败</h3>
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
with st.expander("📚 传感参数优化技术说明", expanded=False):
    st.markdown("""
    <div class='info-box'>
    <h4>🔬 新烟碱农药传感参数优化原理</h4>
    <p>本系统基于<strong>差分脉冲伏安法(DPV)</strong>进行新烟碱农药的电化学检测，使用GBDT机器学习模型优化传感参数并预测最佳电流响应。</p>
    
    <h4>📋 参数说明</h4>
    <ul>
        <li><strong>DT(ml)</strong>: 滴涂量 - 电极表面样品的滴涂体积</li>
        <li><strong>PH</strong>: pH值 - 检测溶液的酸碱度</li>
        <li><strong>SS(mV/s)</strong>: 扫描速率 - 电压扫描的速度</li>
        <li><strong>P(V)</strong>: 检测电压 - 目标化合物的氧化还原电位</li>
        <li><strong>TM(min)</strong>: 孵化时间 - 样品与电极的反应时间</li>
        <li><strong>C0(uM)</strong>: 底液初始浓度 - 电解质中目标物浓度</li>
    </ul>
    
    <h4>🎯 优化目标</h4>
    <p>通过调节各传感参数，实现对吡虫啉、噻虫嗪、噻虫胺等新烟碱类农药的最佳检测响应，提高检测精度和稳定性。</p>
    </div>
    """, unsafe_allow_html=True)

# 页脚信息
st.markdown("---")
footer_info = """
<div style='text-align: center; color: #666; padding: 20px;'>
<p><strong>© 2024 电化学传感器实验室</strong> | 新烟碱农药传感参数优化系统 | 版本: 2.1.0</p>
<p>🔬 基于GBDT算法 | ⚡ 差分脉冲伏安法 | 🎯 智能参数优化</p>
<p><em>特征顺序: DT(ml) → PH → SS(mV/s) → P(V) → TM(min) → C0(uM)</em></p>
</div>
"""
st.markdown(footer_info, unsafe_allow_html=True)