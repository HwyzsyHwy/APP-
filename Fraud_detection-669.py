# -*- coding: utf-8 -*-
"""
电化学传感检测新烟碱农药检测参数预测系统
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
    page_title='电化学传感检测参数预测系统',
    page_icon='⚡',
    layout='wide',
    initial_sidebar_state='expanded'
)

# 强制输入框颜色填充 - 使用更强力的CSS和JavaScript
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
    
    /* 强制所有数字输入框的基础样式 */
    input[type="number"] {
        font-size: 16px !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        padding: 12px !important;
        border-width: 2px !important;
    }
    
    /* 通过data属性强制设置颜色 */
    input[data-color="blue"] {
        background: linear-gradient(135deg, #E3F2FD, #BBDEFB) !important;
        color: #1565C0 !important;
        border: 2px solid #2196F3 !important;
    }
    
    input[data-color="orange"] {
        background: linear-gradient(135deg, #FFF3E0, #FFE0B2) !important;
        color: #E65100 !important;
        border: 2px solid #FF9800 !important;
    }
    
    input[data-color="green"] {
        background: linear-gradient(135deg, #E8F5E8, #C8E6C9) !important;
        color: #2E7D32 !important;
        border: 2px solid #4CAF50 !important;
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
st.markdown("<h1 class='main-title'>电化学传感检测新烟碱农药检测参数预测系统</h1>", unsafe_allow_html=True)

class NeonicotinoidPredictor:
    """新烟碱农药电化学检测预测器"""
    
    def __init__(self):
        self.target_name = "I(uA)"
        self.feature_names = [
            'DT(ml)', 'PH', 'SS(mV/s)', 'P(V)', 'TM(min)', 'C0(uM)'
        ]
        
        self.parameter_ranges = {
            'DT(ml)': {'min': 0.1, 'max': 20.0},
            'PH': {'min': 3.0, 'max': 10.0},
            'SS(mV/s)': {'min': 10.0, 'max': 500.0},
            'P(V)': {'min': -2.0, 'max': 2.0},
            'TM(min)': {'min': 1.0, 'max': 120.0},
            'C0(uM)': {'min': 0.1, 'max': 1000.0}
        }
        
        self.model_loaded = False
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """加载GBDT模型"""
        model_paths = [
            "GBDT.joblib", "./GBDT.joblib", "../GBDT.joblib",
            r"C:\Users\HWY\Desktop\开题-7.2\GBDT.joblib",
            "./models/GBDT.joblib", "../models/GBDT.joblib"
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
        
        data = []
        for feature in self.feature_names:
            data.append(parameters.get(feature, 0.0))
        
        df = pd.DataFrame([data], columns=self.feature_names)
        log(f"输入数据: {dict(zip(self.feature_names, data))}")
        
        try:
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
st.markdown("### 🔬 传感检测参数输入")

# 默认值
default_values = {
    "DT(ml)": 5.0, "PH": 7.0, "SS(mV/s)": 100.0,
    "P(V)": 0.0, "TM(min)": 30.0, "C0(uM)": 50.0
}

# 创建三列布局
col1, col2, col3 = st.columns(3)

parameters = {}

# 第一列 - 蓝色
with col1:
    parameters['DT(ml)'] = st.number_input(
        "DT(ml) - 滴涂量", 
        value=default_values['DT(ml)'], 
        step=0.1,
        key="dt_input",
        format="%.2f",
        help="电极表面的样品滴涂体积"
    )
    
    parameters['SS(mV/s)'] = st.number_input(
        "SS(mV/s) - 扫描速率", 
        value=default_values['SS(mV/s)'], 
        step=10.0,
        key="ss_input",
        format="%.1f",
        help="差分脉冲伏安法的扫描速率"
    )

# 第二列 - 橙色
with col2:
    parameters['PH'] = st.number_input(
        "PH - 溶液pH值", 
        value=default_values['PH'], 
        step=0.1,
        key="ph_input",
        format="%.2f",
        help="检测溶液的pH值"
    )
    
    parameters['P(V)'] = st.number_input(
        "P(V) - 检测电压", 
        value=default_values['P(V)'], 
        step=0.01,
        key="p_input",
        format="%.3f",
        help="差分脉冲伏安法的检测电压"
    )

# 第三列 - 绿色
with col3:
    parameters['TM(min)'] = st.number_input(
        "TM(min) - 孵化时间", 
        value=default_values['TM(min)'], 
        step=5.0,
        key="tm_input",
        format="%.1f",
        help="样品与电极的反应孵化时间"
    )
    
    parameters['C0(uM)'] = st.number_input(
        "C0(uM) - 底液初始浓度", 
        value=default_values['C0(uM)'], 
        step=1.0,
        key="c0_input",
        format="%.1f",
        help="电解质底液中目标物的初始浓度"
    )

# 强制JavaScript颜色应用 - 更精确的方法
st.markdown("""
<script>
window.addEventListener('DOMContentLoaded', function() {
    function forceApplyColors() {
        // 获取所有number类型的输入框
        const allInputs = document.querySelectorAll('input[type="number"]');
        
        // 清除所有已有的data-color属性
        allInputs.forEach(input => {
            input.removeAttribute('data-color');
        });
        
        // 获取三列容器
        const columns = document.querySelectorAll('[data-testid="column"]');
        
        if (columns.length >= 3) {
            // 第一列 - 蓝色
            const col1Inputs = columns[0].querySelectorAll('input[type="number"]');
            col1Inputs.forEach(input => {
                input.setAttribute('data-color', 'blue');
                input.style.setProperty('background', 'linear-gradient(135deg, #E3F2FD, #BBDEFB)', 'important');
                input.style.setProperty('color', '#1565C0', 'important');
                input.style.setProperty('border', '2px solid #2196F3', 'important');
                input.style.setProperty('border-radius', '8px', 'important');
                input.style.setProperty('font-weight', 'bold', 'important');
                input.style.setProperty('font-size', '16px', 'important');
                input.style.setProperty('padding', '12px', 'important');
            });
            
            // 第二列 - 橙色
            const col2Inputs = columns[1].querySelectorAll('input[type="number"]');
            col2Inputs.forEach(input => {
                input.setAttribute('data-color', 'orange');
                input.style.setProperty('background', 'linear-gradient(135deg, #FFF3E0, #FFE0B2)', 'important');
                input.style.setProperty('color', '#E65100', 'important');
                input.style.setProperty('border', '2px solid #FF9800', 'important');
                input.style.setProperty('border-radius', '8px', 'important');
                input.style.setProperty('font-weight', 'bold', 'important');
                input.style.setProperty('font-size', '16px', 'important');
                input.style.setProperty('padding', '12px', 'important');
            });
            
            // 第三列 - 绿色
            const col3Inputs = columns[2].querySelectorAll('input[type="number"]');
            col3Inputs.forEach(input => {
                input.setAttribute('data-color', 'green');
                input.style.setProperty('background', 'linear-gradient(135deg, #E8F5E8, #C8E6C9)', 'important');
                input.style.setProperty('color', '#2E7D32', 'important');
                input.style.setProperty('border', '2px solid #4CAF50', 'important');
                input.style.setProperty('border-radius', '8px', 'important');
                input.style.setProperty('font-weight', 'bold', 'important');
                input.style.setProperty('font-size', '16px', 'important');
                input.style.setProperty('padding', '12px', 'important');
            });
        }
        
        console.log('颜色应用完成，输入框数量:', allInputs.length);
    }
    
    // 立即执行
    forceApplyColors();
    
    // 延迟执行多次
    setTimeout(forceApplyColors, 100);
    setTimeout(forceApplyColors, 500);
    setTimeout(forceApplyColors, 1000);
    setTimeout(forceApplyColors, 2000);
    setTimeout(forceApplyColors, 3000);
    
    // 定期执行
    setInterval(forceApplyColors, 5000);
    
    // 监听DOM变化
    const observer = new MutationObserver(function(mutations) {
        let shouldUpdate = false;
        mutations.forEach(function(mutation) {
            if (mutation.type === 'childList') {
                const addedNodes = Array.from(mutation.addedNodes);
                if (addedNodes.some(node => node.nodeType === 1 && (node.tagName === 'INPUT' || node.querySelector('input')))) {
                    shouldUpdate = true;
                }
            }
        });
        if (shouldUpdate) {
            setTimeout(forceApplyColors, 100);
        }
    });
    
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
});
</script>
""", unsafe_allow_html=True)

# 预测控制按钮
st.markdown("### 🚀 执行预测")
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    predict_clicked = st.button("⚡ 开始预测", use_container_width=True, type="primary")

with col2:
    if st.button("🔄 重置参数", use_container_width=True):
        st.rerun()

with col3:
    show_details = st.checkbox("显示详细信息", value=False)

# 执行预测
if predict_clicked:
    log("=" * 50)
    log("开始新烟碱农药检测参数预测")
    
    warnings = predictor.check_parameter_ranges(parameters)
    st.session_state.warnings = warnings
    
    try:
        result = predictor.predict(parameters)
        st.session_state.prediction_result = result
        st.session_state.prediction_error = None
        log(f"预测完成，电流响应: {result:.4f} uA")
        
    except Exception as e:
        error_msg = str(e)
        st.session_state.prediction_error = error_msg
        st.session_state.prediction_result = None
        log(f"预测失败: {error_msg}")

# 显示警告信息
if st.session_state.warnings:
    st.warning("⚠️ 参数超出建议范围：")
    for warning in st.session_state.warnings:
        st.write(f"• {warning}")

# 显示错误信息
if st.session_state.prediction_error:
    st.error(f"❌ 预测失败: {st.session_state.prediction_error}")

# 结果显示 - 修改为"预测响应电流"
if st.session_state.prediction_result is not None:
    st.markdown("---")
    st.markdown(
        f"""
        <div style='background-color: #1E1E1E; color: white; font-size: 36px; font-weight: bold; 
                    text-align: center; padding: 20px; border-radius: 10px; margin-top: 20px; 
                    border: 2px solid #2E86AB;'>
        🎯 预测响应电流: {st.session_state.prediction_result:.4f} μA
        </div>
        """, 
        unsafe_allow_html=True
    )

# 详细信息显示
if show_details and st.session_state.prediction_result is not None:
    st.markdown("### 📊 预测详情")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**输入参数：**")
        for param, value in parameters.items():
            st.write(f"• {param}: {value}")
    
    with col2:
        st.markdown("**模型信息：**")
        st.write(f"• 模型类型: GBDT")
        st.write(f"• 特征数量: {len(predictor.feature_names)}")
        st.write(f"• 目标变量: {predictor.target_name}")

# 技术说明
st.markdown("---")
st.markdown("### 📖 技术说明")

with st.expander("电化学检测原理"):
    st.markdown("""
    **差分脉冲伏安法 (DPV)** 是检测新烟碱农药的高灵敏度电化学技术：
    
    - **DT(ml)**: 样品滴涂量影响信号强度和检测精度
    - **PH**: 溶液酸碱度影响电极反应和信号稳定性  
    - **SS(mV/s)**: 扫描速率决定检测时间和分辨率
    - **P(V)**: 检测电压设定目标化合物的氧化还原电位
    - **TM(min)**: 孵化时间确保充分的电极表面反应
    - **C0(uM)**: 底液浓度影响基线电流和检测范围
    """)

with st.expander("GBDT模型特点"):
    st.markdown("""
    **梯度提升决策树 (GBDT)** 用于电化学响应预测：
    
    - 高精度回归预测
    - 自动特征重要性分析  
    - 处理非线性关系
    - 抗过拟合能力强
    """)

# 页脚
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666666; font-size: 12px;'>
    © 2024 电化学传感检测系统 | 版本 1.2.0 | 基于GBDT模型
    </div>
    """, 
    unsafe_allow_html=True
)