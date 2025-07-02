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

# 强制输入框内部颜色填充的CSS
st.markdown(
    """
    <style>
    /* 强制覆盖所有输入框的内部颜色 */
    .stNumberInput > div > div > input {
        font-size: 16px !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        border-width: 2px !important;
        padding: 12px !important;
    }
    
    /* 直接通过属性选择器强制设置颜色 */
    input[step="0.1"] {
        background: linear-gradient(135deg, #E3F2FD, #BBDEFB) !important;
        color: #1565C0 !important;
        border: 2px solid #2196F3 !important;
    }
    
    input[step="10"] {
        background: linear-gradient(135deg, #E3F2FD, #BBDEFB) !important;
        color: #1565C0 !important;
        border: 2px solid #2196F3 !important;
    }
    
    input[step="0.01"] {
        background: linear-gradient(135deg, #FFF3E0, #FFE0B2) !important;
        color: #E65100 !important;
        border: 2px solid #FF9800 !important;
    }
    
    input[step="5"] {
        background: linear-gradient(135deg, #E8F5E8, #C8E6C9) !important;
        color: #2E7D32 !important;
        border: 2px solid #4CAF50 !important;
    }
    
    input[step="1"] {
        background: linear-gradient(135deg, #E8F5E8, #C8E6C9) !important;
        color: #2E7D32 !important;
        border: 2px solid #4CAF50 !important;
    }
    
    /* 通过key属性强制设置 */
    input[aria-describedby*="dt_input"] {
        background: linear-gradient(135deg, #E3F2FD, #BBDEFB) !important;
        color: #1565C0 !important;
        border: 2px solid #2196F3 !important;
    }
    
    input[aria-describedby*="ss_input"] {
        background: linear-gradient(135deg, #E3F2FD, #BBDEFB) !important;
        color: #1565C0 !important;
        border: 2px solid #2196F3 !important;
    }
    
    input[aria-describedby*="ph_input"] {
        background: linear-gradient(135deg, #FFF3E0, #FFE0B2) !important;
        color: #E65100 !important;
        border: 2px solid #FF9800 !important;
    }
    
    input[aria-describedby*="p_input"] {
        background: linear-gradient(135deg, #FFF3E0, #FFE0B2) !important;
        color: #E65100 !important;
        border: 2px solid #FF9800 !important;
    }
    
    input[aria-describedby*="tm_input"] {
        background: linear-gradient(135deg, #E8F5E8, #C8E6C9) !important;
        color: #2E7D32 !important;
        border: 2px solid #4CAF50 !important;
    }
    
    input[aria-describedby*="c0_input"] {
        background: linear-gradient(135deg, #E8F5E8, #C8E6C9) !important;
        color: #2E7D32 !important;
        border: 2px solid #4CAF50 !important;
    }
    
    /* 通用强制样式 */
    input[type="number"] {
        background: linear-gradient(135deg, #E3F2FD, #BBDEFB) !important;
        color: #1565C0 !important;
        border: 2px solid #2196F3 !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        font-size: 16px !important;
        padding: 12px !important;
    }
    
    .main-title {
        text-align: center;
        font-size: 32px !important;
        font-weight: bold;
        margin-bottom: 20px;
        color: white !important;
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

# 使用唯一的step值来区分不同的输入框
with col1:
    # DT(ml) - 使用step=0.1 (蓝色)
    parameters['DT(ml)'] = st.number_input(
        "DT(ml) - 滴涂量", 
        value=default_values['DT(ml)'], 
        step=0.1,  # 蓝色标识
        key="dt_input",
        format="%.2f",
        help="电极表面的样品滴涂体积"
    )
    
    # SS(mV/s) - 使用step=10.0 (蓝色)
    parameters['SS(mV/s)'] = st.number_input(
        "SS(mV/s) - 扫描速率", 
        value=default_values['SS(mV/s)'], 
        step=10.0,  # 蓝色标识
        key="ss_input",
        format="%.1f",
        help="差分脉冲伏安法的扫描速率"
    )

with col2:
    # PH - 使用step=0.1 (但通过key区分为橙色)
    parameters['PH'] = st.number_input(
        "PH - 溶液pH值", 
        value=default_values['PH'], 
        step=0.1,
        key="ph_input",
        format="%.2f",
        help="检测溶液的pH值"
    )
    
    # P(V) - 使用step=0.01 (橙色)
    parameters['P(V)'] = st.number_input(
        "P(V) - 检测电压", 
        value=default_values['P(V)'], 
        step=0.01,  # 橙色标识
        key="p_input",
        format="%.3f",
        help="差分脉冲伏安法的检测电压"
    )

with col3:
    # TM(min) - 使用step=5.0 (绿色)
    parameters['TM(min)'] = st.number_input(
        "TM(min) - 孵化时间", 
        value=default_values['TM(min)'], 
        step=5.0,  # 绿色标识
        key="tm_input",
        format="%.1f",
        help="样品与电极的反应孵化时间"
    )
    
    # C0(uM) - 使用step=1.0 (绿色)
    parameters['C0(uM)'] = st.number_input(
        "C0(uM) - 底液初始浓度", 
        value=default_values['C0(uM)'], 
        step=1.0,  # 绿色标识
        key="c0_input",
        format="%.1f",
        help="电解质底液中目标物的初始浓度"
    )

# 强制JavaScript应用输入框内部颜色
st.markdown("""
<script>
// 立即执行
(function() {
    const applyColors = () => {
        const inputs = document.querySelectorAll('input[type="number"]');
        
        inputs.forEach((input, index) => {
            // 移除可能的默认样式
            input.style.setProperty('background', '', 'important');
            input.style.setProperty('color', '', 'important');
            input.style.setProperty('border', '', 'important');
            
            // 根据位置应用颜色
            if (index === 0 || index === 1) {
                // 第一列 - 蓝色
                input.style.setProperty('background', 'linear-gradient(135deg, #E3F2FD, #BBDEFB)', 'important');
                input.style.setProperty('color', '#1565C0', 'important');
                input.style.setProperty('border', '2px solid #2196F3', 'important');
            } else if (index === 2 || index === 3) {
                // 第二列 - 橙色
                input.style.setProperty('background', 'linear-gradient(135deg, #FFF3E0, #FFE0B2)', 'important');
                input.style.setProperty('color', '#E65100', 'important');
                input.style.setProperty('border', '2px solid #FF9800', 'important');
            } else if (index === 4 || index === 5) {
                // 第三列 - 绿色
                input.style.setProperty('background', 'linear-gradient(135deg, #E8F5E8, #C8E6C9)', 'important');
                input.style.setProperty('color', '#2E7D32', 'important');
                input.style.setProperty('border', '2px solid #4CAF50', 'important');
            }
            
            // 通用样式
            input.style.setProperty('border-radius', '8px', 'important');
            input.style.setProperty('font-weight', 'bold', 'important');
            input.style.setProperty('font-size', '16px', 'important');
            input.style.setProperty('padding', '12px', 'important');
        });
    };
    
    // 多次执行确保生效
    setTimeout(applyColors, 500);
    setTimeout(applyColors, 1000);
    setTimeout(applyColors, 2000);
    setTimeout(applyColors, 3000);
    
    // 定期重新应用
    setInterval(applyColors, 5000);
    
    // 监听DOM变化
    const observer = new MutationObserver(applyColors);
    observer.observe(document.body, { childList: true, subtree: true });
})();
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

# 结果显示
if st.session_state.prediction_result is not None:
    st.markdown("---")
    st.markdown(
        f"""
        <div style='background-color: #1E1E1E; color: white; font-size: 36px; font-weight: bold; 
                    text-align: center; padding: 20px; border-radius: 10px; margin-top: 20px; 
                    border: 2px solid #2E86AB;'>
        🎯 预测电流响应: {st.session_state.prediction_result:.4f} μA
        </div>
        """, 
        unsafe_allow_html=True
    )