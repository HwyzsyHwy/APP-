import streamlit as st
import pandas as pd
import numpy as np
import pickle
import traceback
from datetime import datetime

# 页面配置
st.set_page_config(
    page_title="Streamlit",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 基本样式
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    font-family: 'Arial', sans-serif;
}

.main .block-container {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 2rem;
    margin-top: 2rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #20b2aa, #17a2b8) !important;
    border: 3px solid #20b2aa !important;
    color: white !important;
    box-shadow: 0 8px 25px rgba(32, 178, 170, 0.4) !important;
    transform: translateY(-2px) !important;
}

.stButton > button[kind="secondary"] {
    background: rgba(255,255,255,0.1) !important;
    border: 2px solid rgba(255,255,255,0.3) !important;
    color: white !important;
    transition: all 0.3s ease !important;
}

.stNumberInput button {
    border: none !important;
    color: white !important;
    font-weight: bold !important;
    border-radius: 4px !important;
    margin: 0 !important;
}

.stColumn:nth-child(1) .stNumberInput button {
    background-color: #20b2aa !important;
}
.stColumn:nth-child(2) .stNumberInput button {
    background-color: #daa520 !important;
}
.stColumn:nth-child(3) .stNumberInput button {
    background-color: #cd5c5c !important;
}
</style>
""", unsafe_allow_html=True)

# 初始化会话状态
if 'current_page' not in st.session_state:
    st.session_state.current_page = "预测模型"

if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Char Yield"

if 'bottom_button_selected' not in st.session_state:
    st.session_state.bottom_button_selected = "predict"

if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []

def log(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    st.session_state.log_messages.append(log_entry)
    if len(st.session_state.log_messages) > 100:
        st.session_state.log_messages = st.session_state.log_messages[-100:]

# 侧边栏导航
with st.sidebar:
    st.markdown('<div style="text-align: center; margin-bottom: 30px;"><img src="https://raw.githubusercontent.com/HwyzsyHwy/APP-/main/用户.png" style="width: 60px; height: 60px; border-radius: 50%; margin-bottom: 10px;"><p style="color: white; margin: 0;">用户：wy1122</p></div>', unsafe_allow_html=True)
    
    current_page = st.session_state.current_page
    
    if st.button("预测模型", key="nav_predict", use_container_width=True, type="primary" if current_page == "预测模型" else "secondary"):
        st.session_state.current_page = "预测模型"
        st.rerun()
    
    if st.button("执行日志", key="nav_log", use_container_width=True, type="primary" if current_page == "执行日志" else "secondary"):
        st.session_state.current_page = "执行日志"
        st.rerun()

# 主要内容
if st.session_state.current_page == "预测模型":
    st.markdown('<h1 style="color: white; text-align: center; margin-bottom: 30px;">Streamlit</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="color: white; text-align: center; margin-bottom: 30px;">选择预测目标</h3>', unsafe_allow_html=True)
    
    # 模型选择
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔥 Char Yield", key="char_card", use_container_width=True,
                    type="primary" if st.session_state.selected_model == "Char Yield" else "secondary"):
            if st.session_state.selected_model != "Char Yield":
                st.session_state.selected_model = "Char Yield"
                st.session_state.prediction_result = None
                log("切换到模型: Char Yield")
                st.rerun()
    
    with col2:
        if st.button("🛢️ Oil Yield", key="oil_card", use_container_width=True,
                    type="primary" if st.session_state.selected_model == "Oil Yield" else "secondary"):
            if st.session_state.selected_model != "Oil Yield":
                st.session_state.selected_model = "Oil Yield"
                st.session_state.prediction_result = None
                log("切换到模型: Oil Yield")
                st.rerun()
    
    with col3:
        if st.button("💨 Gas Yield", key="gas_card", use_container_width=True,
                    type="primary" if st.session_state.selected_model == "Gas Yield" else "secondary"):
            if st.session_state.selected_model != "Gas Yield":
                st.session_state.selected_model = "Gas Yield"
                st.session_state.prediction_result = None
                log("切换到模型: Gas Yield")
                st.rerun()
    
    st.markdown("---")
    
    # 参数输入
    st.markdown('<h4 style="color: white; text-align: center; margin-bottom: 20px;">输入参数</h4>', unsafe_allow_html=True)
    
    # 默认值
    default_values = {
        "M(wt%)": 7.542,
        "Ash(wt%)": 1.542,
        "VM(wt%)": 82.542,
        "O/C": 0.542,
        "H/C": 1.542,
        "N/C": 0.034,
        "FT(C)": 505.811,
        "HR(C/min)": 29.011,
        "FR(mL/min)": 93.962
    }
    
    feature_categories = {
        "Proximate Analysis": ["M(wt%)", "Ash(wt%)", "VM(wt%)"],
        "Ultimate Analysis": ["O/C", "H/C", "N/C"],
        "Pyrolysis Conditions": ["FT(C)", "HR(C/min)", "FR(mL/min)"]
    }
    
    category_colors = {
        "Proximate Analysis": "#20b2aa",
        "Ultimate Analysis": "#daa520",
        "Pyrolysis Conditions": "#cd5c5c"
    }
    
    col1, col2, col3 = st.columns(3)
    features = {}
    
    # 第一列
    with col1:
        st.markdown('<div style="background: white; color: #333; padding: 12px 20px; border-radius: 25px; text-align: center; font-weight: bold; font-size: 16px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">Proximate Analysis</div>', unsafe_allow_html=True)
        
        for feature in feature_categories["Proximate Analysis"]:
            value = st.number_input(
                feature,
                value=default_values[feature],
                step=0.001,
                format="%.3f",
                key=f"input_{feature}"
            )
            features[feature] = value
    
    # 第二列
    with col2:
        st.markdown('<div style="background: white; color: #333; padding: 12px 20px; border-radius: 25px; text-align: center; font-weight: bold; font-size: 16px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">Ultimate Analysis</div>', unsafe_allow_html=True)
        
        for feature in feature_categories["Ultimate Analysis"]:
            value = st.number_input(
                feature,
                value=default_values[feature],
                step=0.001,
                format="%.3f",
                key=f"input_{feature}"
            )
            features[feature] = value
    
    # 第三列
    with col3:
        st.markdown('<div style="background: white; color: #333; padding: 12px 20px; border-radius: 25px; text-align: center; font-weight: bold; font-size: 16px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">Pyrolysis Conditions</div>', unsafe_allow_html=True)
        
        for feature in feature_categories["Pyrolysis Conditions"]:
            value = st.number_input(
                feature,
                value=default_values[feature],
                step=0.001,
                format="%.3f",
                key=f"input_{feature}"
            )
            features[feature] = value
    
    st.markdown("---")
    
    # 底部按钮
    col1, col2 = st.columns([1, 1])
    
    with col1:
        predict_clicked = st.button("运行预测", use_container_width=True, 
                                   type="primary" if st.session_state.bottom_button_selected == "predict" else "secondary")
        if predict_clicked:
            st.session_state.bottom_button_selected = "predict"
            log("开始预测流程...")
            # 这里可以添加实际的预测逻辑
            st.session_state.prediction_result = 42.5  # 示例结果
            st.rerun()
    
    with col2:
        reset_clicked = st.button("重置输入", use_container_width=True,
                                 type="primary" if st.session_state.bottom_button_selected == "reset" else "secondary")
        if reset_clicked:
            st.session_state.bottom_button_selected = "reset"
            log("重置所有输入值")
            st.session_state.prediction_result = None
            st.rerun()
    
    # 显示预测结果
    if st.session_state.prediction_result is not None:
        st.markdown("---")
        st.markdown(f'<div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;"><h3 style="color: white; margin: 0;">预测结果: {st.session_state.prediction_result:.3f} wt%</h3></div>', unsafe_allow_html=True)

elif st.session_state.current_page == "执行日志":
    st.markdown('<h2 style="color: white; text-align: center; margin-bottom: 30px;">执行日志</h2>', unsafe_allow_html=True)
    
    if st.session_state.log_messages:
        log_content = "<br>".join(st.session_state.log_messages[-50:])
        st.markdown(f'<div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; font-family: monospace; color: white; max-height: 400px; overflow-y: auto;">{log_content}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; text-align: center; color: white;">暂无日志记录</div>', unsafe_allow_html=True)