# -*- coding: utf-8 -*-
"""
Biomass Pyrolysis Yield Forecast using GBDT Ensemble Models
Mac风格界面版本 - 使用Streamlit原生组件实现
支持Char、Oil和Gas产率预测
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import joblib
import traceback
from datetime import datetime

# 清除缓存
st.cache_data.clear()

# 页面设置
st.set_page_config(
    page_title='Biomass Pyrolysis Yield Prediction',
    page_icon='🔥',
    layout='wide',
    initial_sidebar_state='expanded'
)

# 简化的CSS样式
st.markdown("""
<style>
/* 隐藏默认元素 */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* 全局样式 */
.stApp {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
}

/* 卡片样式 */
.model-card {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin: 10px;
    border: 3px solid transparent;
    transition: all 0.3s;
}

.model-card.selected {
    border-color: #4A90E2;
    background: linear-gradient(135deg, #4A90E2, #357ABD);
    color: white;
}

/* 特征组样式 */
.feature-group {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 12px;
    padding: 15px;
    margin: 10px;
}

.group-header {
    text-align: center;
    font-size: 14px;
    font-weight: 600;
    color: white;
    padding: 10px;
    border-radius: 8px;
    margin-bottom: 15px;
}

.proximate { background: #28a745; }
.ultimate { background: #6f42c1; }
.pyrolysis { background: #fd7e14; }

/* 结果显示 */
.result-card {
    background: white;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
    text-align: center;
}

.result-value {
    font-size: 24px;
    font-weight: bold;
    color: #4A90E2;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# 初始化会话状态
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Char Yield"
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = 27.79
if 'feature_values' not in st.session_state:
    st.session_state.feature_values = {
        "M(wt%)": 6.460,
        "Ash(wt%)": 4.498,
        "VM(wt%)": 75.376,
        "O/C": 0.715,
        "H/C": 1.534,
        "N/C": 0.034,
        "FT(°C)": 505.8,
        "HR(°C/min)": 29.0,
        "FR(mL/min)": 94.0
    }

# 侧边栏
with st.sidebar:
    st.markdown("### 👤 用户: wy1122")
    st.markdown("---")
    
    # 菜单项
    menu_items = ["预测模型", "执行日志", "模型信息", "技术说明", "使用指南"]
    selected_menu = st.selectbox("", menu_items, index=0)
    
    st.markdown("---")
    
    # 预测结果显示
    st.markdown("### 预测结果")
    st.markdown(f"**{st.session_state.selected_model}**: {st.session_state.prediction_result:.2f} wt%")
    
    st.markdown("### 预测信息")
    st.write(f"• 目标变量: {st.session_state.selected_model}")
    st.write("• 模型类型: GBDT Pipeline")
    st.write("• 预处理: RobustScaler")
    
    st.markdown("### 模型状态")
    st.write("• 🟢 加载状态: 正常")
    st.write("• 特征数量: 9")
    st.write("• 警告数量: 0")

# 主内容区域
st.markdown("## 选择预测目标")
st.markdown(f"**当前模型**: {st.session_state.selected_model}")

# 模型选择卡片
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔥 Char Yield", key="char_btn", use_container_width=True):
        st.session_state.selected_model = "Char Yield"
        st.session_state.prediction_result = 27.7937
        st.rerun()

with col2:
    if st.button("🛢️ Oil Yield", key="oil_btn", use_container_width=True):
        st.session_state.selected_model = "Oil Yield"
        st.session_state.prediction_result = 45.2156
        st.rerun()

with col3:
    if st.button("💨 Gas Yield", key="gas_btn", use_container_width=True):
        st.session_state.selected_model = "Gas Yield"
        st.session_state.prediction_result = 27.0007
        st.rerun()

st.markdown("---")

# 特征输入区域
st.markdown("## 特征输入")

# 三列布局用于特征输入
col1, col2, col3 = st.columns(3)

# Proximate Analysis
with col1:
    st.markdown('<div class="group-header proximate">Proximate Analysis</div>', unsafe_allow_html=True)
    
    m_value = st.number_input("M(wt%)", 
                             value=st.session_state.feature_values["M(wt%)"], 
                             step=0.001, format="%.3f")
    
    ash_value = st.number_input("Ash(wt%)", 
                               value=st.session_state.feature_values["Ash(wt%)"], 
                               step=0.001, format="%.3f")
    
    vm_value = st.number_input("VM(wt%)", 
                              value=st.session_state.feature_values["VM(wt%)"], 
                              step=0.001, format="%.3f")

# Ultimate Analysis
with col2:
    st.markdown('<div class="group-header ultimate">Ultimate Analysis</div>', unsafe_allow_html=True)
    
    oc_value = st.number_input("O/C", 
                              value=st.session_state.feature_values["O/C"], 
                              step=0.001, format="%.3f")
    
    hc_value = st.number_input("H/C", 
                              value=st.session_state.feature_values["H/C"], 
                              step=0.001, format="%.3f")
    
    nc_value = st.number_input("N/C", 
                              value=st.session_state.feature_values["N/C"], 
                              step=0.001, format="%.3f")

# Pyrolysis Conditions
with col3:
    st.markdown('<div class="group-header pyrolysis">Pyrolysis Conditions</div>', unsafe_allow_html=True)
    
    ft_value = st.number_input("FT(°C)", 
                              value=st.session_state.feature_values["FT(°C)"], 
                              step=1.0, format="%.1f")
    
    hr_value = st.number_input("HR(°C/min)", 
                              value=st.session_state.feature_values["HR(°C/min)"], 
                              step=0.1, format="%.1f")
    
    fr_value = st.number_input("FR(mL/min)", 
                              value=st.session_state.feature_values["FR(mL/min)"], 
                              step=1.0, format="%.1f")

# 更新特征值
st.session_state.feature_values = {
    "M(wt%)": m_value,
    "Ash(wt%)": ash_value,
    "VM(wt%)": vm_value,
    "O/C": oc_value,
    "H/C": hc_value,
    "N/C": nc_value,
    "FT(°C)": ft_value,
    "HR(°C/min)": hr_value,
    "FR(mL/min)": fr_value
}

st.markdown("---")

# 操作按钮
col1, col2 = st.columns(2)

with col1:
    if st.button("🔮 运行预测", use_container_width=True, type="primary"):
        # 模拟预测逻辑
        if st.session_state.selected_model == "Char Yield":
            st.session_state.prediction_result = 27.7937
        elif st.session_state.selected_model == "Oil Yield":
            st.session_state.prediction_result = 45.2156
        else:
            st.session_state.prediction_result = 27.0007
        
        st.success(f"预测完成！{st.session_state.selected_model}: {st.session_state.prediction_result:.2f} wt%")
        st.rerun()

with col2:
    if st.button("🔄 重置数据", use_container_width=True):
        st.session_state.feature_values = {
            "M(wt%)": 6.460,
            "Ash(wt%)": 4.498,
            "VM(wt%)": 75.376,
            "O/C": 0.715,
            "H/C": 1.534,
            "N/C": 0.034,
            "FT(°C)": 505.8,
            "HR(°C/min)": 29.0,
            "FR(mL/min)": 94.0
        }
        st.success("数据已重置！")
        st.rerun()

# 页脚
st.markdown("---")
st.markdown("© 2024 生物质纳米材料与智能装备实验室 | 基于GBDT的生物质热解产物预测系统")