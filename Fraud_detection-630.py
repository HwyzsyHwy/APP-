# -*- coding: utf-8 -*-
"""
Biomass Pyrolysis Yield Forecast using GBDT Ensemble Models
简洁版本 - 使用Streamlit原生组件实现目标布局
支持Char、Oil和Gas产率预测
"""

import streamlit as st
import pandas as pd
import numpy as np

# 页面设置
st.set_page_config(
    page_title='Biomass Pyrolysis Yield Prediction',
    page_icon='🔥',
    layout='wide',
    initial_sidebar_state='collapsed'
)

# 简洁的CSS样式
st.markdown("""
<style>
/* 隐藏默认元素 */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* 全局背景 */
.stApp {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
}

/* 标题样式 */
.main-title {
    color: white;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    margin-bottom: 20px;
}

.current-model {
    color: white;
    text-align: center;
    font-size: 16px;
    margin-bottom: 20px;
}

/* 卡片样式 */
.info-card {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
}

/* 特征组标题 */
.feature-title {
    color: white;
    text-align: center;
    font-weight: bold;
    padding: 8px;
    border-radius: 5px;
    margin-bottom: 10px;
}

.proximate { background: #28a745; }
.ultimate { background: #6f42c1; }
.pyrolysis { background: #fd7e14; }

/* 窗口控制按钮 */
.window-controls {
    position: fixed;
    top: 10px;
    right: 20px;
    display: flex;
    gap: 8px;
    z-index: 1000;
}

.control-dot {
    width: 15px;
    height: 15px;
    border-radius: 50%;
}

.dot-red { background: #ff5f57; }
.dot-yellow { background: #ffbd2e; }
.dot-green { background: #28ca42; }
</style>
""", unsafe_allow_html=True)

# 窗口控制按钮
st.markdown("""
<div class="window-controls">
    <div class="control-dot dot-red"></div>
    <div class="control-dot dot-yellow"></div>
    <div class="control-dot dot-green"></div>
</div>
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

# 主布局：左侧边栏 + 中央区域 + 右侧面板
left_col, center_col, right_col = st.columns([1, 3, 1])

# 左侧边栏
with left_col:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("### 👤 用户: wy1122")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 菜单
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    if st.button("预测模型", use_container_width=True, type="primary"):
        pass
    if st.button("执行日志", use_container_width=True):
        pass
    if st.button("模型信息", use_container_width=True):
        pass
    if st.button("技术说明", use_container_width=True):
        pass
    if st.button("使用指南", use_container_width=True):
        pass
    st.markdown('</div>', unsafe_allow_html=True)

# 中央区域
with center_col:
    # 标题
    st.markdown('<div class="main-title">选择预测目标</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="current-model">当前模型: {st.session_state.selected_model}</div>', unsafe_allow_html=True)
    
    # 模型选择按钮
    model_col1, model_col2, model_col3 = st.columns(3)
    
    with model_col1:
        char_type = "primary" if st.session_state.selected_model == "Char Yield" else "secondary"
        if st.button("🔥 Char Yield", key="char", use_container_width=True, type=char_type):
            st.session_state.selected_model = "Char Yield"
            st.session_state.prediction_result = 27.7937
            st.rerun()
    
    with model_col2:
        oil_type = "primary" if st.session_state.selected_model == "Oil Yield" else "secondary"
        if st.button("🛢️ Oil Yield", key="oil", use_container_width=True, type=oil_type):
            st.session_state.selected_model = "Oil Yield"
            st.session_state.prediction_result = 45.2156
            st.rerun()
    
    with model_col3:
        gas_type = "primary" if st.session_state.selected_model == "Gas Yield" else "secondary"
        if st.button("💨 Gas Yield", key="gas", use_container_width=True, type=gas_type):
            st.session_state.selected_model = "Gas Yield"
            st.session_state.prediction_result = 27.0007
            st.rerun()
    
    st.markdown("---")
    
    # 特征输入区域
    feature_col1, feature_col2, feature_col3 = st.columns(3)
    
    # Proximate Analysis
    with feature_col1:
        st.markdown('<div class="feature-title proximate">Proximate Analysis</div>', unsafe_allow_html=True)
        
        with st.container():
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
    with feature_col2:
        st.markdown('<div class="feature-title ultimate">Ultimate Analysis</div>', unsafe_allow_html=True)
        
        with st.container():
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
    with feature_col3:
        st.markdown('<div class="feature-title pyrolysis">Pyrolysis Conditions</div>', unsafe_allow_html=True)
        
        with st.container():
            ft_value = st.number_input("FT(°C)", 
                                      value=st.session_state.feature_values["FT(°C)"], 
                                      step=1.0, format="%.1f")
            
            hr_value = st.number_input("HR(°C/min)", 
                                      value=st.session_state.feature_values["HR(°C/min)"], 
                                      step=0.1, format="%.1f")
            
            fr_value = st.number_input("FR(mL/min)", 
                                      value=st.session_state.feature_values["FR(mL/min)"], 
                                      step=1.0, format="%.1f")
    
    st.markdown("---")
    
    # 操作按钮
    btn_col1, btn_col2 = st.columns(2)
    
    with btn_col1:
        if st.button("🔮 运行预测", use_container_width=True, type="primary"):
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
            
            # 模拟预测
            if st.session_state.selected_model == "Char Yield":
                st.session_state.prediction_result = 27.7937
            elif st.session_state.selected_model == "Oil Yield":
                st.session_state.prediction_result = 45.2156
            else:
                st.session_state.prediction_result = 27.0007
            
            st.success(f"预测完成！{st.session_state.selected_model}: {st.session_state.prediction_result:.4f} wt%")
            st.rerun()
    
    with btn_col2:
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

# 右侧面板
with right_col:
    # 预测结果
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("### 预测结果")
    st.markdown(f"**{st.session_state.selected_model}**: {st.session_state.prediction_result:.2f} wt%")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 预测信息
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("### 预测信息")
    st.write(f"• 目标变量: {st.session_state.selected_model}")
    st.write(f"• 预测结果: {st.session_state.prediction_result:.4f} wt%")
    st.write("• 模型类型: GBDT Pipeline")
    st.write("• 预处理: RobustScaler")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 模型状态
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("### 模型状态")
    st.write("• 🟢 加载状态: 正常")
    st.write("• 特征数量: 9")
    st.write("• 警告数量: 0")
    st.markdown('</div>', unsafe_allow_html=True)

# 页脚
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; font-size: 12px;'>
© 2024 生物质纳米材料与智能装备实验室 | 基于GBDT的生物质热解产物预测系统
</div>
""", unsafe_allow_html=True)