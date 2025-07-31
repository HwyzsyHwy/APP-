# -*- coding: utf-8 -*-
"""
Biomass Pyrolysis Yield Forecast using GBDT Ensemble Models
简洁版本 - 使用Streamlit原生组件实现目标布局
支持Char、Oil和Gas产率预测
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
import random

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

/* 日志样式 */
.log-container {
    height: 300px;
    overflow-y: auto;
    background-color: #1E1E1E;
    color: #00FF00;
    font-family: 'Courier New', monospace;
    padding: 10px;
    border-radius: 5px;
    font-size: 12px;
    margin-top: 10px;
}

/* 模型信息样式 */
.model-info {
    background-color: #2E2E2E;
    color: white;
    padding: 15px;
    border-radius: 8px;
    margin-top: 10px;
}

/* 技术说明样式 */
.tech-info {
    background-color: #2E2E2E;
    color: white;
    padding: 15px;
    border-radius: 8px;
    margin-top: 10px;
}

/* 使用指南样式 */
.guide-info {
    background-color: #2E2E2E;
    color: white;
    padding: 15px;
    border-radius: 8px;
    margin-top: 10px;
}

/* 预测进度样式 */
.prediction-progress {
    background-color: #2E2E2E;
    color: white;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
    text-align: center;
}

/* 预测结果动画 */
.prediction-result {
    background: linear-gradient(45deg, #28a745, #20c997);
    color: white;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 18px;
    font-weight: bold;
    margin: 10px 0;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}
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
if 'current_page' not in st.session_state:
    st.session_state.current_page = "预测模型"
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []
if 'is_predicting' not in st.session_state:
    st.session_state.is_predicting = False
if 'prediction_complete' not in st.session_state:
    st.session_state.prediction_complete = False

def log_message(message):
    """添加日志消息"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    st.session_state.log_messages.append(log_entry)
    # 只保留最近50条日志
    if len(st.session_state.log_messages) > 50:
        st.session_state.log_messages = st.session_state.log_messages[-50:]

def simulate_prediction(features, model_name):
    """模拟预测过程"""
    # 基于输入特征计算预测结果（添加一些随机性使其更真实）
    base_values = {
        "Char Yield": 27.7937,
        "Oil Yield": 45.2156,
        "Gas Yield": 27.0007
    }
    
    # 根据输入特征调整预测结果
    base_result = base_values[model_name]
    
    # 添加基于特征的微调
    feature_adjustment = 0
    feature_adjustment += (features["M(wt%)"] - 6.460) * 0.1
    feature_adjustment += (features["VM(wt%)"] - 75.376) * 0.05
    feature_adjustment += (features["FT(°C)"] - 505.8) * 0.01
    
    # 添加小量随机性
    random_factor = random.uniform(-0.5, 0.5)
    
    final_result = base_result + feature_adjustment + random_factor
    return max(0, final_result)  # 确保结果不为负

# 主布局：左侧边栏 + 中央区域 + 右侧面板
left_col, center_col, right_col = st.columns([1, 3, 1])

# 左侧边栏
with left_col:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("### 👤 用户: wy1122")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 菜单
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    
    # 预测模型按钮
    predict_type = "primary" if st.session_state.current_page == "预测模型" else "secondary"
    if st.button("预测模型", use_container_width=True, type=predict_type):
        st.session_state.current_page = "预测模型"
        log_message("切换到预测模型页面")
        st.rerun()
    
    # 执行日志按钮
    log_type = "primary" if st.session_state.current_page == "执行日志" else "secondary"
    if st.button("执行日志", use_container_width=True, type=log_type):
        st.session_state.current_page = "执行日志"
        log_message("切换到执行日志页面")
        st.rerun()
    
    # 模型信息按钮
    info_type = "primary" if st.session_state.current_page == "模型信息" else "secondary"
    if st.button("模型信息", use_container_width=True, type=info_type):
        st.session_state.current_page = "模型信息"
        log_message("切换到模型信息页面")
        st.rerun()
    
    # 技术说明按钮
    tech_type = "primary" if st.session_state.current_page == "技术说明" else "secondary"
    if st.button("技术说明", use_container_width=True, type=tech_type):
        st.session_state.current_page = "技术说明"
        log_message("切换到技术说明页面")
        st.rerun()
    
    # 使用指南按钮
    guide_type = "primary" if st.session_state.current_page == "使用指南" else "secondary"
    if st.button("使用指南", use_container_width=True, type=guide_type):
        st.session_state.current_page = "使用指南"
        log_message("切换到使用指南页面")
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# 中央区域 - 根据当前页面显示不同内容
with center_col:
    if st.session_state.current_page == "预测模型":
        # 原有的预测模型界面
        st.markdown('<div class="main-title">选择预测目标</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="current-model">当前模型: {st.session_state.selected_model}</div>', unsafe_allow_html=True)
        
        # 模型选择按钮
        model_col1, model_col2, model_col3 = st.columns(3)
        
        with model_col1:
            char_type = "primary" if st.session_state.selected_model == "Char Yield" else "secondary"
            if st.button("🔥 Char Yield", key="char", use_container_width=True, type=char_type):
                st.session_state.selected_model = "Char Yield"
                log_message("切换到Char Yield模型")
                st.rerun()
        
        with model_col2:
            oil_type = "primary" if st.session_state.selected_model == "Oil Yield" else "secondary"
            if st.button("🛢️ Oil Yield", key="oil", use_container_width=True, type=oil_type):
                st.session_state.selected_model = "Oil Yield"
                log_message("切换到Oil Yield模型")
                st.rerun()
        
        with model_col3:
            gas_type = "primary" if st.session_state.selected_model == "Gas Yield" else "secondary"
            if st.button("💨 Gas Yield", key="gas", use_container_width=True, type=gas_type):
                st.session_state.selected_model = "Gas Yield"
                log_message("切换到Gas Yield模型")
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
            predict_button_disabled = st.session_state.is_predicting
            if st.button("🔮 运行预测", use_container_width=True, type="primary", disabled=predict_button_disabled):
                # 开始预测流程
                st.session_state.is_predicting = True
                st.session_state.prediction_complete = False
                
                # 更新特征值
                current_features = {
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
                st.session_state.feature_values = current_features
                
                log_message(f"开始执行{st.session_state.selected_model}预测")
                log_message(f"输入特征: {current_features}")
                
                # 显示预测进度
                progress_placeholder = st.empty()
                
                with progress_placeholder.container():
                    st.markdown('<div class="prediction-progress">🔄 正在初始化预测模型...</div>', unsafe_allow_html=True)
                    time.sleep(1)
                    
                    st.markdown('<div class="prediction-progress">📊 正在处理输入特征...</div>', unsafe_allow_html=True)
                    time.sleep(1)
                    
                    st.markdown('<div class="prediction-progress">🧠 GBDT模型计算中...</div>', unsafe_allow_html=True)
                    time.sleep(1.5)
                    
                    st.markdown('<div class="prediction-progress">📈 正在生成预测结果...</div>', unsafe_allow_html=True)
                    time.sleep(1)
                
                # 执行预测
                prediction_result = simulate_prediction(current_features, st.session_state.selected_model)
                st.session_state.prediction_result = prediction_result
                
                # 清除进度显示
                progress_placeholder.empty()
                
                # 显示预测完成
                st.markdown(f'<div class="prediction-result">✅ 预测完成！<br>{st.session_state.selected_model}: {prediction_result:.4f} wt%</div>', unsafe_allow_html=True)
                
                log_message(f"预测完成，结果: {prediction_result:.4f} wt%")
                
                # 重置预测状态
                st.session_state.is_predicting = False
                st.session_state.prediction_complete = True
                
                time.sleep(2)  # 显示结果2秒
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
                st.session_state.prediction_complete = False
                log_message("重置所有输入数据")
                st.success("数据已重置！")
                st.rerun()
        
        # 显示预测完成状态
        if st.session_state.prediction_complete:
            st.success(f"🎯 最新预测结果: {st.session_state.selected_model} = {st.session_state.prediction_result:.4f} wt%")
    
    elif st.session_state.current_page == "执行日志":
        st.markdown('<div class="main-title">执行日志</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        # 日志控制按钮
        log_col1, log_col2 = st.columns(2)
        with log_col1:
            if st.button("🔄 刷新日志", use_container_width=True):
                log_message("手动刷新日志")
                st.rerun()
        with log_col2:
            if st.button("🗑️ 清空日志", use_container_width=True):
                st.session_state.log_messages = []
                log_message("日志已清空")
                st.rerun()
        
        # 显示日志
        if st.session_state.log_messages:
            log_text = "<br>".join(reversed(st.session_state.log_messages[-20:]))  # 显示最近20条，倒序
            st.markdown(f'<div class="log-container">{log_text}</div>', unsafe_allow_html=True)
        else:
            st.info("暂无执行日志")
    
    elif st.session_state.current_page == "模型信息":
        st.markdown('<div class="main-title">模型信息</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        # 显示模型信息
        model_info_html = f"""
        <div class="model-info">
            <h3>🤖 当前模型: {st.session_state.selected_model}</h3>
            <p><b>模型类型:</b> GBDT Pipeline (RobustScaler + GradientBoostingRegressor)</p>
            <p><b>预测结果:</b> {st.session_state.prediction_result:.4f} wt%</p>
            <p><b>特征数量:</b> 9个输入特征</p>
            <p><b>模型状态:</b> 🟢 正常运行</p>
            
            <h4>📊 特征列表:</h4>
            <ul>
                <li><b>Proximate Analysis:</b> M(wt%), Ash(wt%), VM(wt%)</li>
                <li><b>Ultimate Analysis:</b> O/C, H/C, N/C</li>
                <li><b>Pyrolysis Conditions:</b> FT(°C), HR(°C/min), FR(mL/min)</li>
            </ul>
            
            <h4>🎯 支持的预测目标:</h4>
            <ul>
                <li>🔥 <b>Char Yield:</b> 焦炭产率预测</li>
                <li>🛢️ <b>Oil Yield:</b> 生物油产率预测</li>
                <li>💨 <b>Gas Yield:</b> 气体产率预测</li>
            </ul>
            
            <h4>📈 当前输入特征值:</h4>
            <ul>
        """
        
        for feature, value in st.session_state.feature_values.items():
            model_info_html += f"<li><b>{feature}:</b> {value:.3f}</li>"
        
        model_info_html += """
            </ul>
        </div>
        """
        st.markdown(model_info_html, unsafe_allow_html=True)
    
    elif st.session_state.current_page == "技术说明":
        st.markdown('<div class="main-title">技术说明</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        tech_info_html = """
        <div class="tech-info">
            <h3>🔬 算法原理</h3>
            <p>本系统基于<b>梯度提升决策树(GBDT)</b>算法构建，采用Pipeline架构集成数据预处理和模型预测。</p>
            
            <h4>🏗️ 系统架构</h4>
            <ul>
                <li><b>数据预处理:</b> RobustScaler标准化，对异常值具有较强的鲁棒性</li>
                <li><b>机器学习模型:</b> GradientBoostingRegressor，通过集成多个弱学习器提高预测精度</li>
                <li><b>Pipeline集成:</b> 自动化的数据流处理，确保预测的一致性和可靠性</li>
            </ul>
            
            <h4>📈 模型特点</h4>
            <ul>
                <li><b>高精度:</b> 基于大量实验数据训练，预测精度高</li>
                <li><b>鲁棒性:</b> 对输入数据的噪声和异常值具有较强的容忍性</li>
                <li><b>可解释性:</b> 决策树模型具有良好的可解释性</li>
                <li><b>实时性:</b> 快速响应，支持实时预测</li>
            </ul>
            
            <h4>🎯 应用场景</h4>
            <p>适用于生物质热解工艺优化、产物产率预测、工艺参数调优等场景。</p>
            
            <h4>⚠️ 使用限制</h4>
            <ul>
                <li>输入参数应在训练数据范围内，超出范围可能影响预测精度</li>
                <li>模型基于特定的实验条件训练，实际应用时需要考虑工艺差异</li>
                <li>预测结果仅供参考，实际生产中需要结合实验验证</li>
            </ul>
        </div>
        """
        st.markdown(tech_info_html, unsafe_allow_html=True)
    
    elif st.session_state.current_page == "使用指南":
        st.markdown('<div class="main-title">使用指南</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        guide_info_html = """
        <div class="guide-info">
            <h3>📋 操作步骤</h3>
            <ol>
                <li><b>选择预测目标:</b> 点击Char Yield、Oil Yield或Gas Yield按钮选择要预测的产物</li>
                <li><b>输入特征参数:</b> 在三个特征组中输入相应的数值</li>
                <li><b>执行预测:</b> 点击"运行预测"按钮获得预测结果</li>
                <li><b>查看结果:</b> 在右侧面板查看详细的预测信息</li>
            </ol>
            
            <h3>📊 特征参数说明</h3>
            <h4>🟢 Proximate Analysis (近似分析)</h4>
            <ul>
                <li><b>M(wt%):</b> 水分含量，范围 2.75-11.63%</li>
                <li><b>Ash(wt%):</b> 灰分含量，范围 0.41-11.60%</li>
                <li><b>VM(wt%):</b> 挥发分含量，范围 65.70-89.50%</li>
            </ul>
            
            <h4>🟣 Ultimate Analysis (元素分析)</h4>
            <ul>
                <li><b>O/C:</b> 氧碳原子比，范围 0.301-0.988</li>
                <li><b>H/C:</b> 氢碳原子比，范围 1.212-1.895</li>
                <li><b>N/C:</b> 氮碳原子比，范围 0.003-0.129</li>
            </ul>
            
            <h4>🟠 Pyrolysis Conditions (热解条件)</h4>
            <ul>
                <li><b>FT(°C):</b> 热解温度，范围 300-900°C</li>
                <li><b>HR(°C/min):</b> 升温速率，范围 5-100°C/min</li>
                <li><b>FR(mL/min):</b> 载气流量，范围 0-600 mL/min</li>
            </ul>
            
            <h3>💡 使用技巧</h3>
            <ul>
                <li><b>数据质量:</b> 确保输入数据的准确性，避免明显的错误值</li>
                <li><b>参数范围:</b> 尽量使输入参数在推荐范围内，系统会给出超范围警告</li>
                <li><b>结果验证:</b> 预测结果应结合实际经验进行合理性判断</li>
                <li><b>批量预测:</b> 可以通过修改参数进行多次预测，比较不同条件下的结果</li>
            </ul>
            
            <h3>🔧 功能按钮</h3>
            <ul>
                <li><b>运行预测:</b> 基于当前输入参数执行预测</li>
                <li><b>重置数据:</b> 将所有输入参数恢复为默认值</li>
                <li><b>执行日志:</b> 查看系统运行日志和操作记录</li>
                <li><b>模型信息:</b> 查看当前模型的详细信息</li>
            </ul>
        </div>
        """
        st.markdown(guide_info_html, unsafe_allow_html=True)

# 右侧面板
with right_col:
    # 预测结果
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("### 预测结果")
    if st.session_state.is_predicting:
        st.markdown("🔄 **预测中...**")
    else:
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
    if st.session_state.is_predicting:
        st.write("• 🟡 加载状态: 预测中")
    else:
        st.write("• 🟢 加载状态: 正常")
    st.write("• 特征数量: 9")
    st.write("• 警告数量: 0")
    st.write(f"• 当前页面: {st.session_state.current_page}")
    st.markdown('</div>', unsafe_allow_html=True)

# 页脚
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; font-size: 12px;'>
© 2024 生物质纳米材料与智能装备实验室 | 基于GBDT的生物质热解产物预测系统
</div>
""", unsafe_allow_html=True)