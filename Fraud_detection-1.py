import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

# 设置页面基本配置
st.set_page_config(page_title="生物质炭产率预测系统", layout="wide")

# 使用CSS自定义样式
st.markdown("""
<style>
    body {
        background-color: #121212;
        color: white;
    }
    .main {
        background-color: #121212;
    }
    .css-1d391kg {
        background-color: #121212;
    }
    .stApp {
        background-color: #121212;
    }
    
    /* 为不同区域设置背景颜色 */
    .proximate-section {
        background-color: #32CD32;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 15px;
    }
    .ultimate-section {
        background-color: #DAA520;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 15px;
    }
    .pyrolysis-section {
        background-color: #FF7F50;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 15px;
    }
    
    /* 直接设置输入框背景色 */
    input[type="number"] {
        background-color: inherit !important;
        color: black !important;
        font-weight: bold !important;
    }
    
    /* 按钮样式 */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 24px;
        border-radius: 4px;
    }
    .clear-button>button {
        background-color: #f44336;
    }
    
    /* 紧凑布局 */
    .row-container {
        display: flex;
        align-items: center;
        margin-bottom: 5px;
    }
    .label {
        flex: 1;
        text-align: right;
        padding-right: 10px;
        font-weight: bold;
        color: black;
    }
    .input-field {
        flex: 1;
    }
</style>
""", unsafe_allow_html=True)

# 标题
st.title("生物质炭产率预测系统")

# 使用st.form来组织输入和按钮
with st.form(key="prediction_form"):
    
    # 创建三列布局
    col1, col2, col3 = st.columns(3)
    
    # 第一列：近分析
    with col1:
        st.markdown('<div class="proximate-section">', unsafe_allow_html=True)
        st.subheader("近分析")
        
        st.markdown('<div class="row-container">', unsafe_allow_html=True)
        st.markdown('<div class="label">M(wt%)</div>', unsafe_allow_html=True)
        m_val = st.number_input("", min_value=0.0, max_value=100.0, value=5.0, step=0.1, key="m_val", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="row-container">', unsafe_allow_html=True)
        st.markdown('<div class="label">Ash(wt%)</div>', unsafe_allow_html=True)
        ash_val = st.number_input("", min_value=0.0, max_value=100.0, value=8.0, step=0.1, key="ash_val", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="row-container">', unsafe_allow_html=True)
        st.markdown('<div class="label">VM(wt%)</div>', unsafe_allow_html=True)
        vm_val = st.number_input("", min_value=0.0, max_value=100.0, value=75.0, step=0.1, key="vm_val", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="row-container">', unsafe_allow_html=True)
        st.markdown('<div class="label">FC(wt%)</div>', unsafe_allow_html=True)
        fc_val = st.number_input("", min_value=0.0, max_value=100.0, value=15.0, step=0.1, key="fc_val", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 第二列：元素分析
    with col2:
        st.markdown('<div class="ultimate-section">', unsafe_allow_html=True)
        st.subheader("元素分析")
        
        st.markdown('<div class="row-container">', unsafe_allow_html=True)
        st.markdown('<div class="label">C(wt%)</div>', unsafe_allow_html=True)
        c_val = st.number_input("", min_value=0.0, max_value=100.0, value=51.0, step=0.1, key="c_val", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="row-container">', unsafe_allow_html=True)
        st.markdown('<div class="label">H(wt%)</div>', unsafe_allow_html=True)
        h_val = st.number_input("", min_value=0.0, max_value=100.0, value=5.7, step=0.1, key="h_val", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="row-container">', unsafe_allow_html=True)
        st.markdown('<div class="label">O(wt%)</div>', unsafe_allow_html=True)
        o_val = st.number_input("", min_value=0.0, max_value=100.0, value=42.6, step=0.1, key="o_val", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="row-container">', unsafe_allow_html=True)
        st.markdown('<div class="label">N(wt%)</div>', unsafe_allow_html=True)
        n_val = st.number_input("", min_value=0.0, max_value=100.0, value=0.6, step=0.1, key="n_val", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="row-container">', unsafe_allow_html=True)
        st.markdown('<div class="label">S(wt%)</div>', unsafe_allow_html=True)
        s_val = st.number_input("", min_value=0.0, max_value=100.0, value=0.1, step=0.1, key="s_val", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 第三列：热解条件
    with col3:
        st.markdown('<div class="pyrolysis-section">', unsafe_allow_html=True)
        st.subheader("热解条件")
        
        st.markdown('<div class="row-container">', unsafe_allow_html=True)
        st.markdown('<div class="label">Temp(℃)</div>', unsafe_allow_html=True)
        temp_val = st.number_input("", min_value=250, max_value=900, value=500, step=10, key="temp_val", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="row-container">', unsafe_allow_html=True)
        st.markdown('<div class="label">Pressure(MPa)</div>', unsafe_allow_html=True)
        press_val = st.number_input("", min_value=0.1, max_value=10.0, value=0.1, step=0.1, key="press_val", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="row-container">', unsafe_allow_html=True)
        st.markdown('<div class="label">Retention Time(min)</div>', unsafe_allow_html=True)
        time_val = st.number_input("", min_value=0, max_value=180, value=30, step=5, key="time_val", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="row-container">', unsafe_allow_html=True)
        st.markdown('<div class="label">Heating Rate(℃/min)</div>', unsafe_allow_html=True)
        rate_val = st.number_input("", min_value=1, max_value=100, value=10, step=1, key="rate_val", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="row-container">', unsafe_allow_html=True)
        st.markdown('<div class="label">Particle size(mm)</div>', unsafe_allow_html=True)
        size_val = st.number_input("", min_value=0.1, max_value=100.0, value=1.0, step=0.1, key="size_val", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 使用两个列来排列按钮
    col1, col2 = st.columns(2)
    with col1:
        predict_button = st.form_submit_button(label="PUSH")
    with col2:
        st.markdown('<div class="clear-button">', unsafe_allow_html=True)
        clear_button = st.form_submit_button(label="CLEAR")
        st.markdown('</div>', unsafe_allow_html=True)

# 模型预测逻辑
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# 当点击PUSH按钮时
if predict_button:
    try:
        # 准备输入数据
        input_data = np.array([[
            m_val, ash_val, vm_val, fc_val,
            c_val, h_val, o_val, n_val, s_val,
            temp_val, press_val, time_val, rate_val, size_val
        ]])
        
        # 加载模型（确保模型文件存在）
        try:
            model = pickle.load(open('gradient_boosting_model.pkl', 'rb'))
            # 进行预测
            prediction = model.predict(input_data)[0]
            st.session_state.prediction_result = prediction
        except FileNotFoundError:
            st.error("模型文件未找到，请确保'gradient_boosting_model.pkl'文件存在于当前目录中")
    except Exception as e:
        st.error(f"预测过程中发生错误: {str(e)}")

# 当点击CLEAR按钮时
if clear_button:
    # 清除session_state中的所有键值
    for key in list(st.session_state.keys()):
        if key != 'prediction_result':  # 保留预测结果
            del st.session_state[key]
    
    # 使用页面重新运行的方式重置所有输入
    try:
        st.rerun()  # 尝试使用新的st.rerun()
    except:
        # 如果st.rerun()不可用，保持静默并继续执行
        pass

# 显示预测结果
if st.session_state.prediction_result is not None:
    st.success(f"产率 (%): {st.session_state.prediction_result:.2f}")