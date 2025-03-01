import streamlit as st
import pickle
import numpy as np
import pandas as pd

# 页面配置
st.set_page_config(page_title="生物质热解产率预测器", layout="wide")

# 更具针对性的CSS
st.markdown("""
<style>
/* 整体页面样式 */
.main {
    background-color: #0E1117;
    color: white;
}

/* 针对number input的输入框部分 */
input[type="number"] {
    background-color: green !important;
    color: white !important;
}

/* 针对number input的按钮部分 */
.stNumberInput button {
    background-color: green !important;
    color: white !important;
}

/* 针对整个number input容器 */
.stNumberInput div[data-baseweb="input"] {
    background-color: green !important;
}

.proximate {
    background-color: green;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
}

.ultimate {
    background-color: yellow;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
    color: black;
}

.pyrolysis {
    background-color: orange;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
    color: black;
}

.prediction {
    margin-top: 20px;
    padding: 20px;
    background-color: #262730;
    border-radius: 5px;
    text-align: center;
}

.btn-push {
    background-color: #4CAF50;
    color: white;
    padding: 10px 24px;
    margin: 10px 2px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}

.btn-clear {
    background-color: #f44336;
    color: white;
    padding: 10px 24px;
    margin: 10px 2px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

# 初始化会话状态
if 'predict_clicked' not in st.session_state:
    st.session_state.predict_clicked = False
if 'clear_pressed' not in st.session_state:
    st.session_state.clear_pressed = False

# 模型和定标器路径
model_options = {
    "Random Forest": "RF_model.pkl",
    "Support Vector Machine": "SVM_model.pkl",
    "K-Nearest Neighbors": "KNN_model.pkl"
}

# 定义加载模型和定标器函数
def load_model(model_name):
    try:
        with open(model_options[model_name], 'rb') as file:
            model = pickle.load(file)
        return model
    except:
        return None

def load_scaler():
    try:
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        return scaler
    except:
        return None

# 标题
st.title("生物质热解产率预测器")

# 创建三列布局
col1, col2, col3 = st.columns(3)

# 默认值
default_values = {
    "M": 5.0, "Ash": 8.0, "VM": 75.0, "FC": 15.0,
    "C": 45.0, "H": 6.0, "O": 40.0, "N": 0.5, "S": 0.1,
    "temp": 500, "heat_rate": 10, "residence_time": 20
}

# 第一列：Proximate Analysis
with col1:
    st.markdown('<div class="proximate">', unsafe_allow_html=True)
    st.markdown("<h3 style='color: white; text-align: center;'>Proximate Analysis</h3>", unsafe_allow_html=True)
    
    # 创建输入字段
    M = st.number_input("M(wt%)", min_value=0.0, max_value=100.0, value=default_values["M"], step=0.1)
    Ash = st.number_input("Ash(wt%)", min_value=0.0, max_value=100.0, value=default_values["Ash"], step=0.1)
    VM = st.number_input("VM(wt%)", min_value=0.0, max_value=100.0, value=default_values["VM"], step=0.1)
    FC = st.number_input("FC(wt%)", min_value=0.0, max_value=100.0, value=default_values["FC"], step=0.1)
    
    st.markdown('</div>', unsafe_allow_html=True)

# 第二列：Ultimate Analysis
with col2:
    st.markdown('<div class="ultimate">', unsafe_allow_html=True)
    st.markdown("<h3 style='color: black; text-align: center;'>Ultimate Analysis</h3>", unsafe_allow_html=True)
    
    C = st.number_input("C(wt%)", min_value=0.0, max_value=100.0, value=default_values["C"], step=0.1)
    H = st.number_input("H(wt%)", min_value=0.0, max_value=100.0, value=default_values["H"], step=0.1)
    O = st.number_input("O(wt%)", min_value=0.0, max_value=100.0, value=default_values["O"], step=0.1)
    N = st.number_input("N(wt%)", min_value=0.0, max_value=100.0, value=default_values["N"], step=0.1)
    S = st.number_input("S(wt%)", min_value=0.0, max_value=100.0, value=default_values["S"], step=0.1)
    
    st.markdown('</div>', unsafe_allow_html=True)

# 第三列：Pyrolysis Conditions
with col3:
    st.markdown('<div class="pyrolysis">', unsafe_allow_html=True)
    st.markdown("<h3 style='color: black; text-align: center;'>Pyrolysis Conditions</h3>", unsafe_allow_html=True)
    
    temp = st.number_input("Temperature(°C)", min_value=100, max_value=1000, value=default_values["temp"], step=10)
    heat_rate = st.number_input("Heating Rate(°C/min)", min_value=1, max_value=100, value=default_values["heat_rate"], step=1)
    residence_time = st.number_input("Residence Time(min)", min_value=1, max_value=120, value=default_values["residence_time"], step=1)
    
    st.markdown('</div>', unsafe_allow_html=True)

# 选择模型
selected_model = st.selectbox("选择模型", list(model_options.keys()))

# 按钮区域
col1, col2 = st.columns(2)

def on_predict_click():
    st.session_state.predict_clicked = True

def on_clear_click():
    st.session_state.clear_pressed = True
    # 重置所有输入值到默认值
    for key in default_values:
        st.session_state[key] = default_values[key]

with col1:
    predict_btn = st.button('PUSH', on_click=on_predict_click, key='predict_button', 
                          help="点击预测产率", use_container_width=True)

with col2:
    clear_btn = st.button('CLEAR', on_click=on_clear_click, key='clear_button', 
                        help="清除所有输入", use_container_width=True)

# 预测逻辑
if st.session_state.predict_clicked:
    st.session_state.predict_clicked = False  # 重置状态
    
    try:
        # 准备输入数据
        input_data = np.array([M, Ash, VM, FC, C, H, O, N, S, temp, heat_rate, residence_time]).reshape(1, -1)
        
        # 加载模型和定标器
        model = load_model(selected_model)
        scaler = load_scaler()
        
        if model is not None and scaler is not None:
            # 缩放数据
            scaled_data = scaler.transform(input_data)
            
            # 预测
            prediction = model.predict(scaled_data)[0]
            
            # 显示预测结果
            st.markdown('<div class="prediction">', unsafe_allow_html=True)
            st.markdown(f"<h2>Yield (%): {prediction:.2f}</h2>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("无法加载模型或定标器")
    except Exception as e:
        st.error(f"预测过程中出错: {str(e)}")

# 清除逻辑
if st.session_state.clear_pressed:
    st.session_state.clear_pressed = False  # 重置状态
    st.rerun()