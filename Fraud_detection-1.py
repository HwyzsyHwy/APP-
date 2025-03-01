import streamlit as st
import pickle
import numpy as np
import pandas as pd

# 页面配置
st.set_page_config(page_title="生物质热解产率预测器", layout="wide")

# 自定义CSS样式
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .proximateAnalysis {
        background-color: green;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        color: white;
    }
    .ultimateAnalysis {
        background-color: yellow;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        color: black;
    }
    .pyrolysisConditions {
        background-color: orange;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        color: black;
    }
    .input-label {
        font-weight: bold;
        display: inline-block;
        width: 120px;
    }
    .stNumberInput {
        margin-top: -15px;
    }
    .button-container {
        display: flex;
        justify-content: space-between;
        margin-top: 20px;
    }
    .result-container {
        background-color: #4CAF50;
        color: white;
        padding: 15px;
        border-radius: 5px;
        margin-top: 20px;
        text-align: center;
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)

# 初始化session_state
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

# 模型选择选项
models = ["随机森林", "XGBoost", "支持向量机", "神经网络"]

# 加载模型和数据处理器函数
def load_model(model_name):
    model_path = f"{model_name}_model.pkl"
    try:
        # 这里只是示例，实际应用请替换为真实的模型加载逻辑
        model = None
        return model
    except:
        st.error(f"无法加载模型：{model_path}")
        return None

def load_scaler():
    scaler_path = "scaler.pkl"
    try:
        # 这里只是示例，实际应用请替换为真实的数据处理器加载逻辑
        scaler = None
        return scaler
    except:
        st.error(f"无法加载数据处理器：{scaler_path}")
        return None

# 标题和描述
st.title("生物质热解产率预测器")
st.write("请输入以下参数来预测生物质热解产率：")

# 创建3列布局
col1, col2, col3 = st.columns(3)

# 默认值
default_values = {
    "M": 5.0, "Ash": 8.0, "VM": 75.0, "FC": 15.0,
    "C": 45.0, "H": 6.0, "O": 45.0, "N": 0.5, "S": 0.1,
    "T": 500, "HR": 10, "HT": 30
}

# 第一列：近似分析（Proximate Analysis）
with col1:
    st.markdown('<div class="proximateAnalysis"><h3>近似分析</h3></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="proximateAnalysis">', unsafe_allow_html=True)
    st.markdown('<span class="input-label">M(wt%):</span>', unsafe_allow_html=True)
    M = st.number_input("", value=default_values["M"], key="M", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="proximateAnalysis">', unsafe_allow_html=True)
    st.markdown('<span class="input-label">Ash(wt%):</span>', unsafe_allow_html=True)
    Ash = st.number_input("", value=default_values["Ash"], key="Ash", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="proximateAnalysis">', unsafe_allow_html=True)
    st.markdown('<span class="input-label">VM(wt%):</span>', unsafe_allow_html=True)
    VM = st.number_input("", value=default_values["VM"], key="VM", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="proximateAnalysis">', unsafe_allow_html=True)
    st.markdown('<span class="input-label">FC(wt%):</span>', unsafe_allow_html=True)
    FC = st.number_input("", value=default_values["FC"], key="FC", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

# 第二列：元素分析（Ultimate Analysis）
with col2:
    st.markdown('<div class="ultimateAnalysis"><h3>元素分析</h3></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="ultimateAnalysis">', unsafe_allow_html=True)
    st.markdown('<span class="input-label">C(wt%):</span>', unsafe_allow_html=True)
    C = st.number_input("", value=default_values["C"], key="C", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="ultimateAnalysis">', unsafe_allow_html=True)
    st.markdown('<span class="input-label">H(wt%):</span>', unsafe_allow_html=True)
    H = st.number_input("", value=default_values["H"], key="H", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="ultimateAnalysis">', unsafe_allow_html=True)
    st.markdown('<span class="input-label">O(wt%):</span>', unsafe_allow_html=True)
    O = st.number_input("", value=default_values["O"], key="O", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="ultimateAnalysis">', unsafe_allow_html=True)
    st.markdown('<span class="input-label">N(wt%):</span>', unsafe_allow_html=True)
    N = st.number_input("", value=default_values["N"], key="N", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="ultimateAnalysis">', unsafe_allow_html=True)
    st.markdown('<span class="input-label">S(wt%):</span>', unsafe_allow_html=True)
    S = st.number_input("", value=default_values["S"], key="S", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

# 第三列：热解条件（Pyrolysis Conditions）
with col3:
    st.markdown('<div class="pyrolysisConditions"><h3>热解条件</h3></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="pyrolysisConditions">', unsafe_allow_html=True)
    st.markdown('<span class="input-label">T(°C):</span>', unsafe_allow_html=True)
    T = st.number_input("", value=default_values["T"], key="T", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="pyrolysisConditions">', unsafe_allow_html=True)
    st.markdown('<span class="input-label">HR(°C/min):</span>', unsafe_allow_html=True)
    HR = st.number_input("", value=default_values["HR"], key="HR", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="pyrolysisConditions">', unsafe_allow_html=True)
    st.markdown('<span class="input-label">HT(min):</span>', unsafe_allow_html=True)
    HT = st.number_input("", value=default_values["HT"], key="HT", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 模型选择
    st.markdown('<div class="pyrolysisConditions">', unsafe_allow_html=True)
    st.markdown('<span class="input-label">模型:</span>', unsafe_allow_html=True)
    model_selection = st.selectbox("", models, index=0, key="model_selection", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

# 按钮容器
st.markdown('<div class="button-container">', unsafe_allow_html=True)
predict_button = st.button("预测", key="predict")
clear_button = st.button("清除", key="clear")
st.markdown('</div>', unsafe_allow_html=True)

# 预测逻辑
if predict_button:
    try:
        # 这里仅作为示例，实际应用应该使用真实的预测逻辑
        # 创建特征数组
        features = np.array([[M, Ash, VM, FC, C, H, O, N, S, T, HR, HT]])
        
        # 这里假设我们在进行模拟预测
        prediction_value = (M + Ash + VM + FC + C + H + O + N + S + T + HR + HT) / 100  # 模拟计算
        st.session_state.prediction = prediction_value * 10  # 假设的预测结果
        
    except Exception as e:
        st.error(f"预测时发生错误: {e}")

# 清除逻辑
if clear_button:
    for key in default_values:
        st.session_state[key] = default_values[key]
    st.session_state.prediction = None
    st.rerun()

# 显示预测结果
if st.session_state.prediction is not None:
    st.markdown(
        f'<div class="result-container">预测产率 (%): {st.session_state.prediction:.2f}</div>',
        unsafe_allow_html=True
    )