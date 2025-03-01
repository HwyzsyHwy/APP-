# -*- coding: utf-8 -*-
"""
Biomass Pyrolysis Yield Forecast
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_squared_error

# 页面设置
st.set_page_config(
    page_title='Biomass Pyrolysis Yield Forecast',
    page_icon='📊',
    layout='wide'
)

# 自定义样式 - 根据图片样式修改
st.markdown(
    """
    <style>
    /* 整体背景设置为深色 */
    .stApp {
        background-color: #0e1117;
    }
    
    /* 主标题样式 */
    .main-title {
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        color: white;
        padding: 10px 0;
        background-color: #1e1e1e;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    
    /* 三个分析部分的样式 */
    .ultimate-section {
        background-color: #c9a21f;
        padding: 5px;
        border-radius: 5px;
        margin-bottom: 10px;
        color: black;
        font-weight: bold;
        text-align: center;
    }
    
    .proximate-section {
        background-color: #4caf50;
        padding: 5px;
        border-radius: 5px;
        margin-bottom: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
    }
    
    .structural-section {
        background-color: #b71c1c;
        padding: 5px;
        border-radius: 5px;
        margin-bottom: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
    }
    
    /* 热解条件部分样式 */
    .pyrolysis-section {
        background-color: #ff7043;
        padding: 5px;
        border-radius: 5px;
        margin-bottom: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
    }
    
    /* 参数值样式 */
    .param-value {
        background-color: #293241;
        padding: 5px 10px;
        border-radius: 5px;
        color: white;
        font-weight: bold;
        margin: 5px 0;
        display: flex;
        justify-content: space-between;
    }
    
    /* 结果显示样式 */
    .result-box {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .yield-label {
        font-size: 22px;
        font-weight: bold;
        color: white;
        margin-right: 15px;
    }
    
    .yield-value {
        background-color: #1e1e1e;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 28px;
        font-weight: bold;
        color: white;
    }
    
    /* 按钮样式 */
    .stButton > button {
        background-color: #e53935;
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        width: 100%;
    }
    
    .clear-button > button {
        background-color: #1e88e5;
        color: white;
    }
    
    /* 滑动条样式调整 */
    .stSlider > div > div {
        background-color: rgba(255, 255, 255, 0.2);
    }
    
    /* 隐藏一些默认Streamlit元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# 主标题
st.markdown("<div class='main-title'>GUI for Bio-Char Yield Prediction based on ELT-PSO Model</div>", unsafe_allow_html=True)

# 隐藏模型选择部分，放在边栏或者更不显眼的位置
with st.sidebar:
    st.header("Model Selection")
    model_name = st.selectbox(
        "Available Models", ["GBDT-Char", "GBDT-Oil", "GBDT-Gas"]
    )
    st.write(f"Current selected model: **{model_name}**")

# 加载模型和Scaler
MODEL_PATHS = {
    "GBDT-Char": "GBDT-Char-1.15.joblib",
    "GBDT-Oil": "GBDT-Oil-1.15.joblib",
    "GBDT-Gas": "GBDT-Gas-1.15.joblib"
}
SCALER_PATHS = {
    "GBDT-Char": "scaler-Char-1.15.joblib",
    "GBDT-Oil": "scaler-Oil-1.15.joblib",
    "GBDT-Gas": "scaler-Gas-1.15.joblib"
}

# 加载函数
def load_model(model_name):
    return joblib.load(MODEL_PATHS[model_name])

def load_scaler(model_name):
    return joblib.load(SCALER_PATHS[model_name])

# 特征分类 - 根据图片调整
feature_categories = {
    "Ultimate Analysis": ["C(wt%)", "H(wt%)", "N(wt%)", "O(wt%)"],
    "Proximate Analysis": ["VM(wt%)", "FC(wt%)", "Ash(wt%)", "M(wt%)"],
    "Pyrolysis Conditions": ["FT(℃)", "HR(℃/min)", "PS(mm)", "FR(mL/min)"]
}

# 创建特征输入界面
features = {}

# 创建三列布局
col1, col2, col3 = st.columns(3)

# 第一列: Ultimate Analysis
with col1:
    st.markdown("<div class='ultimate-section'>Ultimate Analysis</div>", unsafe_allow_html=True)
    for feature in feature_categories["Ultimate Analysis"]:
        st.markdown(f"<div class='param-value'><span>{feature}</span><span id='{feature}'></span></div>", unsafe_allow_html=True)
        if feature == "C(wt%)":
            features[feature] = st.slider(feature, min_value=30.0, max_value=110.0, value=52.05, label_visibility="collapsed")
        elif feature == "H(wt%)":
            features[feature] = st.slider(feature, min_value=0.0, max_value=15.0, value=5.37, label_visibility="collapsed")
        elif feature == "N(wt%)":
            features[feature] = st.slider(feature, min_value=0.0, max_value=5.0, value=0.49, label_visibility="collapsed")
        elif feature == "O(wt%)":
            features[feature] = st.slider(feature, min_value=30.0, max_value=60.0, value=42.1, label_visibility="collapsed")
        
        # 更新显示的值
        st.markdown(
            f"""
            <script>
                document.getElementById('{feature}').textContent = '{features[feature]:.2f}';
            </script>
            """,
            unsafe_allow_html=True
        )

# 第二列: Proximate Analysis
with col2:
    st.markdown("<div class='proximate-section'>Proximate Analysis</div>", unsafe_allow_html=True)
    for feature in feature_categories["Proximate Analysis"]:
        st.markdown(f"<div class='param-value'><span>{feature}</span><span id='{feature}'></span></div>", unsafe_allow_html=True)
        if feature == "VM(wt%)":
            features[feature] = st.slider(feature, min_value=0.0, max_value=110.0, value=73.5, label_visibility="collapsed")
        elif feature == "FC(wt%)":
            features[feature] = st.slider(feature, min_value=0.0, max_value=120.0, value=13.2, label_visibility="collapsed")
        elif feature == "Ash(wt%)":
            features[feature] = st.slider(feature, min_value=0.0, max_value=25.0, value=8.6, label_visibility="collapsed")
        elif feature == "M(wt%)":
            features[feature] = st.slider(feature, min_value=0.0, max_value=20.0, value=4.7, label_visibility="collapsed")

# 第三列: 添加一个结构分析部分和热解条件
with col3:
    st.markdown("<div class='structural-section'>Structural Analysis</div>", unsafe_allow_html=True)
    # 这部分在原代码中没有，但图片中有，我们可以设置为静态值，不用于计算
    st.markdown("<div class='param-value'><span>Lignin (%)</span><span>44</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='param-value'><span>Cellulose (%)</span><span>27.7</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='param-value'><span>HemiCellulose (%)</span><span>21.6</span></div>", unsafe_allow_html=True)
    
    # 间隔
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 热解条件部分
    st.markdown("<div class='pyrolysis-section'>Pyrolysis Condition</div>", unsafe_allow_html=True)
    for feature in feature_categories["Pyrolysis Conditions"]:
        st.markdown(f"<div class='param-value'><span>{feature}</span><span id='{feature}'></span></div>", unsafe_allow_html=True)
        if feature == "FT(℃)":
            features[feature] = st.slider(feature, min_value=250.0, max_value=1100.0, value=500.0, label_visibility="collapsed")
        elif feature == "HR(℃/min)":
            features[feature] = st.slider(feature, min_value=0.0, max_value=200.0, value=10.0, label_visibility="collapsed")
        elif feature == "PS(mm)":
            features[feature] = st.slider(feature, min_value=0.0, max_value=20.0, value=1.5, label_visibility="collapsed")
        elif feature == "FR(mL/min)":
            features[feature] = st.slider(feature, min_value=0.0, max_value=120.0, value=2.0, label_visibility="collapsed")
    
    # 增加RT和SM但不显示在热解条件中
    features["RT(min)"] = 30.0
    features["SM(g)"] = 75.0

# 转换为DataFrame
input_data = pd.DataFrame([features])

# 预测结果部分
st.markdown("<div class='result-box'><span class='yield-label'>Biochar Yield (%)</span><span class='yield-value' id='yield-value'>--</span></div>", unsafe_allow_html=True)

# 按钮行
col1, col2 = st.columns(2)

# 预测按钮和清除按钮
with col1:
    predict_button = st.button("PUSH")
with col2:
    st.markdown("<div class='clear-button'>", unsafe_allow_html=True)
    clear_button = st.button("CLEAR")
    st.markdown("</div>", unsafe_allow_html=True)

if predict_button:
    try:
        # 加载所选模型和Scaler
        model = load_model(model_name)
        scaler = load_scaler(model_name)

        # 数据标准化
        input_data_scaled = scaler.transform(input_data)

        # 预测
        y_pred = model.predict(input_data_scaled)[0]

        # 显示预测结果
        st.markdown(
            f"""
            <script>
                document.getElementById('yield-value').textContent = '{y_pred:.2f}';
            </script>
            """,
            unsafe_allow_html=True
        )
        
        # 为了确保结果显示，也使用Streamlit的方式设置
        st.markdown(f"<div style='text-align: center; font-size: 24px; color: white;'>Predicted Yield: {y_pred:.2f}%</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"预测过程中出现错误: {e}")

if clear_button:
    # 重置不会真正工作，因为Streamlit的状态管理机制，但保留这个按钮以匹配UI
    st.markdown(
        """
        <script>
            document.getElementById('yield-value').textContent = '--';
        </script>
        """,
        unsafe_allow_html=True
    )
    st.experimental_rerun()