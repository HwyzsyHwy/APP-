# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 11:02:43 2022

@author: Kevin Boss
"""
from PIL import Image
import warnings
import streamlit as st
import pandas as pd
import shap
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

plt.style.use('default')

st.set_page_config(
    page_title='Real-Time Fraud Detection',
    page_icon='🕵️‍♀️',
    layout='wide'
)

# Custom styles for title
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        color: white;
    }
    .sub-title {
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        color: white;
    }
    .header-background {
        background-color: #1e1e1e;
        padding: 10px;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Dashboard title with styles
st.markdown("<div class='header-background'><h1 class='main-title'>机器学习： 实时识别出虚假销售</h1><h2 class='sub-title'>实时欺诈检测</h2></div>", unsafe_allow_html=True)

# Sidebar input function
def user_input_features():
    st.sidebar.header('Make a prediction')
    st.sidebar.write('User input parameters below ⬇️')
    a1 = st.sidebar.slider('Action1', -31.0, 3.0, 0.0)
    a2 = st.sidebar.slider('Action2', -5.0, 13.0, 0.0)
    a3 = st.sidebar.slider('Action3', -20.0, 6.0, 0.0)
    a4 = st.sidebar.slider('Action4', -26.0, 7.0, 0.0)
    a5 = st.sidebar.slider('Action5', -4.0, 5.0, 0.0)
    a6 = st.sidebar.slider('Action6', -8.0, 4.0, 0.0)
    a7 = st.sidebar.slider('Sales Amount', 1.0, 5000.0, 1000.0)
    a8 = st.sidebar.selectbox("Gender?", ('Male', 'Female'))
    a9 = st.sidebar.selectbox("Agent Status?", ('Happy', 'Sad', 'Normal'))
    
    return [a1, a2, a3, a4, a5, a6, a7, a8, a9]

# 获取用户输入
outputdf = user_input_features()

# 将用户输入的特征转换为 DataFrame
outputdf = pd.DataFrame([outputdf], columns=[
    'Action1', 'Action2', 'Action3', 'Action4', 'Action5', 
    'Action6', 'Sales Amount', 'Gender', 'Agent Status'
])

# 如果 Gender 和 Agent Status 是分类特征，需要转换为数值
outputdf['Gender'] = outputdf['Gender'].map({'Male': 0, 'Female': 1})
outputdf['Agent Status'] = outputdf['Agent Status'].map({'Happy': 0, 'Sad': 1, 'Normal': 2})

# 加载模型
catmodel = CatBoostClassifier()
try:
    catmodel.load_model('fraud')  # 加载已训练模型
except FileNotFoundError:
    st.error("The model file 'fraud' is not found. Please check the file path.")
    st.stop()

# 添加预测按钮
if st.button("Predict Now"):
    # 模型预测
    try:
        predicted_class = catmodel.predict(outputdf)[0]
        predicted_proba = catmodel.predict_proba(outputdf)
        st.title('Real-Time Predictions')
        st.write(f'Predicted Class: {predicted_class}')
        st.write(f'Prediction Probability: {predicted_proba}')
    except Exception as e:
        st.error(f"Error in model prediction: {e}")

# SHAP 图像可视化
st.title('SHAP Value Analysis')
image_path = 'summary.png'
try:
    image4 = Image.open(image_path)
    st.image(image4, caption='SHAP Summary Plot')
except FileNotFoundError:
    st.warning(f"Image {image_path} not found. Please check the file path.")

# SHAP 值解释
try:
    explainer = shap.Explainer(catmodel)
    shap_values = explainer(outputdf)
    shap.plots.waterfall(shap_values[0])
    st.pyplot(bbox_inches='tight')
except Exception as e:
    st.error(f"Error in SHAP visualization: {e}")
