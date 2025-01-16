# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 11:02:43 2022

@author: Kevin Boss
"""
from PIL import Image
import warnings
import streamlit as st
import pandas as pd
import plotly.express as px
import shap
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

plt.style.use('default')

st.set_page_config(
    page_title='Real-Time Fraud Detection',
    page_icon='🕵️‍♀️',
    layout='wide'
)

# Dashboard title
st.markdown("<h1 style='text-align: center; color: black;'>机器学习： 实时识别出虚假销售</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: black;'>Real-Time Fraud Detection</h1>", unsafe_allow_html=True)

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

# SHAP values visualization
st.title('SHAP Value Analysis')
image_path = 'summary.png'
image4 = Image.open(image_path)
shapdatadf = pd.read_excel('shapdatadf.xlsx', engine='openpyxl')
shapvaluedf = pd.read_excel('shapvaluedf.xlsx', engine='openpyxl')

placeholder5 = st.empty()
with placeholder5.container():
    f1, f2 = st.columns(2)

    with f1:
        st.subheader('Summary plot')
        st.image(image4)
    with f2:
        st.subheader('Dependence plot for features')
        selected_feature = st.selectbox("Choose a feature", shapdatadf.columns)
        fig = px.scatter(x=shapdatadf[selected_feature], 
                         y=shapvaluedf[selected_feature], 
                         color=shapdatadf[selected_feature],
                         color_continuous_scale=['blue', 'red'])
        st.plotly_chart(fig)

# Model prediction
catmodel = CatBoostClassifier()
catmodel.load_model('fraud')  # 加载已训练的模型
outputdf = pd.DataFrame([outputdf], columns=shapdatadf.columns)  # 使用 SHAP 数据集的列名

# 模型预测
predicted_class = catmodel.predict(outputdf)[0]
predicted_proba = catmodel.predict_proba(outputdf)

# 显示预测结果
st.title('Real-Time Predictions')
st.write(f'Predicted Class: {predicted_class}')
st.write(f'Prediction Probability: {predicted_proba}')

# 显示 SHAP 值解释
explainer = shap.Explainer(catmodel)
shap_values = explainer(outputdf)
shap.plots.waterfall(shap_values[0])
st.pyplot(bbox_inches='tight')
