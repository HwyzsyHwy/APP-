# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 11:02:43 2022

@author: Kevin Boss
"""
from PIL import Image
import warnings
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import shap
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import os

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# 设置默认样式
plt.style.use('default')

# Streamlit 页面配置
st.set_page_config(
    page_title='Real-Time Fraud Detection',
    page_icon='🕵️‍♀️',
    layout='wide'
)

# 获取当前工作目录并打印
current_dir = os.getcwd()
st.write(f"Current working directory: {current_dir}")

# 标题
st.markdown("<h1 style='text-align: center; color: black;'>机器学习： 实时识别出虚假销售</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: black;'>Real-Time Fraud Detection</h1>", unsafe_allow_html=True)

# 侧边栏用户输入
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

# 加载数据集
try:
    df = pd.read_excel(os.path.join(current_dir, 'fraud.xlsx'), engine='openpyxl')
    st.title('Dataset')
    if st.button('View some random data'):
        st.write(df.sample(5))
    st.write(f"The dataset is trained on Catboost with a total length of: {len(df)}. 0️⃣ means it's a real transaction, 1️⃣ means it's a Fraud transaction. Data is unbalanced (not⚖️).")
    unbalancedf = pd.DataFrame(df.Class.value_counts())
    st.write(unbalancedf)
except Exception as e:
    st.error(f"Error loading dataset: {e}")

# 可视化
placeholder = st.empty()
with placeholder.container():
    f1, f2, f3 = st.columns(3)
    try:
        with f1:
            fraud_action1 = df[df['Class'] == 1]['Action1']
            real_action1 = df[df['Class'] == 0]['Action1']
            hist_data = [fraud_action1, real_action1]
            fig = ff.create_distplot(hist_data, group_labels=['Fraud', 'Legit'])
            fig.update_layout(title_text='Action 1')
            st.plotly_chart(fig, use_container_width=True)
        with f2:
            fraud_action2 = df[df['Class'] == 1]['Action2']
            real_action2 = df[df['Class'] == 0]['Action2']
            hist_data = [fraud_action2, real_action2]
            fig = ff.create_distplot(hist_data, group_labels=['Fraud', 'Real'])
            fig.update_layout(title_text='Action 2')
            st.plotly_chart(fig, use_container_width=True)
        with f3:
            fraud_action3 = df[df['Class'] == 1]['Action3']
            real_action3 = df[df['Class'] == 0]['Action3']
            hist_data = [fraud_action3, real_action3]
            fig = ff.create_distplot(hist_data, group_labels=['Fraud', 'Real'])
            fig.update_layout(title_text='Action 3')
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error in visualization: {e}")

# SHAP 值分析
st.title('SHAP Value Analysis')
try:
    shapdatadf = pd.read_excel(os.path.join(current_dir, 'shapdatadf.xlsx'), engine='openpyxl')
    shapvaluedf = pd.read_excel(os.path.join(current_dir, 'shapvaluedf.xlsx'), engine='openpyxl')

    f1, f2 = st.columns(2)
    with f1:
        st.subheader('Summary plot')
        image_path = os.path.join(current_dir, 'summary.png')
        if os.path.exists(image_path):
            image4 = Image.open(image_path)
            st.image(image4)
        else:
            st.warning("Summary plot image not found!")
    with f2:
        st.subheader('Dependence plot for features')
        selected_feature = st.selectbox("Choose a feature", shapdatadf.columns)
        fig = px.scatter(
            x=shapdatadf[selected_feature],
            y=shapvaluedf[selected_feature],
            color=shapdatadf[selected_feature],
            color_continuous_scale=['blue', 'red']
        )
        st.plotly_chart(fig)
except Exception as e:
    st.error(f"Error in SHAP value analysis: {e}")

# 模型预测
try:
    catmodel = CatBoostClassifier()
    catmodel.load_model(os.path.join(current_dir, 'fraud'))
    outputdf = pd.DataFrame([outputdf], columns=shapdatadf.columns)
    predicted_class = catmodel.predict(outputdf)[0]
    predicted_proba = catmodel.predict_proba(outputdf)
    st.title('Real-Time Predictions')
    st.write(f'Predicted Class: {predicted_class}')
    st.write(f'Prediction Probability: {predicted_proba}')
except Exception as e:
    st.error(f"Error in model prediction: {e}")

# 绘制 SHAP 水滴图
try:
    explainer = shap.Explainer(catmodel)
    shap_values = explainer(outputdf)
    shap.plots.waterfall(shap_values[0])
    st.pyplot(bbox_inches='tight')
except Exception as e:
    st.error(f"Error in SHAP waterfall plot: {e}")
