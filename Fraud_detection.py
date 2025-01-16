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

outputdf = user_input_features()

# Dataset loading
df = pd.read_excel('fraud.xlsx', engine='openpyxl')

st.title('Dataset')
if st.button('View some random data'):
    st.write(df.sample(5))

st.write(f"The dataset is trained on Catboost with a total length of: {len(df)}. 0️⃣ means it's a real transaction, 1️⃣ means it's a Fraud transaction. Data is unbalanced (not⚖️).")

unbalancedf = pd.DataFrame(df.Class.value_counts())
st.write(unbalancedf)

# Visualization placeholders
placeholder = st.empty()
placeholder2 = st.empty()
placeholder3 = st.empty()

# First set of visualizations
with placeholder.container():
    f1, f2, f3 = st.columns(3)

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
catmodel.load_model('fraud')
outputdf = pd.DataFrame([outputdf], columns=shapdatadf.columns)

predicted_class = catmodel.predict(outputdf)[0]
predicted_proba = catmodel.predict_proba(outputdf)

st.title('Real-Time Predictions')
st.write(f'Predicted Class: {predicted_class}')
st.write(f'Prediction Probability: {predicted_proba}')

# Removed st.set_option
explainer = shap.Explainer(catmodel)
shap_values = explainer(outputdf)
shap.plots.waterfall(shap_values[0])
st.pyplot(bbox_inches='tight')
