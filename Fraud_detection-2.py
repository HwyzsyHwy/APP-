# -*- coding: utf-8 -*-
"""
Biomass Pyrolysis Yield Forecast
"""

import streamlit as st
import pandas as pd
import joblib

# 页面设置
st.set_page_config(
    page_title='Biomass Pyrolysis Yield Forecast',
    page_icon='📊',
    layout='wide'
)

# 自定义样式
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .section {
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
        color: black;
    }
    .ultimate-section {
        background-color: #DAA520;  /* 黄色 */
    }
    .proximate-section {
        background-color: #32CD32;  /* 绿色 */
    }
    .pyrolysis-section {
        background-color: #FF7F50;  /* 橙色 */
    }
    .section-title {
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    .yield-result {
        background-color: #1E1E1E;
        color: white;
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        padding: 15px;
        border-radius: 8px;
        margin-top: 20px;
    }
    .input-row {
        padding: 5px;
        border-radius: 5px;
        margin-bottom: 5px;
    }
    
    /* 自定义输入框样式 */
    .custom-input {
        width: 100%;
        padding: 8px;
        border: none;
        border-radius: 4px;
        font-size: 16px;
        text-align: center;
    }
    
    .green-input {
        background-color: #32CD32 !important;
        color: black !important;
    }
    
    .yellow-input {
        background-color: #DAA520 !important;
        color: black !important;
    }
    
    .orange-input {
        background-color: #FF7F50 !important;
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 主标题
st.markdown("<h1 class='main-title'>GUI for Bio-Char Yield Prediction based on ELT-PSO Model</h1>", unsafe_allow_html=True)

# 初始化会话状态
if 'features' not in st.session_state:
    st.session_state.features = {}

# 定义默认值
default_values = {
    "M(wt%)": 5.0,
    "Ash(wt%)": 8.0,
    "VM(wt%)": 75.0,
    "FC(wt%)": 15.0,
    "C(wt%)": 60.0,
    "H(wt%)": 5.0,
    "N(wt%)": 1.0,
    "O(wt%)": 38.0,
    "PS(mm)": 6.0,
    "SM(g)": 75.0,
    "FT(℃)": 600.0,
    "HR(℃/min)": 50.0,
    "FR(mL/min)": 50.0,
    "RT(min)": 30.0
}

# 初始化特征值
for key, value in default_values.items():
    if key not in st.session_state.features:
        st.session_state.features[key] = value

# 创建自定义HTML输入框的函数
def create_custom_input(feature, color_class, min_val=0.0, max_val=100.0):
    current_value = st.session_state.features.get(feature, default_values[feature])
    
    # 创建带有自定义背景色的HTML输入
    html_input = f"""
    <input 
        type="number" 
        id="{feature}" 
        name="{feature}" 
        value="{current_value}" 
        min="{min_val}" 
        max="{max_val}" 
        step="0.1"
        class="custom-input {color_class}"
        onchange="updateValue(this)"
    >
    <script>
        function updateValue(element) {{
            const value = parseFloat(element.value);
            const min = parseFloat(element.min);
            const max = parseFloat(element.max);
            
            // 验证值是否在有效范围内
            if (value < min) element.value = min;
            if (value > max) element.value = max;
            
            // 保存到表单隐藏字段
            document.getElementById('hidden_{feature}').value = element.value;
        }}
    </script>
    """
    
    # 创建隐藏字段以保存实际值
    hidden_input = f"""
    <input 
        type="hidden" 
        id="hidden_{feature}" 
        name="hidden_{feature}" 
        value="{current_value}"
    >
    """
    
    # 返回完整的HTML
    return html_input + hidden_input

# 特征分类
feature_categories = {
    "Proximate Analysis": ["M(wt%)", "Ash(wt%)", "VM(wt%)", "FC(wt%)"],
    "Ultimate Analysis": ["C(wt%)", "H(wt%)", "N(wt%)", "O(wt%)"],
    "Pyrolysis Conditions": ["PS(mm)", "SM(g)", "FT(℃)", "HR(℃/min)", "FR(mL/min)", "RT(min)"]
}

# 创建表单以捕获输入值
with st.form(key="input_form"):
    # 创建三列布局
    col1, col2, col3 = st.columns(3)
    
    # Proximate Analysis (绿色区域)
    with col1:
        st.markdown("<div class='proximate-section section'><div class='section-title'>Proximate Analysis</div>", unsafe_allow_html=True)
        
        for feature in feature_categories["Proximate Analysis"]:
            # 设置最小最大值
            min_val = 0.0
            max_val = 20.0 if feature == "M(wt%)" else (25.0 if feature == "Ash(wt%)" else (110.0 if feature == "VM(wt%)" else 120.0))
            
            # 两列布局
            col_a, col_b = st.columns([1, 0.5])
            with col_a:
                st.markdown(f"<div class='input-row' style='background-color: #32CD32;'>{feature}</div>", unsafe_allow_html=True)
            with col_b:
                st.markdown(create_custom_input(feature, "green-input", min_val, max_val), unsafe_allow_html=True)
                # 创建一个占位符输入框，但隐藏它
                st.text_input(feature, value=st.session_state.features.get(feature, default_values[feature]), key=f"streamlit_{feature}", label_visibility="collapsed")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Ultimate Analysis (黄色区域)
    with col2:
        st.markdown("<div class='ultimate-section section'><div class='section-title'>Ultimate Analysis</div>", unsafe_allow_html=True)
        
        for feature in feature_categories["Ultimate Analysis"]:
            # 设置最小最大值
            min_val = 30.0 if feature in ["C(wt%)", "O(wt%)"] else 0.0
            max_val = 110.0 if feature == "C(wt%)" else (15.0 if feature == "H(wt%)" else (5.0 if feature == "N(wt%)" else 60.0))
            
            # 两列布局
            col_a, col_b = st.columns([1, 0.5])
            with col_a:
                st.markdown(f"<div class='input-row' style='background-color: #DAA520;'>{feature}</div>", unsafe_allow_html=True)
            with col_b:
                st.markdown(create_custom_input(feature, "yellow-input", min_val, max_val), unsafe_allow_html=True)
                # 创建一个占位符输入框，但隐藏它
                st.text_input(feature, value=st.session_state.features.get(feature, default_values[feature]), key=f"streamlit_{feature}", label_visibility="collapsed")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Pyrolysis Conditions (橙色区域)
    with col3:
        st.markdown("<div class='pyrolysis-section section'><div class='section-title'>Pyrolysis Conditions</div>", unsafe_allow_html=True)
        
        for feature in feature_categories["Pyrolysis Conditions"]:
            # 设置最小最大值
            min_val = 250.0 if feature == "FT(℃)" else (5.0 if feature == "RT(min)" else 0.0)
            max_val = 1100.0 if feature == "FT(℃)" else (200.0 if feature in ["SM(g)", "HR(℃/min)"] else (120.0 if feature == "FR(mL/min)" else (100.0 if feature == "RT(min)" else 20.0)))
            
            # 两列布局
            col_a, col_b = st.columns([1, 0.5])
            with col_a:
                st.markdown(f"<div class='input-row' style='background-color: #FF7F50;'>{feature}</div>", unsafe_allow_html=True)
            with col_b:
                st.markdown(create_custom_input(feature, "orange-input", min_val, max_val), unsafe_allow_html=True)
                # 创建一个占位符输入框，但隐藏它
                st.text_input(feature, value=st.session_state.features.get(feature, default_values[feature]), key=f"streamlit_{feature}", label_visibility="collapsed")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # 预测按钮和清除按钮
    col1, col2 = st.columns(2)
    with col1:
        predict_button = st.form_submit_button("PUSH")
    
    with col2:
        clear_button = st.form_submit_button("CLEAR")

# 处理按钮点击
if clear_button:
    # 重置为默认值
    for key, value in default_values.items():
        st.session_state.features[key] = value
    
    # 重新加载页面以显示更新后的值
    st.experimental_rerun()

# 显示预测结果
if predict_button:
    try:
        # 收集表单中的特征
        features = {key: st.session_state.features[key] for key in default_values.keys()}
        
        # 创建DataFrame
        input_data = pd.DataFrame([features])
        
        # 在这里添加模型预测逻辑
        # 例如：
        # model = load_model(model_name)
        # scaler = load_scaler(model_name)
        # input_data_scaled = scaler.transform(input_data)
        # y_pred = model.predict(input_data_scaled)[0]
        
        # 模拟预测结果
        y_pred = 35.42  # 替换为实际预测
        
        # 显示预测结果
        st.markdown(
            f"<div class='yield-result'>Yield (%) <br> {y_pred:.2f}</div>",
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"预测过程中出现错误: {e}")

# JavaScript代码以确保自定义输入字段的值可以传回Streamlit
st.markdown("""
<script>
// 在页面加载时设置输入框样式
document.addEventListener('DOMContentLoaded', function() {
    // 设置所有输入框的背景颜色
    var greenInputs = document.querySelectorAll('.green-input');
    var yellowInputs = document.querySelectorAll('.yellow-input');
    var orangeInputs = document.querySelectorAll('.orange-input');
    
    greenInputs.forEach(function(input) {
        input.style.backgroundColor = '#32CD32';
    });
    
    yellowInputs.forEach(function(input) {
        input.style.backgroundColor = '#DAA520';
    });
    
    orangeInputs.forEach(function(input) {
        input.style.backgroundColor = '#FF7F50';
    });
});
</script>
""", unsafe_allow_html=True)