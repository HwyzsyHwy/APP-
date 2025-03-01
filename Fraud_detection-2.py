import streamlit as st
import numpy as np
import pickle

# 设置页面配置
st.set_page_config(layout="wide")

# 设置自定义CSS样式 - 简化版
st.markdown("""
<style>
    /* 全局样式 */
    .main {
        background-color: #1E1E1E;
        color: white;
    }
    
    /* 分析部分样式 */
    .proximate-analysis {
        background-color: rgba(46, 139, 87, 0.3);
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
    
    .ultimate-analysis {
        background-color: rgba(255, 215, 0, 0.3);
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
    
    .pyrolysis-conditions {
        background-color: rgba(255, 140, 0, 0.3);
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
    
    /* 标题样式 */
    h1 {
        color: white;
        text-align: center;
    }
    
    h3 {
        margin-top: 0;
    }
    
    /* 按钮样式 */
    .predict-btn {
        background-color: #4CAF50;
        color: white;
    }
    
    .clear-btn {
        background-color: #f44336;
        color: white;
    }
    
    /* 结果样式 */
    .result {
        background-color: #3D3D3D;
        border-radius: 10px;
        padding: 15px;
        margin-top: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# 标题
st.markdown("<h1>生物质热解产率预测</h1>", unsafe_allow_html=True)

# 创建会话状态变量
if 'predicted' not in st.session_state:
    st.session_state['predicted'] = False
    
if 'prediction_result' not in st.session_state:
    st.session_state['prediction_result'] = None

# 定义默认值
default_values = {
    'M': 5.0, 'Ash': 8.0, 'VM': 75.0, 'FC': 15.0,
    'C': 50.0, 'H': 6.0, 'O': 40.0, 'N': 1.0, 'S': 0.1,
    'Temperature': 500, 'Heating_rate': 10, 'Holding_time': 30
}

# 清除函数
def clear_inputs():
    for key in default_values:
        st.session_state[key] = default_values[key]
    st.session_state['predicted'] = False
    st.session_state['prediction_result'] = None

# 主布局
col1, col2, col3 = st.columns(3)

# 第一列 - 近似分析
with col1:
    st.markdown('<div class="proximate-analysis">', unsafe_allow_html=True)
    st.markdown("<h3>近似分析</h3>", unsafe_allow_html=True)
    
    m = st.number_input(
        "M(wt%)", 
        min_value=0.0, 
        max_value=50.0, 
        value=default_values['M'] if 'M' not in st.session_state else st.session_state['M'],
        step=0.1,
        key='M'
    )
    
    ash = st.number_input(
        "Ash(wt%)", 
        min_value=0.0, 
        max_value=50.0, 
        value=default_values['Ash'] if 'Ash' not in st.session_state else st.session_state['Ash'],
        step=0.1,
        key='Ash'
    )
    
    vm = st.number_input(
        "VM(wt%)", 
        min_value=0.0, 
        max_value=100.0, 
        value=default_values['VM'] if 'VM' not in st.session_state else st.session_state['VM'],
        step=0.1,
        key='VM'
    )
    
    fc = st.number_input(
        "FC(wt%)", 
        min_value=0.0, 
        max_value=100.0, 
        value=default_values['FC'] if 'FC' not in st.session_state else st.session_state['FC'],
        step=0.1,
        key='FC'
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

# 第二列 - 元素分析
with col2:
    st.markdown('<div class="ultimate-analysis">', unsafe_allow_html=True)
    st.markdown("<h3>元素分析</h3>", unsafe_allow_html=True)
    
    c = st.number_input(
        "C(wt%)", 
        min_value=0.0, 
        max_value=100.0, 
        value=default_values['C'] if 'C' not in st.session_state else st.session_state['C'],
        step=0.1,
        key='C'
    )
    
    h = st.number_input(
        "H(wt%)", 
        min_value=0.0, 
        max_value=20.0, 
        value=default_values['H'] if 'H' not in st.session_state else st.session_state['H'],
        step=0.1,
        key='H'
    )
    
    o = st.number_input(
        "O(wt%)", 
        min_value=0.0, 
        max_value=100.0, 
        value=default_values['O'] if 'O' not in st.session_state else st.session_state['O'],
        step=0.1,
        key='O'
    )
    
    n = st.number_input(
        "N(wt%)", 
        min_value=0.0, 
        max_value=20.0, 
        value=default_values['N'] if 'N' not in st.session_state else st.session_state['N'],
        step=0.1,
        key='N'
    )
    
    s = st.number_input(
        "S(wt%)", 
        min_value=0.0, 
        max_value=10.0, 
        value=default_values['S'] if 'S' not in st.session_state else st.session_state['S'],
        step=0.01,
        key='S'
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

# 第三列 - 热解条件
with col3:
    st.markdown('<div class="pyrolysis-conditions">', unsafe_allow_html=True)
    st.markdown("<h3>热解条件</h3>", unsafe_allow_html=True)
    
    temperature = st.number_input(
        "温度 (°C)", 
        min_value=200, 
        max_value=1000, 
        value=default_values['Temperature'] if 'Temperature' not in st.session_state else st.session_state['Temperature'],
        step=10,
        key='Temperature'
    )
    
    heating_rate = st.number_input(
        "升温速率 (°C/min)", 
        min_value=1, 
        max_value=100, 
        value=default_values['Heating_rate'] if 'Heating_rate' not in st.session_state else st.session_state['Heating_rate'],
        step=1,
        key='Heating_rate'
    )
    
    holding_time = st.number_input(
        "保温时间 (min)", 
        min_value=0, 
        max_value=120, 
        value=default_values['Holding_time'] if 'Holding_time' not in st.session_state else st.session_state['Holding_time'],
        step=1,
        key='Holding_time'
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

# 按钮布局
col1, col2 = st.columns(2)

with col1:
    predict_button = st.button("预测", key="predict")

with col2:
    clear_button = st.button("清除", key="clear", on_click=clear_inputs)

# 预测逻辑
if predict_button or st.session_state['predicted']:
    st.session_state['predicted'] = True
    
    try:
        # 创建特征数组
        features = [m, ash, vm, fc, c, h, o, n, s, temperature, heating_rate, holding_time]
        
        # 模拟模型预测结果（由于没有实际模型）
        # 在实际应用中，应替换为模型预测逻辑
        prediction = 30.0 + 0.05 * temperature - 0.5 * m + 0.3 * c - 0.2 * ash
        prediction = max(10.0, min(50.0, prediction))  # 将结果限制在10-50范围内
        
        st.session_state['prediction_result'] = prediction
        
        # 显示预测结果
        st.markdown(f'''
        <div class="result">
            <h3>预测结果</h3>
            <h2>产率 (%): {st.session_state["prediction_result"]:.2f}</h2>
        </div>
        ''', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"预测时出错: {e}")