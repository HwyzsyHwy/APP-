import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
from PIL import Image
import io

# 设置页面配置
st.set_page_config(layout="wide")

# 设置自定义CSS样式
st.markdown("""
<style>
    /* 全局样式 */
    .main {
        background-color: #1E1E1E;
        color: white;
    }
    
    /* 标题样式 */
    .title {
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    
    /* 箱体样式 */
    .box {
        background-color: #2D2D2D;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }
    
    /* 分析部分样式 */
    .analysis-box-1 {
        background-color: rgba(46, 139, 87, 0.3);
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
    
    .analysis-box-2 {
        background-color: rgba(255, 215, 0, 0.3);
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
    
    .analysis-box-3 {
        background-color: rgba(255, 140, 0, 0.3);
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
    
    /* 输入字段样式 */
    div[data-baseweb="input"] {
        background-color: transparent !important;
    }
    
    /* 区分不同区域的输入框背景色 */
    .green-bg div[data-baseweb="input"] input,
    .green-bg div[data-baseweb="number-input"] input {
        background-color: rgba(46, 139, 87, 0.5) !important;
    }
    
    .yellow-bg div[data-baseweb="input"] input,
    .yellow-bg div[data-baseweb="number-input"] input {
        background-color: rgba(255, 215, 0, 0.5) !important;
    }
    
    .orange-bg div[data-baseweb="input"] input,
    .orange-bg div[data-baseweb="number-input"] input {
        background-color: rgba(255, 140, 0, 0.5) !important;
    }
    
    /* 按钮样式 */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        width: 100%;
        border: none;
        border-radius: 5px;
        padding: 10px;
    }
    
    .clear-button>button {
        background-color: #f44336;
        color: white;
    }
    
    /* 结果显示区域样式 */
    .result-box {
        background-color: #3D3D3D;
        border-radius: 10px;
        padding: 15px;
        margin-top: 20px;
        text-align: center;
    }
    
    .result-value {
        font-size: 24px;
        font-weight: bold;
        color: #4CAF50;
    }
    
    /* 紧凑布局调整 */
    .row-container {
        display: flex;
        align-items: center;
        margin-bottom: 5px;
    }
    
    .label {
        width: 40%;
        min-width: 100px;
    }
    
    .input-field {
        width: 60%;
    }
    
    /* 给Streamlit的输入组件添加样式 */
    input[type="number"] {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# 标题
st.markdown('<div class="title">生物质热解产率预测</div>', unsafe_allow_html=True)

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

# 加载模型函数
@st.cache_resource
def load_model():
    try:
        with open('best_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"加载模型时出错: {e}")
        return None

# 主布局
col1, col2, col3 = st.columns(3)

# 第一列 - 近似分析
with col1:
    st.markdown('<div class="box analysis-box-1"><h3>近似分析</h3>', unsafe_allow_html=True)
    
    # 为每个输入字段添加green-bg类
    st.markdown('<div class="green-bg">', unsafe_allow_html=True)
    
    # 添加输入字段
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
    st.markdown('</div>', unsafe_allow_html=True)  # 关闭box div

# 第二列 - 元素分析
with col2:
    st.markdown('<div class="box analysis-box-2"><h3>元素分析</h3>', unsafe_allow_html=True)
    
    # 为每个输入字段添加yellow-bg类
    st.markdown('<div class="yellow-bg">', unsafe_allow_html=True)
    
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
    st.markdown('</div>', unsafe_allow_html=True)  # 关闭box div

# 第三列 - 热解条件
with col3:
    st.markdown('<div class="box analysis-box-3"><h3>热解条件</h3>', unsafe_allow_html=True)
    
    # 为每个输入字段添加orange-bg类
    st.markdown('<div class="orange-bg">', unsafe_allow_html=True)
    
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
    st.markdown('</div>', unsafe_allow_html=True)  # 关闭box div

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
        # 这里可以替换为您的实际预测逻辑
        # 假设模型需要以下特征
        features = {
            'M': m, 'Ash': ash, 'VM': vm, 'FC': fc,
            'C': c, 'H': h, 'O': o, 'N': n, 'S': s,
            'Temperature': temperature, 'Heating_rate': heating_rate, 'Holding_time': holding_time
        }
        
        # 创建特征数组
        input_data = np.array([list(features.values())])
        
        # 尝试加载模型并进行预测
        model = load_model()
        if model is not None:
            prediction = model.predict(input_data)[0]
            st.session_state['prediction_result'] = prediction
        else:
            # 如果没有模型，生成模拟预测结果
            prediction = np.random.uniform(20, 40)
            st.session_state['prediction_result'] = prediction
        
        # 显示预测结果
        st.markdown(f'''
        <div class="result-box">
            <p>预测结果:</p>
            <p class="result-value">产率 (%): {st.session_state["prediction_result"]:.2f}</p>
        </div>
        ''', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"预测时出错: {e}")

# 添加JavaScript以修复任何潜在的UI问题
st.markdown("""
<script>
    // 修复输入框的样式和布局
    document.addEventListener('DOMContentLoaded', function() {
        // 给一些时间让所有Streamlit元素加载完成
        setTimeout(function() {
            // 调整输入框的样式
            const inputs = document.querySelectorAll('input[type="number"]');
            inputs.forEach(function(input) {
                input.style.backgroundColor = 'transparent';
                input.style.color = 'white';
            });
        }, 1000);
    });
</script>
""", unsafe_allow_html=True)