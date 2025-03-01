import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(layout="wide")

# 设置整体页面背景为暗色
st.markdown("""
<style>
    .stApp {
        background-color: #1E1E1E;
        color: white;
    }
    
    /* 尝试更精确的CSS选择器 */
    /* 第一列 - 近似分析 (绿色) */
    .green-input input {
        background-color: #32CD32 !important;
        color: black !important;
    }
    
    /* 第二列 - 元素分析 (黄色) */
    .yellow-input input {
        background-color: #DAA520 !important;
        color: black !important;
    }
    
    /* 第三列 - 热解条件 (橙色) */
    .orange-input input {
        background-color: #FF7F50 !important;
        color: black !important;
    }
    
    /* 减少每个部分的边距，使布局更紧凑 */
    .section-container {
        margin: 0;
        padding: 5px;
    }
    
    /* 使标签和输入框距离更近 */
    .stColumn {
        padding: 0 !important;
    }
    
    /* 确保每个输入框的宽度一致 */
    div[data-testid="stNumberInput"] input {
        width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

# 在会话状态中初始化预测结果
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# 加载模型
@st.cache_resource
def load_model():
    model = pickle.load(open('gradient_boosting_model.pkl', 'rb'))
    return model

model = load_model()

# 标题
st.title('生物质热解产率预测')

# 显示预测结果
def display_prediction():
    if st.session_state.prediction_result is not None:
        st.markdown(f"""
        <div style="padding: 20px; background-color: #2E2E2E; border-radius: 5px; margin-top: 20px;">
            <h3 style="color: white;">预测结果</h3>
            <p style="font-size: 24px; color: white;">产率 (%): {st.session_state.prediction_result:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

# 创建输入区域
col1, col2, col3 = st.columns(3)

# 使用类来应用样式
with col1:
    st.markdown('<h3 style="color: #32CD32;">近似分析</h3>', unsafe_allow_html=True)
    
    st.markdown('<div class="green-input">', unsafe_allow_html=True)
    col1_1, col1_2 = st.columns([2, 3])
    with col1_1:
        st.write("M(wt%)")
    with col1_2:
        m = st.number_input("", min_value=0.0, max_value=20.0, value=5.0, step=0.1, key="m", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="green-input">', unsafe_allow_html=True)
    col1_1, col1_2 = st.columns([2, 3])
    with col1_1:
        st.write("Ash(wt%)")
    with col1_2:
        ash = st.number_input("", min_value=0.0, max_value=30.0, value=8.0, step=0.1, key="ash", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="green-input">', unsafe_allow_html=True)
    col1_1, col1_2 = st.columns([2, 3])
    with col1_1:
        st.write("VM(wt%)")
    with col1_2:
        vm = st.number_input("", min_value=50.0, max_value=90.0, value=75.0, step=0.1, key="vm", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="green-input">', unsafe_allow_html=True)
    col1_1, col1_2 = st.columns([2, 3])
    with col1_1:
        st.write("FC(wt%)")
    with col1_2:
        fc = st.number_input("", min_value=5.0, max_value=30.0, value=15.0, step=0.1, key="fc", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<h3 style="color: #DAA520;">元素分析</h3>', unsafe_allow_html=True)
    
    st.markdown('<div class="yellow-input">', unsafe_allow_html=True)
    col2_1, col2_2 = st.columns([2, 3])
    with col2_1:
        st.write("C(wt%)")
    with col2_2:
        c = st.number_input("", min_value=40.0, max_value=60.0, value=48.0, step=0.1, key="c", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="yellow-input">', unsafe_allow_html=True)
    col2_1, col2_2 = st.columns([2, 3])
    with col2_1:
        st.write("H(wt%)")
    with col2_2:
        h = st.number_input("", min_value=4.0, max_value=8.0, value=6.0, step=0.1, key="h", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="yellow-input">', unsafe_allow_html=True)
    col2_1, col2_2 = st.columns([2, 3])
    with col2_1:
        st.write("O(wt%)")
    with col2_2:
        o = st.number_input("", min_value=30.0, max_value=50.0, value=40.0, step=0.1, key="o", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="yellow-input">', unsafe_allow_html=True)
    col2_1, col2_2 = st.columns([2, 3])
    with col2_1:
        st.write("N(wt%)")
    with col2_2:
        n = st.number_input("", min_value=0.0, max_value=3.0, value=0.8, step=0.1, key="n", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<h3 style="color: #FF7F50;">热解条件</h3>', unsafe_allow_html=True)
    
    st.markdown('<div class="orange-input">', unsafe_allow_html=True)
    col3_1, col3_2 = st.columns([2, 3])
    with col3_1:
        st.write("温度(°C)")
    with col3_2:
        temp = st.number_input("", min_value=300, max_value=1000, value=500, step=10, key="temp", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="orange-input">', unsafe_allow_html=True)
    col3_1, col3_2 = st.columns([2, 3])
    with col3_1:
        st.write("停留时间(s)")
    with col3_2:
        time = st.number_input("", min_value=0.0, max_value=200.0, value=10.0, step=1.0, key="time", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="orange-input">', unsafe_allow_html=True)
    col3_1, col3_2 = st.columns([2, 3])
    with col3_1:
        st.write("升温速率(°C/min)")
    with col3_2:
        heating_rate = st.number_input("", min_value=0, max_value=1000, value=10, step=10, key="heating_rate", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="orange-input">', unsafe_allow_html=True)
    col3_1, col3_2 = st.columns([2, 3])
    with col3_1:
        st.write("粒径(mm)")
    with col3_2:
        particle_size = st.number_input("", min_value=0.0, max_value=10.0, value=1.0, step=0.1, key="particle_size", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

# 清除按钮的回调函数
def clear_inputs():
    # 设置所有输入恢复到默认值
    default_values = {
        "m": 5.0, "ash": 8.0, "vm": 75.0, "fc": 15.0,
        "c": 48.0, "h": 6.0, "o": 40.0, "n": 0.8,
        "temp": 500, "time": 10.0, "heating_rate": 10, "particle_size": 1.0
    }
    
    for key, value in default_values.items():
        st.session_state[key] = value
    
    # 清除预测结果
    st.session_state.prediction_result = None

# 按钮区域
col1, col2 = st.columns(2)
with col1:
    if st.button('预测', key='predict_button', help='点击进行预测'):
        try:
            # 收集输入
            input_data = np.array([[
                m, ash, vm, fc,
                c, h, o, n,
                temp, time, heating_rate, particle_size
            ]])
            
            # 进行预测
            prediction = model.predict(input_data)[0]
            st.session_state.prediction_result = prediction
        except Exception as e:
            st.error(f"预测过程中出现错误: {e}")

with col2:
    if st.button('清除', key='clear_button', help='点击清除所有输入', on_click=clear_inputs):
        pass  # 使用on_click回调函数处理

# 显示预测结果
display_prediction()

# 尝试另一种方法使用jQuery更新样式
st.markdown("""
<script>
    // 等待DOM加载完成
    document.addEventListener('DOMContentLoaded', function() {
        // 近似分析 - 绿色背景
        const greenInputs = document.querySelectorAll('.green-input input');
        greenInputs.forEach(input => {
            input.style.backgroundColor = '#32CD32';
            input.style.color = 'black';
        });
        
        // 元素分析 - 黄色背景
        const yellowInputs = document.querySelectorAll('.yellow-input input');
        yellowInputs.forEach(input => {
            input.style.backgroundColor = '#DAA520';
            input.style.color = 'black';
        });
        
        // 热解条件 - 橙色背景
        const orangeInputs = document.querySelectorAll('.orange-input input');
        orangeInputs.forEach(input => {
            input.style.backgroundColor = '#FF7F50';
            input.style.color = 'black';
        });
    });
</script>
""", unsafe_allow_html=True)