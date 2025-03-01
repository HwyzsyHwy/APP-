import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 页面配置
st.set_page_config(page_title="生物质热解产率预测器", layout="wide")

# 初始化会话状态
if 'predict_result' not in st.session_state:
    st.session_state.predict_result = None

if 'input_values' not in st.session_state:
    st.session_state.input_values = {
        "M": 5.0,
        "Ash": 8.0,
        "VM": 75.0,
        "FC": 15.0,
        "C": 45.0,
        "H": 6.0,
        "O": 45.0,
        "N": 0.5,
        "S": 0.1,
        "Temperature": 500.0,
        "Heating_Rate": 10.0,
        "Residence_Time": 30.0
    }

# 自定义HTML和CSS
st.markdown("""
<style>
/* 设置整体背景为深色 */
.stApp {
    background-color: #1E1E1E;
    color: white;
}

/* 标题样式 */
.header-green {
    background-color: green;
    padding: 10px;
    border-radius: 5px;
    color: white;
    text-align: center;
    margin-bottom: 10px;
}

.header-yellow {
    background-color: #CCCC00;
    padding: 10px;
    border-radius: 5px;
    color: black;
    text-align: center;
    margin-bottom: 10px;
}

.header-orange {
    background-color: #FFA500;
    padding: 10px;
    border-radius: 5px;
    color: black;
    text-align: center;
    margin-bottom: 10px;
}

/* 输入框样式 */
.input-container {
    display: flex;
    align-items: center;
    margin-bottom: 10px;
    padding: 8px;
    border-radius: 5px;
}

.green-container {
    background-color: green;
    color: white;
}

.yellow-container {
    background-color: #CCCC00;
    color: black;
}

.orange-container {
    background-color: #FFA500;
    color: black;
}

.input-label {
    flex: 1;
    margin-right: 10px;
}

.custom-input {
    background-color: inherit;
    color: inherit;
    border: 1px solid #ccc;
    border-radius: 3px;
    padding: 5px;
    width: 80px;
}

/* 按钮样式 */
.button-container {
    display: flex;
    justify-content: space-between;
    margin-top: 20px;
}

.predict-button, .clear-button {
    padding: 10px 20px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-weight: bold;
}

.predict-button {
    background-color: #4CAF50;
    color: white;
}

.clear-button {
    background-color: #f44336;
    color: white;
}

/* 结果展示区域 */
.result-area {
    background-color: #333333;
    padding: 20px;
    border-radius: 5px;
    margin-top: 20px;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# 设置JavaScript以更新Streamlit session state
st.markdown("""
<script>
document.addEventListener('DOMContentLoaded', function() {
    // 为所有自定义输入框添加事件监听
    var inputs = document.querySelectorAll('.custom-input');
    inputs.forEach(function(input) {
        input.addEventListener('change', function() {
            const key = this.getAttribute('data-key');
            const value = parseFloat(this.value);
            
            // 发送消息到Streamlit
            window.parent.postMessage({
                type: 'streamlit:setComponentValue',
                key: key,
                value: value
            }, '*');
        });
    });
});
</script>
""", unsafe_allow_html=True)

# 标题
st.title("生物质热解产率预测器")

# 创建三列布局
col1, col2, col3 = st.columns(3)

# 第一列：Proximate Analysis (绿色)
with col1:
    st.markdown('<div class="header-green">Proximate Analysis</div>', unsafe_allow_html=True)
    
    # 使用自定义HTML创建输入框
    st.markdown(f"""
    <div class="input-container green-container">
        <div class="input-label">M(wt%)</div>
        <input type="number" class="custom-input" data-key="M" value="{st.session_state.input_values['M']}" min="0" max="100" step="0.1" />
    </div>
    
    <div class="input-container green-container">
        <div class="input-label">Ash(wt%)</div>
        <input type="number" class="custom-input" data-key="Ash" value="{st.session_state.input_values['Ash']}" min="0" max="100" step="0.1" />
    </div>
    
    <div class="input-container green-container">
        <div class="input-label">VM(wt%)</div>
        <input type="number" class="custom-input" data-key="VM" value="{st.session_state.input_values['VM']}" min="0" max="100" step="0.1" />
    </div>
    
    <div class="input-container green-container">
        <div class="input-label">FC(wt%)</div>
        <input type="number" class="custom-input" data-key="FC" value="{st.session_state.input_values['FC']}" min="0" max="100" step="0.1" />
    </div>
    """, unsafe_allow_html=True)

# 第二列：Ultimate Analysis (黄色)
with col2:
    st.markdown('<div class="header-yellow">Ultimate Analysis</div>', unsafe_allow_html=True)
    
    # 使用自定义HTML创建输入框
    st.markdown(f"""
    <div class="input-container yellow-container">
        <div class="input-label">C(wt%)</div>
        <input type="number" class="custom-input" data-key="C" value="{st.session_state.input_values['C']}" min="0" max="100" step="0.1" />
    </div>
    
    <div class="input-container yellow-container">
        <div class="input-label">H(wt%)</div>
        <input type="number" class="custom-input" data-key="H" value="{st.session_state.input_values['H']}" min="0" max="100" step="0.1" />
    </div>
    
    <div class="input-container yellow-container">
        <div class="input-label">O(wt%)</div>
        <input type="number" class="custom-input" data-key="O" value="{st.session_state.input_values['O']}" min="0" max="100" step="0.1" />
    </div>
    
    <div class="input-container yellow-container">
        <div class="input-label">N(wt%)</div>
        <input type="number" class="custom-input" data-key="N" value="{st.session_state.input_values['N']}" min="0" max="10" step="0.1" />
    </div>
    
    <div class="input-container yellow-container">
        <div class="input-label">S(wt%)</div>
        <input type="number" class="custom-input" data-key="S" value="{st.session_state.input_values['S']}" min="0" max="10" step="0.1" />
    </div>
    """, unsafe_allow_html=True)

# 第三列：Pyrolysis Conditions (橙色)
with col3:
    st.markdown('<div class="header-orange">Pyrolysis Conditions</div>', unsafe_allow_html=True)
    
    # 使用自定义HTML创建输入框
    st.markdown(f"""
    <div class="input-container orange-container">
        <div class="input-label">Temperature(°C)</div>
        <input type="number" class="custom-input" data-key="Temperature" value="{st.session_state.input_values['Temperature']}" min="200" max="1000" step="10" />
    </div>
    
    <div class="input-container orange-container">
        <div class="input-label">Heating Rate(°C/min)</div>
        <input type="number" class="custom-input" data-key="Heating_Rate" value="{st.session_state.input_values['Heating_Rate']}" min="1" max="100" step="1" />
    </div>
    
    <div class="input-container orange-container">
        <div class="input-label">Residence Time(min)</div>
        <input type="number" class="custom-input" data-key="Residence_Time" value="{st.session_state.input_values['Residence_Time']}" min="1" max="120" step="1" />
    </div>
    """, unsafe_allow_html=True)

# 模型选择
model_options = ['Random Forest', 'XGBoost', 'LightGBM']
selected_model = st.selectbox("选择模型", model_options)

# 按钮区域
st.markdown("""
<div class="button-container">
    <button class="predict-button" id="predict-btn">预测</button>
    <button class="clear-button" id="clear-btn">清除</button>
</div>

<script>
document.getElementById('predict-btn').addEventListener('click', function() {
    window.parent.postMessage({
        type: 'streamlit:setComponentValue',
        key: 'predict_clicked',
        value: true
    }, '*');
});

document.getElementById('clear-btn').addEventListener('click', function() {
    window.parent.postMessage({
        type: 'streamlit:setComponentValue',
        key: 'clear_clicked',
        value: true
    }, '*');
});
</script>
""", unsafe_allow_html=True)

# 模拟按钮点击行为
predict_button = st.button("预测", key="predict_button", label_visibility="collapsed")
clear_button = st.button("清除", key="clear_button", label_visibility="collapsed")

# 加载模型和缩放器的函数
def load_model_and_scaler(model_name):
    try:
        if model_name == 'Random Forest':
            model = pickle.load(open('rf_model.pkl', 'rb'))
        elif model_name == 'XGBoost':
            model = pickle.load(open('xgb_model.pkl', 'rb'))
        else:  # LightGBM
            model = pickle.load(open('lgbm_model.pkl', 'rb'))
        
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        return model, scaler
    except Exception as e:
        st.error(f"模型加载错误: {str(e)}")
        return None, None

# 处理预测按钮点击
if predict_button or ('predict_clicked' in st.session_state and st.session_state.predict_clicked):
    if 'predict_clicked' in st.session_state:
        st.session_state.predict_clicked = False
    
    try:
        # 加载模型和缩放器
        model, scaler = load_model_and_scaler(selected_model)
        if model and scaler:
            # 收集所有输入特征
            input_values = st.session_state.input_values
            features = np.array([[
                input_values["M"], input_values["Ash"], input_values["VM"], input_values["FC"],
                input_values["C"], input_values["H"], input_values["O"], input_values["N"], input_values["S"],
                input_values["Temperature"], input_values["Heating_Rate"], input_values["Residence_Time"]
            ]])
            # 特征缩放
            features_scaled = scaler.transform(features)
            # 预测
            prediction = model.predict(features_scaled)[0]
            st.session_state.predict_result = prediction
    except Exception as e:
        st.error(f"预测错误: {str(e)}")

# 处理清除按钮点击
if clear_button or ('clear_clicked' in st.session_state and st.session_state.clear_clicked):
    if 'clear_clicked' in st.session_state:
        st.session_state.clear_clicked = False
    
    # 重置所有输入值
    st.session_state.input_values = {
        "M": 5.0,
        "Ash": 8.0,
        "VM": 75.0,
        "FC": 15.0,
        "C": 45.0,
        "H": 6.0,
        "O": 45.0,
        "N": 0.5,
        "S": 0.1,
        "Temperature": 500.0,
        "Heating_Rate": 10.0,
        "Residence_Time": 30.0
    }
    st.session_state.predict_result = None
    st.rerun()

# 展示预测结果
if st.session_state.predict_result is not None:
    st.markdown(f"""
    <div class="result-area">
        <h3>预测结果</h3>
        <p>Yield (%): {st.session_state.predict_result:.2f}</p>
    </div>
    """, unsafe_allow_html=True)