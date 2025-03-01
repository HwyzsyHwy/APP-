import streamlit as st
import numpy as np
import pickle
from streamlit_elements import elements, mui, html

# 安装依赖库：pip install streamlit-elements

# 页面配置
st.set_page_config(page_title="生物质热解产率预测器", layout="wide")

# 自定义CSS样式
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .section-title {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 15px;
        text-align: center;
    }
    .proximateAnalysis {
        color: green;
    }
    .ultimateAnalysis {
        color: yellow;
    }
    .pyrolysisConditions {
        color: orange;
    }
    .button-container {
        display: flex;
        justify-content: space-between;
        margin-top: 20px;
    }
    .result-container {
        background-color: #4CAF50;
        color: white;
        padding: 15px;
        border-radius: 5px;
        margin-top: 20px;
        text-align: center;
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)

# 初始化session_state
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

if 'input_values' not in st.session_state:
    st.session_state.input_values = {
        "M": 5.0, "Ash": 8.0, "VM": 75.0, "FC": 15.0,
        "C": 45.0, "H": 6.0, "O": 45.0, "N": 0.5, "S": 0.1,
        "T": 500, "HR": 10, "HT": 30
    }

# 模型选择选项
models = ["随机森林", "XGBoost", "支持向量机", "神经网络"]

# 加载模型和数据处理器函数
def load_model(model_name):
    model_path = f"{model_name}_model.pkl"
    try:
        # 这里只是示例，实际应用请替换为真实的模型加载逻辑
        model = None
        return model
    except:
        st.error(f"无法加载模型：{model_path}")
        return None

def load_scaler():
    scaler_path = "scaler.pkl"
    try:
        # 这里只是示例，实际应用请替换为真实的数据处理器加载逻辑
        scaler = None
        return scaler
    except:
        st.error(f"无法加载数据处理器：{scaler_path}")
        return None

# 更新输入值的回调函数
def update_value(key, value):
    st.session_state.input_values[key] = value

# 标题和描述
st.title("生物质热解产率预测器")
st.write("请输入以下参数来预测生物质热解产率：")

# 使用Streamlit Elements创建自定义输入框
with elements("custom_inputs"):
    with mui.Grid(container=True, spacing=2):
        # 第一列：近似分析（Proximate Analysis）
        with mui.Grid(item=True, xs=4):
            mui.Typography("近似分析", 
                           variant="h5", 
                           className="section-title proximateAnalysis")
            
            with mui.Paper(elevation=3, sx={"p": 2, "backgroundColor": "green", "mb": 2}):
                with mui.Grid(container=True, spacing=1):
                    with mui.Grid(item=True, xs=6):
                        mui.Typography("M(wt%):", color="white")
                    with mui.Grid(item=True, xs=6):
                        mui.TextField(
                            defaultValue=st.session_state.input_values["M"],
                            type="number",
                            inputProps={"step": 0.1, "min": 0, "max": 100},
                            onChange=lambda e: update_value("M", float(e["target"]["value"])),
                            fullWidth=True,
                            sx={
                                "& .MuiInputBase-input": {"color": "white"},
                                "& .MuiOutlinedInput-root": {
                                    "& fieldset": {"borderColor": "white"},
                                    "&:hover fieldset": {"borderColor": "white"},
                                    "&.Mui-focused fieldset": {"borderColor": "white"}
                                }
                            }
                        )
            
            with mui.Paper(elevation=3, sx={"p": 2, "backgroundColor": "green", "mb": 2}):
                with mui.Grid(container=True, spacing=1):
                    with mui.Grid(item=True, xs=6):
                        mui.Typography("Ash(wt%):", color="white")
                    with mui.Grid(item=True, xs=6):
                        mui.TextField(
                            defaultValue=st.session_state.input_values["Ash"],
                            type="number",
                            inputProps={"step": 0.1, "min": 0, "max": 100},
                            onChange=lambda e: update_value("Ash", float(e["target"]["value"])),
                            fullWidth=True,
                            sx={
                                "& .MuiInputBase-input": {"color": "white"},
                                "& .MuiOutlinedInput-root": {
                                    "& fieldset": {"borderColor": "white"},
                                    "&:hover fieldset": {"borderColor": "white"},
                                    "&.Mui-focused fieldset": {"borderColor": "white"}
                                }
                            }
                        )
            
            with mui.Paper(elevation=3, sx={"p": 2, "backgroundColor": "green", "mb": 2}):
                with mui.Grid(container=True, spacing=1):
                    with mui.Grid(item=True, xs=6):
                        mui.Typography("VM(wt%):", color="white")
                    with mui.Grid(item=True, xs=6):
                        mui.TextField(
                            defaultValue=st.session_state.input_values["VM"],
                            type="number",
                            inputProps={"step": 0.1, "min": 0, "max": 100},
                            onChange=lambda e: update_value("VM", float(e["target"]["value"])),
                            fullWidth=True,
                            sx={
                                "& .MuiInputBase-input": {"color": "white"},
                                "& .MuiOutlinedInput-root": {
                                    "& fieldset": {"borderColor": "white"},
                                    "&:hover fieldset": {"borderColor": "white"},
                                    "&.Mui-focused fieldset": {"borderColor": "white"}
                                }
                            }
                        )
            
            with mui.Paper(elevation=3, sx={"p": 2, "backgroundColor": "green", "mb": 2}):
                with mui.Grid(container=True, spacing=1):
                    with mui.Grid(item=True, xs=6):
                        mui.Typography("FC(wt%):", color="white")
                    with mui.Grid(item=True, xs=6):
                        mui.TextField(
                            defaultValue=st.session_state.input_values["FC"],
                            type="number",
                            inputProps={"step": 0.1, "min": 0, "max": 100},
                            onChange=lambda e: update_value("FC", float(e["target"]["value"])),
                            fullWidth=True,
                            sx={
                                "& .MuiInputBase-input": {"color": "white"},
                                "& .MuiOutlinedInput-root": {
                                    "& fieldset": {"borderColor": "white"},
                                    "&:hover fieldset": {"borderColor": "white"},
                                    "&.Mui-focused fieldset": {"borderColor": "white"}
                                }
                            }
                        )
        
        # 第二列：元素分析（Ultimate Analysis）
        with mui.Grid(item=True, xs=4):
            mui.Typography("元素分析", 
                           variant="h5", 
                           className="section-title ultimateAnalysis")
            
            with mui.Paper(elevation=3, sx={"p": 2, "backgroundColor": "yellow", "mb": 2}):
                with mui.Grid(container=True, spacing=1):
                    with mui.Grid(item=True, xs=6):
                        mui.Typography("C(wt%):", color="black")
                    with mui.Grid(item=True, xs=6):
                        mui.TextField(
                            defaultValue=st.session_state.input_values["C"],
                            type="number",
                            inputProps={"step": 0.1, "min": 0, "max": 100},
                            onChange=lambda e: update_value("C", float(e["target"]["value"])),
                            fullWidth=True,
                            sx={
                                "backgroundColor": "yellow",
                                "& .MuiInputBase-input": {"color": "black"},
                                "& .MuiOutlinedInput-root": {
                                    "& fieldset": {"borderColor": "black"},
                                    "&:hover fieldset": {"borderColor": "black"},
                                    "&.Mui-focused fieldset": {"borderColor": "black"}
                                }
                            }
                        )
            
            with mui.Paper(elevation=3, sx={"p": 2, "backgroundColor": "yellow", "mb": 2}):
                with mui.Grid(container=True, spacing=1):
                    with mui.Grid(item=True, xs=6):
                        mui.Typography("H(wt%):", color="black")
                    with mui.Grid(item=True, xs=6):
                        mui.TextField(
                            defaultValue=st.session_state.input_values["H"],
                            type="number",
                            inputProps={"step": 0.1, "min": 0, "max": 100},
                            onChange=lambda e: update_value("H", float(e["target"]["value"])),
                            fullWidth=True,
                            sx={
                                "backgroundColor": "yellow",
                                "& .MuiInputBase-input": {"color": "black"},
                                "& .MuiOutlinedInput-root": {
                                    "& fieldset": {"borderColor": "black"},
                                    "&:hover fieldset": {"borderColor": "black"},
                                    "&.Mui-focused fieldset": {"borderColor": "black"}
                                }
                            }
                        )
            
            with mui.Paper(elevation=3, sx={"p": 2, "backgroundColor": "yellow", "mb": 2}):
                with mui.Grid(container=True, spacing=1):
                    with mui.Grid(item=True, xs=6):
                        mui.Typography("O(wt%):", color="black")
                    with mui.Grid(item=True, xs=6):
                        mui.TextField(
                            defaultValue=st.session_state.input_values["O"],
                            type="number",
                            inputProps={"step": 0.1, "min": 0, "max": 100},
                            onChange=lambda e: update_value("O", float(e["target"]["value"])),
                            fullWidth=True,
                            sx={
                                "backgroundColor": "yellow",
                                "& .MuiInputBase-input": {"color": "black"},
                                "& .MuiOutlinedInput-root": {
                                    "& fieldset": {"borderColor": "black"},
                                    "&:hover fieldset": {"borderColor": "black"},
                                    "&.Mui-focused fieldset": {"borderColor": "black"}
                                }
                            }
                        )
            
            with mui.Paper(elevation=3, sx={"p": 2, "backgroundColor": "yellow", "mb": 2}):
                with mui.Grid(container=True, spacing=1):
                    with mui.Grid(item=True, xs=6):
                        mui.Typography("N(wt%):", color="black")
                    with mui.Grid(item=True, xs=6):
                        mui.TextField(
                            defaultValue=st.session_state.input_values["N"],
                            type="number",
                            inputProps={"step": 0.1, "min": 0, "max": 100},
                            onChange=lambda e: update_value("N", float(e["target"]["value"])),
                            fullWidth=True,
                            sx={
                                "backgroundColor": "yellow",
                                "& .MuiInputBase-input": {"color": "black"},
                                "& .MuiOutlinedInput-root": {
                                    "& fieldset": {"borderColor": "black"},
                                    "&:hover fieldset": {"borderColor": "black"},
                                    "&.Mui-focused fieldset": {"borderColor": "black"}
                                }
                            }
                        )
            
            with mui.Paper(elevation=3, sx={"p": 2, "backgroundColor": "yellow", "mb": 2}):
                with mui.Grid(container=True, spacing=1):
                    with mui.Grid(item=True, xs=6):
                        mui.Typography("S(wt%):", color="black")
                    with mui.Grid(item=True, xs=6):
                        mui.TextField(
                            defaultValue=st.session_state.input_values["S"],
                            type="number",
                            inputProps={"step": 0.1, "min": 0, "max": 100},
                            onChange=lambda e: update_value("S", float(e["target"]["value"])),
                            fullWidth=True,
                            sx={
                                "backgroundColor": "yellow",
                                "& .MuiInputBase-input": {"color": "black"},
                                "& .MuiOutlinedInput-root": {
                                    "& fieldset": {"borderColor": "black"},
                                    "&:hover fieldset": {"borderColor": "black"},
                                    "&.Mui-focused fieldset": {"borderColor": "black"}
                                }
                            }
                        )
        
        # 第三列：热解条件（Pyrolysis Conditions）
        with mui.Grid(item=True, xs=4):
            mui.Typography("热解条件", 
                           variant="h5", 
                           className="section-title pyrolysisConditions")
            
            with mui.Paper(elevation=3, sx={"p": 2, "backgroundColor": "orange", "mb": 2}):
                with mui.Grid(container=True, spacing=1):
                    with mui.Grid(item=True, xs=6):
                        mui.Typography("T(°C):", color="black")
                    with mui.Grid(item=True, xs=6):
                        mui.TextField(
                            defaultValue=st.session_state.input_values["T"],
                            type="number",
                            inputProps={"step": 1, "min": 0, "max": 1000},
                            onChange=lambda e: update_value("T", float(e["target"]["value"])),
                            fullWidth=True,
                            sx={
                                "backgroundColor": "orange",
                                "& .MuiInputBase-input": {"color": "black"},
                                "& .MuiOutlinedInput-root": {
                                    "& fieldset": {"borderColor": "black"},
                                    "&:hover fieldset": {"borderColor": "black"},
                                    "&.Mui-focused fieldset": {"borderColor": "black"}
                                }
                            }
                        )
            
            with mui.Paper(elevation=3, sx={"p": 2, "backgroundColor": "orange", "mb": 2}):
                with mui.Grid(container=True, spacing=1):
                    with mui.Grid(item=True, xs=6):
                        mui.Typography("HR(°C/min):", color="black")
                    with mui.Grid(item=True, xs=6):
                        mui.TextField(
                            defaultValue=st.session_state.input_values["HR"],
                            type="number",
                            inputProps={"step": 1, "min": 0, "max": 100},
                            onChange=lambda e: update_value("HR", float(e["target"]["value"])),
                            fullWidth=True,
                            sx={
                                "backgroundColor": "orange",
                                "& .MuiInputBase-input": {"color": "black"},
                                "& .MuiOutlinedInput-root": {
                                    "& fieldset": {"borderColor": "black"},
                                    "&:hover fieldset": {"borderColor": "black"},
                                    "&.Mui-focused fieldset": {"borderColor": "black"}
                                }
                            }
                        )
            
            with mui.Paper(elevation=3, sx={"p": 2, "backgroundColor": "orange", "mb": 2}):
                with mui.Grid(container=True, spacing=1):
                    with mui.Grid(item=True, xs=6):
                        mui.Typography("HT(min):", color="black")
                    with mui.Grid(item=True, xs=6):
                        mui.TextField(
                            defaultValue=st.session_state.input_values["HT"],
                            type="number",
                            inputProps={"step": 1, "min": 0, "max": 100},
                            onChange=lambda e: update_value("HT", float(e["target"]["value"])),
                            fullWidth=True,
                            sx={
                                "backgroundColor": "orange",
                                "& .MuiInputBase-input": {"color": "black"},
                                "& .MuiOutlinedInput-root": {
                                    "& fieldset": {"borderColor": "black"},
                                    "&:hover fieldset": {"borderColor": "black"},
                                    "&.Mui-focused fieldset": {"borderColor": "black"}
                                }
                            }
                        )
            
            with mui.Paper(elevation=3, sx={"p": 2, "backgroundColor": "orange", "mb": 2}):
                with mui.Grid(container=True, spacing=1):
                    with mui.Grid(item=True, xs=6):
                        mui.Typography("模型:", color="black")
                    with mui.Grid(item=True, xs=6):
                        mui.Select(
                            defaultValue=models[0],
                            sx={
                                "backgroundColor": "orange",
                                "color": "black",
                                "& .MuiOutlinedInput-root": {
                                    "& fieldset": {"borderColor": "black"},
                                    "&:hover fieldset": {"borderColor": "black"},
                                    "&.Mui-focused fieldset": {"borderColor": "black"}
                                }
                            }
                        ).children(
                            *[mui.MenuItem(value=model, children=model) for model in models]
                        )

# 按钮列
col1, col2 = st.columns(2)
with col1:
    predict_button = st.button("预测", key="predict")
with col2:
    clear_button = st.button("清除", key="clear")

# 预测逻辑
if predict_button:
    try:
        # 获取当前输入值
        input_values = st.session_state.input_values
        
        # 这里仅作为示例，实际应用应该使用真实的预测逻辑
        # 创建特征数组
        features = np.array([[
            input_values["M"], input_values["Ash"], input_values["VM"], input_values["FC"],
            input_values["C"], input_values["H"], input_values["O"], input_values["N"], input_values["S"],
            input_values["T"], input_values["HR"], input_values["HT"]
        ]])
        
        # 这里假设我们在进行模拟预测
        prediction_value = sum(input_values.values()) / 100  # 模拟计算
        st.session_state.prediction = prediction_value  # 假设的预测结果
        
    except Exception as e:
        st.error(f"预测时发生错误: {e}")

# 清除逻辑
if clear_button:
    st.session_state.input_values = {
        "M": 5.0, "Ash": 8.0, "VM": 75.0, "FC": 15.0,
        "C": 45.0, "H": 6.0, "O": 45.0, "N": 0.5, "S": 0.1,
        "T": 500, "HR": 10, "HT": 30
    }
    st.session_state.prediction = None
    st.rerun()

# 显示预测结果
if st.session_state.prediction is not None:
    st.markdown(
        f'<div class="result-container">预测产率 (%): {st.session_state.prediction:.2f}</div>',
        unsafe_allow_html=True
    )