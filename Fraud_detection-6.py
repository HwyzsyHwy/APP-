import numpy as np
import pandas as pd
import streamlit as st
import os
import joblib
import logging
import json
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger("Pyrolysis_Predictor")

class DirectPredictor:
    """
    直接预测器 - 无需依赖external模块，直接加载保存的模型进行预测
    """
    def __init__(self, models_dir=None):
        """
        初始化预测器
        
        参数:
            models_dir: 模型保存目录，默认查找常见位置
        """
        self.models_dir = self._find_model_directory() if models_dir is None else models_dir
        logger.info(f"使用模型目录: {self.models_dir}")
        
        self.models = []
        self.model_weights = None
        self.final_scaler = None
        self.feature_names = None
        self.metadata = None
        self.target_name = "Char Yield(%)"
        
        # 加载所有需要的模型组件
        try:
            self._load_all_components()
        except Exception as e:
            logger.error(f"加载模型组件时出错: {str(e)}")
            raise
    
    def _find_model_directory(self):
        """查找模型目录"""
        # 常见的模型目录位置
        common_locations = [
            "Char_Yield_Model",  # 当前目录
            "../Char_Yield_Model",  # 上一级目录
            "models/Char_Yield_Model",  # models子目录
            "C:/Users/HWY/Desktop/方-3/Char_Yield_Model"  # 绝对路径
        ]
        
        for location in common_locations:
            if os.path.exists(location) and os.path.isdir(location):
                return location
        
        # 如果找不到预设位置，尝试在当前目录下查找
        current_dir = os.getcwd()
        for item in os.listdir(current_dir):
            item_path = os.path.join(current_dir, item)
            if os.path.isdir(item_path) and "model" in item.lower():
                return item_path
        
        # 如果依然找不到，抛出错误
        raise FileNotFoundError("无法找到模型目录，请手动指定models_dir参数")
    
    def _load_all_components(self):
        """加载所有模型组件"""
        # 加载元数据
        metadata_path = os.path.join(self.models_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            self.feature_names = self.metadata.get('feature_names', None)
            logger.info(f"已加载元数据，特征列表: {self.feature_names}")
        else:
            # 如果找不到元数据，使用默认特征顺序
            self.feature_names = [
                'PT(°C)', 'RT(min)', 'HT(°C/min)', 
                'C(%)', 'H(%)', 'O(%)', 'N(%)',
                'Ash(%)', 'VM(%)', 'FC(%)'
            ]
            logger.warning(f"未找到元数据文件，使用默认特征顺序: {self.feature_names}")
        
        # 加载模型
        models_dir = os.path.join(self.models_dir, 'models')
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.startswith('model_') and f.endswith('.joblib')]
            model_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # 确保按顺序加载
            
            for model_file in model_files:
                model_path = os.path.join(models_dir, model_file)
                self.models.append(joblib.load(model_path))
            logger.info(f"已加载 {len(self.models)} 个模型")
        else:
            raise FileNotFoundError(f"模型目录不存在: {models_dir}")
        
        # 加载权重
        weights_path = os.path.join(self.models_dir, 'model_weights.npy')
        if os.path.exists(weights_path):
            self.model_weights = np.load(weights_path)
            logger.info(f"已加载模型权重，形状: {self.model_weights.shape}")
        else:
            # 如果找不到权重文件，使用平均权重
            self.model_weights = np.ones(len(self.models)) / len(self.models)
            logger.warning("未找到权重文件，使用平均权重")
        
        # 加载缩放器
        scaler_path = os.path.join(self.models_dir, 'final_scaler.joblib')
        if os.path.exists(scaler_path):
            self.final_scaler = joblib.load(scaler_path)
            logger.info("已加载标准化器")
        else:
            raise FileNotFoundError(f"标准化器文件不存在: {scaler_path}")
    
    def predict(self, data):
        """
        预测Char Yield(%)
        
        参数:
            data: DataFrame或字典，包含特征数据
            
        返回:
            预测的Char Yield(%)值
        """
        try:
            # 处理输入数据
            if isinstance(data, dict):
                # 如果是字典，转换为DataFrame
                data = pd.DataFrame([data])
            
            # 确保特征顺序正确
            if self.feature_names:
                # 检查是否缺少特征
                missing_features = set(self.feature_names) - set(data.columns)
                if missing_features:
                    error_msg = f"输入数据缺少以下特征: {missing_features}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                # 重排特征顺序 - 这是解决预测不准确的关键
                data = data[self.feature_names]
                logger.info(f"特征已按正确顺序排列: {self.feature_names}")
            
            # 标准化数据
            X_scaled = self.final_scaler.transform(data)
            logger.info(f"数据已标准化，形状: {X_scaled.shape}")
            
            # 使用所有模型进行预测
            all_predictions = np.zeros((data.shape[0], len(self.models)))
            for i, model in enumerate(self.models):
                all_predictions[:, i] = model.predict(X_scaled)
            
            # 计算加权平均
            weighted_pred = np.sum(all_predictions * self.model_weights.reshape(1, -1), axis=1)
            logger.info(f"预测结果: {weighted_pred}")
            
            return weighted_pred
            
        except Exception as e:
            logger.error(f"预测过程中出错: {str(e)}")
            raise

# 使用Streamlit创建Web应用
st.set_page_config(
    page_title="生物质热解产率预测系统",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 应用标题
st.title("Prediction of crop biomass pyrolysis yield based on CatBoost ensemble modeling")

# 创建三列布局
col1, col2, col3 = st.columns(3)

# 第一列: 元素分析
with col1:
    st.subheader("Ultimate Analysis")
    c_pct = st.number_input("C(%)", min_value=0.0, max_value=100.0, value=38.3, step=0.1)
    h_pct = st.number_input("H(%)", min_value=0.0, max_value=100.0, value=5.5, step=0.1)
    o_pct = st.number_input("O(%)", min_value=0.0, max_value=100.0, value=55.2, step=0.1)
    n_pct = st.number_input("N(%)", min_value=0.0, max_value=100.0, value=0.6, step=0.1)

# 第二列: 近似分析
with col2:
    st.subheader("Proximate Analysis")
    ash_pct = st.number_input("Ash(%)", min_value=0.0, max_value=100.0, value=6.6, step=0.1)
    vm_pct = st.number_input("VM(%)", min_value=0.0, max_value=100.0, value=81.1, step=0.1)
    fc_pct = st.number_input("FC(%)", min_value=0.0, max_value=100.0, value=10.3, step=0.1)

# 第三列: 热解条件
with col3:
    st.subheader("Pyrolysis Conditions")
    pt_c = st.number_input("PT(°C)", min_value=200.0, max_value=1000.0, value=500.0, step=1.0)
    ht_c_min = st.number_input("HT(°C/min)", min_value=1.0, max_value=100.0, value=10.0, step=1.0)
    rt_min = st.number_input("RT(min)", min_value=0.0, max_value=500.0, value=60.0, step=1.0)

# 预测按钮
col1, col2 = st.columns([5, 1])
with col2:
    predict_button = st.button("PUSH", type="primary")
    clear_button = st.button("CLEAR")

# 预测结果显示区域
result_container = st.container()

with result_container:
    # 预测结果
    st.subheader("Char Yield (wt%)")
    
    # 如果点击预测按钮
    if predict_button:
        try:
            # 准备输入数据 - 确保与训练时的特征顺序完全一致
            input_data = {
                'PT(°C)': pt_c,
                'RT(min)': rt_min,
                'HT(°C/min)': ht_c_min,
                'C(%)': c_pct,
                'H(%)': h_pct,
                'O(%)': o_pct,
                'N(%)': n_pct,
                'Ash(%)': ash_pct,
                'VM(%)': vm_pct,
                'FC(%)': fc_pct
            }
            
            # 记录输入数据
            logger.info(f"输入数据: {input_data}")
            
            # 创建预测器并预测
            predictor = DirectPredictor()
            result = predictor.predict(pd.DataFrame([input_data]))[0]
            
            # 显示结果
            st.header(f"{result:.2f}")
            
            # 记录调试信息
            logger.info(f"预测结果: {result:.2f}")
            
            # 添加到调试信息区域
            with st.expander("Debug Information"):
                st.write(f"输入参数: PT(°C): {pt_c}, H(%):{h_pct}, N(%):{n_pct}, Ash(%):{ash_pct}")
                st.write(f"O(%):{o_pct}, FC(%):{fc_pct}, RT(min):{rt_min}")
                st.write(f"VM(%):{vm_pct}, HT(°C/min):{ht_c_min}, C(%):{c_pct}")
                
                st.write("预测过程:")
                st.write(f"模型目录: {predictor.models_dir}")
                st.write(f"加载了 {len(predictor.models)} 个模型")
                st.write(f"特征顺序: {predictor.feature_names}")
                st.write(f"预测结果: {result:.2f}")
                
        except Exception as e:
            st.error(f"预测失败: {str(e)}")
            logger.error(f"预测失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

# 模型信息区域
with st.expander("About the Model"):
    st.write("This application uses a CatBoost ensemble model to predict char yield in biomass pyrolysis.")
    
    st.subheader("Key Factors Affecting Char Yield:")
    st.markdown("""
    * **Pyrolysis Temperature**: Higher temperature generally decreases char yield
    * **Residence Time**: Longer residence time generally increases char yield
    * **Biomass Composition**: Carbon content and ash content significantly affect the final yield
    """)
    
    st.write("The model was trained using 10-fold cross-validation with optimized hyperparameters, achieving high prediction accuracy (R² = 0.93, RMSE = 3.39 on test set).")

# 添加页脚
st.markdown("---")
st.caption("© 2023 Biomass Pyrolysis Research Team. All rights reserved.")