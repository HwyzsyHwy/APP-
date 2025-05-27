# -*- coding: utf-8 -*-
"""
Biomass Pyrolysis Yield Forecast using Multiple Ensemble Models
完全修复版本 - 支持Stacking和CatBoost+XGBoost模型混合使用
支持Char、Oil和Gas产率预测
从同一仓库直接加载模型文件
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import traceback
import matplotlib.pyplot as plt
from datetime import datetime
import requests
import tempfile

# 清除缓存，强制重新渲染
st.cache_data.clear()

# 页面设置
st.set_page_config(
    page_title='Biomass Pyrolysis Yield Prediction',
    page_icon='🔥',
    layout='wide',
    initial_sidebar_state='expanded'
)

# 自定义样式
st.markdown(
    """
    <style>
    /* 全局字体设置 */
    html, body, [class*="css"] {
        font-size: 16px !important;
    }
    
    /* 标题 */
    .main-title {
        text-align: center;
        font-size: 32px !important;
        font-weight: bold;
        margin-bottom: 20px;
        color: white !important;
    }
    
    /* 区域样式 */
    .section-header {
        color: white;
        font-weight: bold;
        font-size: 22px;
        text-align: center;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    
    /* 输入标签样式 */
    .input-label {
        padding: 5px;
        border-radius: 5px;
        margin-bottom: 5px;
        font-size: 18px;
        color: white;
    }
    
    /* 结果显示样式 */
    .yield-result {
        background-color: #1E1E1E;
        color: white;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        padding: 15px;
        border-radius: 8px;
        margin-top: 20px;
    }
    
    /* 强制应用白色背景到输入框 */
    [data-testid="stNumberInput"] input {
        background-color: white !important;
        color: black !important;
    }
    
    /* 增大按钮的字体 */
    .stButton button {
        font-size: 18px !important;
    }
    
    /* 警告样式 */
    .warning-box {
        background-color: rgba(255, 165, 0, 0.2);
        border-left: 5px solid orange;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    
    /* 错误样式 */
    .error-box {
        background-color: rgba(255, 0, 0, 0.2);
        border-left: 5px solid red;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    
    /* 成功样式 */
    .success-box {
        background-color: rgba(0, 128, 0, 0.2);
        border-left: 5px solid green;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
    }
    
    /* 日志样式 */
    .log-container {
        height: 300px;
        overflow-y: auto;
        background-color: #1E1E1E;
        color: #00FF00;
        font-family: 'Courier New', monospace;
        padding: 10px;
        border-radius: 5px;
        font-size: 14px !important;
    }
    
    /* 模型选择器样式 */
    .model-selector {
        background-color: #2E2E2E;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    
    /* 模型切换按钮组样式 */
    div[data-testid="stHorizontalBlock"] [data-testid="stButton"] {
        margin: 0 5px;
    }
    
    /* 填满屏幕 */
    .stApp {
        width: 100%;
        min-width: 100%;
        margin: 0 auto;
    }
    
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    
    /* 侧边栏模型信息样式 */
    .sidebar-model-info {
        background-color: #2E2E2E;
        padding: 10px;
        border-radius: 5px;
        margin-top: 20px;
    }
    
    /* 性能指标样式 */
    .performance-metrics {
        background-color: #2E2E2E;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }
    
    /* 技术说明样式 */
    .tech-info {
        background-color: #2E2E2E;
        padding: 15px;
        border-radius: 8px;
        margin-top: 20px;
    }
    
    /* 下载进度样式 */
    .download-progress {
        background-color: #2E2E2E;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 创建侧边栏日志区域
log_container = st.sidebar.container()
log_container.markdown("<h3>执行日志</h3>", unsafe_allow_html=True)
log_text = st.sidebar.empty()

# 初始化日志字符串
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []

def log(message):
    """记录日志到侧边栏和会话状态"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    st.session_state.log_messages.append(log_entry)
    # 只保留最近的100条日志
    if len(st.session_state.log_messages) > 100:
        st.session_state.log_messages = st.session_state.log_messages[-100:]
    
    # 更新日志显示
    log_text.markdown(
        f"<div class='log-container'>{'<br>'.join(st.session_state.log_messages)}</div>", 
        unsafe_allow_html=True
    )

# 记录启动日志
log("应用启动 - 同仓库模型加载版本")
log("支持Stacking和CatBoost+XGBoost模型混合使用")

# 初始化会话状态 - 添加模型选择功能
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Char Yield"  # 默认选择Char产率模型
    log(f"初始化选定模型: {st.session_state.selected_model}")

# 清除旧的缓存格式，重新初始化
if 'model_cache' not in st.session_state:
    st.session_state.model_cache = {}
    log("初始化模型缓存")
else:
    # 检查缓存格式，如果是旧格式则清除
    for key, value in list(st.session_state.model_cache.items()):
        if not isinstance(value, dict) or 'pipeline' not in value or 'type' not in value:
            log(f"清除旧格式缓存: {key}")
            del st.session_state.model_cache[key]
    
# 更新主标题以显示当前选定的模型
st.markdown("<h1 class='main-title'>基于集成模型的生物质热解产物预测系统</h1>", unsafe_allow_html=True)

# 添加模型选择区域 - 修改为三个按钮一排
st.markdown("<div class='model-selector'>", unsafe_allow_html=True)
st.markdown("<h3>选择预测目标</h3>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    char_button = st.button(" Char Yield", 
                           key="char_button", 
                           help="预测焦炭产率 (wt%)", 
                           use_container_width=True,
                           type="primary" if st.session_state.selected_model == "Char Yield" else "secondary")
with col2:
    oil_button = st.button(" Oil Yield", 
                          key="oil_button", 
                          help="预测生物油产率 (wt%)", 
                          use_container_width=True,
                          type="primary" if st.session_state.selected_model == "Oil Yield" else "secondary")
with col3:
    gas_button = st.button(" Gas Yield", 
                          key="gas_button", 
                          help="预测气体产率 (wt%)", 
                          use_container_width=True,
                          type="primary" if st.session_state.selected_model == "Gas Yield" else "secondary")

# 处理模型选择 - 修改为切换模型时不重置输入值
if char_button and st.session_state.selected_model != "Char Yield":
    st.session_state.selected_model = "Char Yield"
    st.session_state.prediction_result = None
    st.session_state.warnings = []
    log(f"切换到模型: {st.session_state.selected_model}")
    st.rerun()

if oil_button and st.session_state.selected_model != "Oil Yield":
    st.session_state.selected_model = "Oil Yield"
    st.session_state.prediction_result = None
    st.session_state.warnings = []
    log(f"切换到模型: {st.session_state.selected_model}")
    st.rerun()

if gas_button and st.session_state.selected_model != "Gas Yield":
    st.session_state.selected_model = "Gas Yield"
    st.session_state.prediction_result = None
    st.session_state.warnings = []
    log(f"切换到模型: {st.session_state.selected_model}")
    st.rerun()

st.markdown(f"<p style='text-align:center;'>当前模型: <b>{st.session_state.selected_model}</b></p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

class ModelPredictor:
    """完全修复的预测器类 - 支持从同一仓库加载模型文件"""
    
    def __init__(self, target_model="Char Yield"):
        self.target_name = target_model
        self.model_type = "Unknown"  # 初始化为Unknown，避免None
        self.pipeline = None
        self.model_loaded = False
        self.model_path = None
        
        # 定义正确的特征顺序（与训练时一致）
        self.feature_names = [
            'M(wt%)', 'Ash(wt%)', 'VM(wt%)', 'FC(wt%)', 
            'C(wt%)', 'H(wt%)', 'N(wt%)', 'O(wt%)', 
            'PS(mm)', 'SM(g)', 'FT(℃)', 'HR(℃/min)', 
            'FR(mL/min)', 'RT(min)'
        ]
        
        # 定义UI到模型的特征映射关系
        self.ui_to_model_mapping = {
            'FT(°C)': 'FT(℃)',        # UI上显示为°C，而模型使用℃
            'HR(°C/min)': 'HR(℃/min)'  # UI上显示为°C/min，而模型使用℃/min
        }
        
        # 反向映射，用于显示
        self.model_to_ui_mapping = {v: k for k, v in self.ui_to_model_mapping.items()}
        
        # 训练范围不变
        self.training_ranges = self._set_training_ranges()
        self.last_features = {}  # 存储上次的特征值
        self.last_result = None  # 存储上次的预测结果
        
        # 初始化模型
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化模型 - 先尝试本地加载，再尝试从同仓库下载"""
        log(f"初始化{self.target_name}模型")
        
        # 尝试从缓存加载
        if self._load_from_cache():
            log(f"从缓存成功加载{self.target_name}模型，类型: {self.model_type}")
            return
        
        # 尝试本地加载
        log(f"缓存中未找到{self.target_name}模型，尝试本地加载")
        self.model_path = self._find_local_model()
        if self.model_path and self._load_pipeline():
            self._save_to_cache()
            log(f"成功从本地加载并缓存{self.target_name}模型，类型: {self.model_type}")
            return
        
        # 本地没有，从同仓库下载
        log(f"本地未找到{self.target_name}模型，开始从同仓库下载")
        self.model_path = self._download_model_from_repo()
        if self.model_path:
            if self._load_pipeline():
                self._save_to_cache()
                log(f"成功下载并缓存{self.target_name}模型，类型: {self.model_type}")
            else:
                log(f"加载{self.target_name}模型失败")
        else:
            log(f"未能下载{self.target_name}模型文件")
    
    def _load_from_cache(self):
        """从缓存中加载模型"""
        if self.target_name in st.session_state.model_cache:
            cached_data = st.session_state.model_cache[self.target_name]
            if isinstance(cached_data, dict) and 'pipeline' in cached_data and 'type' in cached_data:
                self.pipeline = cached_data['pipeline']
                self.model_type = cached_data['type']
                self.model_loaded = True
                return True
        return False
    
    def _save_to_cache(self):
        """保存模型到缓存"""
        if self.pipeline is not None and self.model_type != "Unknown":
            st.session_state.model_cache[self.target_name] = {
                'pipeline': self.pipeline,
                'type': self.model_type
            }
            log(f"模型已保存到缓存: {self.target_name} ({self.model_type})")
    
    def _find_local_model(self):
        """查找本地模型文件"""
        # 为不同产率目标设置不同的模型文件名
        model_files = {
            "Char Yield": [
                "Stacking-CatBoost-XGBoost-Char-improved.joblib",
                "CatBoost-Char-improved.joblib", 
                "XGBoost-Char-improved.joblib",
                "Char-model.joblib"
            ],
            "Oil Yield": [
                "Stacking-CatBoost-XGBoost-Oil-improved.joblib",
                "CatBoost-Oil-improved.joblib",
                "XGBoost-Oil-improved.joblib", 
                "Oil-model.joblib"
            ],
            "Gas Yield": [
                "Stacking-CatBoost-XGBoost-Gas-improved.joblib",
                "CatBoost-Gas-improved.joblib",
                "XGBoost-Gas-improved.joblib",
                "Gas-model.joblib"
            ]
        }
        
        target_files = model_files.get(self.target_name, [])
        
        # 本地搜索路径
        local_paths = [
            ".",
            "./models",
            "./model_files",
            "../models",
            "../model_files"
        ]
        
        log(f"在本地搜索{self.target_name}模型文件...")
        
        for path in local_paths:
            if not os.path.exists(path):
                continue
                
            for filename in target_files:
                file_path = os.path.join(path, filename)
                if os.path.exists(file_path):
                    log(f"找到本地模型文件: {file_path}")
                    # 根据文件名识别模型类型
                    if 'stacking' in filename.lower():
                        self.model_type = "Stacking"
                    elif 'catboost' in filename.lower():
                        self.model_type = "CatBoost"
                    elif 'xgboost' in filename.lower():
                        self.model_type = "XGBoost"
                    else:
                        self.model_type = "CatBoost"  # 默认
                    return file_path
        
        log("本地未找到模型文件")
        return None
    
    def _get_repo_info(self):
        """获取当前仓库信息"""
        # 尝试从环境变量获取仓库信息（适用于GitHub Actions等CI环境）
        repo_owner = os.environ.get('GITHUB_REPOSITORY_OWNER')
        repo_name = os.environ.get('GITHUB_REPOSITORY')
        
        if repo_name and '/' in repo_name:
            repo_owner, repo_name = repo_name.split('/', 1)
        
        # 如果环境变量不可用，使用默认值或从Streamlit secrets获取
        if not repo_owner or not repo_name:
            try:
                repo_owner = st.secrets.get("GITHUB_OWNER", "your-username")
                repo_name = st.secrets.get("GITHUB_REPO", "your-repo-name")
            except:
                # 如果secrets也不可用，使用占位符
                repo_owner = "your-username"
                repo_name = "your-repo-name"
        
        return repo_owner, repo_name
    
    def _download_model_from_repo(self):
        """从同一仓库下载模型文件"""
        repo_owner, repo_name = self._get_repo_info()
        
        # 构建GitHub raw URL
        base_url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/main"
        
        # 可能的模型文件路径
        model_paths = [
            "models",
            "model_files", 
            ".",
            "data/models"
        ]
        
        # 为不同产率目标设置不同的模型文件名
        model_files = {
            "Char Yield": [
                "Stacking-CatBoost-XGBoost-Char-improved.joblib",
                "CatBoost-Char-improved.joblib", 
                "XGBoost-Char-improved.joblib",
                "Char-model.joblib"
            ],
            "Oil Yield": [
                "Stacking-CatBoost-XGBoost-Oil-improved.joblib",
                "CatBoost-Oil-improved.joblib",
                "XGBoost-Oil-improved.joblib", 
                "Oil-model.joblib"
            ],
            "Gas Yield": [
                "Stacking-CatBoost-XGBoost-Gas-improved.joblib",
                "CatBoost-Gas-improved.joblib",
                "XGBoost-Gas-improved.joblib",
                "Gas-model.joblib"
            ]
        }
        
        target_files = model_files.get(self.target_name, [])
        
        log(f"从仓库 {repo_owner}/{repo_name} 下载{self.target_name}模型文件...")
        
        # 显示下载进度
        progress_placeholder = st.empty()
        
        # 尝试不同路径和文件名组合
        for model_path in model_paths:
            for filename in target_files:
                progress_placeholder.markdown(
                    f"<div class='download-progress'>正在尝试下载: {model_path}/{filename}</div>", 
                    unsafe_allow_html=True
                )
                
                if model_path == ".":
                    file_url = f"{base_url}/{filename}"
                else:
                    file_url = f"{base_url}/{model_path}/{filename}"
                
                downloaded_path = self._download_file(file_url, filename)
                if downloaded_path:
                    progress_placeholder.empty()
                    return downloaded_path
        
        progress_placeholder.markdown(
            "<div class='error-box'>模型下载失败，请确保模型文件存在于仓库中</div>", 
            unsafe_allow_html=True
        )
        
        return None
    
    def _download_file(self, file_url, filename):
        """下载单个文件"""
        try:
            log(f"尝试下载: {file_url}")
            
            # 发送HTTP请求下载文件
            response = requests.get(file_url, timeout=60, stream=True)
            
            if response.status_code == 200:
                # 创建临时文件保存模型
                temp_dir = tempfile.gettempdir()
                local_path = os.path.join(temp_dir, f"biomass_model_{filename}")
                
                # 写入文件，显示下载进度
                total_size = int(response.headers.get('content-length', 0))
                downloaded_size = 0
                
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            
                            # 更新下载进度
                            if total_size > 0:
                                progress = (downloaded_size / total_size) * 100
                                if downloaded_size % (8192 * 10) == 0:  # 每下载约80KB记录一次
                                    log(f"下载进度: {progress:.1f}%")
                
                log(f"成功下载模型文件: {filename} ({downloaded_size} bytes)")
                
                # 根据文件名识别模型类型
                if 'stacking' in filename.lower():
                    self.model_type = "Stacking"
                elif 'catboost' in filename.lower():
                    self.model_type = "CatBoost"
                elif 'xgboost' in filename.lower():
                    self.model_type = "XGBoost"
                else:
                    self.model_type = "CatBoost"  # 默认
                
                return local_path
                
            else:
                log(f"文件不存在: {file_url} (状态码: {response.status_code})")
                return None
                
        except requests.exceptions.Timeout:
            log(f"下载{filename}超时")
            return None
        except requests.exceptions.ConnectionError:
            log(f"下载{filename}时网络连接错误")
            return None
        except requests.exceptions.RequestException as e:
            log(f"下载{filename}时网络错误: {str(e)}")
            return None
        except Exception as e:
            log(f"下载{filename}时出错: {str(e)}")
            return None
    
    def _load_pipeline(self):
        """加载Pipeline模型 - 自动识别模型类型"""
        if not self.model_path:
            log("模型路径为空，无法加载")
            return False
        
        try:
            log(f"加载Pipeline模型: {self.model_path}")
            self.pipeline = joblib.load(self.model_path)
            
            # 验证是否能进行预测
            if hasattr(self.pipeline, 'predict'):
                log(f"模型加载成功，类型: {type(self.pipeline).__name__}")
                
                # 重新识别模型类型（更准确）
                identified_type = self._identify_model_type()
                if identified_type != "Unknown":
                    self.model_type = identified_type
                    log(f"重新识别的模型类型: {self.model_type}")
                
                self.model_loaded = True
                
                # 尝试识别Pipeline的组件
                if hasattr(self.pipeline, 'named_steps'):
                    components = list(self.pipeline.named_steps.keys())
                    log(f"Pipeline组件: {', '.join(components)}")
                    
                    # 检查是否为Stacking模型
                    if 'stacking' in components:
                        log("检测到Stacking模型组件")
                        self.model_type = "Stacking"
                    elif 'model' in components:
                        # 尝试识别具体的模型类型
                        model_component = self.pipeline.named_steps['model']
                        model_class_name = type(model_component).__name__
                        log(f"检测到模型组件: {model_class_name}")
                        if 'GBDT' in model_class_name or 'GradientBoosting' in model_class_name:
                            self.model_type = "GBDT"
                        elif 'XGB' in model_class_name:
                            self.model_type = "XGBoost"
                        elif 'CatBoost' in model_class_name:
                            self.model_type = "CatBoost"
                        elif 'RandomForest' in model_class_name:
                            self.model_type = "RandomForest"
                
                log(f"最终确定的模型类型: {self.model_type}")
                
                # 如果是下载的临时文件，清理它
                if self.model_path.startswith(tempfile.gettempdir()):
                    try:
                        os.remove(self.model_path)
                        log("已清理临时模型文件")
                    except:
                        pass
                
                return True
            else:
                log("加载的对象没有predict方法，不能用于预测")
                self.model_loaded = False
                return False
                
        except Exception as e:
            log(f"加载模型出错: {str(e)}")
            log(traceback.format_exc())
            self.model_loaded = False
            return False
    
    def _identify_model_type(self):
        """自动识别模型类型"""
        if not self.pipeline:
            return "Unknown"
        
        try:
            if hasattr(self.pipeline, 'named_steps'):
                if 'stacking' in self.pipeline.named_steps:
                    return "Stacking"
                elif 'model' in self.pipeline.named_steps:
                    model = self.pipeline.named_steps['model']
                    model_name = type(model).__name__
                    if 'GradientBoosting' in model_name:
                        return "GBDT"
                    elif 'XGB' in model_name:
                        return "XGBoost"
                    elif 'CatBoost' in model_name:
                        return "CatBoost"
                    elif 'RandomForest' in model_name:
                        return "RandomForest"
            
            # 如果是直接的模型对象
            model_name = type(self.pipeline).__name__
            if 'Stacking' in model_name:
                return "Stacking"
            elif 'GradientBoosting' in model_name:
                return "GBDT"
            elif 'XGB' in model_name:
                return "XGBoost"
            elif 'CatBoost' in model_name:
                return "CatBoost"
            elif 'RandomForest' in model_name:
                return "RandomForest"
                
        except Exception as e:
            log(f"识别模型类型时出错: {str(e)}")
        
        return "Unknown"
    
    def _set_training_ranges(self):
        """设置训练数据的范围 - 使用正确的特征名称"""
        ranges = {
            'M(wt%)': {'min': 2.750, 'max': 12.640},
            'Ash(wt%)': {'min': 0.780, 'max': 29.510},
            'VM(wt%)': {'min': 51.640, 'max': 89.500},
            'FC(wt%)': {'min': 0.100, 'max': 23.900},
            'C(wt%)': {'min': 22.490, 'max': 53.300},
            'H(wt%)': {'min': 3.303, 'max': 8.200},
            'N(wt%)': {'min': 0.170, 'max': 4.870},
            'O(wt%)': {'min': 34.000, 'max': 73.697},
            'PS(mm)': {'min': 0.075, 'max': 10.000},
            'SM(g)': {'min': 3.000, 'max': 125.000},
            'FR(mL/min)': {'min': 0.000, 'max': 600.000},
            'RT(min)': {'min': 15.000, 'max': 90.000}
        }
        
        # 添加映射后的特征范围
        ranges['FT(℃)'] = {'min': 250.000, 'max': 900.000}
        ranges['HR(℃/min)'] = {'min': 1.000, 'max': 100.000}
        
        # 为UI特征也添加相同的范围
        ranges['FT(°C)'] = ranges['FT(℃)']
        ranges['HR(°C/min)'] = ranges['HR(℃/min)']
        
        return ranges
    
    def check_input_range(self, features):
        """检查输入值是否在训练数据范围内"""
        warnings = []
        
        for feature, value in features.items():
            # 获取映射后的特征名
            mapped_feature = self.ui_to_model_mapping.get(feature, feature)
            range_info = self.training_ranges.get(mapped_feature)
            
            if range_info:
                if value < range_info['min'] or value > range_info['max']:
                    warning = f"{feature}: {value:.2f} (超出训练范围 {range_info['min']:.2f} - {range_info['max']:.2f})"
                    warnings.append(warning)
                    log(f"警告: {warning}")
        
        return warnings
    
    def _prepare_features(self, features):
        """准备特征，处理特征名称映射和顺序"""
        # 创建一个空的DataFrame，所有特征初始化为0
        model_features = {feature: 0.0 for feature in self.feature_names}
        
        # 首先将UI特征映射到模型特征名称
        for ui_feature, value in features.items():
            model_feature = self.ui_to_model_mapping.get(ui_feature, ui_feature)
            if model_feature in self.feature_names:
                model_features[model_feature] = value
                if ui_feature != model_feature:
                    log(f"特征映射: '{ui_feature}' -> '{model_feature}'")
        
        # 创建DataFrame并按照正确顺序排列列
        df = pd.DataFrame([model_features])
        
        # 确保列顺序与训练时一致
        df = df[self.feature_names]
        
        log(f"准备好的特征，列顺序: {list(df.columns)}")
        return df
    
    def predict(self, features):
        """预测方法 - 支持多种模型类型"""
        # 检查输入是否有变化
        features_changed = False
        if self.last_features:
            for feature, value in features.items():
                if feature in self.last_features and abs(self.last_features[feature] - value) > 0.001:
                    features_changed = True
                    break
        else:
            # 第一次预测
            features_changed = True
        
        # 如果输入没有变化且有上次结果，直接返回上次结果
        if not features_changed and self.last_result is not None:
            log("输入未变化，使用上次的预测结果")
            return self.last_result
        
        # 保存当前特征
        self.last_features = features.copy()
        
        # 准备特征数据
        log(f"开始准备{len(features)}个特征数据")
        features_df = self._prepare_features(features)
        
        # 尝试使用Pipeline进行预测
        if self.model_loaded and self.pipeline is not None:
            try:
                log(f"使用{self.model_type} Pipeline模型预测")
                # 直接使用Pipeline进行预测，包含所有预处理步骤
                result = float(self.pipeline.predict(features_df)[0])
                log(f"{self.model_type} Pipeline预测结果: {result:.2f}")
                self.last_result = result
                return result
            except Exception as e:
                log(f"Pipeline预测失败: {str(e)}")
                log(traceback.format_exc())
                # 如果预测失败，尝试重新初始化模型
                log("尝试重新初始化模型...")
                self._initialize_model()
                if self.model_loaded:
                    try:
                        # 再次尝试预测
                        result = float(self.pipeline.predict(features_df)[0])
                        log(f"重新初始化后预测结果: {result:.2f}")
                        self.last_result = result
                        return result
                    except Exception as new_e:
                        log(f"重新初始化后预测仍然失败: {str(new_e)}")
        
        # 如果到这里，说明预测失败，返回错误提示
        log("所有预测尝试都失败，请检查模型文件")
        raise ValueError("模型预测失败。请确保模型文件存在于仓库中。")
    
    def get_model_info(self):
        """获取模型信息摘要 - 支持多种模型类型"""
        # 根据实际模型类型设置描述 - 更新为CatBoost + XGBoost
        model_descriptions = {
            "Stacking": "Stacking集成模型 (CatBoost + XGBoost)",
            "GBDT": "梯度提升决策树模型",
            "XGBoost": "XGBoost梯度提升模型",
            "CatBoost": "CatBoost梯度提升模型",
            "RandomForest": "随机森林模型",
            "Unknown": "未知模型类型"
        }
        
        model_type_desc = model_descriptions.get(self.model_type, f"{self.model_type}模型")
        
        info = {
            "模型类型": model_type_desc,
            "目标变量": self.target_name,
            "特征数量": len(self.feature_names),
            "模型状态": "已加载" if self.model_loaded else "未加载",
            "数据源": "同仓库加载"
        }
        
        if self.model_loaded and self.pipeline is not None:
            try:
                if hasattr(self.pipeline, 'named_steps'):
                    pipeline_steps = list(self.pipeline.named_steps.keys())
                    info["Pipeline组件"] = ", ".join(pipeline_steps)
                    
                    # 根据模型类型显示不同的信息
                    if self.model_type == "Stacking" and 'stacking' in self.pipeline.named_steps:
                        stacking_model = self.pipeline.named_steps['stacking']
                        info["集成方法"] = "StackingRegressor"
                        
                        # 安全地显示基学习器信息
                        base_learners = []
                        try:
                            estimators_info = None
                            
                            if hasattr(stacking_model, 'estimators_') and stacking_model.estimators_ is not None:
                                estimators_info = stacking_model.estimators_
                                source = "estimators_"
                            elif hasattr(stacking_model, 'estimators') and stacking_model.estimators is not None:
                                estimators_info = stacking_model.estimators
                                source = "estimators"
                            
                            if estimators_info is not None:
                                log(f"获取基学习器信息来源: {source}, 类型: {type(estimators_info)}")
                                
                                if isinstance(estimators_info, (list, tuple)):
                                    for i, item in enumerate(estimators_info):
                                        try:
                                            if hasattr(item, 'fit') and hasattr(item, 'predict'):
                                                base_learners.append(f"估计器{i+1}: {type(item).__name__}")
                                            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                                                name, estimator = item[0], item[1]
                                                base_learners.append(f"{name}: {type(estimator).__name__}")
                                            elif isinstance(item, (list, tuple)) and len(item) == 1:
                                                base_learners.append(f"估计器{i+1}: {type(item[0]).__name__}")
                                            else:
                                                base_learners.append(f"估计器{i+1}: {type(item).__name__}")
                                        except Exception as item_error:
                                            base_learners.append(f"估计器{i+1}: 解析错误")
                                            log(f"解析估计器{i+1}时出错: {str(item_error)}")
                                
                                if base_learners:
                                    info["基学习器"] = ", ".join(base_learners)
                                else:
                                    info["基学习器"] = "未能解析基学习器信息"
                            else:
                                info["基学习器"] = "未找到基学习器信息"
                                
                        except Exception as e:
                            info["基学习器"] = f"获取失败: {str(e)}"
                            log(f"获取基学习器信息时出错: {str(e)}")
                        
                        # 安全地显示元学习器信息
                        try:
                            if hasattr(stacking_model, 'final_estimator_') and stacking_model.final_estimator_ is not None:
                                meta_learner = type(stacking_model.final_estimator_).__name__
                                info["元学习器"] = meta_learner
                            elif hasattr(stacking_model, 'final_estimator') and stacking_model.final_estimator is not None:
                                meta_learner = type(stacking_model.final_estimator).__name__
                                info["元学习器配置"] = meta_learner
                            else:
                                info["元学习器"] = "未找到元学习器信息"
                        except Exception as e:
                            info["元学习器"] = f"获取失败: {str(e)}"
                            log(f"获取元学习器信息时出错: {str(e)}")
                    
                    elif 'model' in self.pipeline.named_steps:
                        # 单一模型的情况
                        model_component = self.pipeline.named_steps['model']
                        info["算法类型"] = type(model_component).__name__
                        
                        # 尝试获取模型参数信息
                        try:
                            if hasattr(model_component, 'get_params'):
                                params = model_component.get_params()
                                key_params = {}
                                
                                # 根据模型类型提取关键参数
                                if self.model_type == "GBDT":
                                    for param in ['n_estimators', 'learning_rate', 'max_depth']:
                                        if param in params:
                                            key_params[param] = params[param]
                                elif self.model_type in ["XGBoost", "CatBoost"]:
                                    for param in ['n_estimators', 'learning_rate', 'max_depth']:
                                        if param in params:
                                            key_params[param] = params[param]
                                elif self.model_type == "RandomForest":
                                    for param in ['n_estimators', 'max_depth', 'max_features']:
                                        if param in params:
                                            key_params[param] = params[param]
                                
                                if key_params:
                                    info["关键参数"] = ", ".join([f"{k}={v}" for k, v in key_params.items()])
                        except Exception as e:
                            log(f"获取模型参数时出错: {str(e)}")
                            
            except Exception as e:
                info["错误"] = f"获取模型信息时出错: {str(e)}"
                log(f"获取模型信息时出错: {str(e)}")
                
        return info

# 初始化预测器 - 使用当前选择的模型
predictor = ModelPredictor(target_model=st.session_state.selected_model)

# 在侧边栏添加模型信息
model_info = predictor.get_model_info()
model_info_html = "<div class='sidebar-model-info'><h3>关于模型</h3>"
for key, value in model_info.items():
    model_info_html += f"<p><b>{key}</b>: {value}</p>"

model_info_html += "</div>"
st.sidebar.markdown(model_info_html, unsafe_allow_html=True)

# 添加仓库配置说明
repo_owner, repo_name = predictor._get_repo_info()
st.sidebar.markdown(f"""
<div class='sidebar-model-info'>
<h3>仓库信息</h3>
<p><b>仓库:</b> {repo_owner}/{repo_name}</p>
<p><b>加载策略:</b> 本地优先，远程备用</p>
<p><b>支持路径:</b> models/, model_files/, 根目录</p>
</div>
""", unsafe_allow_html=True)

# 其余代码保持不变...
# [这里包含所有的UI代码，与之前相同]

# 初始化会话状态
if 'clear_pressed' not in st.session_state:
    st.session_state.clear_pressed = False
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'warnings' not in st.session_state:
    st.session_state.warnings = []
if 'prediction_error' not in st.session_state:
    st.session_state.prediction_error = None
if 'feature_values' not in st.session_state:
    # 初始化存储所有特征输入值的字典
    st.session_state.feature_values = {}

# 定义默认值 - 从图表中提取均值作为默认值
default_values = {
    "M(wt%)": 6.57,
    "Ash(wt%)": 5.87,
    "VM(wt%)": 74.22,
    "FC(wt%)": 13.32,
    "C(wt%)": 45.12,
    "H(wt%)": 5.95,
    "N(wt%)": 1.50,
    "O(wt%)": 47.40,
    "PS(mm)": 1.23,
    "SM(g)": 27.03,
    "FT(°C)": 505.24,
    "HR(°C/min)": 27.81,
    "FR(mL/min)": 87.42,
    "RT(min)": 36.88
}

# 特征分类
feature_categories = {
    "Proximate Analysis": ["M(wt%)", "Ash(wt%)", "VM(wt%)", "FC(wt%)"],
    "Ultimate Analysis": ["C(wt%)", "H(wt%)", "N(wt%)", "O(wt%)"],
    "Pyrolysis Conditions": ["PS(mm)", "SM(g)", "FT(°C)", "HR(°C/min)", "FR(mL/min)", "RT(min)"]
}

# 颜色配置
category_colors = {
    "Ultimate Analysis": "#501d8a",  
    "Proximate Analysis": "#1c8041",  
    "Pyrolysis Conditions": "#e55709" 
}

# 创建三列布局
col1, col2, col3 = st.columns(3)

# 使用字典存储所有输入值
features = {}

# Proximate Analysis - 第一列
with col1:
    category = "Proximate Analysis"
    color = category_colors[category]
    st.markdown(f"<div class='section-header' style='background-color: {color};'>{category}</div>", unsafe_allow_html=True)
    
    for feature in feature_categories[category]:
        if st.session_state.clear_pressed:
            value = default_values[feature]
        else:
            # 先从会话状态获取值，如果不存在则使用默认值
            value = st.session_state.feature_values.get(feature, default_values[feature])
        
        col_a, col_b = st.columns([1, 0.5])
        with col_a:
            st.markdown(f"<div class='input-label' style='background-color: {color};'>{feature}</div>", unsafe_allow_html=True)
        with col_b:
            # 不再限制输入范围，但仍然使用默认值
            features[feature] = st.number_input(
                "", 
                value=float(value), 
                step=0.01,
                key=f"{category}_{feature}",
                format="%.2f",
                label_visibility="collapsed"
            )

# Ultimate Analysis - 第二列
with col2:
    category = "Ultimate Analysis"
    color = category_colors[category]
    st.markdown(f"<div class='section-header' style='background-color: {color};'>{category}</div>", unsafe_allow_html=True)
    
    for feature in feature_categories[category]:
        if st.session_state.clear_pressed:
            value = default_values[feature]
        else:
            value = st.session_state.feature_values.get(feature, default_values[feature])
        
        col_a, col_b = st.columns([1, 0.5])
        with col_a:
            st.markdown(f"<div class='input-label' style='background-color: {color};'>{feature}</div>", unsafe_allow_html=True)
        with col_b:
            # 不再限制输入范围
            features[feature] = st.number_input(
                "", 
                value=float(value), 
                step=0.01,
                key=f"{category}_{feature}",
                format="%.2f",
                label_visibility="collapsed"
            )

# Pyrolysis Conditions - 第三列
with col3:
    category = "Pyrolysis Conditions"
    color = category_colors[category]
    st.markdown(f"<div class='section-header' style='background-color: {color};'>{category}</div>", unsafe_allow_html=True)
    
    for feature in feature_categories[category]:
        if st.session_state.clear_pressed:
            value = default_values[feature]
        else:
            value = st.session_state.feature_values.get(feature, default_values[feature])
        
        col_a, col_b = st.columns([1, 0.5])
        with col_a:
            st.markdown(f"<div class='input-label' style='background-color: {color};'>{feature}</div>", unsafe_allow_html=True)
        with col_b:
            # 不再限制输入范围
            features[feature] = st.number_input(
                "", 
                value=float(value), 
                step=0.01,
                key=f"{category}_{feature}",
                format="%.2f",
                label_visibility="collapsed"
            )

# 调试信息：显示所有当前输入值
with st.expander("显示当前输入值", expanded=False):
    debug_info = "<ul style='columns: 3;'>"
    for feature, value in features.items():
        debug_info += f"<li>{feature}: {value:.2f}</li>"
    debug_info += "</ul>"
    st.markdown(debug_info, unsafe_allow_html=True)

# 重置状态
if st.session_state.clear_pressed:
    st.session_state.feature_values = {}
    st.session_state.clear_pressed = False

# 预测结果显示区域
result_container = st.container()

# 预测按钮区域
col1, col2 = st.columns([1, 1])

with col1:
    predict_clicked = st.button("🔮 运行预测", use_container_width=True, type="primary")
    if predict_clicked:
        log("开始预测，获取最新输入值...")
        
        # 切换模型后需要重新初始化预测器
        if predictor.target_name != st.session_state.selected_model:
            log(f"检测到模型变更，重新初始化预测器: {st.session_state.selected_model}")
            predictor = ModelPredictor(target_model=st.session_state.selected_model)
        
        # 保存当前输入到会话状态
        st.session_state.feature_values = features.copy()
        
        log(f"开始{st.session_state.selected_model}预测，输入特征数: {len(features)}")
        
        # 检查输入范围
        warnings = predictor.check_input_range(features)
        st.session_state.warnings = warnings
        
        # 计算FC(wt%)是否满足FC(wt%) = 100 - Ash(wt%) - VM(wt%)的约束
        calculated_fc = 100 - features['Ash(wt%)'] - features['VM(wt%)']
        if abs(calculated_fc - features['FC(wt%)']) > 0.5:  # 允许0.5%的误差
            st.session_state.warnings.append(
                f"FC(wt%)值 ({features['FC(wt%)']:.2f}) 与计算值 (100 - Ash - VM = {calculated_fc:.2f}) 不符，这可能影响预测准确性。"
            )
            log(f"警告: FC(wt%)值与计算值不符: {features['FC(wt%)']:.2f} vs {calculated_fc:.2f}")
        
        # 执行预测
        try:
            # 确保预测器已正确初始化
            if not predictor.model_loaded:
                log("模型未加载，尝试重新初始化")
                predictor._initialize_model()
                if not predictor.model_loaded:
                    st.error("无法加载模型。请确保模型文件存在于仓库中。")
                    st.session_state.prediction_error = "模型加载失败"
                    st.rerun()
            
            result = predictor.predict(features)
            if result is not None:
                st.session_state.prediction_result = float(result)
                log(f"预测成功: {st.session_state.prediction_result:.2f}")
                st.session_state.prediction_error = None
            else:
                log("警告: 预测结果为空")
                st.session_state.prediction_error = "预测结果为空"
        except Exception as e:
            st.session_state.prediction_error = str(e)
            log(f"预测错误: {str(e)}")
            log(traceback.format_exc())
            st.error(f"预测过程中发生错误: {str(e)}")

with col2:
    if st.button("🔄 重置输入", use_container_width=True):
        log("重置所有输入值")
        st.session_state.clear_pressed = True
        st.session_state.prediction_result = None
        st.session_state.warnings = []
        st.session_state.prediction_error = None
        st.rerun()

# 显示预测结果
if st.session_state.prediction_result is not None:
    st.markdown("---")
    
    # 显示主预测结果
    result_container.markdown(f"<div class='yield-result'>{st.session_state.selected_model}: {st.session_state.prediction_result:.2f} wt%</div>", unsafe_allow_html=True)
    
    # 显示模型信息
    if not predictor.model_loaded:
        result_container.markdown(
            "<div class='error-box'><b>⚠️ 错误：</b> 模型未成功加载，无法执行预测。请检查模型文件是否存在于仓库中。</div>", 
            unsafe_allow_html=True
        )
    
    # 显示警告
    if st.session_state.warnings:
        warnings_html = "<div class='warning-box'><b>⚠️ 警告：部分输入可能影响预测精度</b><ul>"
        for warning in st.session_state.warnings:
            warnings_html += f"<li>{warning}</li>"
        warnings_html += "</ul><p>请根据提示调整输入值以获得更准确的预测。</p></div>"
        result_container.markdown(warnings_html, unsafe_allow_html=True)
    
    # 显示预测信息
    with st.expander("预测信息", expanded=False):
        st.markdown(f"""
        - **目标变量:** {st.session_state.selected_model}
        - **预测结果:** {st.session_state.prediction_result:.2f} wt%
        - **使用模型:** {predictor.model_type} Pipeline模型
        - **数据源:** 同仓库加载
        """)
    
    # 技术说明部分 - 根据模型类型动态调整，更新为CatBoost + XGBoost
    with st.expander("技术说明", expanded=False):
        if predictor.model_type == "Stacking":
            tech_content = """
            <div class='tech-info'>
            <p>本模型基于Stacking集成算法创建，结合了CatBoost和XGBoost两种强大的机器学习算法，预测生物质热解产物分布。模型使用生物质的元素分析、近似分析数据和热解条件作为输入，计算热解炭、热解油和热解气体产量。</p>
            
            <p><b>模型架构：</b></p>
            <ul>
                <li><b>基学习器1：</b> CatBoost - 梯度提升算法，对类别特征处理优秀，具有强大的特征学习能力</li>
                <li><b>基学习器2：</b> XGBoost - 极端梯度提升算法，在结构化数据上表现卓越</li>
                <li><b>元学习器：</b> Ridge回归 - 线性回归器，用于组合基学习器的预测结果</li>
                <li><b>数据预处理：</b> RobustScaler - 对异常值不敏感的标准化器</li>
            </ul>
            
            <p><b>部署特点：</b></p>
            <ul>
                <li><b>智能加载：</b> 优先使用本地模型文件，本地不存在时自动从同仓库下载</li>
                <li><b>智能缓存：</b> 首次加载后本地缓存，提高后续预测速度</li>
                <li><b>多路径支持：</b> 支持models/、model_files/、根目录等多种文件组织方式</li>
            </ul>
            
            <p><b>特别提醒：</b></p>
            <ul>
                <li>输入参数建议在训练数据的分布范围内，以保证软件的预测精度</li>
                <li>由于模型训练时FC(wt%)通过100-Ash(wt%)-VM(wt%)公式转换得出，所以用户使用此软件进行预测时也建议使用此公式对FC(wt%)进行计算</li>
                <li>所有特征的训练范围都基于真实训练数据的统计信息，如输入超出范围将会收到提示</li>
                <li>Stacking模型通过交叉验证训练，有效防止过拟合，提高泛化能力</li>
                <li>首次使用时如果本地无模型文件，需要网络连接从仓库下载</li>
            </ul>
            </div>
            """
        else:
            tech_content = f"""
            <div class='tech-info'>
            <p>本模型基于{predictor.model_type}算法创建，用于预测生物质热解产物分布。模型使用生物质的元素分析、近似分析数据和热解条件作为输入，计算热解炭、热解油和热解气体产量。</p>
            
            <p><b>模型特点：</b></p>
            <ul>
                <li><b>算法类型：</b> {predictor.model_type}</li>
                <li><b>数据预处理：</b> RobustScaler - 对异常值不敏感的标准化器</li>
                <li><b>特征工程：</b> 14个关键特征，包括元素分析、近似分析和热解条件</li>
                <li><b>智能加载：</b> 本地优先，远程备用的加载策略</li>
            </ul>
            
            <p><b>特别提醒：</b></p>
            <ul>
                <li>输入参数建议在训练数据的分布范围内，以保证软件的预测精度</li>
                <li>由于模型训练时FC(wt%)通过100-Ash(wt%)-VM(wt%)公式转换得出，所以用户使用此软件进行预测时也建议使用此公式对FC(wt%)进行计算</li>
                <li>所有特征的训练范围都基于真实训练数据的统计信息，如输入超出范围将会收到提示</li>
                <li>首次使用时如果本地无模型文件，需要网络连接从仓库下载</li>
            </ul>
            </div>
            """
        
        st.markdown(tech_content, unsafe_allow_html=True)

elif st.session_state.prediction_error is not None:
    st.markdown("---")
    error_html = f"""
    <div class='error-box'>
        <h3>预测失败</h3>
        <p>{st.session_state.prediction_error}</p>
        <p>请检查：</p>
        <ul>
            <li>确保模型文件存在于仓库的models/、model_files/或根目录中</li>
            <li>如果是首次使用，确保网络连接正常以下载模型文件</li>
            <li>确保输入数据符合模型要求</li>
            <li>检查FC(wt%)是否满足 100-Ash(wt%)-VM(wt%) 约束</li>
        </ul>
    </div>
    """
    st.markdown(error_html, unsafe_allow_html=True)

# 添加页脚
st.markdown("---")
footer = """
<div style='text-align: center;'>
<p>© 2024 生物质纳米材料与智能装备实验室. 版本: 7.1.0 (同仓库智能加载版本)</p>
<p>模型文件优先本地加载，本地不存在时自动从同仓库下载</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)