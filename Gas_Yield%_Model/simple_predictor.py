
import numpy as np
import pandas as pd
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class Gas_Yield%Predictor:
    """
    简化版预测器 - 用于加载保存的模型进行预测和分析
    """
    def __init__(self, models_dir=None):
        """
        初始化预测器
        
        参数:
            models_dir: 模型保存目录，默认使用当前文件所在目录
        """
        if models_dir is None:
            # 如果未指定目录，使用当前文件所在目录
            models_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.models_dir = models_dir
        self.models = []
        self.model_weights = None
        self.final_scaler = None
        self.feature_names = None
        self.metadata = None
        self.target_name = "Gas Yield(%)"
        
        # 加载所有需要的模型组件
        self._load_all_components()
    
    def _load_all_components(self):
        """加载所有模型组件"""
        # 加载元数据
        metadata_path = os.path.join(self.models_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            self.feature_names = self.metadata.get('feature_names', None)
            self.performance = self.metadata.get('performance', {})
        else:
            raise FileNotFoundError(f"元数据文件未找到: {metadata_path}")
        
        # 加载模型
        models_dir = os.path.join(self.models_dir, 'models')
        model_files = [f for f in os.listdir(models_dir) if f.startswith('model_') and f.endswith('.joblib')]
        
        self.models = []
        for i in range(len(model_files)):
            model_path = os.path.join(models_dir, f'model_{i}.joblib')
            if os.path.exists(model_path):
                self.models.append(joblib.load(model_path))
        
        # 加载权重
        weights_path = os.path.join(self.models_dir, 'model_weights.npy')
        if os.path.exists(weights_path):
            self.model_weights = np.load(weights_path)
        
        # 加载缩放器
        scaler_path = os.path.join(self.models_dir, 'final_scaler.joblib')
        if os.path.exists(scaler_path):
            self.final_scaler = joblib.load(scaler_path)
        
        print(f"模型已加载，包含 {len(self.models)} 个子模型")
        print(f"预测目标: {self.target_name}")
    
    def predict(self, data):
        """
        预测Gas Yield(%)
        
        参数:
            data: DataFrame或字典，包含特征数据
            
        返回:
            预测的Gas Yield(%)值
        """
        # 处理输入数据
        if isinstance(data, dict):
            # 如果是字典，转换为DataFrame
            data = pd.DataFrame([data])
        
        # 确保特征顺序正确
        if self.feature_names is not None:
            # 检查是否缺少特征
            missing_features = set(self.feature_names) - set(data.columns)
            if missing_features:
                raise ValueError(f"输入数据缺少以下特征: {missing_features}")
            
            # 重排特征顺序
            data = data[self.feature_names]
        
        # 标准化数据
        X_scaled = self.final_scaler.transform(data)
        
        # 使用所有模型进行预测
        all_predictions = np.zeros((data.shape[0], len(self.models)))
        for i, model in enumerate(self.models):
            all_predictions[:, i] = model.predict(X_scaled)
        
        # 计算加权平均
        weighted_pred = np.sum(all_predictions * self.model_weights.reshape(1, -1), axis=1)
        
        return weighted_pred
    
    def plot_prediction(self, X, y_true, figsize=(5, 4.5)):
        """
        绘制预测值与实际值的对比图
        
        参数:
            X: 特征数据
            y_true: 实际值
            figsize: 图表大小
        """
        # 预测
        y_pred = self.predict(X)
        
        # 计算性能指标
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制散点图
        ax.scatter(y_true, y_pred, c='#4b6f89', 
                  label=f'RMSE: {rmse:.2f}, R²: {r2:.2f}', 
                  alpha=0.7, s=50, edgecolor='w', linewidth=0.5)
        
        # 确定图表范围
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        margin = (max_val - min_val) * 0.1
        plot_min = min_val - margin
        plot_max = max_val + margin
        
        # 绘制理想拟合线
        ax.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', color='black', label='Ideal fit')
        
        # 添加直方图
        ax_histx = ax.inset_axes([0, 1, 1, 0.1])  # x轴上的直方图
        ax_histy = ax.inset_axes([1, 0, 0.1, 1])  # y轴上的直方图
        
        # 绘制密度图
        sns.kdeplot(y_pred, ax=ax_histx, fill=True, bw_adjust=1, alpha=0.7)
        sns.kdeplot(y_pred, ax=ax_histy, fill=True, vertical=True, bw_adjust=1, alpha=0.7)
        
        # 隐藏直方图的刻度标签
        ax_histx.axis('off')
        ax_histy.axis('off')
        
        # 设置图表格式
        ax.set_xlabel(f'True {self.target_name}', fontsize=12, fontweight='normal')
        ax.set_ylabel(f'Predicted {self.target_name}', fontsize=12, fontweight='normal')
        ax.tick_params(axis='both', which='major', width=1, color='black')
        
        # 设置轴范围
        ax.set_xlim([plot_min, plot_max])
        ax.set_ylim([plot_min, plot_max])
        
        # 添加图例
        ax.legend(fontsize=10, loc='upper left', framealpha=0.6)
        
        plt.tight_layout()
        plt.show()
        
        return rmse, r2
    
    def get_importance(self, plot=True, figsize=(10, 6)):
        """
        获取特征重要性
        
        参数:
            plot: 是否绘制图表
            figsize: 图表大小
            
        返回:
            特征重要性DataFrame
        """
        # 计算特征重要性
        importance = np.zeros(len(self.feature_names))
        for i, model in enumerate(self.models):
            model_importance = model.get_feature_importance()
            importance += model_importance * self.model_weights[i]
        
        # 创建DataFrame
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # 绘制图表
        if plot:
            plt.figure(figsize=figsize)
            sns.barplot(x='Importance', y='Feature', data=importance_df)
            plt.title('特征重要性', fontsize=14)
            plt.xlabel('重要性得分', fontsize=12)
            plt.ylabel('特征', fontsize=12)
            plt.tight_layout()
            plt.show()
        
        return importance_df
    
    def summary(self):
        """打印模型摘要信息"""
        print(f"模型目标: {self.target_name}")
        print(f"特征数量: {len(self.feature_names)}")
        print(f"特征列表: {', '.join(self.feature_names)}")
        
        if self.performance:
            print(f"模型性能:")
            print(f"  训练集 RMSE: {self.performance.get('train_rmse', 'unknown'):.2f}")
            print(f"  训练集 R²: {self.performance.get('train_r2', 'unknown'):.4f}")
            print(f"  测试集 RMSE: {self.performance.get('test_rmse', 'unknown'):.2f}")
            print(f"  测试集 R²: {self.performance.get('test_r2', 'unknown'):.4f}")


# 简单使用示例:
if __name__ == "__main__":
    # 1. 初始化预测器 (不需要指定路径，会自动使用当前目录)
    predictor = Gas_Yield%Predictor()
    
    # 2. 打印模型信息
    predictor.summary()
    
    # 3. 单样本预测
    sample = {
        'PT(°C)': 500,
        'RT(min)': 10,
        # 添加其他特征...
    }
    # 确保包含所有必要特征
    for feature in predictor.feature_names:
        if feature not in sample:
            sample[feature] = 0  # 设置默认值
    
    result = predictor.predict([sample])[0]
    print(f"\n预测 Gas Yield(%): {result:.2f}")
    
    # 4. 获取特征重要性
    importance = predictor.get_importance()
    print("\n特征重要性排名:")
    print(importance.head())
