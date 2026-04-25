1. 项目简介
本实验通过计算机辅助方法，对分子的甜度指标（logSw）进行定量结构-活性关系（QSAR）建模。相比于上一节的深度学习方法，本节侧重于可解释性机器学习，通过经典物理化学描述符来理解影响分子甜度的关键因素。

2. 核心技术栈
化学信息学: RDKit (用于计算分子描述符及 Morgan 指纹)
特征降维: t-SNE (化学空间可视化)
机器学习算法: LightGBM (梯度提升决策树回归)
模型解释: SHAP (基于博弈论的特征贡献度分析)
自动建模辅助: LazyPredict (快速筛选基准模型)
3. 工作流程 (Workflow)
特征提取 (Featurization):
计算 9 类核心物理化学描述符（分子量 MW、脂水分配系数 logP、极性表面积 TPSA 等）。
生成 2048 位的 Morgan 指纹 (Radius=2) 以捕捉局部结构信息。
探索性数据分析 (EDA):
t-SNE 可视化: 将高维描述符空间映射到 2D 平面，观察高甜度与低甜度分子的分布差异。
统计分析: 通过小提琴图对比不同类别间核心描述符的分布。
模型训练:
使用 LightGBM 进行回归预测。
引入早停机制（Early Stopping）防止过拟合。
可解释性分析 (Explainable AI):
使用 SHAP Summary Plot 评估全局特征重要性。
使用 SHAP Waterfall Plot 对特定高活性分子进行个体贡献度拆解。
4. 关键发现与可视化说明
Chemical Space: 观察 t-SNE 图可以判断甜度分子在化学空间上是否存在聚集效应。
SHAP 分析:
全局视角: 哪些因素（如 logP 或氢键供体）对甜度影响最大？
局部视角: 针对具体某个甜度极高的分子，是什么结构特征导致了预测值的升高？
5. 环境准备
Bash

pip install rdkit-pypi seaborn sklearn lightgbm shap joblib
6. 文件结构与产出
analysis_script.py: 核心 Python 代码。
lgbm_model.pkl: 训练好的 LightGBM 模型。
preprocessing_info.pkl: 包含特征选择和均值填充等预处理元数据。
可视化结果: 包含 logSw 分布图、t-SNE 散点图、回归拟合图及 SHAP 解释图。

