# 1. 安装依赖库 (Colab环境)
!pip install -q rdkit-pypi seaborn lazypredict xgboost lightgbm shap

# 2. 全局导入
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# RDKit
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Draw

# Sklearn & Models
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import linregress
from lightgbm import LGBMRegressor
import shap

# 3. 全局颜色与绘图配置
COLOR_HIGH = '#4B74B2'       # RGB(75,116,178)
COLOR_LOW = '#D9412B'        # RGB(217,65,43)
COLOR_PERFECT = '#90BEE0'    # RGB(144,190,224)
PALETTE = [COLOR_HIGH, COLOR_LOW]

sns.set_theme(style="whitegrid", rc={"axes.facecolor": "white", "figure.facecolor": "white"})
import warnings
warnings.filterwarnings('ignore')


def calculate_all_features(smiles):
    """一次性计算描述符和指纹，提高效率"""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None, None

    # 计算描述符
    desc = {
        'MW': Descriptors.MolWt(mol),
        'logP': Descriptors.MolLogP(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'TPSA': Descriptors.TPSA(mol),
        'RotatableBonds': Descriptors.NumRotatableBonds(mol),
        'NumAromaticRings': Descriptors.NumAromaticRings(mol),
        'NumStereocenters': len(Chem.FindMolChiralCenters(mol, includeUnassigned=False)),
        'HallKierAlpha': Descriptors.HallKierAlpha(mol)
    }

    # 计算指纹 (2048位)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    fp_array = np.array(fp)

    return desc, fp_array


print(">>> 开始加载和处理数据...")
df = pd.read_csv("/content/SweetpredDB.csv")

# 提取特征
features = df['Smiles'].apply(lambda x: pd.Series(calculate_all_features(x)))
features.columns = ['Descriptors', 'Fingerprints']

# 拼接数据并清洗无效SMILES
df = pd.concat([df, features], axis=1).dropna(subset=['Descriptors'])
print(f"有效分子数：{len(df)} / 原始数据：{len(df) + features['Descriptors'].isna().sum()}")

# 展开描述符字典为DataFrame列
desc_df = pd.DataFrame(df['Descriptors'].tolist(), index=df.index)
# 展开指纹为DataFrame列
fp_df = pd.DataFrame(df['Fingerprints'].tolist(), index=df.index, columns=[f"Bit_{i}" for i in range(2048)])

# 添加分类标签
df['Sweetness_Class'] = np.where(df['logSw'] >= 3, 'High', 'Low')

# 最终的主数据集 (包含所有描述符和目标值)
df_main = pd.concat([df[['Smiles', 'logSw', 'Sweetness_Class']], desc_df], axis=1)

# 缺失值填充 (均值)
numeric_cols = desc_df.columns
df_main[numeric_cols] = df_main[numeric_cols].fillna(df_main[numeric_cols].mean())

print(">>> 数据处理完成！")
print(">>> 正在生成可视化图表...")

fig = plt.figure(figsize=(20, 15), dpi=150)

# 1. logSw 分布图
ax1 = plt.subplot(2, 2, 1)
sns.histplot(data=df_main, x='logSw', bins=30, kde=True, color=COLOR_HIGH, edgecolor='white', ax=ax1)
ax1.set_title('Distribution of logSw', fontsize=14, fontweight='bold')
# 加粗KDE曲线
for line in ax1.lines:
    line.set_color('black')
    line.set_linewidth(1.5)

# 2. t-SNE 降维可视化 (基于描述符)
ax2 = plt.subplot(2, 2, 2)
scaler = StandardScaler()
desc_scaled = scaler.fit_transform(desc_df)
tsne_result = TSNE(n_components=2, random_state=42).fit_transform(desc_scaled)

for cls, color in zip(['High', 'Low'], [COLOR_HIGH, COLOR_LOW]):
    mask = (df_main['Sweetness_Class'] == cls)
    ax2.scatter(tsne_result[mask, 0], tsne_result[mask, 1],
                c=color, s=30, alpha=0.7, edgecolor='w', label=f'{cls} (n={sum(mask)})')
ax2.set_title('t-SNE Chemical Space (Descriptors)', fontsize=14, fontweight='bold')
ax2.legend()

# 3. 核心描述符小提琴图
important_descs = ['MW', 'logP', 'TPSA', 'NumHDonors']
for i, desc in enumerate(important_descs):
    ax = plt.subplot(4, 4, 8 + i + 1)
    sns.violinplot(data=df_main, x='Sweetness_Class', y=desc, palette=PALETTE, order=['High', 'Low'], ax=ax)
    ax.set_title(desc, fontsize=12)
    ax.set_xlabel('')

plt.tight_layout()
plt.show()# 1. 准备训练数据 (基于描述符)
X = df_main.drop(['Smiles', 'logSw', 'Sweetness_Class'], axis=1)
y = df_main['logSw']

# 方差过滤
selector = VarianceThreshold(threshold=0.0)
X_selected = selector.fit_transform(X)
selected_features = X.columns[selector.get_support()].tolist()
X = pd.DataFrame(X_selected, columns=selected_features)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. 训练LGBM模型 (启用早停机制防止过拟合)
print(">>> 开始训练 LightGBM 模型...")
model = LGBMRegressor(n_estimators=1000, learning_rate=0.02, random_state=42, verbose=-1)

# 注意：LightGBM的新版本使用 callbacks 来实现 early stopping
from lightgbm import early_stopping, log_evaluation
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_metric='mse',
    callbacks=[early_stopping(stopping_rounds=50, verbose=False), log_evaluation(0)]
)

# 3. 评估与绘图
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
slope, intercept, _, _, _ = linregress(y_test, y_pred)

plt.figure(figsize=(8, 8), dpi=100)
sns.scatterplot(x=y_test, y=y_pred, color=COLOR_HIGH, alpha=0.8, edgecolor='w')

# 绘制回归线和完美拟合线
min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
plt.plot(y_test, slope * y_test + intercept, color=COLOR_PERFECT, lw=2, label='Regression line')
plt.plot([min_val, max_val], [min_val, max_val], '--', color=COLOR_LOW, lw=2, label='Perfect fit')

plt.title(f'LGBM Prediction (RMSE: {rmse:.3f}, R²: {r2:.3f})', fontsize=14, pad=15)
plt.xlabel('True logSw'), plt.ylabel('Predicted logSw')
plt.legend(loc='lower right')
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

# 4. 保存模型与预处理信息
joblib.dump(model, 'lgbm_model.pkl')
joblib.dump({
    'feature_columns': X.columns.tolist(),
    'numeric_means': df_main[numeric_cols].mean().to_dict(),
    'selected_features': selected_features
}, 'preprocessing_info.pkl')
print(">>> 模型已保存。")
print(">>> 生成 SHAP 可解释性分析...")
explainer = shap.TreeExplainer(model)
# 为了计算速度和清晰度，使用测试集计算SHAP
shap_values = explainer(X_test)

# 1. 特征重要性汇总图 (Summary Plot)
plt.figure(figsize=(10, 6), dpi=120)
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP Feature Effects", fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# 2. 单个样本深入分析 (随机抽取一个高甜度分子)
sample_idx = y_test.idxmax() # 取测试集中甜度最高的分子
smiles = df_main.loc[sample_idx, 'Smiles']
mol = Chem.MolFromSmiles(smiles)

# 绘制分子结构
print(f"\n>>> 分析分子 SMILES: {smiles}")
print(f"真实 logSw: {y_test.loc[sample_idx]:.3f} | 预测 logSw: {model.predict(X_test.loc[[sample_idx]])[0]:.3f}")
display(Draw.MolToImage(mol, size=(300, 300)))

# 绘制该分子的 SHAP 瀑布图 (Waterfall plot，最直观的单样本解释图)
# 找到该样本在 X_test 中的局部索引
local_idx = X_test.index.get_loc(sample_idx)

plt.figure(figsize=(10, 5), dpi=120)
shap.plots.waterfall(shap_values[local_idx], max_display=10, show=False)
plt.title("How features contributed to this specific prediction", pad=20)
plt.tight_layout()
plt.show()
