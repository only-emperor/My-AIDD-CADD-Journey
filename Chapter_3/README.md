# Chapter 3 - SMILES 避坑指南 for AIDD 开发者

> 写给 AI 制药（AIDD）开发者的分子字符串表示实践指南  
> 目标：快速理解 SMILES、常见变体、工程坑点，以及训练前该做的数据清洗。

---

## 目录

- [1. 为什么 AIDD 一定会遇到 SMILES](#1-为什么-aidd-一定会遇到-smiles)
- [2. 什么是 SMILES](#2-什么是-smiles)
- [3. SMILES 基础语法](#3-smiles-基础语法)
  - [3.1 原子](#31-原子)
  - [3.2 化学键](#32-化学键)
  - [3.3 分支](#33-分支)
  - [3.4 环](#34-环)
  - [3.5 方括号原子](#35-方括号原子)
- [4. 大坑：同一个分子可以有很多种写法](#4-大坑同一个分子可以有很多种写法)
- [5. AIDD 常见的几种 SMILES 变体](#5-aidd-常见的几种-smiles-变体)
  - [5.1 Canonical SMILES](#51-canonical-smiles)
  - [5.2 Randomized SMILES](#52-randomized-smiles)
  - [5.3 SELFIES](#53-selfies)
  - [5.4 Isomeric SMILES](#54-isomeric-smiles)
  - [5.5 Tokenization](#55-tokenization)
- [6. 工程里最容易踩的坑](#6-工程里最容易踩的坑)
- [7. 一个推荐的数据预处理流程](#7-一个推荐的数据预处理流程)
- [8. RDKit 示例代码](#8-rdkit-示例代码)
- [9. 训练前检查清单](#9-训练前检查清单)
- [10. 总结](#10-总结)

---

## 1. 为什么 AIDD 一定会遇到 SMILES

做 AI 药物研发（AIDD）时，模型并不能直接理解“化学结构图”。

我们必须先把分子转成计算机能处理的形式，例如：

- 文本序列
- 图结构
- 分子指纹
- 3D 坐标

其中，**SMILES** 是最常见的一种分子文本表示。

它的优势很直接：

- 一行字符串就能表示一个分子
- 便于保存到 CSV / JSON / 数据库
- 可以直接送进 NLP 模型
- RDKit、Open Babel 等工具链支持完善

但也正因为它把“图结构”压缩成了“字符串”，所以它会带来很多额外问题：

- 同一个分子可能有多种写法
- 对 tokenizer 很敏感
- 可能丢失立体信息
- 在生成任务里容易产生非法字符串

所以对 AIDD 开发者来说，SMILES 是绕不过去的基础设施。

---

## 2. 什么是 SMILES

SMILES 的全称是：

**Simplified Molecular Input Line Entry System**

你可以把它理解成化学分子的“字符串表示法”。

例如，乙醇（CH3-CH2-OH）的 SMILES 是：

```text
CCO
这行字符串描述了：

原子是什么
原子之间怎么连接
某些键类型
某些立体信息（如果显式写出来）
它不是完整的 3D 结构，但足够适合很多 AIDD 任务里的数据处理和模型训练。

3. SMILES 基础语法
3.1 原子
原子直接写元素符号：

text

C N O S P F Cl Br I
例如：

C：碳
N：氮
O：氧
Cl：氯
注意两点：

氢原子通常省略
SMILES 会按照常见价态自动补全隐式氢。

芳香原子通常用小写表示
例如苯环中的碳写作：

text

c1ccccc1
3.2 化学键
常见键类型：

单键：通常省略
双键：=
三键：#
示例：

text

CC
C=O
C#N
3.3 分支
分支用括号 () 表示。

例如：

text

CC(O)C
表示主链上有一个支链 O。

3.4 环
环结构通过数字标记闭合位置。

例如环己烷：

text

C1CCCCC1
苯：

text

c1ccccc1
两个相同数字表示这两个位置需要连起来形成一个环。

3.5 方括号原子
当原子包含特殊信息时，需要写进方括号 []，例如：

电荷
同位素
显式氢
特殊价态
示例：

text

[Na+]
[NH4+]
[O-]
[13C]
这类 token 在后续 tokenizer 处理中很重要，不能随便切碎。

4. 大坑：同一个分子可以有很多种写法
SMILES 最大的坑之一是：

同一个分子，不一定只有一种写法。

例如乙醇既可以写成：

text

CCO
也可以写成：

text

OCC
它们在化学上是同一个分子，但模型会把它们当成两条不同样本。

这会导致：

重复样本
数据泄漏
模型学习到字符串表面形式，而不是分子本身
所以在真正训练模型前，第一步往往不是建模，而是先做规范化。

5. AIDD 常见的几种 SMILES 变体
5.1 Canonical SMILES
为了解决“同物不同名”的问题，通常会把 SMILES 转成 Canonical SMILES。

它的思路是：

用固定算法，把同一个分子映射成一个标准写法。

例如：

Python

from rdkit import Chem

smiles_a = "CCO"
smiles_b = "OCC"

mol_a = Chem.MolFromSmiles(smiles_a)
mol_b = Chem.MolFromSmiles(smiles_b)

std_a = Chem.MolToSmiles(mol_a, canonical=True)
std_b = Chem.MolToSmiles(mol_b, canonical=True)

print(std_a)  # CCO
print(std_b)  # CCO
注意：

Canonical SMILES 很适合做去重和统一格式
但它依赖具体工具实现，不同工具结果不一定完全一致
同一个项目里最好固定同一个工具和版本
5.2 Randomized SMILES
Canonical SMILES 追求唯一，
而 Randomized SMILES 则故意让同一个分子产生多种等价写法。

它常用于：

数据增强
分子表示学习
基于 SMILES 的序列模型训练
好处：

减少模型对固定遍历顺序的依赖
提高泛化能力
缓解过拟合
代价：

数据量会膨胀
训练时间会增加
如果采样不当，可能引入分布偏差
5.3 SELFIES
SMILES 在分子生成任务里有个很大的问题：很脆弱。

例如：

text

C(C
C1CC
这类字符串因为括号或环没闭合，无法解析成合法分子。

SELFIES 的目标就是解决这个问题。

它是一种更鲁棒的分子字符串表示，特点是：

按照 SELFIES 规则生成的字符串，几乎总能解码为合法分子。

因此，在下面这些任务中通常更推荐 SELFIES：

de novo 分子生成
强化学习分子设计
基于语言模型的分子生成
但要注意：

SELFIES 解决的是“字符串合法”，不是“药物合理”。
它不保证生成的分子：

可合成
稳定
有药效
像药
5.4 Isomeric SMILES
普通 SMILES 主要描述原子连接关系。
但药物研发里，立体化学信息经常决定活性和毒性。

Isomeric SMILES 会额外保留：

@ / @@：手性中心
/ 和 \：双键几何异构
适用场景：

Docking 前数据准备
构效关系分析
手性敏感的 ADMET 任务
3D 构象相关任务
注意：

Isomeric SMILES 包含的是立体信息，不是完整的 3D 坐标。
它不能替代 conformer、SDF 或真实三维结构。

5.5 Tokenization
如果把 SMILES 送进 Transformer、BERT 或其他语言模型，分词方式非常关键。

一个经典错误是：

text

Cl
被错误切成：

C
l
这会直接破坏化学语义。

类似高风险 token 还包括：

Br
[NH3+]
[C@@H]
%10
@@
/ 和 \
所以实践中应优先使用：

化学规则驱动的 tokenizer
化学专用词表
或在化学预切分后再做 BPE / WordPiece
不要直接拿通用 NLP tokenizer 暴力切。

6. 工程里最容易踩的坑
6.1 数据集中存在非法 SMILES
真实数据集里经常有解析失败样本。原因可能包括：

非法字符串
编码错误
历史遗留格式
混合物 / 金属配合物写法异常
训练前必须统计：

原始样本数
可解析样本数
解析失败样本数
6.2 盐形式、溶剂化物、混合物
很多数据库里的分子不是“纯净的小分子”，而是：

hydrochloride
sodium salt
hydrate
mixture
如果建模目标是活性母核，通常需要进一步处理：

去盐（desalting）
中和（neutralization）
标准化（standardization）
6.3 互变异构体和质子化状态
同一个化合物可能在数据集中以不同形式出现：

tautomer
不同 protonation state
如果不统一，会导致：

重复计数
标签分散
训练目标不一致
6.4 不小心丢了立体信息
如果在预处理时没有明确保留 stereochemistry，
可能会把原本不同活性的分子误合并。

因此必须根据任务明确：

是否保留 @、@@
是否保留 /、\
train / val / test 是否使用同一规则
6.5 SMILES 很方便，但不是万能表示
SMILES 适合很多任务，但不是唯一解。

在很多场景下，应该同时考虑：

分子图（Graph）
分子指纹（Morgan / ECFP）
分子描述符
3D conformer
蛋白序列 / 结构
尤其是更复杂的 AIDD 任务，单靠字符串往往不够。

7. 一个推荐的数据预处理流程
建议的最小流程如下：

读取原始 SMILES
用 RDKit 解析
删除无法解析的样本
转成 Canonical SMILES
去重
根据任务决定是否：
去盐
中和
处理 tautomer
保留立体信息
再进入 tokenizer / featurizer / 模型
按任务选择表示方式：

任务	推荐表示
性质预测 / QSAR	Canonical SMILES，可选 Randomized SMILES
分子生成	SELFIES 或 SMILES + validity check
对接 / 立体敏感任务	Isomeric SMILES + 3D conformer
传统机器学习基线	Fingerprints / descriptors
图神经网络	RDKit 分子图
8. RDKit 示例代码
8.1 判断 SMILES 是否合法
Python

from rdkit import Chem

def is_valid_smiles(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

print(is_valid_smiles("CCO"))   # True
print(is_valid_smiles("C(C"))   # False
8.2 转换为 Canonical SMILES
Python

from rdkit import Chem

def canonicalize_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)

print(canonicalize_smiles("OCC"))  # CCO
8.3 批量处理 CSV 中的 SMILES
Python

import pandas as pd
from rdkit import Chem

def canonicalize_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)

df = pd.read_csv("molecules.csv")   # 假设有一列叫 smiles
df["canonical_smiles"] = df["smiles"].apply(canonicalize_smiles)
df = df.dropna(subset=["canonical_smiles"])
df = df.drop_duplicates(subset=["canonical_smiles"])

print(df.head())
8.4 保留立体信息
Python

from rdkit import Chem

smiles = "F[C@H](Cl)Br"
mol = Chem.MolFromSmiles(smiles)

print(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
9. 训练前检查清单
数据质量
 所有 SMILES 都能被 RDKit 正常解析
 已统计无效样本数量
 已完成 canonicalization
 已去重
 已确认是否需要去盐 / 中和 / 标准化
 已确认是否处理 tautomer / protonation state
表示方式
 当前任务是否适合用 SMILES
 如果是生成任务，是否应改用 SELFIES
 是否必须保留 stereochemistry
 是否需要图 / 指纹 / 3D 表示作为补充
Tokenization
 没有把 Cl、Br 错误拆分
 能正确处理 [NH3+] 这类 bracket atoms
 能正确处理 @@、/、\
 训练和推理阶段使用同一套 tokenizer
可复现性
 固定 RDKit 版本
 固定 preprocessing 代码
 固定 canonicalization / standardization 规则
 train / val / test 使用完全一致的处理流程
10. 总结
如果只记住三件事：

先 canonicalize，再去重，再训练
按任务选表示：预测可用 SMILES，生成优先考虑 SELFIES，立体相关任务必须保留 isomeric 信息
不要用通用 tokenizer 生切 SMILES
SMILES 是 AIDD 里非常好用的分子表示，但它不是完美表示。
它保留了很多有价值的结构信息，也会丢失很多上下文和 3D 细节。

真正重要的不是“会不会用 SMILES”，而是：
