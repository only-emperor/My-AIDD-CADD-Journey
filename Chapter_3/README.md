SMILES 避坑指南 for AIDD Developers
写给 AI 制药（AIDD）开发者的分子字符串表示实践指南
目标：快速理解 SMILES、常见变体、工程坑点与数据预处理最佳实践。

Table of Contents
1. Why SMILES matters in AIDD
2. What is SMILES
3. Basic SMILES syntax
3.1 Atoms
3.2 Bonds
3.3 Branches
3.4 Rings
3.5 Bracket atoms
4. Canonical SMILES
5. Common SMILES variants used in AIDD
5.1 Randomized SMILES
5.2 SELFIES
5.3 Isomeric SMILES
5.4 Tokenization
6. Common pitfalls in real-world pipelines
7. Recommended preprocessing workflow
8. RDKit examples
9. Checklist before training
10. References
1. Why SMILES matters in AIDD
在 AI 药物研发（AIDD）中，模型不能直接理解“化学结构图”。
我们需要先把分子转换成机器可处理的表示形式，例如：

文本序列
图结构
分子指纹
3D 坐标
其中，SMILES 是目前最常见、最工程友好的分子文本表示方式之一，因为它：

易存储：一行字符串即可表示一个分子
易处理：适合 CSV / JSON / SQL
易建模：可以直接送入 NLP 模型
生态成熟：RDKit、Open Babel、各类 benchmark 都支持
但也正因为 SMILES 是“把图结构拍扁成字符串”，它会带来一系列问题：

同一分子可有多种写法
对 tokenizer 非常敏感
可能丢失立体信息
在分子生成任务中很脆弱
因此，AIDD 项目里“会用 SMILES”远远不够，关键在于知道它的边界与坑点。

2. What is SMILES
SMILES 的全称是：

Simplified Molecular Input Line Entry System

它是一种把分子结构编码成字符串的规则系统。

例如，乙醇（CH3-CH2-OH）的 SMILES 为：

text

CCO
SMILES 本质上描述的是：

原子
原子之间的连接关系
某些键类型
某些立体信息（如果显式写出）
它不是完整的 3D 分子结构表示，但足以覆盖大量 2D 化学信息表达和机器学习输入需求。

3. Basic SMILES syntax
3.1 Atoms
原子通常直接写元素符号：

text

C N O S P F Cl Br I
示例：

C: carbon
N: nitrogen
O: oxygen
Cl: chlorine
注意：

大多数氢原子默认省略
SMILES 会根据常见价态自动补全隐式氢。

芳香原子通常用小写表示
例如苯环中的碳写作 c：

text

c1ccccc1
3.2 Bonds
常见键类型表示如下：

单键：通常省略
双键：=
三键：#
示例：

text

CC
C=O
C#N
3.3 Branches
分支结构使用括号 () 表示。

例如：

text

CC(O)C
可理解为主链 C-C-C，中间的碳原子连出一个 O 支链。

3.4 Rings
环结构使用数字标记断开和闭合位置。

例如，环己烷：

text

C1CCCCC1
苯：

text

c1ccccc1
复杂环系可能会出现多个环编号。

3.5 Bracket atoms
当原子包含以下特殊信息时，需要使用方括号 []：

电荷
同位素
显式氢
非默认价态
某些立体信息
示例：

text

[Na+]
[NH4+]
[O-]
[13C]
这类 token 在 tokenizer 中必须被正确处理，否则很容易破坏化学语义。

4. Canonical SMILES
Problem
同一个分子可以有多种合法的 SMILES 写法。

例如，乙醇可以写成：

text

CCO
OCC
它们表示的是同一个分子，但对机器学习模型来说可能会被视为不同样本。

Solution
使用 Canonical SMILES 将同一分子统一映射成一个标准字符串表示。

在工程上，这一步通常用于：

去重
统一数据格式
降低数据泄漏风险
保证训练 / 推理输入一致性
RDKit example
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
Important note
Canonical SMILES 通常依赖具体工具实现。
不同工具或不同版本之间的 canonical 结果不一定完全一致。

因此建议：

在同一项目中统一使用同一工具链
固定 RDKit / Open Babel 版本
不要混用不同 canonical 规则生成的数据
5. Common SMILES variants used in AIDD
5.1 Randomized SMILES
What it is
Randomized SMILES 会对同一个分子随机选择不同遍历路径，从而生成多个不同但等价的 SMILES。

Why it is useful
它常被用作数据增强，帮助模型减少对固定书写顺序的依赖。

适用场景：

分子性质预测
分子表示学习
基于 SMILES 的预训练
分子生成模型训练
Pros
增强模型鲁棒性
缓解过拟合
提升序列模型泛化能力
Cons
数据量膨胀
训练时间增加
采样策略不当时会引入分布偏差
5.2 SELFIES
Motivation
SMILES 在分子生成任务中容易产生非法字符串，例如：

text

C(C
C1CC
这类字符串无法被解析成合法分子。

What SELFIES solves
SELFIES 是一种更鲁棒的分子字符串表示方式。
其设计目标是：生成的字符串几乎总能解码为合法分子。

Best use cases
推荐优先用于：

de novo 分子生成
强化学习分子设计
基于语言模型的分子生成 / 优化
Caveat
SELFIES 保证的是“语法上可解码为合法分子”，不保证：

分子可合成
分子稳定
分子有药效
分子具有药物相似性
5.3 Isomeric SMILES
Why it matters
普通 SMILES 主要描述连接关系，默认不完整表达立体化学信息。
但在药物研发中，手性和双键几何构型经常直接影响生物活性。

What it includes
Isomeric SMILES 在普通 SMILES 基础上加入：

@ / @@：手性中心
/ 和 \：双键顺反 / 几何异构
When to use it
以下任务建议保留 isomeric 信息：

对接（Docking）前数据准备
构效关系分析（SAR/QSAR）
手性敏感的 ADMET 任务
3D 构象相关任务
Important note
Isomeric SMILES 包含立体化学信息，但不是完整的 3D 结构表示。
它不能替代：

conformer generation
3D coordinates
SDF / MOL blocks
5.4 Tokenization
Why generic tokenizers fail
通用 NLP tokenizer 往往不了解化学语义。
例如：

text

Cl
如果被错误切成：

C
l
则化学意义被破坏。

类似问题还包括：

Br
[NH3+]
[C@@H]
%10 这类双位数环编号
/、\、@@
Recommended practice
使用化学感知的 tokenizer：

基于正则规则的 tokenizer
化学专用词表
在化学预切分基础上再做 BPE / WordPiece / SentencePiece
Engineering implication
在 SMILES-based 模型里，tokenizer 往往显著影响：

词表大小
序列长度
训练稳定性
最终性能
不要把 tokenizer 当作可忽略的细节。

6. Common pitfalls in real-world pipelines
6.1 Invalid SMILES in datasets
真实数据集中经常存在：

非法 SMILES
编码问题
历史遗留格式
解析失败样本
务必在训练前显式统计：

原始样本数
有效样本数
无效样本数
去重后样本数
6.2 Salts / solvents / mixtures
很多数据库中的分子并不是“干净的单一小分子”，而可能包含：

盐形式
溶剂化物
混合物
counter ions
示例：

hydrochloride
sodium salt
hydrate
如果任务目标是建模“活性母核”，通常需要：

去盐（desalting）
中和（neutralization）
标准化（standardization）
6.3 Tautomers and protonation states
同一化合物可能以不同形式存在：

不同互变异构体（tautomers）
不同质子化状态
这会导致：

重复统计
样本标签分散
训练不一致
在某些项目中需要进一步做：

tautomer canonicalization
pH-aware standardization
6.4 Losing stereochemistry unintentionally
如果数据预处理时没有显式保留 stereochemistry，
你可能会把原本不同的活性分子错误合并。

因此要根据任务明确：

是否需要保留 @, @@, /, \
是否允许去除立体信息
训练集和测试集处理规则是否一致
6.5 SMILES is convenient, not perfect
SMILES 是高性价比表示，但不是万能表示。
在很多任务中，你应考虑与其他表示联合使用：

分子图（Graph）
分子指纹（ECFP / Morgan）
分子描述符
3D conformers
蛋白序列 / 结构信息
7. Recommended preprocessing workflow
下面是一套适合大多数 AIDD 项目的基础流程。

Minimal pipeline
读取原始 SMILES
使用 RDKit 解析
丢弃无法解析的样本
Canonicalize
去重
视任务决定是否：
去盐
中和
互变异构体标准化
保留立体信息
再进入 tokenizer / featurizer / model pipeline
Suggested policy by task
Task	Recommended representation
Property prediction / QSAR	Canonical SMILES, optional Randomized SMILES
Molecular generation	SELFIES or SMILES + validity checking
Docking / 3D-sensitive tasks	Isomeric SMILES + conformers
Classical ML baseline	Fingerprints / descriptors
Graph models	Molecular graph from RDKit
8. RDKit examples
8.1 Validate a SMILES string
Python

from rdkit import Chem

def is_valid_smiles(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

print(is_valid_smiles("CCO"))   # True
print(is_valid_smiles("C(C"))   # False
8.2 Canonicalize SMILES
Python

from rdkit import Chem

def canonicalize_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)

print(canonicalize_smiles("OCC"))  # CCO
8.3 Canonicalize a CSV column
Python

import pandas as pd
from rdkit import Chem

def canonicalize_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)

df = pd.read_csv("molecules.csv")  # assume a column named "smiles"
df["canonical_smiles"] = df["smiles"].apply(canonicalize_smiles)
df = df.dropna(subset=["canonical_smiles"])
df = df.drop_duplicates(subset=["canonical_smiles"])

print(df.head())
8.4 Preserve stereochemistry if needed
Python

from rdkit import Chem

smiles = "F[C@H](Cl)Br"
mol = Chem.MolFromSmiles(smiles)

print(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
9. Checklist before training
在把带有 SMILES 的数据集送进模型之前，建议至少完成以下检查。

Data quality
 所有 SMILES 都能被 RDKit 正常解析
 已统计无效样本数量
 已完成 canonicalization
 已去重
 已确认是否需要去盐 / 中和 / 标准化
 已确认是否处理 tautomer / protonation state
Representation choice
 当前任务是否适合使用 SMILES
 是否应改用 SELFIES（如果是生成任务）
 是否必须保留 stereochemistry
 是否还需要图 / 指纹 / 3D 表示作为补充
Tokenization
 没有把 Cl、Br 等元素错误拆分
 能正确处理 bracket atoms，如 [NH3+]
 能正确处理 stereochemistry tokens，如 @@
 训练 / 推理阶段使用完全一致的 tokenizer
Reproducibility
 固定 RDKit 版本
 固定 preprocessing 代码
 固定 canonicalization / standardization 规则
 保证 train / val / test 使用同一套处理逻辑
10. References
Weininger D. SMILES, a chemical language and information system.
RDKit Documentation: https://www.rdkit.org/
SELFIES: https://github.com/aspuru-guzik-group/selfies
Open Babel: http://openbabel.org/
ChemBERTa and related molecular language model papers
