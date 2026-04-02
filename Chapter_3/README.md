🧬 SMILES 与 AIDD：分子图的文本化表示
在人工智能药物发现（AIDD）领域，将复杂的三维/二维分子结构转化为计算机能够高效处理的数据格式是第一步。其中，SMILES（Simplified Molecular-Input Line-Entry System）是最核心、最广泛使用的分子表示方法。

本指南将系统介绍 SMILES 的基础知识，以及它在深度学习和药物发现中的演变与高级应用。

1. SMILES 表示基础
1.1 什么是 SMILES？
SMILES（简化分子线性输入规范）是一种用 ASCII 字符串明确描述分子结构的线性符号系统。

本质：将分子的图结构（节点为原子，边为化学键）通过深度优先遍历（DFS）展平为一维字符串。
优势：极其紧凑、人类可读、极其适合计算机存储，且可以直接作为自然语言处理（NLP）模型的输入（如 RNN, Transformer）。
1.2 基本语法
SMILES 的语法规则简洁而强大，主要包含以下几个核心要素：

原子 (Atoms)：用元素符号表示。首字母大写，次字母小写（如 C, N, O, Cl）。
芳香性：芳香环上的原子通常用小写字母表示（如 c, n, o 表示苯环或吡啶环上的原子）。
氢原子：在标准 SMILES 中，连接在碳及其他杂原子上的氢通常是隐式的（省略不写）。如果需要特别标明同位素、电荷或手性，需用方括号括起来，如 [NH3+], [13C]。
化学键 (Bonds)：
单键：默认省略，或用 - 表示。
双键：= （如 C=O 为甲醛）。
三键：# （如 C#N 为氰基）。
芳香键：通常省略，或用 : 表示。
分支 (Branches)：使用圆括号 () 表示主链上的分支。
例如：异丁烷 (Isobutane) 表示为 CC(C)C。
闭环 (Rings)：通过在环的断开处打上相同的数字标签来表示环的闭合。
例如：环己烷表示为 C1CCCCC1，苯环表示为 c1ccccc1。若数字超过9，需用 %，如 C%10。
立体化学 (Stereochemistry)：
手性中心：用 @ 和 @@ 表示（基于观察方向的逆时针和顺时针）。例如 L-丙氨酸为 N[C@@H](C)C(=O)O。
双键顺反：用 / 和 \ 表示方向。
断开的结构 (Disconnected Structures)：用 . 分隔（常用于盐类或混合物）。
例如：氯化钠为 [Na+].[Cl-]。
1.3 规范性 (Canonicalization)
痛点：由于遍历分子图的起始原子和路径可以不同，同一个分子可以有多个完全合法的 SMILES 字符串。

例如，乙醇可以写成：CCO, OCC, C(O)C。
这在数据库去重和机器学习中是致命的（会导致模型认为它们是不同的分子）。
解决方案：Canonical SMILES（规范化 SMILES）
通过一套标准化的图同构算法（如 Morgan 算法），为每个分子生成唯一的字符串路径。
在 AIDD 实践中，通常使用 RDKit 库来进行规范化：

Python

from rdkit import Chem

# 不同的输入 SMILES
smiles1 = "CCO"
smiles2 = "OCC"

# 转换为 RDKit Mol 对象
mol1 = Chem.MolFromSmiles(smiles1)
mol2 = Chem.MolFromSmiles(smiles2)

# 转换为 Canonical SMILES
canon1 = Chem.MolToSmiles(mol1) # 默认输出 canonical SMILES
canon2 = Chem.MolToSmiles(mol2)

print(canon1) # 输出: CCO
print(canon2) # 输出: CCO (两者现在完全一致)
2. AIDD 中常见的 SMILES 表示与变体
在深度学习中，直接使用标准 SMILES 有时会遇到语法无效（模型生成的字符串无法还原为分子）等问题。因此，在 GitHub 开源项目和 AIDD 论文中，常出现以下几种 SMILES 的进阶表示和变体：

2.1 Randomized SMILES (SMILES 数据增强)
虽然数据库需要唯一的 Canonical SMILES，但在模型训练阶段，故意生成同一个分子的不同合法的 SMILES 字符串，是一种极其有效的数据增强 (Data Augmentation) 手段。

作用：提高生成模型（如 VAE, GAN）或预测模型（如基于 Transformer 的属性预测）的泛化能力，防止过拟合。
代码实现 (RDKit)：
Python

def randomize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    # doRandom=True 每次打乱原子顺序
    return Chem.MolToSmiles(mol, canonical=False, doRandom=True)

print(randomize_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")) # 可能会输出: O=C(O)c1ccccc1OC(C)=O 等
2.2 SELFIES (100% 鲁棒的表示法)
背景：SMILES 的语法很严格（如括号必须闭合，环数字必须成对）。如果 AI 随机改变 SMILES 中的一个字符，通常会生成非法的乱码（Invalid Molecule）。
SELFIES (SELF-referencIng Embedded Strings) 是近年来极其火热的替代方案。它的设计保证了任何随机的 SELFIES 字符串都能映射到一个合法的分子图。
应用：在分子生成任务（如强化学习设计新药）中全面取代基础 SMILES。
例子：[C][C][O]
2.3 DeepSMILES
背景：解决 SMILES 中环的数字匹配和括号匹配问题对神经网络带来的挑战。
特点：DeepSMILES 不使用成对的数字来闭环，而是使用只在环结尾出现的单一符号；同时使用特殊的后缀符号来替代成对的括号。
应用：主要用于简化 RNN 的学习难度，但目前流行度略逊于 SELFIES。
2.4 Tokenized SMILES (词元化的 SMILES)
在将 SMILES 送入大语言模型（LLM）或 Transformer 之前，需要对其进行 Tokenization（分词）。AIDD 中常见的切分方式有：

字符级 (Character-level)：逐字符切分，如 ['C', 'C', '=', 'O']。（缺点：会将 Cl 切分成 C 和 l，导致化学意义丧失）。
正则表达式切分 (Regex/Smiles-pe)：基于化学规则，将 [NH3+], Cl, Br 等视为一个完整的 Token。这是目前 ChemBERTa 等模型的主流做法。
BPE (Byte Pair Encoding)：让模型自己学习高频词汇串（如 c1ccccc1 可能被当成一个单词），类似于 NLP 中的标准操作。
2.5 Isomeric SMILES (同分异构 SMILES)
在涉及到靶点结合（Docking）或空间构象影响药效的任务中，普通的 SMILES 是不够的。必须使用包含手性（@, @@）和顺反异构（/, \）的 Isomeric SMILES。在 RDKit 中，通常默认开启。


