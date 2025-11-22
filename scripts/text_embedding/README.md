# Text Embedding 模块

本文件夹包含EEGClip项目中用于文本嵌入处理的相关文件，主要用于生成和处理脑电图(EEG)相关的文本描述嵌入向量。

## 文件说明

### [`tuh_description.csv`](tuh_description.csv)
**作用**: TUH Abnormal EEG 数据集的医生报告文本数据

- 包含来自 TUH Abnormal 脑电图数据集的医生诊断报告文本
- 这些报告是医生对 EEG 记录的描述和诊断意见
- 数据来源: [`braindecode.datasets.TUHAbnormal`](../text_embedding.py:54)
- 通过 [`text_preprocessing()`](../text_embedding.py:71) 函数进行预处理
- 用于生成文本嵌入向量的原始数据源

**可能包含的字段**:
- 患者ID
- 报告文本 (report)
- 诊断标签
- 其他元数据

### [`zc_sentences_emb_dict.json`](zc_sentences_emb_dict.json)
**作用**: 零样本分类句子的嵌入向量字典

- "zc" 代表 "zero-shot"（零样本）
- 存储预定义的 EEG 相关零样本句子的文本嵌入向量
- 通过 [`sentence_embedder()`](../text_embedding.py:26) 函数生成嵌入向量
- 用于 EEGClip 模型的零样本分类任务

**包含的分类维度**:
- **病理状态**: 正常 vs 异常脑电记录
- **性别**: 女性 vs 男性患者  
- **年龄**: 超过50岁 vs 低于50岁
- **药物治疗**: 使用 vs 未使用抗癫痫药物
- **组合特征**: 性别+病理、年龄+病理等组合
- **脑电特征**: 特定的EEG模式描述（如alpha节律、尖波等）

### [`text_embedding.py`](text_embedding.py)
**作用**: 文本嵌入生成脚本

- 使用预训练语言模型生成文本嵌入
- 主要模型: "WhereIsAI/UAE-Large-V1"
- 提供批量文本嵌入处理功能
- 支持零样本句子嵌入生成

**主要功能**:
- [`sentence_embedder()`](text_embedding.py:26): 单句嵌入生成
- 批量处理EEG报告文本
- 零样本句子嵌入字典构建
- 嵌入向量保存和加载

## 在 EEGClip 项目中的作用

这两个数据文件共同支持 EEGClip 的文本-EEG 对比学习：

1. **训练数据**: `tuh_description.csv` 提供真实的医生报告文本作为训练数据
2. **评估基准**: `zc_sentences_emb_dict.json` 提供预定义的零样本分类基准
3. **对比学习**: 两者都用于生成文本嵌入，与 EEG 信号嵌入进行对比学习

## 使用方式

```python
# 生成文本嵌入
from scripts.text_embedding.text_embedding import sentence_embedder

# 加载零样本嵌入字典
import json
with open("scripts/text_embedding/zc_sentences_emb_dict.json", "r") as f:
    zc_embeddings = json.load(f)

# 使用嵌入进行零样本分类
# (具体使用方式参考 EEGClip 主项目文档)
```

## 相关配置

这些文件的路径在 [`configs/preprocess_config.py`](../../configs/preprocess_config.py) 中定义：
- `embs_df_path`: 文本嵌入数据文件路径
- `zc_sentences_emb_dict_path`: 零样本句子嵌入字典路径