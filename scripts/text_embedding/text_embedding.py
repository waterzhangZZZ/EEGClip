import copy
import json
import os
import sys

# Ensure repository root is on sys.path so top-level imports like
# `import configs.preprocess_config` work when running this script
# directly (python scripts/text_embedding/text_embedding.py).
# 将EEGClip的绝对地址加入sys.path
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import configs.preprocess_config as preprocess_config
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

# check nb of available GPUs
# 查看可用的GPU数量，并输出
print(torch.cuda.device_count())
# 输入：句子、分词器、模型、最大长度
# 过程：分词->模型推理->提取嵌入向量
# 输出：numpy格式的嵌入向量
def sentence_embedder(sentence,tokenizer,model,max_length=512):
    with torch.no_grad():
        desc_tokenized = tokenizer(sentence, return_tensors="pt", max_length=max_length, padding=True, truncation=True).to(device)
        outputs = model(**desc_tokenized)
        emb = outputs.to_tuple()[0][0][0].detach().cpu().numpy()
    return emb


device = torch.device("cuda")

# 模型配置
model_name = "WhereIsAI/UAE-Large-V1" # "medalpaca/medalpaca-13b" #
max_length = 2512
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, device_map="auto")


# 数据集加载部分
"""
从braindecode库导入TUHAbnormal数据集类
TUHAbnormal是一个公开的EEG数据集，包含异常脑电记录
设置并行工作线程数为32
创造TUHAbnormal数据集示例：
path数据集路径；recording_ids=None加载所有目录；target_name="report"目标字段是医生报告文本；
preload=False不预加载数据到内存；add_physician_reports=True添加医生报告到数据集；
n_jobs=num_workers使用并行作业加载数据
"""
"""
from braindecode.datasets import TUHAbnormal
num_workers = 32
tuh_data_dir = preprocess_config.tuh_data_dir

dataset = TUHAbnormal(
    path=tuh_data_dir,
    recording_ids=None,#range(n_recordings_to_load),  # loads the n chronologically first recordings
    target_name="report",  # age, gender, pathology
    preload=False,
    add_physician_reports=True,
    n_jobs=num_workers)

# ## Preprocessing

# text preprocessing
from EEGClip.text_preprocessing import text_preprocessing
# 对数据集的描述文本进行预处理，返回一个包含预处理后文本的DataFrame
embs_df = text_preprocessing(dataset.description)
# 从DataFrame中只保留"report"列，过滤掉其他不需要的列
embs_df = embs_df[["report"]]

embs_df = pd.read_csv("scripts/text_embedding/embs_df.csv")
"""



# 备选模型列表
# 记录其他可用的文本嵌入模型
"""
bert-base-uncased
"BAAI/bge-large-en-v1.5"
medicalai/ClinicalBERT
hkunlp/instructor-xl
microsoft/BioGPT-Large-PubMedQA
microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
mixedbread-ai/mxbai-embed-large-v1
"""




#model.to(device)
# 批量文本嵌入处理
# 逐句处理和批量处理
# 将EEG报告文本进行批量嵌入生成，保存到CSV文件
"""
# 初始化嵌入列表
embs = []

# 遍历DataFrame中的每个报告文本，i是索引，r是报告文本内容
for i, r in enumerate(embs_df["report"]):
    if i % 100:
        print(i, "/", len(embs_df["report"]))
    # 对单个句子调用嵌入函数生成嵌入向量
    emb = sentence_embedder(r, tokenizer, model, max_length)

    # 将嵌入向量添加到列表中
    embs.append(emb)


# # do this in batches
# batch_size = 8

# embs = []
# for i in range(0, len(embs_df["report"]), batch_size):
#     print(i, "/", len(embs_df["report"]))
#     sentences = embs_df["report"].iloc[i:i + batch_size].to_list()

#     embs_batch = sentence_embedder(sentences, tokenizer, model, max_length)
#     embs.append(embs_batch)

# 将嵌入列表转换为numpy数组，然后转为列表，作为新列添加到DataFrame，列名为模型名称
embs_df[model_name] = np.array(embs).tolist()
# 将包含嵌入向量的DataFrame保存到CSV文件
embs_df.to_csv("scripts/text_embedding/embs_df.csv")
"""

### Encoding of zero-shot sentences
# 编码零样本句子

zc_sentences_dict = {
     "additional_sentences": {
        "s0": "excessive beta activity",
        "s1": "10 Hz alpha rhythm present during wakefulness",
        "s2": "Left temporal sharp waves",
        "s3": "Right frontal focal slowing",
        "s4": "10 Hz posterior dominant rhythm in occipital regions",
        "s5": "Left temporal sharp waves with a frequency of 6 Hz",
        "s6": "3 Hz spike-and-wave discharges",
        "s7": "Asymmetric alpha rhythm, higher amplitude on right",
        "s8": "10 Hz alpha rhythm maximal at O1 and O2",
        "s9": "6 Hz theta activity diffusely present",
        "s10": "25 Hz beta spindles in central regions",
        "s11": "Left temporal 4 Hz slow waves",


    }
    }



# 零样本句子字典：
# 1.病理状态：正常vs异常
# 2.性别：女性vs男性
# 3.年龄：超过50vs低于50
# 4.抗癫痫药物使用情况：未使用vs使用
# 5.性别+病理状态组合
# 6.年龄+病理状态组合
# 7.性别+年龄组合
"""
"pathological": {
    "s0": "This is a normal recording, from an healthy patient",
    "s1": "This an pathological recording, from a diseased patient  ",
},
"gender": {"s0": "The patient is female",
            "s1": "The patient is male"},
"under_50": {
    "s0": "The patient is over 50 years old",
    "s1": "The patient is under 50 years old",
},
"medication": {
    "s0": "The patient is not taking anti-epileptic medication",
    "s1": "The patient is taking anti-epileptic medication",
},
"pathological_gender": {
    "s00": "This is a normal recording, from an healthy male patient",
    "s01": "This is a normal recording, from an healthy female patient",
    "s10": "This an pathological recording, from a diseased male patient",
    "s11": "This an pathological recording, from a diseased female patient"
},
"pathological_under_50": {
    "s00": "This is a normal recording, from an healthy patient over 50 years old",
    "s01": "This is a normal recording, from an healthy patient under 50 years old",
    "s10": "This an pathological recording, from a diseased patient over 50 years old",
    "s11": "This an pathological recording, from a diseased patient under 50 years old"
},
"gender_under_50": {
    "s00": "The patient is a male over 50 years old",
    "s01": "The patient is a male under 50 years old",
    "s10": "The patient is a female over 50 years old",
    "s11": "The patient is a female under 50 years old"
},
"""


# 运行零样本嵌入，对于zc_sentences_dict定义的EEG相关句子生成嵌入向量，并保存到JSON文件
zc_sentences_model_emb_dict = copy.deepcopy(zc_sentences_dict)
for label, sentences in zc_sentences_dict.items():
    for l, s in sentences.items():
        zc_sentences_model_emb_dict[label][l] = sentence_embedder(s, tokenizer, model, max_length).tolist()




with open(preprocess_config.zc_sentences_emb_dict_path, "r") as f:
            zc_sentences_emb_dict = json.load(f)


zc_sentences_emb_dict[model_name] = zc_sentences_model_emb_dict


with open(preprocess_config.zc_sentences_emb_dict_path, "w") as f:
    json.dump(zc_sentences_emb_dict, f)
