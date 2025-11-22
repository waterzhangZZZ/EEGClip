import os       # 用于操作系统相关的功能，如文件路径操作
import socket   # 用于网络通信和主机名检测
import sys

import numpy as np
from braindecode.preprocessing import Preprocessor      # EEG数据预处理库

# 检测当前所在的平台，未来不一定会调用
is_windows = sys.platform.startswith('win')
is_linux = sys.platform.startswith('linux')

# __file__内置变量，包含当前执行脚本的文件名，例如"configs/preprocess_config.py"
# os.path.abspath(__file__)-获取绝对路径，例如"D:/code/EEGClip/configs/preprocess_config.py"
# os.path.dirname(path)-获取目录路径，返回路径的目录部分(去掉文件名)，例如"D:/code/EEGClip/configs"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))   # 获取当前文件所在目录的绝对路径

"""
paths to the different datasets/models. Feel free to modify
数据集/模型的路径，可以随意修改
"""
# 存放模型训练过程中产生的所有输出文件
results_dir = os.path.join(ROOT_DIR, "..", "results") + os.sep
# TUH EEG数据集目录-存放原始EEG脑电图数据文件，EDF文件格式
tuh_data_dir = os.path.join(ROOT_DIR, "..", "data", "tuh_eeg_abnormal", "v2.0.0", "edf") + os.sep
# 文本嵌入数据文件，CSV文件格式
embs_df_path = os.path.join(ROOT_DIR, "..", "scripts", "text_embedding", "embs_df.csv")
# 句子嵌入字典，json格式，提供与计算的文本特征，加速模型训练
zc_sentences_emb_dict_path = os.path.join(ROOT_DIR, "..", "scripts", "text_embedding", "zc_sentences_emb_dict.json")


# path to models trained on various tasks. Handy for baselines comparisons
# 用于各种任务训练的模型路径，方便与基础比较和模型加载

model_paths = {
    # EEGClip模型的128维版本检查点路径
    "eegclip128": results_dir
    + "wandb/EEGClip/1lgwz214/checkpoints/epoch=6-step=42609.ckpt",
    # "eegclip":results_dir + "models/EEGClip_100_medicalai/ClinicalBERT_64.ckpt",
    # 使用BERT编码器的EEGClip模型检查点
    "eegclip_bert": results_dir + "wandb/EEGClip/kg9zhzgx/checkpoints/epoch=11-step=10692.ckpt",
    # 使用Instructor编码器的EEGClip模型检查点
    "eegclip_instructor": results_dir + "wandb/EEGClip/xv65fc7j/checkpoints/epoch=15-step=14256.ckpt",
    # 标准EEGClip模型检查点
    "eegclip": results_dir
    + "wandb/EEGClip/v90rgytb/checkpoints/epoch=19-step=17820.ckpt",
    # 病理检测任务训练的模型检查点
    "pathological_task": results_dir
    + "wandb/EEGClip_few_shot/1vljui8s/checkpoints/epoch=9-step=7100.ckpt",
    # 50岁以下患者任务训练的模型检查点
    "under_50_task": results_dir
    + "wandb/EEGClip_few_shot/akl12j6m/checkpoints/epoch=9-step=7100.ckpt",
}

# EEG数据处理参数设置
n_max_minutes = 3       # 最大处理时长（分钟），超过此时间的EEG数据将被截断
sfreq = 100             # 重采样频率（Hz），将EEG数据重采样到100Hz

# 标准化的EEG通道名称列表
# 这些是按照国际10-20系统排列的脑电图电极位置
ar_ch_names = sorted(
    [
        "EEG A1-REF",
        "EEG A2-REF",
        "EEG FP1-REF",
        "EEG FP2-REF",
        "EEG F3-REF",
        "EEG F4-REF",
        "EEG C3-REF",
        "EEG C4-REF",
        "EEG P3-REF",
        "EEG P4-REF",
        "EEG O1-REF",
        "EEG O2-REF",
        "EEG F7-REF",
        "EEG F8-REF",
        "EEG T3-REF",
        "EEG T4-REF",
        "EEG T5-REF",
        "EEG T6-REF",
        "EEG FZ-REF",
        "EEG CZ-REF",
        "EEG PZ-REF",
    ]
)


# EEG数据预处理链
# 这些预处理器将按顺序应用于原始EEG数据
preprocessors = [
    # DONE : correct the order of preprocessing steps ?
    # 选择指定的EEG通道并按照指定顺序排列
    Preprocessor(fn="pick_channels", ch_names=ar_ch_names, ordered=True),
    # 截取EEG数据的时间窗口：从0秒开始，到n_max_minutes*60秒结束
    Preprocessor("crop", tmin=0, tmax=n_max_minutes * 60, include_tmax=True),
    # 将电压值从伏特转化为微伏（乘1e6）
    Preprocessor(fn=lambda x: x * 1e6, apply_on_array=True),
    # 将电压限制在-800到800微伏之间，去除异常值
    Preprocessor(fn=lambda x: np.clip(x, -800, 800), apply_on_array=True),
    # convert from volt to microvolt, directly modifying the numpy array
    # 将电压转换为微伏，直接修改 numpy 数组
    # 设置EEG参考电极：使用所有电极的平均值进行参考
    Preprocessor("set_eeg_reference", ref_channels="average"),
    # 数据归一化，将数据除以30进行缩放
    Preprocessor(fn=lambda x: x / 30, apply_on_array=True),  # this seemed best
    # 重采样：将数据重采样到指定的采样频率（100Hz）
    Preprocessor(fn="resample", sfreq=sfreq),
]
