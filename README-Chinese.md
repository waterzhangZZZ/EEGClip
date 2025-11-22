# EEG-CLIP：从自然语言描述中学习脑电图表示

EEG-CLIP 是一个对比学习框架，将脑电图时间序列数据与其对应的临床文本描述在共享嵌入空间中对齐。受 CLIP（对比语言-图像预训练）的启发，EEG-CLIP 能够实现多功能的脑电图表示学习，以在低数据量情况下改进病理检测和其他下游任务。

<p align="center">
  <img src="results/publication_plots/few_shot_under_50.png" alt="EEG-CLIP 在年龄分类任务上的结果" width="600"/>
</p>

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/tidiane-camaret/EEGClip.git
cd EEGClip

# 安装依赖
pip install -r requirements.txt
pip install -e .
```

*注意：您可能需要修改 `configs/preprocess_config.py` 中的数据集和预训练模型路径*

### 1. 预处理

使用 ClinicalBERT 模型生成文本嵌入：

```bash
python scripts/text_embedding/text_embedding.py
```

### 2. 训练 EEGClip

在 TUH EEG 异常语料库上训练 EEGClip 模型：

```bash
python scripts/eegclip_train_eval.py
```

### 3. 评估

#### 3.1 标准分类

使用冻结的脑电图编码器训练和评估分类器：

```bash
python scripts/classif/classification_tuh.py
```

选项：
- `--task_name`: 分类任务（病理状态、年龄、性别等）
- `--train_frac`: 控制少样本实验的训练数据大小
- `--weights`: 选择预训练权重
- `--freeze_encoder`: 切换脑电图编码器冻结

#### 3.2 零样本分类

评估模型仅使用文本提示对脑电图记录进行分类的能力，无需额外训练：

```bash
python scripts/classif/classification_zero_shot_tuh.py
```

## 引用

如果您在研究中使用了 EEG-CLIP，请引用我们的论文：

```bibtex
@ARTICLE{10.3389/frobt.2025.1625731,
AUTHOR={Camaret Ndir, Tidiane  and Schirrmeister , Robin T.  and Ball , Tonio },
TITLE={EEG-CLIP: learning EEG representations from natural language descriptions},
JOURNAL={Frontiers in Robotics and AI},
VOLUME={Volume 12 - 2025},
YEAR={2025},
URL={https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1625731},
DOI={10.3389/frobt.2025.1625731},
ISSN={2296-9144}]
```

## 数据集

本工作使用天普大学医院 (TUH) 脑电图语料库：

```bibtex
@article{obeid_temple_2016,
  title={The Temple University Hospital EEG Data Corpus},
  volume={10},
  journal={Frontiers in Neuroscience},
  author={Obeid, Iyad and Picone, Joseph},
  year={2016},
  pages={196}
}
```

## 许可证

本项目采用 MIT 许可证 - 有关详细信息，请参阅 LICENSE 文件。