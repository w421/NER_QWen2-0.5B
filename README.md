# 金融文本信息抽取模型

基于Qwen2-0.5B模型的金融领域信息抽取项目，可以从金融文本中抽取各种实体及其属性。本项目采用LoRA技术进行高效微调，支持动态抽取各种实体类型和属性。

## 目录

- [项目介绍](#项目介绍)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [使用指南](#使用指南)
- [模型评估](#模型评估)
- [示例](#示例)
- [注意事项](#注意事项)

## 项目介绍

本项目旨在解决金融文本中实体及其属性的自动抽取任务。与传统的固定类型抽取任务不同，该模型能够适应多种抽取需求，根据提供的schema动态识别和抽取各种实体及其属性。

主要特点:
- 基于Qwen2-0.5B模型，占用资源少
- 使用LoRA技术进行参数高效微调，降低训练成本
- 支持动态schema，可以抽取任意指定类型的实体和属性
- 适用于各类金融文本，如财报、新闻、公告等
- 提供完整的训练、评估和推理流程

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- PEFT 0.4+
- 其他依赖见 `requirements.txt`

安装依赖:
```bash
pip install torch transformers datasets accelerate peft scikit-learn tqdm
```

## 快速开始

### 完整流程

使用以下命令运行完整的数据准备、训练、评估流程:

```bash
python run_pipeline.py
```

### 交互式演示

训练完成后，可以通过以下命令启动交互式演示:

```bash
python inference.py --mode interactive
```

或者通过pipeline脚本:

```bash
python run_pipeline.py --skip_data_prep --skip_training --skip_evaluation --interactive
```

## 项目结构

- `prepare_data.py`: 数据准备脚本，将JSON格式数据转换为模型训练格式
- `finetune_qwen_lora.py`: 使用LoRA方法微调Qwen2-0.5B模型
- `inference.py`: 模型推理脚本，支持交互式和批量推理
- `evaluate.py`: 评估模型在测试集上的性能
- `run_pipeline.py`: 运行完整工作流的主脚本
- `data/`: 数据目录，包含训练和验证数据
- `trained_model/`: 保存微调后的模型
- `evaluation_results/`: 评估结果输出目录

## 使用指南

### 数据准备

数据格式应为JSON，包含文本、抽取schema和抽取结果，如下所示:

```json
{
  "text": "金融文本内容",
  "schema": [
    {
      "entity_type": "实体类型",
      "attributes": ["属性1", "属性2"]
    }
  ],
  "result": [
    {
      "entities": ["实体名称"],
      "attributes": {
        "属性1": "值1",
        "属性2": "值2"
      }
    }
  ],
  "id": "唯一标识"
}
```

执行以下命令准备数据:

```bash
python prepare_data.py
```

### 模型微调

使用以下命令进行模型微调:

```bash
python finetune_qwen_lora.py
```

可以修改脚本中的配置参数，如`EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`等。

注意: 在CPU上训练会比较慢，建议使用较小的批次大小和训练轮数。

### 推理

#### 交互式推理

```bash
python inference.py --mode interactive
```

#### 批量推理

```bash
python inference.py --mode file --input your_data.json --output predictions.json
```

## 模型评估

执行以下命令评估模型性能:

```bash
python evaluate.py --test_file data/valid.json
```

评估指标包括:
- 精确匹配率(JSON完全匹配)
- 属性级别的精确率、召回率和F1分数

## 示例

### 交互式示例

1. 启动交互式推理:
```bash
python inference.py --mode interactive
```

2. 输入金融文本:
```
阿里巴巴集团宣布启动香港上市计划，拟发行5亿股普通股，每股定价不超过188港元，预计募集资金约940亿港元。
```

3. 输入schema:
```
IPO信息:公司名称,上市地点,发行股份数,每股定价,预计募资额
```

4. 模型会返回抽取结果:
```json
{
  "result": [
    {
      "entities": ["阿里巴巴集团香港上市"],
      "attributes": {
        "公司名称": "阿里巴巴集团",
        "上市地点": "香港",
        "发行股份数": "5亿股",
        "每股定价": "不超过188港元",
        "预计募资额": "约940亿港元"
      }
    }
  ]
}
```

## 注意事项

1. **训练数据量**: 本项目使用的示例数据集较小（50条），实际应用中建议使用更大规模的数据集（500条以上）获得更好效果。

2. **CPU训练**: 在CPU上训练模型会非常慢，单轮训练可能需要数小时。如有条件，建议使用GPU加速训练。

3. **模型大小**: Qwen2-0.5B是一个相对较小的模型，如果有更高的性能要求，建议使用更大的模型版本如Qwen2-7B。

4. **Schema设计**: schema的设计对抽取效果有重要影响，建议使用准确、简洁的实体类型和属性。

5. **内存占用**: 即使是0.5B参数的模型，在CPU上运行也需要较大内存(至少8GB)，请确保系统资源充足。
