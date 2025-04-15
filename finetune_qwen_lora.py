"""
使用LoRA方法在CPU上微调Qwen2-0.5B模型，用于金融文本信息抽取任务
"""
import os
import torch
import json
from datasets import Dataset,load_dataset
from sympy import print_glsl
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    logging
)
from peft import (
    get_peft_model,
    LoraConfig,
    prepare_model_for_kbit_training,
    TaskType
)
from tqdm import tqdm

# 日志配置
logging.set_verbosity_info()

# 模型和数据配置
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
# MODEL_NAME = "../Qwen2.5-3B-Instruct"
# MODEL_NAME = "Qwen/Qwen2-0.5B"
OUTPUT_DIR = "trained_model"
TRAIN_FILE = "data/train.json"
VALID_FILE = "data/valid.json"

# 训练配置
MAX_LENGTH = 2048  # 最大序列长度
BATCH_SIZE = 2  # 在CPU上需要小批次
GRADIENT_ACCUMULATION_STEPS = 8  # 梯度累积以增大有效批次大小
LEARNING_RATE = 2e-4  # 学习率
EPOCHS = 5  # 训练轮数
LORA_RANK = 8  # LoRA秩
LORA_ALPHA = 16  # LoRA缩放因子
LORA_DROPOUT = 0.1  # LoRA dropout
WARMUP_RATIO = 0.1  # 预热比例

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_jsonl_dataset(file_path):
    """加载JSONL格式的数据集"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return Dataset.from_list(data)


def preprocess_data(examples, tokenizer):
    """改进后的预处理函数，包含严格验证"""
    texts = []
    valid_indices = []  # 记录有效样本的原始索引

    for idx, chat in enumerate(examples["messages"]):
        try:
            # 验证每条消息格式
            assert isinstance(chat, list), "messages必须是列表"
            for msg in chat:
                assert isinstance(msg, dict), "消息必须是字典"
                assert "role" in msg and "content" in msg, "缺少role/content字段"
                assert msg["role"] in ("system", "user", "assistant"), f"非法role: {msg['role']}"

            # 应用模板
            text = tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=False
            )
            if not isinstance(text, str) or len(text) < 10:
                raise ValueError("生成文本无效")

            texts.append(text)
            valid_indices.append(idx)
        except Exception as e:
            print(f"过滤样本{idx}: {str(e)}")
            continue

    if not texts:
        raise ValueError("所有样本均被过滤，请检查数据格式")

    # Tokenization
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt"
    )

    # 标签处理（仅计算assistant部分的loss）
    labels = tokenized["input_ids"].clone()
#     print('有效输入', valid_indices)
    # for i, idx in enumerate(valid_indices):
    #     assistant_content = None
    #     for msg in examples["messages"][idx]:
    #         if msg["role"] == "assistant":
    #             assistant_content = msg["content"]
    #             break
    #
    #     if assistant_content:
    #         assistant_tokens = tokenizer(assistant_content, add_special_tokens=False).input_ids
    #         seq_len = len(tokenized["input_ids"][i])
    #         assistant_len = len(assistant_tokens)
    #         labels[i, :seq_len - assistant_len] = -100
    #     else:
    #         labels[i, :] = -100

    for i, chat in enumerate(examples["messages"]):
        # 忽略padding
        labels[i, tokenized["attention_mask"][i] == 0] = -100

    tokenized["labels"] = labels
#     print('tokenlized结果', tokenized)
    # return tokenized
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": tokenized["labels"]
    }
def create_lora_config():
    """创建LoRA配置"""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        # 为所有线性层应用LoRA
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    )

def main():
    print(f"开始微调 {MODEL_NAME} 模型...")
    print(f"使用设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")

    # 加载tokenizer
    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"已加载tokenizer: {tokenizer.__class__.__name__}")

    # 加载数据集
    print("加载数据集...")
    train_data = load_jsonl_dataset(TRAIN_FILE)
    valid_data = load_jsonl_dataset(VALID_FILE)
    print(f"训练集: {len(train_data)}条, 验证集: {len(valid_data)}条")

    # 加载模型
    print("加载基础模型...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",  # 自动选择设备
        torch_dtype=torch.float32,  # CPU上使用float32
        trust_remote_code=True
    )
    print(f"模型参数数量: {model.num_parameters()}")

    # 将模型修改为LoRA版本
    print("创建LoRA模型...")
    peft_config = create_lora_config()
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 创建训练参数
    print("配置训练参数...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        report_to="none",  # 禁用Wandb等报告
    )

    # 创建数据集
    print("创建数据集...")
    train_dataset = {"messages": [item["messages"] for item in train_data]}
    valid_dataset = {"messages": [item["messages"] for item in valid_data]}

    # 预处理数据
    print("预处理训练数据...")
    tokenized_train = preprocess_data(train_dataset, tokenizer)
    tokenized_valid = preprocess_data(valid_dataset, tokenizer)

    tokenized_train = Dataset.from_dict({
        "input_ids": tokenized_train["input_ids"],
        "attention_mask": tokenized_train["attention_mask"],
        "labels": tokenized_train["labels"]
    })


    tokenized_valid = Dataset.from_dict({
        "input_ids": tokenized_valid["input_ids"],
        "attention_mask": tokenized_valid["attention_mask"],
        "labels": tokenized_valid["labels"]
    })

    # 创建数据收集器和训练器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        data_collator=data_collator,
    )
    
    # 开始训练
    print("开始训练...")
    print(f"Dataset length: {len(tokenized_train)}", tokenized_train[0])
    trainer.train()
    
    # 保存模型
    print("保存模型...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"微调完成！模型已保存到 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()