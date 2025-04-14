"""
准备数据集，将JSON格式的数据转换为指令微调格式并划分训练/验证集
"""
import json
import random
import os
from sklearn.model_selection import train_test_split

# 输入输出文件配置
INPUT_FILE = "sample_data6.json"  # 包含50条样本数据的JSON文件
OUTPUT_DIR = "data"  # 输出目录
TRAIN_FILE = os.path.join(OUTPUT_DIR, "train.json")  # 训练数据集
VALID_FILE = os.path.join(OUTPUT_DIR, "valid.json")  # 验证数据集
TEST_SIZE = 0.2  # 验证集比例

# Qwen2系列模型的对话格式模板
SYSTEM_PROMPT = "你是一个金融信息抽取助手。请根据提供的schema从文本中抽取实体及其属性，并以JSON格式输出结果。"

def load_data(file_path):
    """加载原始JSON数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"加载了 {len(data)} 条原始数据")
    return data

def format_schema(schema):
    """格式化schema为文本形式"""
    formatted_schema = []
    for s in schema:
        entity_type = s.get("entity_type", s.get("entity", "实体类型"))  # 兼容两种可能的字段名
        attributes = s.get("attributes", [])
        formatted_schema.append(f"实体类型: {entity_type}, 属性: {', '.join(attributes)}")
    return '\n'.join(formatted_schema)

def format_result(result):
    """格式化抽取结果为JSON字符串"""
    # 确保结果格式统一
    return json.dumps(result, ensure_ascii=False, indent=2)

def convert_to_chat_format(data):
    """将原始数据转换为模型所需的对话格式"""
    formatted_data = []
    
    for item in data:
        # 提取字段
        text = item["text"]
        schema = item["schema"]
        result = item["result"]
        
        # 构建用户指令，包含文本和schema
        user_instruction = f"请从以下金融文本中抽取实体及其属性，按照给定的schema进行抽取：\n\n文本：{text}\n\nSchema：\n{format_schema(schema)}\n\n请以JSON格式输出结果。"
        
        # 构建模型期望响应
        expected_response = format_result(result)
        
        # 构建符合Qwen2系列模型格式的对话样本
        chat_sample = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_instruction},
                {"role": "assistant", "content": expected_response}
            ]
        }
        
        formatted_data.append(chat_sample)
    
    print(f"转换了 {len(formatted_data)} 条对话格式数据")
    return formatted_data

def save_jsonl(data, file_path):
    """将数据保存为JSONL格式"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"已保存到 {file_path}")

def main():
    # 加载原始数据
    raw_data = load_data(INPUT_FILE)
    
    # 转换为模型所需的对话格式
    formatted_data = convert_to_chat_format(raw_data)
    
    # 划分训练集和验证集
    train_data, valid_data = train_test_split(formatted_data, test_size=TEST_SIZE, random_state=42)
    print(f"划分数据集: 训练集 {len(train_data)} 条，验证集 {len(valid_data)} 条")
    
    # 保存为JSONL格式
    save_jsonl(train_data, TRAIN_FILE)
    save_jsonl(valid_data, VALID_FILE)
    
    print("数据准备完成！")

if __name__ == "__main__":
    main()
