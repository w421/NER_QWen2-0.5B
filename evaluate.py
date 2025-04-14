"""
评估微调后的Qwen2-0.5B模型在金融信息抽取任务上的性能
"""
import json
import os
import argparse
import numpy as np
from inference import load_model_and_tokenizer, generate_extraction, parse_json_result
from sklearn.metrics import precision_score, recall_score, f1_score

# 默认配置
DEFAULT_TEST_FILE = "data/valid.json"  # 默认使用验证集作为测试
OUTPUT_DIR = "evaluation_results"

def load_test_data(file_path):
    """加载测试数据集"""
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.endswith('.jsonl'):
            # JSONL格式
            data = [json.loads(line) for line in f.readlines()]
        else:
            # JSON格式
            data = json.load(f)
    
    print(f"加载了 {len(data)} 条测试数据")
    return data

def normalize_json(obj):
    """规范化JSON对象，使其更容易比较"""
    if isinstance(obj, dict):
        # 对字典内容进行排序
        return {k: normalize_json(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, list):
        # 对列表内容进行排序(如果可能)
        try:
            return sorted(normalize_json(x) for x in obj)
        except TypeError:
            # 如果列表元素不可比较，则保持原顺序
            return [normalize_json(x) for x in obj]
    else:
        # 基本类型直接返回
        return obj

def exact_match(pred, gold):
    """计算两个JSON对象是否完全匹配"""
    try:
        normalized_pred = normalize_json(pred)
        normalized_gold = normalize_json(gold)
        return normalized_pred == normalized_gold
    except Exception as e:
        print(f"比较错误: {str(e)}")
        return False

def attribute_level_metrics(predictions, ground_truths):
    """计算属性级别的精确率、召回率和F1"""
    all_pred_attrs = []
    all_gold_attrs = []
    
    for pred, gold in zip(predictions, ground_truths):
        # 收集预测的属性-值对
        pred_attrs = []
        for entity_pred in pred:
            if "attributes" in entity_pred and isinstance(entity_pred["attributes"], dict):
                for attr, value in entity_pred["attributes"].items():
                    pred_attrs.append((attr, str(value)))
        
        # 收集真实的属性-值对
        gold_attrs = []
        for entity_gold in gold:
            if "attributes" in entity_gold and isinstance(entity_gold["attributes"], dict):
                for attr, value in entity_gold["attributes"].items():
                    gold_attrs.append((attr, str(value)))
        
        # 将属性-值对转换为二进制标签
        all_attrs = list(set(pred_attrs).union(set(gold_attrs)))
        
        pred_binary = [1 if attr in pred_attrs else 0 for attr in all_attrs]
        gold_binary = [1 if attr in gold_attrs else 0 for attr in all_attrs]
        
        all_pred_attrs.extend(pred_binary)
        all_gold_attrs.extend(gold_binary)
    
    # 如果没有预测或真实标签，返回零
    if not all_pred_attrs or not all_gold_attrs:
        return 0, 0, 0
    
    # 计算指标
    precision = precision_score(all_gold_attrs, all_pred_attrs, zero_division=0)
    recall = recall_score(all_gold_attrs, all_pred_attrs, zero_division=0)
    f1 = f1_score(all_gold_attrs, all_pred_attrs, zero_division=0)
    
    return precision, recall, f1

def evaluate_model(model, tokenizer, test_data):
    """评估模型性能"""
    results = []
    exact_matches = 0
    
    predictions = []
    ground_truths = []
    
    for idx, item in enumerate(test_data):
        print(f"评估第 {idx+1}/{len(test_data)} 条数据...")
        
        # 解析数据
        if "messages" in item:  # JSONL格式的对话数据
            # 提取用户消息和助手响应
            for i, msg in enumerate(item["messages"]):
                if msg["role"] == "user":
                    # 从用户消息中提取文本和schema
                    user_content = msg["content"]
                    text_match = user_content.split("文本：", 1)
                    if len(text_match) > 1:
                        text = text_match[1].split("\n\nSchema")[0].strip()
                    else:
                        text = ""
                    
                    schema_match = user_content.split("Schema：", 1)
                    if len(schema_match) > 1:
                        schema_text = schema_match[1].strip()
                        # 解析schema文本
                        schema_lines = schema_text.split("\n")
                        schema = []
                        for line in schema_lines:
                            if line and "实体类型:" in line:
                                parts = line.split("实体类型:", 1)[1].split("属性:", 1)
                                if len(parts) == 2:
                                    entity_type = parts[0].strip()
                                    attributes = [a.strip() for a in parts[1].strip().split(",")]
                                    schema.append({"entity_type": entity_type, "attributes": attributes})
                    else:
                        schema = []
                elif msg["role"] == "assistant" and i > 0:  # 确保这是对用户消息的回复
                    # 提取助手响应作为标准答案
                    gold_response = msg["content"]
                    gold_result = parse_json_result(gold_response)
        else:  # JSON格式的单条数据
            text = item.get("text", "")
            schema = item.get("schema", [])
            gold_result = item.get("result", [])
        
        # 使用模型生成抽取结果
        generation = generate_extraction(model, tokenizer, text, schema)
        
        # 解析生成的JSON
        pred_result = parse_json_result(generation)
        
        # 检查是否完全匹配
        is_match = exact_match(pred_result, gold_result)
        if is_match:
            exact_matches += 1
        
        # 收集结果用于属性级别评估
        predictions.append(pred_result)
        ground_truths.append(gold_result)
        
        # 记录详细结果
        result = {
            "id": item.get("id", f"test_{idx}"),
            "text": text,
            "schema": schema,
            "gold": gold_result,
            "prediction": pred_result,
            "is_match": is_match
        }
        results.append(result)
    
    # 计算整体指标
    exact_match_rate = exact_matches / len(test_data) if test_data else 0
    
    # 计算属性级别指标
    try:
        attr_precision, attr_recall, attr_f1 = attribute_level_metrics(predictions, ground_truths)
    except Exception as e:
        print(f"计算属性级别指标时出错: {str(e)}")
        attr_precision, attr_recall, attr_f1 = 0, 0, 0
    
    # 汇总评估结果
    summary = {
        "test_size": len(test_data),
        "exact_match_count": exact_matches,
        "exact_match_rate": exact_match_rate,
        "attribute_precision": attr_precision,
        "attribute_recall": attr_recall,
        "attribute_f1": attr_f1
    }
    
    return results, summary

def main():
    parser = argparse.ArgumentParser(description="评估金融信息抽取模型")
    parser.add_argument('--test_file', type=str, default=DEFAULT_TEST_FILE, 
                        help="测试数据文件路径(JSON或JSONL格式)")
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help="评估结果输出目录")
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer()
    print("模型加载完成")
    
    # 加载测试数据
    test_data = load_test_data(args.test_file)
    
    # 评估模型
    print("开始评估...")
    results, summary = evaluate_model(model, tokenizer, test_data)
    
    # 保存评估结果
    results_file = os.path.join(args.output_dir, "detailed_results.json")
    summary_file = os.path.join(args.output_dir, "evaluation_summary.json")
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # 打印评估摘要
    print("\n=== 评估摘要 ===")
    print(f"测试样本数: {summary['test_size']}")
    print(f"精确匹配数: {summary['exact_match_count']}")
    print(f"精确匹配率: {summary['exact_match_rate']:.4f}")
    print(f"属性级精确率: {summary['attribute_precision']:.4f}")
    print(f"属性级召回率: {summary['attribute_recall']:.4f}")
    print(f"属性级F1分数: {summary['attribute_f1']:.4f}")
    print(f"详细结果已保存到: {results_file}")
    print(f"评估摘要已保存到: {summary_file}")

if __name__ == "__main__":
    main()
