"""
使用微调后的Qwen2.5模型进行金融信息抽取推理
"""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
import re

# 模型配置
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
# BASE_MODEL = "../Qwen2.5-3B-Instruct"
LORA_MODEL = "trained_model"

# 系统提示词
SYSTEM_PROMPT = "你是一个金融信息抽取助手。请根据提供的schema从文本中抽取实体及其属性，并以JSON格式输出结果。"

def format_schema(schema):
    """格式化schema为文本形式"""
    formatted_schema = []
    for s in schema:
        entity_type = s.get("entity_type", s.get("entity", "实体类型"))
        attributes = s.get("attributes", [])
        formatted_schema.append(f"实体类型: {entity_type}, 属性: {', '.join(attributes)}")
    return '\n'.join(formatted_schema)

def load_model_and_tokenizer(base_model=BASE_MODEL, lora_model=LORA_MODEL):
    """加载模型和分词器"""
    print(f"加载基础模型 {base_model}...")

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.float32,  # CPU上使用float32
        trust_remote_code=True
    )

    print(f"加载LoRA适配器 {lora_model}...")
    # 加载LoRA适配器
    model = PeftModel.from_pretrained(base_model, lora_model)
    model.eval()  # 设置为评估模式

    return model, tokenizer

def generate_extraction(model, tokenizer, text, schema, max_new_tokens=1024, temperature=0.1):
    """生成金融信息抽取结果"""
    # 构建用户指令
    user_instruction = f"请从以下金融文本中抽取实体及其属性，按照给定的schema进行抽取：\n\n文本：{text}\n\nSchema：\n{format_schema(schema)}\n\n请以JSON格式输出结果。"

    # 构建messages列表
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_instruction}
    ]

    # 使用chat_template格式化输入
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 对输入进行标记化
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 生成回复
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),
            pad_token_id=tokenizer.pad_token_id,
        )

    # 解码并处理回复
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 提取assistant部分的回复
    assistant_text = generated_text.split("<|assistant|>")[-1].strip()

    return assistant_text

def parse_json_result(text):
    """从文本中提取JSON结果"""
    # print('抽取结果---------', text)
    try:
        # 尝试直接解析文本
        data = json.loads(text)
        return data
    except json.JSONDecodeError:
        # 尝试用正则表达式匹配 JSON 部分
        json_match = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError as e:
                return {"error": f"JSON解析错误: {str(e)}", "text": text}
        else:
            return {"error": "无法从文本中提取有效的JSON"}
    # except json.JSONDecodeError:
    #     # 如果直接解析失败，尝试从文本中提取JSON部分
    #     try:
    #         # 寻找可能的JSON开始和结束位置
    #         start_idx = text.find('{')
    #         end_idx = text.rfind('}') + 1
    #
    #         if start_idx >= 0 and end_idx > start_idx:
    #             json_str = text[start_idx:end_idx]
    #             return json.loads(json_str)
    #         else:
    #             # 检查是否有JSON数组
    #             start_idx = text.find('[')
    #             end_idx = text.rfind(']') + 1
    #
    #             if start_idx >= 0 and end_idx > start_idx:
    #                 json_str = text[start_idx:end_idx]
    #                 return json.loads(json_str)
    #             else:
    #                 return {"error": "无法从文本中提取有效的JSON"}
    #     except Exception as e:
    #         return {"error": f"JSON解析错误: {str(e)}", "text": text}

def extract_from_file(model, tokenizer, input_file, output_file):
    """从文件中批量提取信息"""
    # 加载输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []

    # 处理每条数据
    for idx, item in enumerate(data):
        print(f"处理第 {idx+1}/{len(data)} 条数据...")

        text = item["text"]
        schema = item["schema"]

        # 生成抽取结果
        extraction_text = generate_extraction(model, tokenizer, text, schema)

        # 解析JSON结果
        extraction_result = parse_json_result(extraction_text)

        # 构建结果字典
        result = {
            "id": item.get("id", f"test_{idx}"),
            "text": text,
            "schema": schema,
            "ground_truth": item.get("result", []),
            "prediction": extraction_result
        }

        results.append(result)

    # 保存结果到输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"已处理 {len(results)} 条数据，结果保存到 {output_file}")
    return results

def interactive_demo(model, tokenizer):
    """交互式演示"""
    print("\n=== 金融信息抽取交互演示 ===")
    print("输入'exit'退出")

    while True:
        print("\n请输入金融文本:")
        text = input()
        if text.lower() == 'exit':
            break

        print("\n请输入schema (格式: 实体类型:属性1,属性2,...), 例如: 公司:公司名称,市值,行业")
        schema_input = input()
        if schema_input.lower() == 'exit':
            break

        try:
            # 解析schema输入
            schema_parts = schema_input.split(':')
            entity_type = schema_parts[0].strip()
            attributes = [attr.strip() for attr in schema_parts[1].split(',')]

            schema = [{"entity_type": entity_type, "attributes": attributes}]

            # 生成抽取结果
            print("\n正在生成...")
            extraction_text = generate_extraction(model, tokenizer, text, schema)

            print("\n=== 抽取结果 ===")
            print(extraction_text)

        except Exception as e:
            print(f"错误: {str(e)}")

def transform_prediction_to_result(file, result):
    # 加载输入文件
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    output_data = []
    for item in data:
        result_item = item["prediction"]
        output_item = {
            "id": item["id"],
            "result": result_item
        }
        output_data.append(output_item)

        # 保存结果到输出文件
        with open(result, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
    return output_data

def main():
    parser = argparse.ArgumentParser(description="金融信息抽取推理工具")
    parser.add_argument('--mode', type=str, choices=['interactive', 'file'], default='interactive',
                        help="运行模式: interactive(交互式) 或 file(从文件)")
    parser.add_argument('--input', type=str, help="输入文件路径(JSON格式)")
    parser.add_argument('--output', type=str, default="predictions.json", help="输出文件路径")
    parser.add_argument('--result', type=str, default="result.json", help="输出文件路径")
    args = parser.parse_args()

    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer()
    print("模型加载完成！")

    if args.mode == 'interactive':
        interactive_demo(model, tokenizer)
    else:
        if not args.input:
            print("错误: 在file模式下必须指定输入文件。")
            return
        extract_from_file(model, tokenizer, args.input, args.output)
        transform_prediction_to_result(args.output, args.result)
if __name__ == "__main__":
    main()
