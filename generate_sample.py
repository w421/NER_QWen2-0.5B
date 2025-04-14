import json
import requests
import time
import uuid
import random
import os

# API配置
API_URL = "https://api.siliconflow.cn/v1"
API_KEY = "sk-rgjzpxswicphsnjyzrhhlqpwglgwtctildwwpknpeqxrzmtl"
MODEL = "deepseek-ai/DeepSeek-V3"

# 输出文件配置
OUTPUT_FILE = "sample_data5.json"
NUM_SAMPLES = 50  # 生成数据总数
BATCH_SIZE = 10   # 每次API调用生成的数据条数

# 金融文本类型
TEXT_TYPES = [
    "公司财报", "行业分析", "市场评论", "政策解读", "研究报告",
    "新闻公告", "投资建议", "风险提示", "交易通知", "监管通告"
]

def generate_batch_prompt(batch_size=10):
    """创建用于批量生成金融数据的prompt"""
    # 随机选择几种不同的文本类型
    selected_types = random.sample(TEXT_TYPES, min(batch_size, len(TEXT_TYPES)))
    if len(selected_types) < batch_size:
        # 如果类型不够，则重复使用一些类型
        selected_types.extend(random.choices(TEXT_TYPES, k=batch_size-len(selected_types)))
    
    prompt = f"""作为一个专业的金融信息抽取系统，请帮我一次性生成{batch_size}条不同的金融数据。

对于每条数据，请完成以下任务：
1. 首先，生成一段真实、专业的中文金融文本，长度在50-150字之间。每条数据的文本主题和内容应该不同。
2. 然后，根据生成的文本内容，设计1个合适的实体抽取schema，schema包含一个实体类型和2-5个相关属性。
3. 最后，严格按照设计的schema，从文本中抽取出所有相关实体及其属性值。

你必须严格按照以下格式输出一个包含{batch_size}个数据项的JSON数组：

```json
[
  {{
    "text": "第1条金融文本",
    "schema": [
      {{
        "entity_type": "实体类型",
        "attributes": ["属性1", "属性2", ...]
      }}
    ],
    "result": [
      {{
        "entities": ["实体名称"],
        "attributes": {{
          "属性1": "值1",
          "属性2": "值2",
          ...
        }}
      }},
      // 如果有多个满足schema的实体，需要列出所有实体的抽取结果
    ]
  }},
  // ... 共{batch_size}个数据项
]
```

请确保：
1. 所有生成的金融文本各不相同且主题多样化，可以包括以下类型的内容：{', '.join(selected_types)}
2. 每个schema必须根据其对应文本内容设计，不能是固定的或重复的。
3. 实体类型和属性必须与文本内容密切相关，在文本中能够找到对应的实体和属性值。
4. 对于每条数据，result必须包含文本中所有符合schema的实体及其属性值。
5. 所有实体和属性名称、属性值必须使用中文。
6. 只输出JSON数组，不要有其他解释文字。
"""
    return prompt

def call_llm_api(prompt):
    """调用LLM API生成批量数据"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    data = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.8,  # 稍微提高温度以增加多样性
        "max_tokens": 4096   # 增加token上限以适应批量生成
    }
    
    try:
        response = requests.post(API_URL + "/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        # 提取JSON部分
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()
        return json.loads(content)
    except Exception as e:
        print(f"API调用出错: {str(e)}")
        print(f"返回内容: {content if 'content' in locals() else '无内容'}")
        return None

def validate_batch_data(data_batch):
    """验证批量数据格式是否正确"""
    if not isinstance(data_batch, list):
        print("验证失败: 返回的不是JSON数组")
        return False
    
    valid_items = []
    for data in data_batch:
        if not isinstance(data, dict):
            continue
        
        required_fields = ["text", "schema", "result"]
        if not all(field in data for field in required_fields):
            continue
        
        # 验证schema
        if not isinstance(data["schema"], list) or len(data["schema"]) == 0:
            continue
        
        schema_valid = True
        for schema in data["schema"]:
            # 注意这里修改了字段名从 entity 到 entity_type，适应文件中最新的格式
            if "entity_type" not in schema or "attributes" not in schema:
                schema_valid = False
                break
            if not isinstance(schema["attributes"], list):
                schema_valid = False
                break
        
        if not schema_valid:
            continue
        
        # 验证result
        if not isinstance(data["result"], list):
            continue
        
        result_valid = True
        for item in data["result"]:
            if "entities" not in item or "attributes" not in item:
                result_valid = False
                break
            if not isinstance(item["entities"], list) or not isinstance(item["attributes"], dict):
                result_valid = False
                break
        
        if not result_valid:
            continue
        
        valid_items.append(data)
    
    print(f"验证通过: {len(valid_items)}/{len(data_batch)} 条数据有效")
    return valid_items

def initialize_output_file(output_file):
    """初始化输出文件为空数组"""
    # 如果文件不存在或者为空，写入一个空数组
    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False)

def append_to_json_file(new_data, output_file):
    """将新数据追加到JSON数组文件中"""
    try:
        # 读取现有数据
        with open(output_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    data = []
            except json.JSONDecodeError:
                data = []
        
        # 添加新数据
        data.extend(new_data)
        
        # 写回文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        print(f"追加到文件时出错: {str(e)}")
        return False

def generate_dataset(total_samples, batch_size, output_file):
    """分批生成指定数量的测试数据集并立即追加到文件"""
    initialize_output_file(output_file)
    total_generated = 0
    
    # 计算需要的批次数
    num_batches = (total_samples + batch_size - 1) // batch_size  # 向上取整
    
    for batch in range(num_batches):
        current_batch_size = min(batch_size, total_samples - total_generated)
        print(f"正在生成第 {batch+1}/{num_batches} 批数据 ({current_batch_size}条)...")
        
        # 生成批量数据的prompt
        prompt = generate_batch_prompt(current_batch_size)
        
        # 重试机制
        max_retries = 3
        batch_data = []
        
        for retry in range(max_retries):
            try:
                data_batch = call_llm_api(prompt)
                if data_batch:
                    valid_items = validate_batch_data(data_batch)
                    if valid_items:
                        # 添加id字段
                        for item in valid_items:
                            item["id"] = str(uuid.uuid4())[:8]  # 生成唯一ID
                        
                        batch_data = valid_items
                        break
                
                print(f"数据验证失败，重试 {retry+1}/{max_retries}")
            except Exception as e:
                print(f"处理错误: {str(e)}, 重试 {retry+1}/{max_retries}")
            
            # 等待一段时间后重试
            time.sleep(3)
        
        # 如果成功生成了数据，立即追加到文件
        if batch_data:
            print(f"成功生成 {len(batch_data)} 条数据，正在写入文件...")
            append_to_json_file(batch_data, output_file)
            total_generated += len(batch_data)
            print(f"当前进度: {total_generated}/{total_samples} 条数据")
        
        # 控制API调用速率
        time.sleep(2)
    
    return total_generated

def main():
    print(f"开始生成包含{NUM_SAMPLES}条数据的金融测试数据集，每批{BATCH_SIZE}条...")
    total_generated = generate_dataset(NUM_SAMPLES, BATCH_SIZE, OUTPUT_FILE)
    print(f"数据生成完成! 共生成 {total_generated} 条数据，已保存到 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
