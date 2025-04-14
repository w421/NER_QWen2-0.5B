import json
import requests
import time
import uuid
import random
from tqdm import tqdm

# API配置
API_URL = "https://api.siliconflow.cn/v1"
API_KEY = "sk-rgjzpxswicphsnjyzrhhlqpwglgwtctildwwpknpeqxrzmtl"
MODEL = "alibaba/Qwen2-7B-Instruct"

# 输出文件配置
OUTPUT_FILE = "financial_test_data.json"
NUM_SAMPLES = 1000  # 生成的数据条数

# 金融实体类型和属性的示例，用于指导LLM生成多样化的schema
ENTITY_TYPE_EXAMPLES = [
    "公司", "银行", "基金", "股票", "债券", "证券", "投资项目", "资产",
    "金融产品", "保险", "期货", "货币", "指数", "交易所", "投资者", "高管",
    "项目", "经济指标", "行业", "政策", "监管机构", "风险", "股东", "分红",
    "财务指标", "市场", "交易", "并购", "融资", "利率", "汇率"
]

ATTRIBUTE_EXAMPLES = [
    "名称", "代码", "价格", "涨跌幅", "市值", "收益率", "成交量", "总资产",
    "净资产", "营收", "利润", "增长率", "负债率", "估值", "评级", "风险等级",
    "所属行业", "成立时间", "上市时间", "地区", "持股比例", "持有期限",
    "总收入", "净利润", "每股收益", "毛利率", "净利率", "资产负债率",
    "流动比率", "速动比率", "净资产收益率", "总资产收益率", "营业收入增长率",
    "净利润增长率", "市盈率", "市净率", "市销率", "股息率", "发行价",
    "发行规模", "募资金额", "投资回报率", "到期时间", "期限", "违约率"
]

# 金融领域文本类型
TEXT_TYPES = [
    "公司财报", "行业分析", "市场评论", "政策解读", "研究报告",
    "新闻公告", "投资建议", "风险提示", "交易通知", "监管通告"
]

def generate_prompt(text_type=None):
    """创建用于生成金融数据的prompt"""
    if text_type is None:
        text_type = random.choice(TEXT_TYPES)
    
    prompt = f"""作为一个专业的金融信息抽取系统，请帮我完成以下任务：

1. 首先，生成一段真实、专业的中文金融{text_type}文本，长度在50-150字之间。文本应该包含各种金融实体和它们的属性。

2. 然后，根据你生成的文本内容，设计1个合适的实体抽取schema，schema包含一个实体类型和2-5个相关属性。实体类型应该是文本中实际出现的实体类型，属性是这些实体实际具有的属性，参考示例格式但内容必须随文本变化。

3. 最后，请严格按照你设计的schema，从你生成的文本中抽取出所有相关实体及其属性值，并按照指定的JSON格式输出结果。

你必须严格按照以下格式输出：

```json
{{
  "text": "你生成的金融文本",
  "schema": [
    {{
      "entity_type": "实体类型",
      "attributes": ["属性1", "属性2", ...]
    }},
    // 每个text只有一个schema，schema中只有一个entity_type,就是说每个文本抽取一个实体类型就行，不需要多个实体类型
  ],
  "result": [
    {{
      "entities": ["实体名称1"],
      "attributes": {{
        "属性1": "值1",
        "属性2": "值2",
        ...
      }}
    }},
    // 如果文本中有多个满足schema的实体，需要列出所有实体的抽取结果
  ]
}}
```

注意：
1. text字段必须是你自己生成的金融文本，不要使用已有的文本。
2. schema必须根据你生成的文本内容设计，而不是固定的。
3. 实体类型和属性必须与文本内容密切相关，并且在文本中能够找到对应的实体和属性值。
4. result必须包含文本中所有符合schema的实体及其属性值。
5. 只输出JSON格式结果，不要有其他任何解释文字。
6. 所有实体和属性名称、属性值必须使用中文。
"""
    return prompt

def call_llm_api(prompt):
    """调用LLM API生成数据"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    data = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 2048
    }
    
    try:
        response = requests.post(API_URL + "/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        # 提取JSON部分
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        return json.loads(content)
    except Exception as e:
        print(f"API调用出错: {str(e)}")
        return None

def validate_data(data):
    """验证数据格式是否正确"""
    if not isinstance(data, dict):
        return False
    
    required_fields = ["text", "schema", "result"]
    if not all(field in data for field in required_fields):
        return False
    
    # 验证schema
    if not isinstance(data["schema"], list) or len(data["schema"]) == 0:
        return False
    
    for schema in data["schema"]:
        if "entity_type" not in schema or "attributes" not in schema:
            return False
        if not isinstance(schema["attributes"], list):
            return False
    
    # 验证result
    if not isinstance(data["result"], list):
        return False
    
    for item in data["result"]:
        if "entities" not in item or "attributes" not in item:
            return False
        if not isinstance(item["entities"], list) or not isinstance(item["attributes"], dict):
            return False
    
    return True

def generate_dataset(num_samples):
    """生成指定数量的测试数据集"""
    dataset = []
    
    for i in tqdm(range(num_samples), desc="生成数据"):
        # 随机选择一种金融文本类型
        text_type = random.choice(TEXT_TYPES)
        prompt = generate_prompt(text_type)
        
        # 重试机制
        max_retries = 3
        for retry in range(max_retries):
            try:
                data = call_llm_api(prompt)
                if data and validate_data(data):
                    # 添加id字段
                    data_with_id = {
                        "id": str(uuid.uuid4())[:8],  # 生成唯一ID
                        "text": data["text"],
                        "schema": data["schema"],
                        "result": data["result"]
                    }
                    dataset.append(data_with_id)
                    break
                else:
                    print(f"数据验证失败，重试 {retry+1}/{max_retries}")
            except Exception as e:
                print(f"处理错误: {str(e)}, 重试 {retry+1}/{max_retries}")
            
            # 等待一段时间后重试
            time.sleep(2)
        
        # 控制API调用速率
        time.sleep(1)
    
    return dataset

def save_dataset(dataset, output_file):
    """保存数据集到JSON文件"""
    with open(output_file, 'a', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"已生成数据集文件: {output_file}")

def main():
    print(f"开始生成包含{NUM_SAMPLES}条数据的金融测试数据集...")
    dataset = generate_dataset(NUM_SAMPLES)
    save_dataset(dataset, OUTPUT_FILE)
    print(f"成功生成{len(dataset)}条数据!")

if __name__ == "__main__":
    main()
