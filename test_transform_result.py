import json

FILE='predictions.json'
RESULT='result.json'

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


if __name__ == "__main__":
    transform_prediction_to_result(FILE, RESULT)