import json

FILE= 'sample_data10_ori.json'
RESULT='sample_data10.json'

def clean_data(file, result):
    # 加载输入文件
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print('sample_data10_ori', len(data))
    output_data = []
    for item in data:
        result_item = []
        s_attr = item["schema"][0]["attributes"]
        for res in item["result"]:
            attr = res["attributes"]
            flag = True
            if len(attr)  == len(s_attr):
                for key, val in attr.items():
                    if val == "" or val is None or val == "null":
                        flag = False
            if flag:
                result_item.append(res)
        output_item = {
            "text": item["text"],
            "id": item["id"],
            "schema": item["schema"],
            "result": result_item
        }
        if output_item["result"] != []:
            output_data.append(output_item)

        # 保存结果到输出文件
        with open(result, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
    return output_data


if __name__ == "__main__":
    with open('sample_data6.json', 'r', encoding='utf-8') as f:
        data1 = json.load(f)
    print('sample_data6条数', len(data1))

    with open('sample_data9.json', 'r', encoding='utf-8') as f:
        data2 = json.load(f)
    print('sample_data9条数', len(data2))

    print("总条数", len(data1) +len(data2))

    clean_data(FILE, RESULT)

    with open('sample_data10.json', 'r', encoding='utf-8') as f:
        data3 = json.load(f)
    print('sample_data10条数', len(data3))