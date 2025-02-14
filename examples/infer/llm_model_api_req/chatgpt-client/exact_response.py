import re
import os
import sys
import pandas as pd

def extract_and_format_content(line, wenjuan_type_dict):
    # 使用正则表达式提取photo_id和answer字段的内容
    photo_id_match = re.search(r"photo_id is (\d+)", line)
    answer_match = re.search(r'answer: "(.*?)"', line, re.DOTALL)
    valid_match = re.search(r"【结果", line)
    TP,TN,FP,FN =0,0,0,0
    if not valid_match:
        print("answer is ", line)
        return None, TP,TN,FP,FN
    if photo_id_match and answer_match:
        photo_id = photo_id_match.group(1)
        content = answer_match.group(1)
        # answer_match
        content = content.replace('\n', '').replace('\r', '').replace(',', '，')
        wenjuan_type = wenjuan_type_dict.get(photo_id, 'null')

        # 提取结果中的满意/不满意
        result_match = re.search(r"【结果.*(不满意)", content)
        result = result_match.group(1).strip() if result_match else '满意'
        # 判断wenjuan_type和结果
        if wenjuan_type == '问卷优质':
            if result == '满意':
                TP += 1
            else :
                FN += 1
        else:
            if result == '满意':
                FP += 1
            else :
                TN += 1
        result_flag = TP + TN
        return f"{photo_id},{result_flag},{wenjuan_type},{content}", TP,TN,FP,FN
    return None, TP,TN,FP,FN

def process_file(input_file_path, output_file_path, wenjuan_type_dict):
    formatted_lines = []
    total_line = 0
    TP,TN,FP,FN = 0,0,0,0
    with open(input_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        # 使用正则表达式分割每个数据块
        blocks = re.split(r'\n(?=photo_id is)', content)
        for block in blocks:
            if total_line == 0:
                print("block is ", block)
            formatted_line,TP_tmp,TN_tmp,FP_tmp,FN_tmp = extract_and_format_content(block, wenjuan_type_dict)
            if formatted_line:
                formatted_lines.append(formatted_line)
            total_line += 1
            TP += TP_tmp
            TN += TN_tmp
            FP += FP_tmp
            FN += FN_tmp

    with open(output_file_path, 'w', encoding='utf-8') as file:
        for formatted_line in formatted_lines:
            file.write(formatted_line + '\n')
    accuracy = 1.0 * (TP+TN) / (TP+TN+FP+FN)
    recall = 1.0 * TP / (TP + FN)
    print("end process, total_line is ", total_line, ", accuracy is ", accuracy, ",recall is ", recall)
    print("total_cnt : ", TP,TN,FP,FN)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python format_content.py <input_file_path> <output_file_path>")
        sys.exit(1)

    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    if not os.path.exists(input_file_path):
        print(f"Error: Input file '{input_file_path}' does not exist.")
        sys.exit(1)

    # 读取Excel文件并创建photo_id到wenjuan_type的映射
    excel_file_path = '/home/huqigen03/0207_1k.xlsx'
    if not os.path.exists(excel_file_path):
        print(f"Error: Excel file '{excel_file_path}' does not exist.")
        sys.exit(1)

    # 使用openpyxl引擎读取Excel文件
    df = pd.read_excel(excel_file_path, engine='openpyxl')
    # 确保photo_id作为字符串处理
    wenjuan_type_dict = df.set_index(df['photo_id'].astype(str))['wenjuan_type'].to_dict()

    process_file(input_file_path, output_file_path, wenjuan_type_dict)

    print(f"格式化的内容已保存到 {output_file_path}")
