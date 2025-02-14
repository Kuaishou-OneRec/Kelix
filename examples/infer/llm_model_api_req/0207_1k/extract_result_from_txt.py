# -*- coding: utf-8 -*-
"""
File: extract_result_from_txt.py

Description:
    该脚本用于从输入文本文件中提取和格式化包含 photo_id 和 content 的内容，并基于问卷类型对结果进行分类统计。
    脚本主要实现以下功能：
      1. 解析每一行文本，通过正则表达式提取出照片ID (photo_id) 和对话内容 (content)。
         - 支持两种不同格式的 content 字段匹配。
      2. 对提取的内容进行预处理（如清除换行符，替换逗号）。
      3. 根据照片对应的问卷类型（从 Excel 文件中读取形成字典），判断该内容是否表达"满意"或"不满意"。
      4. 根据问卷类型和表达结果计算统计指标：
         - TP（True Positive）、TN（True Negative）、FP（False Positive）、FN（False Negative）。
      5. 将格式化后的内容以特定格式写入到输出文件中。
      6. 在处理完所有行后，输出总体分类准确率与召回率。

Usage:
    python extract_result_from_txt.py <input_file_path> <output_file_path> <wenjuan_file_path>

    例如：
    python extract_result_from_txt.py input.txt output.txt wenjuan.xlsx

Author: [Your Name or the original author]
Date: [Date]
"""

import re
import os
import sys
import pandas as pd

def extract_and_format_content(line, wenjuan_type_dict):
    # 使用正则表达式提取photo_id和content字段的内容
    photo_id_match = re.search(r"photo_id is (\d+)", line)
    print("start process photo: ", photo_id_match.group(1))
    
    # 支持两种格式的content匹配
    content_match_1 = re.search(r"content='(.*?)', (?:role='assistant'|refusal=None, role='assistant')", line, re.DOTALL)
    content_match_2 = re.search(r"'content': '(.*?)', 'tool_calls'", line, re.DOTALL)

    content = None
    if content_match_1:
        content = content_match_1.group(1)
    elif content_match_2:
        content = content_match_2.group(1)

    TP, TN, FP, FN = 0, 0, 0, 0
    if photo_id_match and content:
        photo_id = photo_id_match.group(1)
        # 去除content中的换行符，并替换英文逗号为中文逗号
        content = content.replace('\n', '').replace('\r', '').replace(',', '，')
        wenjuan_type = wenjuan_type_dict.get(photo_id, 'null')

        # 提取结果中的满意/不满意
        result_match = re.search(r"【结果.*(不满意)", content)
        result = result_match.group(1).strip() if result_match else '满意'
        # 判断wenjuan_type和结果
        if wenjuan_type == '问卷优质':
            if result == '满意':
                TP += 1
            else:
                FN += 1
        else:
            if result == '满意':
                FP += 1
            else:
                TN += 1
        result_flag = TP + TN
        # print("result is ", result, ", wenjuan_type is ", wenjuan_type, ", result_match succ" if result_match else ", result_match null", ", content is ", content, TP, TN, FP, FN)

        return f"{photo_id},{result_flag},{wenjuan_type},{content}", TP, TN, FP, FN
    return None, TP, TN, FP, FN

def process_file(input_file_path, output_file_path, wenjuan_type_dict):
    formatted_lines = []
    total_line = 0
    TP, TN, FP, FN = 0, 0, 0, 0
    with open(input_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            formatted_line, TP_tmp, TN_tmp, FP_tmp, FN_tmp = extract_and_format_content(line, wenjuan_type_dict)
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
    accuracy = 1.0 * (TP + TN) / (TP + TN + FP + FN)
    recall = 1.0 * TP / (TP + FN)
    print("end process, total_line is ", total_line, ", accuracy is ", accuracy, ",recall is ", recall)
    print("total_cnt : ", TP, TN, FP, FN)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python format_content.py <input_file_path> <output_file_path> <wenjuan_file_path>")
        sys.exit(1)

    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    wenjuan_file_path = sys.argv[3]

    if not os.path.exists(input_file_path):
        print(f"Error: Input file '{input_file_path}' does not exist.")
        sys.exit(1)

    if not os.path.exists(wenjuan_file_path):
        print(f"Error: Wenjuan file '{wenjuan_file_path}' does not exist.")
        sys.exit(1)

    # 读取0205_text_cmt.xlsx文件
    wenjuan_df = pd.read_excel(wenjuan_file_path)
    wenjuan_type_dict = dict(zip(wenjuan_df['photo_id'].astype(str), wenjuan_df['wenjuan_type'].astype(str)))

    process_file(input_file_path, output_file_path, wenjuan_type_dict)

    print(f"格式化的内容已保存到 {output_file_path}")
