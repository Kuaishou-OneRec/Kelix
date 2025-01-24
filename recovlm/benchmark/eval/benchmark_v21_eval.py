#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys,os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import logging
import glob
import time
import json
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from google.protobuf import text_format

from kess.framework import (
    ClientOption,
    GrpcClient,
    KessOption,
)

sys.path.insert(0, 'eval')
from mmu.mmu_chat_gpt_pb2 import MmuChatGptRequest,MmuChatGptResponse
from mmu.mmu_chat_gpt_pb2_grpc import (
    MmuChatGptServiceStub,
)

def chat(grpc_client: GrpcClient, timeout: float, question: str):
    # try:
    #构造request, biz需根据实际申请的进行修改
    biz = 'luoxinchen_b585bcc7_gpt-4o-2024-08-06'

    request = MmuChatGptRequest(biz=biz)
    request.session_id = 'test'
    request.req_id = '1000'
    request.query = question
    #发起请求
    count = 1
    while True:
        print("try {} times".format(count))
        #发起请求
        resp = grpc_client.Chat(request, timeout=timeout)
        if resp.status.code == 1 and resp.answer != "UNKNOWN ERROR":
            question, answer = list(resp.query_history)
            print("chat done")
            return question, answer
        else:
            print("Error occurred, retry")
            print(resp.status)
        time.sleep(10)
        count += 1

def calc_acc(total_time, output_dir):
    correct_keys = []
    with open(os.path.join(output_dir, "res.txt"), "w") as fw:
        print("Total time: {}".format(total_time))
        fw.write("Total time: {}\n\n".format(total_time))
        res_dict = {}
        for f in glob.glob("{}/*".format(output_dir)):
            # print(f)
            with open(f) as fr:
                if 'res.txt' in f:
                    continue

                line = fr.readlines()[0]

                values = line.strip().split("\t")
                try:
                    key, img_path, preview_link, question, model_output, label_answer, flag, question_type = values
                except:
                    print(question)
                    continue
                if "-" not in question_type:
                    question_type = question_type + "-" + question_type
                class_1, class_2 = question_type.split("-")
                if class_1 not in res_dict:
                    res_dict[class_1] = {}
                if class_2 not in res_dict[class_1]:
                    res_dict[class_1][class_2] = []
                res_dict[class_1][class_2].append(flag)
                if flag == 1:
                    correct_keys.append(key)
        keys = list(res_dict.keys())
        keys.sort()
        return_string = ""
        return_dict = {}
        for k1 in keys:
            if k1 == "纯语言问题":
                gpt4_total_score, model_total_score, instance_num = 0, 0, 0
                keys2 = list(res_dict[k1])
                keys2.sort()
                for k2 in keys2:
                    score_1 = [float(i.split("<->")[0]) for i in res_dict[k1][k2]]
                    score_2 = [float(i.split("<->")[1]) for i in res_dict[k1][k2]]
                    instance_num += len(score_1)
                    gpt4_total_score += sum(score_1)
                    model_total_score += sum(score_2)
                    print("{}: gpt4_score: {:.3f}, model_score: {:.3f}".format(k2, sum(score_1)/len(score_1), sum(score_2)/len(score_2)))
                    fw.write("{}: gpt4_score: {:.3f}, model_score: {:.3f}\n".format(k2, sum(score_1)/len(score_1), sum(score_2)/len(score_2)))

                print("\n{}: gpt4_score: {:.3f}, model_score: {:.3f}".format(k1, gpt4_total_score/instance_num, model_total_score/instance_num))
                print("-"*50 + "\n")

                fw.write("\n{}: gpt4_score: {:.3f}, model_score: {:.3f}\n".format(k1, gpt4_total_score/instance_num, model_total_score/instance_num))
                fw.write("-"*50 + "\n\n")
            else:
                total, correct = 0, 0
                keys2 = list(res_dict[k1])
                keys2.sort()
                for k2 in keys2:
                    v2 = [int(i) for i in res_dict[k1][k2]]
                    total += len(v2)
                    correct += sum(v2)
                    print("{}: {}/{}, accuracy: {:.3f}".format(k2, sum(v2), len(v2), \
                                                        sum(v2) / len(v2)))
                    return_dict[k2] = "{:.3f}".format(sum(v2) / len(v2))

                    fw.write("{}: {}/{}, accuracy: {:.3f}\n".format(k2, sum(v2), len(v2), \
                                                        sum(v2) / len(v2)))

                print("\n{}: {}/{}, accuracy: {:.3f}\n".format(k1, correct, total, \
                                                    correct / total))
                print("-"*50 + "\n")
                return_string = return_string + k1 + "{:.3f}".format(correct / total) + "\t|\t"

                fw.write("\n{}: {}/{}, accuracy: {:.3f}\n\n".format(k1, correct, total, \
                                                    correct / total))
                fw.write("-"*50 + "\n\n")
                return_dict[k1] = "{:.3f}".format(correct / total)
        return return_string, return_dict, correct_keys

#服务名不要改动
client_option = ClientOption(
    biz_def='mmu',
    grpc_service_name='mmu-chat-gpt-service',
    grpc_stub_class=MmuChatGptServiceStub,
)
client = GrpcClient(client_option)


def process_(inp, output_dir):
    fin_tag = False
    print("process_ start")
    while not fin_tag:
        key, img_path, preview_link, class_1, class_2, que, question, model_output, label_answer, question_type = inp
        ques, anw = chat(client, 1000, que)
        print(f"anw is {anw}")
        result = anw.split("####")[1]
        # result = "回答正确"
        if "纯语言问题" in question_type:
            score_1, score_2 = result.split("\n")[0].split()
            reason = "|".join(result.split("\n")[1:])
            flag = "<->".join([score_1, score_2, reason])
        else:
            flag = -1
            if "回答正确" in result:
                flag = 1
            elif "回答错误" in result:
                flag = 0
            else:
                print(result)
        if flag != -1:
            with open("{}/{}".format(output_dir, key), "w") as f_write:
                f_write.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(key, img_path, preview_link, question, model_output, label_answer, flag, question_type))
                f_write.flush()
                fin_tag = True
    print("process_ done")


def check_already_exam(check_path):
    xx = glob.glob(check_path)
    answer_lib = {}
    for y in tqdm(xx):
        # print(y)
        try:
            if ".txt" not in y:
                line = open(y).readlines()[0].strip()
                s = line.split('\t')
                key = s[0] + "++++" + s[-4]
                try:
                    answer_lib[key].append(s[-2])
                except:
                    answer_lib[key] = [s[-2]]
        except:
            continue
    final_answer_lib = {}
    for k in answer_lib:
        if len(set(answer_lib[k])) == 1:
            final_answer_lib[k] = answer_lib[k][0]
    return final_answer_lib

def calc_gpt4_parallel(check_path, input_file, eval_item, output_dir):
    final_answer_lib = check_already_exam(check_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    start_time = time.time()
    input_list = []

    already = glob.glob(output_dir + "/*")
    already = {al.split("/")[-1] for al in already}

    with open(input_file, 'r') as f_open:
        for line in f_open:
            tmp_dict = json.loads(line.strip())

            key = tmp_dict['key']
            if key in already:
                continue
            question = tmp_dict['question']
            answer = tmp_dict['answer']
            model_output = tmp_dict['model_output']
            # 为了兼容模型输出中带了 \nHuman: 的情况
            if isinstance(model_output, list):
                model_output = model_output[0]
            model_output = model_output.split("\n")[0]
            question_type = tmp_dict['question_type']

            if question_type not in eval_item:
                continue

            if key + "++++" + model_output in final_answer_lib:
                with open("{}/{}".format(output_dir, key), "w") as f_write:
                    f_write.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(key, tmp_dict['image_path'], tmp_dict['preview_link'], question, model_output, answer, final_answer_lib[key + "++++" + model_output], question_type))
                continue


            if "-" not in question_type:
                question_type = question_type + "-" + question_type
            class_1, class_2 = question_type.split("-")

            if class_1 == "纯语言问题":
                prompt = \
                        '''
                        你是一个有帮助且精准的助手，用于检查答案的质量。问题【{}】
                        The Start 助理1的回答
                        【{}】
                        The End 助理1的回答
                        The Start 助理2的回答
                        【{}】
                        The End 助理2的回答
                        我们希望请您就上述用户问题中两个AI助手的表现提供反馈意见。请您评估它们的响应是否有帮助、相关性、准确性、详细程度。每个助手都会获得一个1到10分的总体评分，其中得分越高表示综合表现越好。请您先输出一行仅包含两个值，分别表示助手1和2的评分。这两个评分需要用空格分隔。在接下来的行中，请您提供全面的评估说明，避免任何潜在的偏见
                        '''
                que = prompt.format(question, answer, model_output)
            else:
                prompt = '''给你一个问题、标准答案、回答，请根据参考答案判断回答是否正确。问题：{}。标准答案：{}。回答：{}。请以“回答正确”或“回答错误”的格式返回答案。'''
                que = prompt.format(question, answer, model_output)

            input_list.append([key, tmp_dict['image_path'], tmp_dict['preview_link'], class_1, class_2, \
                            que, question, model_output, answer, question_type])

    from multiprocessing import Pool
    p = Pool(5)
    for inp in input_list:
        p.apply_async(process_, args=(inp, output_dir, ))
        # process_(inp, output_dir)
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')

    end_time = time.time()
    print("Time: {}".format(end_time - start_time))
    return_string, return_dict, correct_keys = calc_acc(end_time - start_time, output_dir)
    return return_dict, correct_keys

def load_data(infer_outputs_file):
    data_list = []
    with open(infer_outputs_file, 'r') as f_open:
        data_list = json.load(f_open)["annotations"]
    return data_list

def eval_Benchmark_v21(save_path, file_output_path):
        data_list = load_data(file_output_path)
        src = [item for item in data_list if "task_type" in item and item["task_type"] == "Benchmark_v21"]
        out_file_name = os.path.join(save_path, "infer_result.txt")
        f_write = open(out_file_name, 'w')
        for item in src:
            item["preview_link"] = "null"
            f_write.write(json.dumps(item, ensure_ascii=False)+"\n")
        f_write.close()

        # 配置机评输出路径
        gpt_tmp_result_dir = os.path.join(save_path, "gpt_tmp_res")
        eval_item = ["内容理解-对象检测", "内容理解-对象识别", "内容理解-属性分类", "内容理解-场景分类", "内容理解-行为识别", "OCR能力-基础能力", "OCR能力-进阶能力", "逻辑推理-数学能力", "逻辑推理-空间能力"]
        # eval_item = ["内容理解-对象检测", "内容理解-对象识别", "内容理解-属性分类", "内容理解-场景分类", "内容理解-行为识别"]
        # 计算 gpt-4 机评结果
        check_path = gpt_tmp_result_dir + "/*"
        core_acc, correct_keys = calc_gpt4_parallel(check_path, out_file_name, eval_item, gpt_tmp_result_dir)

        # 合并 gpt-4 机评结果, 用于可视化
        combined_file = "{}/combined_gpt4_result.txt".format(save_path)
        os.system("cat {}/* > {}".format(gpt_tmp_result_dir, combined_file))

        # gpt-4 机评结果可视化
        visualize_item = ["image_path", "key", "question", "answer", "model_output"]
        gpt_vis_dir = os.path.join(save_path, "GPT_eval_vis")
        #vis_main(out_file_name, combined_file, gpt_vis_dir, visualize_item)

        return core_acc, correct_keys
