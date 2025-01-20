#encoding=utf-8

import json
from argparse import ArgumentParser
import torch
import os
import json
from tqdm import tqdm
from PIL import Image
import math
import multiprocessing
from multiprocessing import Pool, Queue, Manager
from transformers import AutoModelForCausalLM, AutoTokenizer

OCRBench_score = {"Regular Text Recognition":0,"Irregular Text Recognition":0,"Artistic Text Recognition":0,"Handwriting Recognition":0,
"Digit String Recognition":0,"Non-Semantic Text Recognition":0,"Scene Text-centric VQA":0,"Doc-oriented VQA":0,
"Key Information Extraction":0,"Handwritten Mathematical Expression Recognition":0}
AllDataset_score = {"IIIT5K":0,"svt":0,"IC13_857":0,"IC15_1811":0,"svtp":0,"ct80":0,"cocotext":0,"ctw":0,"totaltext":0,"HOST":0,"WOST":0,"WordArt":0,"IAM":0,"ReCTS":0,"ORAND":0,"NonSemanticText":0,"SemanticText":0,
"STVQA":0,"textVQA":0,"ocrVQA":0,"ESTVQA":0,"ESTVQA_cn":0,"docVQA":0,"infographicVQA":0,"ChartQA":0,"ChartQA_Human":0,"FUNSD":0,"SROIE":0,"POIE":0,"HME100k":0}

def eval_OCRBench(data, fw_e, fw_c):
    error_dict = []
    correct_dict = []
    for i in range(len(data)):
        cur_row = {}
        cur_correct_row = {}
        data_type = data[i]["question_type"]
        dataset_name = data[i]["dataset"]
        answers = data[i]["answer"]
        if data[i].get('predict',0)==0:
            continue
        predict = data[i]['predict']
        cur_row["answer"] = answers
        cur_row["predict"] = data[i]['predict']
        cur_row["messages"] = data[i]["messages"]
        cur_row["image"] = data[i]["image"]
        cur_row["dataset_name"] = dataset_name
        data[i]['result'] = 0
        if dataset_name == "HME100k":
            if type(answers)==list:
                for j in range(len(answers)):
                    answer = answers[j].strip().replace("\n"," ").replace(" ","")
                    predict = predict.strip().replace("\n"," ").replace(" ","")
                    if answer in predict:
                        data[i]['result'] = 1
            else:
                answers = answers.strip().replace("\n"," ").replace(" ","")
                predict = predict.strip().replace("\n"," ").replace(" ","")
                if answers in predict:
                    data[i]['result'] = 1
        else:
            if type(answers)==list:
                for j in range(len(answers)):
                    answer = answers[j].lower().strip().replace("\n"," ")
                    predict = predict.lower().strip().replace("\n"," ")
                    if answer in predict:
                        data[i]['result'] = 1
            else:
                answers = answers.lower().strip().replace("\n"," ")
                predict = predict.lower().strip().replace("\n"," ")
                if answers in predict:
                    data[i]['result'] = 1

        if data[i]['result'] != 1:
            error_dict.append(cur_row)
        else:
            correct_dict.append(cur_row)
    
    json.dump(error_dict, fw_e, indent=4, separators=(',', ':'))
    json.dump(correct_dict, fw_c, indent=4, separators=(',', ':'))
    for i in range(len(data)):
        if data[i].get("result",100)==100:
            continue
        OCRBench_score[data[i]['question_type']] += data[i]['result']
    recognition_score = OCRBench_score['Regular Text Recognition']+OCRBench_score['Irregular Text Recognition']+OCRBench_score['Artistic Text Recognition']+OCRBench_score['Handwriting Recognition']+OCRBench_score['Digit String Recognition']+OCRBench_score['Non-Semantic Text Recognition']
    final_score = recognition_score+OCRBench_score['Scene Text-centric VQA']+OCRBench_score['Doc-oriented VQA']+OCRBench_score['Key Information Extraction']+OCRBench_score['Handwritten Mathematical Expression Recognition']
    return final_score
