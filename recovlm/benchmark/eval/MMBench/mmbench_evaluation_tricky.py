import pandas as pd
import json
import random
from collections import Counter

'''
This script provides metric calculation for mmbench_dev with the same accuarcy algo as OpenCompass server
'''

class MMBenchEvaluation:
    def __init__(self, predict_dict, original_file_path):
        self.index2predictions = predict_dict
        datas = pd.read_parquet(original_file_path)
        glb_opts = ['A', 'B', 'C', 'D']
        self.index2answer = {}
        self.index2choices = {}
        self.index2rawanswer = {}
        for idx in range(len(datas)):
            data = datas.iloc[idx]
            
            choices = []
            for opt in glb_opts:
                if not pd.isna(data[opt]):
                    choices.append(data[opt])
            self.index2choices[data['index']] = choices

            self.index2answer[data['index']] = glb_opts.index(data['answer'])
            self.index2rawanswer[data['index']] = choices[glb_opts.index(data['answer'])] 

    def most_common_elements(self, lst):
        counter = Counter(lst)
        max_count = max(counter.values())
        most_common = [element for element, count in counter.items() if count == max_count]
        return random.choice(most_common) # random sample from random choice

    def eval(self):
        identity_indexes = list(set([int(_ % 1e6) for _ in self.index2predictions.keys()]))
        correct = 0
        total = 0
        for index in identity_indexes:
            raw_preds = []
            raw_answer = []
            for _ in range(4):
                cycle_index = int(_ * 1e6 + index)
                if self.index2predictions.get(cycle_index, None) is not None:
                    raw_answer = self.index2rawanswer[cycle_index]
                    raw_pred = self.index2choices[cycle_index][self.index2predictions[cycle_index]]
                    raw_preds.append(raw_pred)

            if len(set(raw_preds)) == 1:
                if raw_preds[0] == raw_answer:
                    correct += 1
            else:
                result = self.most_common_elements(raw_preds)
                if result == raw_answer:
                    correct += 1
            total += 1

        return [correct, total, correct / total * 100]