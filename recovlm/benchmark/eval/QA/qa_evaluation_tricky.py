import pandas as pd
import json
import random
from collections import Counter
import editdistance

'''
This script provides metric calculation for qa with the same accuarcy algo as OpenCompass server
'''


class ANLSEvaluator:
    def __init__(self, answers, predicts, resuce_fn=max):
        
        self.answers = answers
        self.predicts = predicts
        self.get_edit_distance = editdistance.eval
        self.resuce_fn = resuce_fn

    def get_anls(self, s1, s2):
        s1 = s1.lower().rstrip(".").strip()
        s2 = s2.lower().rstrip(".").strip()
        iou = 1 - self.get_edit_distance(s1, s2) / max(len(s1), len(s2))
        # anls = iou if iou >= 0.5 else 0.0
        anls = iou
        return anls

    def eval(self):
        
        total = 0
        anls_dict = dict()
        reduce_anls_dict = dict()
        reduce_anls_list = list()

        keys_list = list()
        for key in self.answers:
            assert key in self.predicts

            keys_list.append(key)
            predict = self.predicts[key]

            answer_list = self.answers[key]
            if isinstance(answer_list, str):
                answer_list = [answer_list]
            
            anls_list = list()
            for answer in answer_list:
                anls = self.get_anls(predict, answer)
                anls_list.append(anls)
            reduce_anls = self.resuce_fn(anls_list)

            anls_dict[key] = anls_list
            reduce_anls_dict[key] = reduce_anls
            reduce_anls_list.append(reduce_anls)

        avg_anls = sum(reduce_anls_list) / len(reduce_anls_list)
        return (anls_dict, reduce_anls_dict, avg_anls), keys_list