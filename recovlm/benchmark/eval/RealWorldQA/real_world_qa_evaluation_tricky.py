import pandas as pd
import json
import random
from collections import Counter

'''
This script provides metric calculation for realworldqa with the same accuarcy algo as OpenCompass server
'''

class RealWorldQAEvaluation:
    def __init__(self, answers, predicts):
        self.answers = answers
        self.predicts = predicts

    def eval(self):

        correct = 0
        total = 0
        correct_keys = set()

        for key in self.answers:
            assert key in self.predicts
            total += 1
            answer = self.answers[key].rstrip(".").lower()
            predict = self.predicts[key].rstrip(".").lower()
            if answer == predict:
                correct += 1
                correct_keys.add(key)

        return [correct, total, correct / total * 100], correct_keys