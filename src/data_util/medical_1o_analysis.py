from data_util.gpt_utils import get_response
from configs.data_configs import *
from configs.templates import new_attr_extract_template # test new prompt
from configs.key import _API_KEY

from openai import OpenAI
import json
import copy
import argparse
import os
from collections import Counter
from tqdm import tqdm
current = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(current))

if __name__ == "__main__":
    # https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT
    with open(f'{root_path}/data/medical_o1_reasoning_SFT/medical_o1_sft.json') as fin: # loading new data
        data = json.load(fin) # <class 'list'>, list of dictionary
        # print(len(data)) # 25371
        
    # Define bin thresholds
    thresholds = list(range(50, 750, 50))
    data_len = Counter({f'len<{t}': 0 for t in thresholds})
    data_len['len>=700'] = 0

    # Count occurrences in bins
    for item in data:
        question_length = len(item['Question'])
        for t in thresholds:
            if question_length < t:
                data_len[f'len<{t}'] += 1
                break
        else:
            data_len['len>=700'] += 1

    print(data_len)
    print(dict(data_len))
    # {'len<50': 173, 'len<100': 1481, 'len<150': 4235, 'len<200': 4270, 'len<250': 3655, 'len<300': 2808, 'len<350': 2081, 'len<400': 1598, 'len<450': 1068, 'len<500': 733, 'len<550': 521, 'len<600': 365, 'len<650': 313, 'len<700': 266, 'len>=700': 1804}