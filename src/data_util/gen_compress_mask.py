from data_util.gpt_utils import get_response
from configs.data_configs import *
from configs.templates import compress_mask_template, fill_compress_template
from configs.key import _API_KEY

import random
from openai import OpenAI
import json
import copy
import argparse
import os
from tqdm import tqdm
current = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(current))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default=root_path)
    parser.add_argument("--temperature", type=float, default=1,
        help = "temperature for text generation")
    parser.add_argument("--max_tokens", type=int, default=1000,
        help = "max new token for text generation")
    parser.add_argument("--gpt_model", type=str, default="gpt-4o")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    with open(f'{root_path}/data/mmlu/private_attrs_ner.json') as fin:
        data = json.load(fin)
    random.shuffle(data)
    
    with open(f'{root_path}/data/mmlu/compress_gpt.json') as fin:
        cpr_data = json.load(fin)
    query2cpr = {}
    for sample in cpr_data:
        question, compresion = sample["question"], sample["compression"]
        query2cpr[question] = compresion

    output_path = f'{args.root_path}/data/mmlu/compress_gpt_mask.json'
    if os.path.exists(output_path):
        with open(output_path) as fin:
            output_data = json.load(fin)
    else:
        output_data = []

    pre_questions = []
    for sample in output_data:
        pre_questions.append(sample["question"])

    client = OpenAI(api_key=_API_KEY)

    with tqdm(total=len(data), unit='batch') as pbar:
        for sample in data:
            org_question, attrs = sample["question"], sample["private attributes"]
            if org_question in pre_questions:
                pbar.update(1)
                continue
            mask_question = copy.deepcopy(org_question)
            if len(attrs) == 0 and org_question in query2cpr.keys():
                result = query2cpr[org_question]
            else:
                for attr in attrs:
                    mask_question = mask_question.replace(attr, "#MASK")
                prompt = compress_mask_template.format(question=mask_question)
                # print(prompt)
                result = get_response(client, prompt, args)
                # print(result)
                if len(attrs) > 0:
                    prompt = fill_compress_template.format(org_question=org_question, compression=result)
                    result = get_response(client, prompt, args)
                    # print(result)
            output = copy.deepcopy(sample)
            output["compression"] = result
            output_data.append(output)
            with open(f'{args.root_path}/data/mmlu/compress_gpt_mask.json', 'w') as fout:
                json.dump(output_data, fout, indent=4)
            pbar.update(1)
            # breaks