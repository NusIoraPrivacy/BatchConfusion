from data_util.gpt_utils import get_response
from configs.data_configs import *
from configs.templates import compress_template
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
    with open(f'{root_path}/data/mmlu/raw_data.json') as fin:
        data = json.load(fin)
    random.shuffle(data)
    client = OpenAI(api_key=_API_KEY)
    output_data = []
    with tqdm(total=len(data), unit='batch') as pbar:
        for sample in data:
            question = sample["question"]
            prompt = compress_template.format(question=question)
            # print(prompt)
            result = get_response(client, prompt, args)
            # print(result)
            output = copy.deepcopy(sample)
            output["compression"] = result
            output_data.append(output)
            with open(f'{args.root_path}/data/mmlu/compress_gpt.json', 'w') as fout:
                json.dump(output_data, fout, indent=4)
            pbar.update(1)
            # breaks