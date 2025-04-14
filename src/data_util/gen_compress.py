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
    parser.add_argument("--data_name", type=str, default="medical_o1_reasoning_SFT")
    parser.add_argument("--in_file_name", type=str, default="compress_raw_100.json")
    parser.add_argument("--out_file_name", type=str, default="compress_gpt_new990.json")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    with open(f'{args.root_path}/data/{args.data_name}/{args.in_file_name}') as fin:
        data = json.load(fin)
    # print(len(data))
    random.shuffle(data)
    client = OpenAI(api_key=_API_KEY)
    output_data = []
    with tqdm(total=len(data), unit='batch') as pbar:
        for sample in data:
            new_sample = {}
            for key in sample:
                if key == "Question":
                    new_sample["question"] = sample["Question"]
                elif key == "Answer":
                    new_sample["response"] = sample["Answer"]
                else:
                    new_sample[key] = sample[key]
            sample = new_sample
            question = sample["question"]
            prompt = compress_template.format(question=question)
            # print(prompt)
            result = get_response(client, prompt, args)
            # print(result)
            output = copy.deepcopy(sample)
            output["compression"] = result
            output_data.append(output)
            with open(f'{args.root_path}/data/{args.data_name}/{args.out_file_name}', 'w') as fout:
                json.dump(output_data, fout, indent=4)
            pbar.update(1)
            # breaks