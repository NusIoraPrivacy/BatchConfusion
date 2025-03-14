from configs.data_configs import *
from data_util.gpt_utils import get_response
from configs.templates import attr_extract_template
from configs.key import _API_KEY

from sklearn.metrics import accuracy_score, f1_score
from sentence_transformers import SentenceTransformer

import random
import copy
from openai import OpenAI
import json
import argparse
import os
from tqdm import tqdm
current = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(current))

def standard_ans(ans, labels):
    ans = ans.strip(".")
    ans = ans.strip()
    pred = 1
    for label in labels:
        if label in ans:
            pred = label
    return pred

def get_prediction(query, choices, client, args):
    prompt = (f"Question: {query}\n Please select one of the options, and output A-D only:\n"
                f"A: {choices[0]}\n B: {choices[1]}\n C: {choices[2]}\n D: {choices[3]}"
                "Remember to output only a single character from A to D!")
    # print(prompt)
    raw_pred = get_response(client, prompt, args)
    # print(raw_pred)
    pred = standard_ans(raw_pred, candidate_labels)
    return pred

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default=root_path)
    parser.add_argument("--temperature", type=float, default=1,
        help = "temperature for text generation")
    parser.add_argument("--max_tokens", type=int, default=1000,
        help = "max new token for text generation")
    parser.add_argument("--local_cpr", type=str, default=None,
        help = "path of local compression files")
    parser.add_argument("--remote_cpr", type=str, default="medical_o1_sft.json",
        help = "path of gpt compression files")
    parser.add_argument("--data_name", type=str, default="medical_o1_reasoning_SFT")
    parser.add_argument("--gpt_model", type=str, default="gpt-4o-mini")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    if args.local_cpr is None:
        with open(f'{args.root_path}/data/{args.data_name}/{args.remote_cpr}') as fin:
            data = json.load(fin)
        random.shuffle(data)
        n_train = int(len(data)*0.8)
        data = data[n_train:]
    else:
        with open(f'{args.root_path}/result/{args.data_name}/{args.local_cpr}') as fin:
            data = json.load(fin)
    # print(len(data))

    candidate_labels = ["A", "B", "C", "D"]

    client = OpenAI(api_key=_API_KEY)
    
    if args.local_cpr is None:
        local_suffix = ""
    else:
        local_suffix = "_local"
    output_path = f'{args.root_path}/result/mmlu/answer_{args.gpt_model}{local_suffix}.json'
    output_data = []
    labels, origin_predictions, compress_predictions = [], [], []
    with tqdm(total=len(data), unit='batch') as pbar:
        for i, sample in enumerate(data):
            original_query, compress_query, choices, label = sample["question"], sample["compression"], sample["choices"], sample["answer"]
            if args.local_cpr is not None:
                compress_query = sample["predicted compression"]
            origin_pred = get_prediction(original_query, choices, client, args)
            compress_pred = get_prediction(compress_query, choices, client, args)
            # print(origin_pred)
            # print(compress_pred)
            label = candidate_labels[label]
            labels.append(label)
            origin_predictions.append(origin_pred)
            compress_predictions.append(compress_pred)
            pbar.update(1)
            org_acc = accuracy_score(labels, origin_predictions)
            org_f1 = f1_score(labels, origin_predictions, average="macro")
            compress_acc = accuracy_score(labels, compress_predictions)
            compress_f1 = f1_score(labels, compress_predictions, average="macro")
            pbar.set_postfix(origin_acc=org_acc, origin_f1=org_f1, compress_acc=compress_acc, compress_f1=compress_f1)
            output = copy.deepcopy(sample)
            output["origin_pred"] = origin_pred
            output["compress_pred"] = compress_pred
            output_data.append(output)
            if (i+1) % 10 == 0:
                with open(output_path, 'w') as fout:
                    json.dump(output_data, fout, indent=4)
