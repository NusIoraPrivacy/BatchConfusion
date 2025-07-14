from configs.data_configs import *
from data_util.gpt_utils import get_response
from configs.templates import attr_extract_template
from configs.key import _API_KEY
from evaluation.utils import create_client

from sklearn.metrics import accuracy_score, f1_score
from sentence_transformers import SentenceTransformer

import logging
import random
import copy
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
    raw_pred = get_response(client, prompt, args, args.gpt_model)
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
    parser.add_argument("--local_epoch", type=str, default="epoch_1.json",
        help = "path of local compression files")
    parser.add_argument("--local_model", type=str, default="Qwen3-1.7B",
        help = "path of local compression files")
    parser.add_argument("--remote_cpr", type=str, default="fina_fake_qattr_none_zero.json",
        help = "path of gpt compression files")
    parser.add_argument("--data_name", type=str, default="mmlu_fina")
    parser.add_argument("--gpt_model", type=str, default="gpt-4o-mini")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    if args.local_model is None:
        with open(f'{args.root_path}/data/{args.data_name}/{args.remote_cpr}') as fin:
            data = json.load(fin)
        # random.shuffle(data)
        # n_train = int(len(data)*0.8)
        # data = data[n_train:]
    else:
        with open(f'{args.root_path}/result/{args.data_name}/compression/{args.local_model}/{args.local_epoch}') as fin:
            data = json.load(fin)
            
        with open(f'{args.root_path}/data/{args.data_name}/{args.remote_cpr}') as fin:
            data_truth = json.load(fin)
        query2sample = {}
        for sample in data_truth:
            question = sample["question"]
            query2sample[question] = sample
        
        for sample in data:
            question = sample["question"]
            link_sample = query2sample[question]
            sample["choices"] = link_sample["choices"]
            sample["answer"] = link_sample["answer"]

    log_root = f"{args.root_path}/result/{args.data_name}/logs/compression"
    if not os.path.exists(log_root):
        os.makedirs(log_root)
    if args.local_model is None:
        file_name = f"full_question-{args.gpt_model}.log"
    else:
        file_name = f"{args.local_model}-{args.local_epoch}-{args.gpt_model}.log"
    file_path = f"{log_root}/{file_name}"
    logging.basicConfig(
        filename=file_path,
        filemode="w",  # use 'a' to append
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )

    candidate_labels = ["A", "B", "C", "D"]

    client = create_client(args.gpt_model)
    
    labels, compress_predictions = [], []
    with tqdm(total=len(data), unit='batch') as pbar:
        for i, sample in enumerate(data):
            original_query, compress_query, choices, label = sample["question"], sample["compression"], sample["choices"], sample["answer"]
            if args.local_model is not None:
                compress_query = sample["predicted compression"]
                # print(compress_query)
            # origin_pred = get_prediction(original_query, choices, client, args)
            compress_pred = get_prediction(compress_query, choices, client, args)
            # print(origin_pred)
            # print(compress_pred)
            label = candidate_labels[label]
            labels.append(label)
            # origin_predictions.append(origin_pred)
            compress_predictions.append(compress_pred)
            pbar.update(1)
            # org_acc = accuracy_score(labels, origin_predictions)
            # org_f1 = f1_score(labels, origin_predictions, average="macro")
            compress_acc = accuracy_score(labels, compress_predictions)
            compress_f1 = f1_score(labels, compress_predictions, average="macro")
            pbar.set_postfix(compress_acc=compress_acc, compress_f1=compress_f1)
            logging.info(
                f"Iteration {i+1}/{len(data)} - "
                f"Accuracy {compress_acc} - "
                f"F1 {compress_f1}"
            )
