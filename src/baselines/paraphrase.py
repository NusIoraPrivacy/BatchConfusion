from configs.data_configs import *
from evaluation.utils import create_client
from baselines.utils import *

from sklearn.metrics import accuracy_score, f1_score

import logging
import random
import copy
import json
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
    parser.add_argument("--remote_cpr", type=str, default="fina_fake_qattr_none_zero.json",
        help = "path of gpt compression files")
    parser.add_argument("--data_name", type=str, default="mmlu_fina")
    parser.add_argument("--gpt_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--para_model", type=str, default="eugenesiow/bart-paraphrase")
    parser.add_argument("--epsilon", type=int, default=1)
    parser.add_argument("--data_size", type=int, default=400)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    with open(f'{args.root_path}/data/{args.data_name}/{args.remote_cpr}') as fin:
        data = json.load(fin)
    random.shuffle(data)
    data = data[:args.data_size]

    para_tokenizer, para_model = load_para_model(args.para_model)
    for sample in tqdm(data):
        question = sample["question"]
        # print(question)
        question = paraphrase(para_tokenizer, para_model, question, args.para_model, args.epsilon, args)
        sample["question"] = question
    del para_model
    
    log_root = f"{args.root_path}/result/{args.data_name}/logs/dp/paraphrase"
    if not os.path.exists(log_root):
        os.makedirs(log_root)
    file_name = f"epsilon-{args.epsilon}-{args.gpt_model}.log"

    file_path = f"{log_root}/{file_name}"
    logging.basicConfig(
        filename=file_path,
        filemode="w",  # use 'a' to append
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )

    candidate_labels = ["A", "B", "C", "D"]

    client = create_client(args.gpt_model)
    
    labels, predictions = [], []
    with tqdm(total=len(data), unit='batch') as pbar:
        for i, sample in enumerate(data):
            query, choices, label = sample["question"], sample["choices"], sample["answer"]
            # print(query)
            pred = get_prediction(query, choices, client, args)
            # print(origin_pred)
            # print(compress_pred)
            label = candidate_labels[label]
            labels.append(label)
            # origin_predictions.append(origin_pred)
            predictions.append(pred)
            pbar.update(1)
            # org_acc = accuracy_score(labels, origin_predictions)
            # org_f1 = f1_score(labels, origin_predictions, average="macro")
            acc = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions, average="macro")
            pbar.set_postfix(acc=acc, f1=f1)
            logging.info(
                f"Iteration {i+1}/{len(data)} - "
                f"Accuracy {acc} - "
                f"F1 {f1}"
            )
