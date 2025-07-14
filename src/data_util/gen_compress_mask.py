from data_util.gpt_utils import get_response
from configs.data_configs import *
from configs.templates import compress_mask_template, fill_compress_template
from configs.key import _API_KEY

import random
import logging
from openai import OpenAI
import json
import copy
import argparse
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
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
    raw_pred = get_response(client, prompt, args, gpt_model=args.gpt_model_resp)
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
    parser.add_argument("--gpt_model", type=str, default="gpt-4o")
    parser.add_argument("--gpt_model_resp", type=str, default="gpt-4o-mini")
    parser.add_argument("--data_name", type=str, default="mmlu_fina")
    parser.add_argument("--in_file_name", type=str, default="fina_fake_qcattr_none_zero.json")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    with open(f'{root_path}/data/{args.data_name}/{args.in_file_name}') as fin:
        data = json.load(fin)
        
    log_root = f"{args.root_path}/result/{args.data_name}/logs"
    if not os.path.exists(log_root):
        os.makedirs(log_root)
    file_name = f"mask-cpr-{args.gpt_model}-{args.gpt_model_resp}.log"
    file_path = f"{log_root}/{file_name}"

    logging.basicConfig(
        filename=file_path,
        filemode="w",  # use 'a' to append
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )

    candidate_labels = ["A", "B", "C", "D"]
    pre_questions = []
    # for sample in output_data:
    #     pre_questions.append(sample["question"])

    client = OpenAI(api_key=_API_KEY)
    labels, compress_predictions = [], []

    with tqdm(total=len(data), unit='batch') as pbar:
        for cnt, sample in enumerate(data):
            org_question, attrs, choices, label = sample["question"], sample["filtered private attributes question"], sample["choices"], sample["answer"]
            if org_question in pre_questions:
                pbar.update(1)
                continue
            # obtain compressed question
            mask_question = copy.deepcopy(org_question)
            for attr in attrs:
                mask_question = mask_question.replace(attr, "#MASK")
            prompt = compress_mask_template.format(question=mask_question)
            # print(prompt)
            mask_cpr = get_response(client, prompt, args)
            # print(result)
            if len(attrs) > 0:
                prompt = fill_compress_template.format(org_question=org_question, compression=mask_cpr)
                unmask_cpr = get_response(client, prompt, args)
                # print(result)
            
            # query model
            compress_pred = get_prediction(unmask_cpr, choices, client, args)
            label = candidate_labels[label]
            labels.append(label)
            compress_predictions.append(compress_pred)
            acc = accuracy_score(labels, compress_predictions)
            f1 = f1_score(labels, compress_predictions, average="macro")
            pbar.set_postfix(acc=acc, f1=f1)
            logging.info(
                f"Iteration {cnt+1}/{len(data)} - "
                f"Accuracy {acc} - "
                f"F1 {f1}"
            )
            logging.info(f"Compression: {unmask_cpr}")
            pbar.update(1)