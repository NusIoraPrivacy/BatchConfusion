from configs.data_configs import *
from data_util.gpt_utils import get_response
from configs.key import _API_KEY
from configs.templates import evaluation_template

from sklearn.metrics import accuracy_score, f1_score
from sentence_transformers import SentenceTransformer

from rouge import Rouge
from sacrebleu.metrics import BLEU

import re
import ast
import random
import copy
from openai import OpenAI
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
    parser.add_argument("--local_cpr", type=str, default=None,
        help = "path of local compression files")
    parser.add_argument("--remote_cpr", type=str, default="financial_datasets_compress.json",
        help = "path of gpt compression files")
    parser.add_argument("--data_name", type=str, default="financial_datasets")
    parser.add_argument("--gpt_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--eval_gpt_model", type=str, default="gpt-4o")
    parser.add_argument("--out_file_name", type=str, default="cpr_v1.json")
    parser.add_argument("--query_origin", type=bool, default=True)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    if args.local_cpr is None:
        with open(f'{args.root_path}/data/{args.data_name}/{args.remote_cpr}') as fin:
            data = json.load(fin)
        # random.shuffle(data)
        # n_train = int(len(data)*0.96)
        # data = data[n_train:]
    else:
        with open(f'{args.root_path}/result/{args.data_name}/{args.local_cpr}') as fin:
            data = json.load(fin)
    print(len(data))
    one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
    one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")
    client = OpenAI(api_key=_API_KEY)
    bleu_scorer = BLEU(effective_order=True)
    rouge_scorer = Rouge()
    if args.local_cpr is None:
        local_suffix = ""
    else:
        local_suffix = "_local"
    output_dir = f"{args.root_path}/result/{args.data_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if args.out_file_name is None:
        output_path = f'{output_dir}/answer_{args.gpt_model}{local_suffix}.json'
    else:
        output_path = f'{output_dir}/{args.out_file_name}'
    output_data = []
    # ref_answers, predictions = [], []
    org_rougeLs, org_blues, cpr_rougeLs, cpr_blues = [], [], [], []
    with tqdm(total=len(data), unit='batch') as pbar:
        for i, sample in enumerate(data):
            output = copy.deepcopy(sample)
            original_query, ref_answer, compress_query = sample["question"], sample["response"], sample["compression"]
            # print(original_query)
            if args.local_cpr is not None:
                compress_query = sample["predicted compression"]
            # original prediction
            if args.query_origin:
                original_query = original_query + " Please answer as concise as possible."
                pred = get_response(client, original_query, args)
                # print(pred)
                try:
                    score = rouge_scorer.get_scores(hyps=pred, refs=ref_answer)
                    rougeL = score[0]["rouge-l"]["f"]
                except ValueError as e:
                        rougeL = 0
                try:
                    blue_score = bleu_scorer.sentence_score(hypothesis=pred, references=[ref_answer])
                    blue = blue_score.score/100
                except ValueError as e:
                    blue = 0
                org_rougeLs.append(rougeL)
                org_blues.append(blue)
                avg_org_rougeL = sum(org_rougeLs)/len(org_rougeLs)
                avg_org_blue = sum(org_blues)/len(org_blues)

                output["origin_prediction"] = pred
                output["origin_blue"] = blue
                output["origin_rougeL"] = rougeL

            # compression prediction
            compress_query = compress_query + " Please answer as concise as possible."
            pred = get_response(client, compress_query, args)
            # print(pred)
            # ref_answers.append(ref_answer)
            # predictions.append(pred)
            try:
                score = rouge_scorer.get_scores(hyps=pred, refs=ref_answer)
                rougeL = score[0]["rouge-l"]["f"]
            except ValueError as e:
                    rougeL = 0
            try:
                blue_score = bleu_scorer.sentence_score(hypothesis=pred, references=[ref_answer])
                blue = blue_score.score/100
            except ValueError as e:
                blue = 0
            cpr_rougeLs.append(rougeL)
            cpr_blues.append(blue)
            avg_cpr_rougeL = sum(cpr_rougeLs)/len(cpr_rougeLs)
            avg_cpr_blue = sum(cpr_blues)/len(cpr_blues)
            
            output["compress_prediction"] = pred
            output["compress_blue"] = blue
            output["compress_rougeL"] = rougeL
            
            output_data.append(output)
            if args.query_origin:
                pbar.set_postfix(org_blue=avg_org_blue, org_rougeL=avg_org_rougeL, cpr_blue=avg_cpr_blue, cpr_rougeL=avg_cpr_rougeL)
            else:
                pbar.set_postfix(cpr_blue=avg_cpr_blue, cpr_rougeL=avg_cpr_rougeL)
            pbar.update(1)
            # if (i+1) % 10 == 0:
            if i % 10 == 0 or (i+1) == len(data):
                with open(output_path, 'w') as fout:
                    json.dump(output_data, fout, indent=4)
