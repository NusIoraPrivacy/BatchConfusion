from configs.data_configs import *
from data_util.gpt_utils import get_response
from configs.key import _API_KEY
from configs.templates import evaluation_template
from evaluation.utils import create_client

from sklearn.metrics import accuracy_score, f1_score
from sentence_transformers import SentenceTransformer

from rouge import Rouge
from sacrebleu.metrics import BLEU

import logging
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
    parser.add_argument("--local_epoch", type=str, default="epoch_1.json",
        help = "path of local compression files")
    parser.add_argument("--local_model", type=str, default="Qwen3-1.7B",
        help = "path of local compression files")
    parser.add_argument("--remote_cpr", type=str, default="compress_fake_cattr_multi_4omini_4.json",
        help = "path of gpt compression files")
    parser.add_argument("--data_name", type=str, default="legal-qa-v1")
    parser.add_argument("--query_online_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--eval_online_model", type=str, default="gpt-4o")
    parser.add_argument("--answer_key", type=str, default="response")
    parser.add_argument("--query_key", type=str, default="question")
    parser.add_argument("--cpr_key", type=str, default=["compression", "predicted compression"][1])
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    if args.local_model is None:
        with open(f'{args.root_path}/data/{args.data_name}/{args.remote_cpr}') as fin:
            data = json.load(fin)

    else:
        with open(f'{args.root_path}/result/{args.data_name}/compression/{args.local_model}/{args.local_epoch}') as fin:
            data = json.load(fin)
            
        with open(f'{args.root_path}/data/{args.data_name}/{args.remote_cpr}') as fin:
            data_truth = json.load(fin)
        query2sample = {}
        for sample in data_truth:
            question = sample[args.query_key]
            query2sample[question] = sample
        
        for sample in data:
            question = sample[args.query_key]
            link_sample = query2sample[question]
            sample[args.answer_key] = link_sample[args.answer_key] 

    log_root = f"{args.root_path}/result/{args.data_name}/logs/compression"
    if not os.path.exists(log_root):
        os.makedirs(log_root)
    if args.local_model is None:
        file_name = f"full_question-{args.query_online_model}.log"
    else:
        file_name = f"{args.local_model}-{args.local_epoch}-{args.query_online_model}.log"
    file_path = f"{log_root}/{file_name}"
    logging.basicConfig(
        filename=file_path,
        filemode="w",  # use 'a' to append
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )

    # print(len(data))
    one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
    one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")
    query_client = create_client(args.query_online_model)
    eval_client = create_client(args.eval_online_model)
    bleu_scorer = BLEU(effective_order=True)
    rouge_scorer = Rouge()

    # ref_answers, predictions = [], []
    cpr_rougeLs, cpr_blues, cpr_ratings = [], [], []
    with tqdm(total=len(data), unit='batch') as pbar:
        for i, sample in enumerate(data):
            original_query, ref_answer, compress_query = sample[args.query_key], sample[args.answer_key], sample[args.cpr_key]

            # # original prediction
            # if args.local_cpr is None:
            #     pred = get_response(client, original_query, args)
            #     # print(pred)
            #     try:
            #         score = rouge_scorer.get_scores(hyps=pred, refs=ref_answer)
            #         rougeL = score[0]["rouge-l"]["f"]
            #     except ValueError as e:
            #             rougeL = 0
            #     try:
            #         blue_score = bleu_scorer.sentence_score(hypothesis=pred, references=[ref_answer])
            #         blue = blue_score.score/100
            #     except ValueError as e:
            #         blue = 0
            #     org_rougeLs.append(rougeL)
            #     org_blues.append(blue)
            #     avg_org_rougeL = sum(org_rougeLs)/len(org_rougeLs)
            #     avg_org_blue = sum(org_blues)/len(org_blues)
                
            #     prompt = evaluation_template.format(question=original_query, answer=pred)
            #     _eval = get_response(client, prompt, args, args.eval_gpt_model)
            #     # print(_eval)
            #     match = re.search(one_score_pattern, _eval)
            #     if not match:
            #         match = re.search(one_score_pattern_backup, _eval)

            #     if match:
            #         rating = ast.literal_eval(match.groups()[0])
            #     else:
            #         rating = -1
            #     if rating >= 0:
            #         org_ratings.append(rating)
            #     output["origin_rating"] = rating
            #     avg_org_rating = sum(org_ratings)/len(org_ratings)

            # compression prediction
            pred = get_response(query_client, compress_query, args, args.query_online_model)
            # print(pred)
            # ref_answers.append(ref_answer)
            # predictions.append(pred)
            try:
                # print(pred)
                # print(ref_answer)
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
            
            prompt = evaluation_template.format(question=original_query, answer=pred)
            _eval = get_response(eval_client, prompt, args, args.eval_online_model)
            # print(_eval)
            match = re.search(one_score_pattern, _eval)
            if not match:
                match = re.search(one_score_pattern_backup, _eval)

            if match:
                rating = ast.literal_eval(match.groups()[0])
            else:
                rating = -1
            if rating >= 0:
                cpr_ratings.append(rating)
            avg_cpr_rating = sum(cpr_ratings)/len(cpr_ratings)
            pbar.set_postfix(cpr_blue=avg_cpr_blue, cpr_rougeL=avg_cpr_rougeL, cpr_rating=avg_cpr_rating)
            pbar.update(1)
            logging.info(
                f"Iteration {i+1}/{len(data)} - "
                f"RougeL {avg_cpr_rougeL} - "
                f"Blue {avg_cpr_blue} - "
                f"Rating {avg_cpr_rating}"
            )

