from data_util.gpt_utils import get_response, get_response_multiprompts
from configs.data_configs import *
from configs.templates import compress_template, compress_reflect_template_oa_v2, evaluation_template
from configs.key import _API_KEY

from rouge import Rouge
from sacrebleu.metrics import BLEU
from sklearn.metrics import accuracy_score, f1_score
import random
from openai import OpenAI
import json
import copy
import argparse
import re
import ast
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

def extract_content(tag, text):
    # Find the starting position of the tag
    start_idx = text.find(tag)

    # If tag is not found, return None
    if start_idx == -1:
        return ""
    
    # Extract the content after the tag
    content_after_tag = text[start_idx+len(tag):].strip()
    return content_after_tag

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default=root_path)
    parser.add_argument("--temperature", type=float, default=1,
        help = "temperature for text generation")
    parser.add_argument("--max_tokens", type=int, default=1000,
        help = "max new token for text generation")
    parser.add_argument("--gpt_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--gpt_model_eval", type=str, default="gpt-4o")
    parser.add_argument("--data_name", type=str, default="financial_datasets")
    parser.add_argument("--rating_thd", type=int, default=7)
    parser.add_argument("--in_file_name", type=str, default="cpr_v2.json")
    parser.add_argument("--out_file_name", type=str, default="cpr_v3.json")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    with open(f'{root_path}/result/{args.data_name}/{args.in_file_name}') as fin:
        data = json.load(fin)
    # random.shuffle(data)
    client = OpenAI(api_key=_API_KEY)
    bleu_scorer = BLEU(effective_order=True)
    rouge_scorer = Rouge()
    output_data = []
    labels = []
    predictions = []
    one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
    one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")
    with tqdm(total=len(data), unit='batch') as pbar:
        rougeLs, blues = [], []
        for sample in data:
            org_question, cpr_question, ref_ans, cpr_pred, org_blue, cpr_blue, org_rougeL, cpr_rougeL = sample["question"], sample["compression"], sample["response"], sample["compress_prediction"], sample["origin_blue"], sample["compress_blue"], sample["origin_rougeL"], sample["compress_rougeL"]
            output = copy.deepcopy(sample)
            prompt_old = compress_template.format(question=org_question)
            prompts = [prompt_old]
            # obtain new compression result
            prompts.append(cpr_question)
            prompt_new = compress_reflect_template_oa_v2.format(ref_ans=ref_ans, bad_ans=cpr_pred)
            prompts.append(prompt_new)
            result = get_response_multiprompts(client, prompts, args, args.gpt_model_eval)
            # print(result)
            cpr_question = extract_content("#thecompression:", result)
            cpr_query = cpr_question + " Please answer as concise as possible."
            # print(cpr_question)
            # query gpt
            cpr_pred = get_response(client, cpr_query, args, args.gpt_model)
            try:
                score = rouge_scorer.get_scores(hyps=cpr_pred, refs=ref_ans)
                rougeL = score[0]["rouge-l"]["f"]
            except ValueError as e:
                    rougeL = 0
            try:
                blue_score = bleu_scorer.sentence_score(hypothesis=cpr_pred, references=[ref_ans])
                blue = blue_score.score/100
            except ValueError as e:
                blue = 0
            # print(compress_pred)
            output["compression"] = cpr_question
            output["compress_prediction"] = cpr_pred
            output["compress_blue"] = blue
            output["compress_rougeL"] = rougeL
            
            rougeLs.append(rougeL)
            blues.append(blue)
            avg_blue = sum(blues)/len(blues)
            avg_rougeL = sum(rougeLs)/len(rougeLs)
            pbar.set_postfix(blue=avg_blue, rougeL=avg_rougeL)
            output_data.append(output)
            with open(f'{args.root_path}/result/{args.data_name}/{args.out_file_name}', 'w') as fout:
                json.dump(output_data, fout, indent=4)
            pbar.update(1)