from data_util.gpt_utils import get_response, get_response_multiprompts
from configs.data_configs import *
from configs.templates import compress_template, compress_reflect_template_oa, evaluation_template
from configs.key import _API_KEY

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
    parser.add_argument("--data_name", type=str, default="medical_o1_reasoning_SFT")
    parser.add_argument("--rating_thd", type=int, default=7)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    with open(f'{root_path}/result/{args.data_name}/answer_gpt-4o-mini.json') as fin:
        data = json.load(fin)
    # random.shuffle(data)
    client = OpenAI(api_key=_API_KEY)
    output_data = []
    labels = []
    predictions = []
    one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
    one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")
    with tqdm(total=len(data), unit='batch') as pbar:
        ratings = []
        for sample in data:
            org_question, cpr_question, ref_ans, cpr_pred, rating = sample["question"], sample["compression"], sample["response"], sample["compress_prediction"], sample["compress_rating"]
            output = copy.deepcopy(sample)
            if rating <= args.rating_thd:
                success = False
                n_try = 0
                prompt_old = compress_template.format(question=org_question)
                prompts = [prompt_old]
                while not success:
                    # obtain new compression result
                    prompts.append(cpr_question)
                    prompt_new = compress_reflect_template_oa.format(ref_ans=ref_ans, rating=rating, bad_ans=cpr_pred)
                    prompts.append(prompt_new)
                    result = get_response_multiprompts(client, prompts, args)
                    # print(result)
                    cpr_question = extract_content("#thecompression:", result)
                    # print(cpr_question)
                    # query gpt
                    cpr_pred = get_response(client, cpr_question, args, args.gpt_model)

                    prompt = evaluation_template.format(question=org_question, answer=cpr_pred)
                    _eval = get_response(client, prompt, args, args.gpt_model_eval)
                    # print(_eval)
                    match = re.search(one_score_pattern, _eval)
                    if not match:
                        match = re.search(one_score_pattern_backup, _eval)

                    if match:
                        rating = ast.literal_eval(match.groups()[0])
                    else:
                        rating = -1

                    if rating > args.rating_thd:
                        success = True
                    n_try += 1
                    if n_try >= 5:
                        break
                # print(compress_pred)
                output["compression"] = cpr_question
                output["compress_prediction"] = cpr_pred
                output["compress_rating"] = rating
                
                ratings.append(rating)
                avg_rating = sum(ratings)/len(ratings)
                pbar.set_postfix(rating=avg_rating)
            output_data.append(output)
            with open(f'{args.root_path}/result/{args.data_name}/compress_{args.gpt_model}_v2.json', 'w') as fout:
                json.dump(output_data, fout, indent=4)
            pbar.update(1)