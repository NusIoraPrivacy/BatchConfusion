from data_util.gpt_utils import get_response, get_response_multiprompts
from configs.data_configs import *
from configs.templates import compress_template, compress_reflect_template
from configs.key import _API_KEY

from sklearn.metrics import accuracy_score, f1_score
import random
from openai import OpenAI
import json
import copy
import argparse
import re
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

def get_prediction(query, choices, client, args):
    prompt = (f"Question: {query}\n Please select one of the options, and output A-D only:\n"
                f"A: {choices[0]}\n B: {choices[1]}\n C: {choices[2]}\n D: {choices[3]}"
                "Remember to output only a single character from A to D!")
    # print(prompt)
    raw_pred = get_response(client, prompt, args, args.gpt_model_resp)
    # print(raw_pred)
    pred = standard_ans(raw_pred, candidate_answers)
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
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    with open(f'{root_path}/result/{args.data_name}/answer_gpt-4o-mini.json') as fin:
        data = json.load(fin)
    random.shuffle(data)
    client = OpenAI(api_key=_API_KEY)
    output_data = []
    candidate_answers = ["A", "B", "C", "D"]
    labels = []
    predictions = []
    with tqdm(total=len(data), unit='batch') as pbar:
        for sample in data:
            org_question, cpr_question, answer, org_pred, cpr_pred = sample["question"], sample["compression"], sample["answer"], sample["origin_pred"], sample["compress_pred"]
            try:
                org_pred = candidate_answers.index(org_pred)
            except:
                org_pred = 0
            try:
                cpr_pred = candidate_answers.index(cpr_pred)
            except:
                cpr_pred = 0
            output = copy.deepcopy(sample)
            if org_pred != cpr_pred and org_pred == answer:
                choices = sample["choices"]
                true_ans = choices[answer]
                success = False
                n_try = 0
                prompt_old = compress_template.format(question=org_question)
                prompts = [prompt_old]
                while not success:
                    wrong_ans = choices[cpr_pred]
                    # obtain new compression result
                    prompts.append(cpr_question)
                    prompt_new = compress_reflect_template.format(true_ans=true_ans, wrong_ans=wrong_ans)
                    prompts.append(prompt_new)
                    result = get_response_multiprompts(client, prompts, args)
                    # print(result)
                    cpr_question = extract_content("#thecompression:", result)
                    # print(result)
                    # query gpt
                    cpr_pred = get_prediction(cpr_question, choices, client, args)
                    cpr_pred = candidate_answers.index(cpr_pred)
                    if cpr_pred == answer:
                        success = True
                    n_try += 1
                    if n_try >= 5:
                        break
                # print(compress_pred)
                output["compression"] = cpr_question
                output["compress_pred"] = candidate_answers[cpr_pred]
                
                labels.append(answer)
                predictions.append(cpr_pred)
                acc = accuracy_score(labels, predictions)
                f1 = f1_score(labels, predictions, average="macro")
                pbar.set_postfix(acc=acc, f1=f1)
            output_data.append(output)
            with open(f'{args.root_path}/result/{args.data_name}/compress_{args.gpt_model}_v2.json', 'w') as fout:
                json.dump(output_data, fout, indent=4)
            pbar.update(1)