from torch.utils.data import DataLoader
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, default_data_collator
from accelerate import infer_auto_device_map, init_empty_weights
import torch.nn.functional as F

import logging
import time
from tqdm import tqdm
import random
import json
import os
import argparse
import re
from collections import OrderedDict
import math
current = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(current))

models = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]
data_names = ["legal-qa-v1", "medical_o1_reasoning_SFT", "mmlu_fina", "twitter"]
fake_keys = ['fake attributes question', 'fake attributes compression', "fake attributes text"]
priv_keys = ["filtered private attributes question", "filtered private attributes compression", "filtered private attributes", "filtered private attributes text"]
query_keys = ["question", "compression", "text"]

def get_model_tokenizer(model_name, args):
    if "Qwen" in model_name or "Llama" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map = "auto")
    elif "bart" in model_name or "t5" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(args.device)
    elif "gpt2" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(args.device)
    return model, tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default=root_path)
    parser.add_argument("--temperature", type=float, default=1,
        help = "temperature for text generation")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--data_name", type=str, default=data_names[2])
    parser.add_argument("--in_file_name", type=str, default="fina_fake_qcattr_none_zero.json")
    parser.add_argument("--out_file_name", type=str, default="fake_cattr_random_v3_0.5.json")
    parser.add_argument("--fake_key", type=str, default=fake_keys[1])
    parser.add_argument("--priv_key", type=str, default=priv_keys[1])
    parser.add_argument("--query_key", type=str, default=query_keys[1])
    parser.add_argument("--model_name", type=str, default=models[0])
    parser.add_argument("--test_only", type=bool, default=False)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--thd", type=float, default=0.5)
    parser.add_argument("--decay_weight", type=float, default=0.1)
    parser.add_argument("--sample_mul", type=float, default=2)
    parser.add_argument("--max_length", type=int, default=100) # cpr: 100; full: 250
    args = parser.parse_args()
    return args

def get_probs(model, org_attrs, fake_attr_list, org_query, loss_function, args):
    snippet_list = []
    snippet = org_query
    snippet_list = []
    for fake_attrs in fake_attr_list:
        this_snippet = "" + snippet
        # print(this_snippet)
        for org_attr, fake_attr in zip(org_attrs, fake_attrs):
            this_snippet = this_snippet.replace(org_attr, fake_attr)
        snippet_list.append(this_snippet)
    inputs = tokenizer(snippet_list, return_tensors="pt", padding="longest")
    # print(inputs['input_ids'][0])
    # print(inputs['input_ids'].shape)
    if inputs['input_ids'].shape[-1] > args.max_length:
        inputs = tokenizer(snippet_list, return_tensors="pt", padding="max_length", max_length=args.max_length, truncation=True)
    # print(inputs['input_ids'].shape)
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)
    # input_ids = inputs['input_ids'].to(model.device)
    # target_ids[:, :start_idx] = -100
    # print(inputs)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        probs = (probs[...,0]).tolist()

    return probs

def get_sample_number(fake_attrs, conf=0.95, ratio=0.5):
    n_attrs = len(fake_attrs)
    t = max(int(n_attrs * 0.2), 1)
    n_candidates = 0
    for fake_attr_list in fake_attrs:
        n_candidates = max(n_candidates, len(fake_attr_list))
    n_comb = math.comb(n_attrs, t)
    n_samples = int((math.log(1-conf)-math.log(n_comb))/math.log(1-1/n_candidates)/ratio)
    return n_samples

def get_unique_fake_attrs(fake_attrs):
    seen = set()
    new_fake_attrs = []
    for combination in zip(*fake_attrs):
        if combination not in seen:
            seen.add(combination)
            new_fake_attrs.append(list(combination))
    return new_fake_attrs

if __name__ == "__main__":
    args = parse_args()
    print(args)
    log_root = f"{args.root_path}/result/{args.data_name}/logs"
    if not os.path.exists(log_root):
        os.makedirs(log_root)
    file_name = (args.out_file_name).replace(".json", "")
    model_name = (args.model_name).split("/")[-1]
    file_name = f"pjdgen-{file_name}-{model_name}.log"
    file_path = f"{log_root}/{file_name}"
    logging.basicConfig(
        filename=file_path,
        filemode="w",  # use 'a' to append
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )
    model_name = "-".join((args.model_name).split("/"))
    model_path = f"{args.root_path}/model/{args.data_name}/{model_name}"
    model, tokenizer = get_model_tokenizer(model_path, args)
    with open(f'{args.root_path}/data/{args.data_name}/{args.in_file_name}') as fin:
        data = json.load(fin)

    loss_function = torch.nn.CrossEntropyLoss(reduction="none")
    out_path = f'{args.root_path}/result/{args.data_name}/{args.out_file_name}'
    if os.path.exists(out_path):
        with open(out_path) as fin:
            outputs = json.load(fin)
    else:
        outputs = []

    prev_questions = []
    for sample in outputs:
        prev_questions.append(sample[args.query_key])

    times = []
    all_samples = []
    total_sample_sizes = []
    with tqdm(total=len(data)) as pbar:
        for cnt, sample in enumerate(data):
            # print(sample.keys())
            if sample[args.query_key] in prev_questions:
                pbar.update(1)
                continue
            # print(sample.keys())
            priv_attrs, fake_attrs, query = sample[args.priv_key], sample[args.fake_key], sample[args.query_key]
            if len(fake_attrs) and len(query) == 0:
                pbar.update(1)
                continue
            fake_key = args.fake_key
            # sample fake attributes
            n_samples = get_sample_number(fake_attrs, ratio=1/args.sample_mul)
            beam = []
            sample_fake_attrs = []
            for org_attr, fake_attr_list in zip(priv_attrs, fake_attrs):
                fake_attr_list = list(set(fake_attr_list))
                sample_fakes = random.choices(fake_attr_list, k = n_samples)
                sample_fake_attrs.append(sample_fakes)
            sample_fake_attrs = get_unique_fake_attrs(sample_fake_attrs)
            # print("attribute length:", len(priv_attrs))
            total_sample_sizes.append(len(sample_fake_attrs))
            t1 = time.time()
            
            itrs = len(sample_fake_attrs)//args.batch_size + 1
            scores = []
            for i in range(itrs):
                this_seq_list = sample_fake_attrs[(i*args.batch_size):((i+1)*args.batch_size)]
                if len(this_seq_list) > 0:
                    # print(this_seq_list)
                    this_scores = get_probs(model, priv_attrs, this_seq_list, query, loss_function, args)
                    scores.extend(this_scores)
            
            t2 = time.time()
            times.append(t2-t1)
            sample[fake_key] = []
            for score, seq in zip(scores, sample_fake_attrs):
                if score < args.thd:
                    sample[fake_key].append(seq)
            n_samples = len(sample[fake_key])
            all_samples.append(n_samples)
            outputs.append(sample)
            with open(out_path, 'w') as fout:
                json.dump(outputs, fout, indent=4)
            pbar.update(1)
            avg_time = sum(times)/len(times)
            avg_sample_size = sum(all_samples)/len(all_samples)
            avg_total_sample = sum(total_sample_sizes)/len(total_sample_sizes)
            pbar.set_postfix(time=avg_time, n_sample=n_samples, total_n_sample=len(sample_fake_attrs))
            logging.info(
                f"Iteration {cnt+1}/{len(data)} - "
                f"Average total sample size {avg_total_sample} - "
                f"Average sample size {avg_sample_size} - "
                f"Average time {avg_time}s"
            )