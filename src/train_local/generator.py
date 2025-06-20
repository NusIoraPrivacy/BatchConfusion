from transformers import default_data_collator
from torch.utils.data import DataLoader
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import infer_auto_device_map, init_empty_weights

import time
from tqdm import tqdm
import random
import json
import os
import argparse
import re
from collections import OrderedDict
current = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(current))

models = ["Qwen/Qwen2.5-1.5B-Instruct", "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.1-8B", "openai-community/gpt2"]
fake_keys = ['fake attributes question', 'fake attributes compression']

def get_model_tokenizer(model_name, args):
    if "Qwen" in model_name or "Llama" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        with init_empty_weights():
            model = AutoModelForCausalLM.from_pretrained(model_name)
        # device_map = infer_auto_device_map(model, max_memory={0: "2GiB", 1: "2GiB", 2: "2GiB", 3: "2GiB",}, 
        #             no_split_module_classes=['MixtralDecoderLayer', "LlamaDecoderLayer", "Phi3DecoderLayer"])
        # print(device_map)
        device_map = OrderedDict({'model.embed_tokens': 0, 'model.layers.0': 1, 'model.layers.1': 1, 'model.layers.2': 1, 'model.layers.3': 1, 'model.layers.4': 1, 'model.layers.5': 1, 'model.layers.6': 1, 'model.layers.7': 1, 'model.layers.8': 2, 'model.layers.9': 2, 'model.layers.10': 2, 'model.layers.11': 2, 'model.layers.12': 2, 'model.layers.13': 2, 'model.layers.14': 2, 'model.layers.15': 2, 'model.norm': 2, 'model.rotary_emb': 2, 'lm_head': 0})
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map = device_map)
    elif "bart" in model_name or "t5" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(args.device)
    elif "gpt2" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(args.device)
    return model, tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default=root_path)
    parser.add_argument("--temperature", type=float, default=1,
        help = "temperature for text generation")
    parser.add_argument("--max_tokens", type=int, default=1000,
        help = "max new token for text generation")
    parser.add_argument("--token_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--data_name", type=str, default="mmlu_fina")
    parser.add_argument("--in_file_name", type=str, default="fina_fake_qcattr_none_zero.json")
    parser.add_argument("--out_file_name", type=str, default="compress_fake_qattr.json")
    parser.add_argument("--fake_key", type=str, default=fake_keys[0])
    parser.add_argument("--model_name", type=str, default=models[1])
    parser.add_argument("--test_only", type=bool, default=False)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--thd", type=float, default=5.5)
    parser.add_argument("--decay_weight", type=float, default=0.1)
    parser.add_argument("--max_k", type=float, default=500)
    args = parser.parse_args()
    return args

# def get_surprisal(model, org_attrs, fake_attrs, start_idx=0, end=False):
#     match = re.search(org_attrs[-1], org_query)
#     if match:
#         if i == 0:
#             start_idx, end_idx = match.span()
#         else:
#             end_idx = match.span()[-1]
#         if end:
#             snippet = org_query
#         else:
#             snippet = org_query[:end_idx]
#         for org_attr, fake_attr in zip(org_attrs, fake_attrs):
#             snippet = snippet.replace(org_attr, fake_attr)
#         inputs = tokenizer(snippet, return_tensors="pt")
#         input_ids = inputs['input_ids'].to(model.device)
#         target_ids = input_ids.clone()
#         target_ids[:, :start_idx] = -100

#         with torch.no_grad():
#             outputs = model(input_ids, labels=target_ids)
#             neg_log_likelihood = (outputs.loss).item()
#     else:
#         neg_log_likelihood = 0
#     return neg_log_likelihood

def get_surprisal(model, org_attrs, fake_attr_list, org_query, loss_function, start_idx=0, end=False):
    match = re.search(re.escape(org_attrs[-1]), org_query)
    if match:
        snippet_list = []
        if i == 0:
            start_idx, end_idx = match.span()
        else:
            end_idx = match.span()[-1]
        if end:
            snippet = org_query
        else:
            snippet = org_query[:end_idx]
        snippet_list = []
        for fake_attrs in fake_attr_list:
            this_snippet = "" + snippet
            for org_attr, fake_attr in zip(org_attrs, fake_attrs):
                this_snippet = this_snippet.replace(org_attr, fake_attr)
            snippet_list.append(this_snippet)
        inputs = tokenizer(snippet_list, return_tensors="pt", padding="longest")
        for key in inputs:
            inputs[key] = inputs[key].to(model.device)
        # input_ids = inputs['input_ids'].to(model.device)
        target_ids = inputs['input_ids'].clone()
        target_ids[:, :start_idx] = -100

        with torch.no_grad():
            outputs = model(**inputs, labels=target_ids)
            logits = outputs.logits
            shift_labels = target_ids[..., 1:].contiguous()
            shift_logits = logits[..., :-1, :].contiguous()
            loss = loss_function(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_logits.size(0), shift_logits.size(1))
            # Resize and average loss per sample
            neg_log_likelihoods = loss.sum(axis=1)
            label_cnt = shift_labels.ne(-100).sum(axis=1)
            neg_log_likelihoods = (neg_log_likelihoods/label_cnt).tolist()
    else:
        neg_log_likelihoods = [0] * len(fake_attr_list)
    return neg_log_likelihoods

if __name__ == "__main__":
    args = parse_args()
    model, tokenizer = get_model_tokenizer(args.model_name, args)
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
        prev_questions.append(sample["question"])
    
    with tqdm(total=len(data)) as pbar:
        for sample in data:
            # print(sample.keys())
            if sample["question"] in prev_questions:
                continue
            priv_attrs, priv_cpr_attrs, fake_attrs = sample["filtered private attributes question"], sample["filtered private attributes compression"], sample[args.fake_key]
            # priv_attrs, priv_cpr_attrs, fake_attrs = sample["filtered private attributes"], sample["filtered private attributes compression"], sample[args.fake_key]
            fake_key = args.fake_key

            org_query, cpr_query = sample["question"], sample["compression"]
            # print("attribute length:", len(priv_attrs))
            t1 = time.time()
            top_k = 1
            beam = [([], 100)]
            cur_attrs = []
            for i, (attr, fake_attr_list) in enumerate(zip(priv_attrs, fake_attrs)):
                # print(len(fake_attr_list))
                fake_attr_list = list(set(fake_attr_list))
                cur_attrs.append(attr)
                top_k = top_k * len(fake_attr_list)
                if i >= 1:
                    top_k = min(args.max_k, int(top_k * args.decay_weight))
                # print("Top k:", top_k)
                if i == 0:
                    match = re.search(attr, org_query)
                    if match:
                        start_idx = match.span()[0]
                    else:
                        start_idx = 0
                new_beam = []
                new_seq_list = []
                for seq, score in beam:
                    # for fake_attr in fake_attr_list:
                    #     new_seq = seq + [fake_attr]
                    #     score = get_surprisal(model, cur_attrs, new_seq, org_query, loss_function, start_idx=start_idx)
                    #     # print(new_seq)
                    #     # print(score)
                    #     if score < args.thd:
                    #         new_beam.append((new_seq, score))
                    for fake_attr in fake_attr_list:
                        new_seq = seq + [fake_attr]
                        new_seq_list.append(new_seq)
                itrs = len(new_seq_list)//args.batch_size + 1
                scores = []
                for i in range(itrs):
                    this_seq_list = new_seq_list[(i*args.batch_size):((i+1)*args.batch_size)]
                    if len(this_seq_list) > 0:
                        this_scores = get_surprisal(model, cur_attrs, this_seq_list, org_query, loss_function, start_idx=start_idx)
                        scores.extend(this_scores)
                # # print(new_seq)
                # print(scores)
                for score, new_seq in zip(scores, new_seq_list):
                    # if score < args.thd:
                    new_beam.append((new_seq, score))
                beam = sorted(new_beam, key=lambda x: x[1])[:top_k]
                # print("beam length:", len(beam))
            sample[fake_key] = []
            for seq, score in beam:
                sample[fake_key].append(seq)
            outputs.append(sample)
            with open(out_path, 'w') as fout:
                json.dump(outputs, fout, indent=4)
            pbar.update(1)