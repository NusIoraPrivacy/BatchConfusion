from configs.data_configs import *
from data_util.gpt_utils import get_response
from configs.templates import attr_extract_template
from configs.key import _API_KEY
from configs.globals import token_cost_dict

from sklearn.metrics import accuracy_score, f1_score
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import numpy as np
import racplusplus
import tiktoken

import copy
from openai import OpenAI
import json
import argparse
import os
import time
from tqdm import tqdm
current = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(current))

def num_tokens_from_string(string):
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

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
    raw_pred = get_response(client, prompt, args)
    # print(raw_pred)
    pred = standard_ans(raw_pred, candidate_labels)
    return pred

def cluster_sentences(all_sentence_dicts, embedder, args):
    all_sentences = []
    for sent_dict in all_sentence_dicts:
        all_sentences.append(sent_dict["question"])
    
    corpus_embeddings = embedder.encode(all_sentences)
    # print(corpus_embeddings.shape)
    if args.cluster_mode == "agglomerative":
        clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=args.cluster_thd)
        # fit model and predict clusters
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_
        
    elif args.cluster_mode == "rac":
        cluster_assignment = racplusplus.rac(corpus_embeddings, args.cluster_thd, None, 1000, 8, "euclidean")

    else:
        print("Invalid cluster mode")

    new_sentence_dicts = []
    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        this_sent_dict = copy.deepcopy(all_sentence_dicts[sentence_id])
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []
        clustered_sentences[cluster_id].append((this_sent_dict["question"], this_sent_dict["query cost"]))
        new_query = clustered_sentences[cluster_id][0][0]
        attrs = this_sent_dict["private attrs"]
        for attr in attrs:
            new_query = new_query.replace("#MASK", attr, 1)
        this_sent_dict["question"] = new_query
        new_sentence_dicts.append(this_sent_dict)

    return new_sentence_dicts, clustered_sentences

fake_keys = ['fake attributes question', 'fake attributes compression', "fake attributes text"]
priv_keys = ["filtered private attributes question", "filtered private attributes compression", "filtered private attributes", "filtered private attributes text"]
query_keys = ["question", "compression", "text"]
resp_keys = ["prediction", "compress_prediction", "origin_pred", "compress_pred"]
data2key = {
    "legal-qa-v1": {
        "full": ["question", "filtered private attributes", "fake attributes question", "response"],
        "compress": ["compression", "filtered private attributes compression", "fake attributes compression", "response"]
        },
    "medical_o1_reasoning_SFT": {
        "full": ["question", "filtered private attributes question", "fake attributes question", "response"],
        "compress": ["compression", "filtered private attributes compression", "fake attributes compression", "response"]
        },
    "mmlu_fina": {
        "full": ["question", "filtered private attributes question", "fake attributes question", "origin_pred"],
        "compress": ["compression", "filtered private attributes compression", "fake attributes compression", "compress_pred"]
        }
}

common_col_keys = ["question", "private attributes", "fake attributes", "response"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default=root_path)
    parser.add_argument("--temperature", type=float, default=1,
        help = "temperature for text generation")
    parser.add_argument("--max_tokens", type=int, default=1000,
        help = "max new token for text generation")
    parser.add_argument("--gpt_model", type=str, default="gpt-4o")
    parser.add_argument("--cluster_thd", type=float, default=0.3)
    parser.add_argument("--n_fake", type=int, default=10)
    parser.add_argument("--ratio_retain", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=10)
    parser.add_argument("--file_name", type=str, default="fake_cattr_random_Qwen2.5-0.5B-Instruct_10.json")
    parser.add_argument("--data_name", type=str, default="mmlu_fina")
    parser.add_argument("--data_mode", type=str, default=["full", "compress"][1])
    parser.add_argument("--fake_key", type=str, default=fake_keys[1])
    parser.add_argument("--priv_key", type=str, default=priv_keys[1])
    parser.add_argument("--query_key", type=str, default=query_keys[1])
    parser.add_argument("--resp_key", type=str, default=resp_keys[3])
    parser.add_argument("--sent_transformer", type=str, default='paraphrase-distilroberta-base-v1')
    parser.add_argument("--sample_size", type=int, default=3000)
    parser.add_argument("--cluster_mode", type=str, default=["agglomerative", "rac"][1])
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    if args.data_name == "mix":
        data = []
        query_key,  priv_key, fake_key, resp_key = common_col_keys[0], common_col_keys[1], common_col_keys[2], common_col_keys[3]
        for data_name in data2key:
            col_keys = data2key[data_name][args.data_mode]
            with open(f'{root_path}/result/{data_name}/{args.file_name}') as fin:
                this_data = json.load(fin)
            for sample in this_data:
                data.append({
                    query_key: sample[col_keys[0]], priv_key: sample[col_keys[1]], fake_key: sample[col_keys[2]], resp_key: sample[col_keys[3]]
                })
                # except:
                #     print(sample)
                #     print(data_name)
    else:
        query_key,  priv_key, fake_key, resp_key = args.query_key,  args.priv_key, args.fake_key, args.resp_key
        with open(f'{root_path}/result/{args.data_name}/{args.file_name}') as fin:
            data = json.load(fin)
    # sample data with replacement
    proportions = np.random.dirichlet(np.repeat(args.alpha, len(data)))
    # print(len(data))
    sampled_data = []
    sampled_indices = list(np.random.choice(len(data), args.sample_size, p=proportions, replace=True))
    # print(len(sampled_indices))
    for idx in sampled_indices:
        sampled_data.append(data[idx])
    data = sampled_data
    candidate_labels = ["A", "B", "C", "D"]

    # client = OpenAI(api_key=_API_KEY)
    # group questions
    sentence_dict = defaultdict(list)
    # construct masked group by number of private attributes
    query_cost = 0
    for sample in data:
        priv_attrs, fake_attrs, query, response = sample[priv_key], sample[fake_key], sample[query_key], sample[resp_key]
        n_in_tokens = num_tokens_from_string(query)
        n_out_tokens = num_tokens_from_string(response)
        cost = (token_cost_dict[args.gpt_model][0]*n_in_tokens + token_cost_dict[args.gpt_model][1]*n_out_tokens) * (len(fake_attrs)+1)
        query_cost += cost
        # answer, choices = sample["answer"], sample["choices"]
        for attr in priv_attrs:
            query = query.replace(attr, "#MASK")
        query_dict = {"question": query, "private attrs": priv_attrs, "query cost": cost}
        sentence_dict[len(priv_attrs)].append(query_dict)
    query_cost /= len(data)
    print("Query cost for sentence before batching:", query_cost)

    # cluster each sentence in the dictionary
    new_sentence_dict = []
    query_cost = 0
    embedder = SentenceTransformer(args.sent_transformer, device="cuda")
    all_times = []
    for n in sentence_dict:
        print("Cluster for original sentences:", n)
        all_sentence_dicts = sentence_dict[n]
        t1 = time.time()
        if len(all_sentence_dicts) == 1:
            this_new_sentence_dict, n_sent = all_sentence_dicts, 1
            this_cost = this_new_sentence_dict[0]["query cost"]
            query_cost += this_cost
        else:
            this_new_sentence_dict, this_clusters = cluster_sentences(all_sentence_dicts, embedder, args)
            n_sent = len(this_clusters)
            for cluster_id in this_clusters:
                this_cost = 0
                all_pairs = this_clusters[cluster_id]
                for query, cost in all_pairs:
                    this_cost = max(this_cost, cost)
                query_cost += this_cost
        t2 = time.time()
        all_times.append(t2-t1)
        new_sentence_dict.extend(this_new_sentence_dict)
    query_cost /= len(data)
    print("Query cost for sentence after batching:", query_cost)
    avg_time = sum(all_times)/len(all_times)
    print("Average batching time:", avg_time)
    
    # # retreive responses
    # output_data = []
    # labels, origin_predictions, compress_predictions = [], [], []
    # with tqdm(total=len(data), unit='batch') as pbar:
    #     for i, sample in enumerate(data):
    #         original_query, compress_query, choices, label = sample["question"], sample["compression"], sample["choices"], sample["answer"]
    #         origin_pred = get_prediction(original_query, choices, client, args)
    #         compress_pred = get_prediction(compress_query, choices, client, args)
    #         label = candidate_labels[label]
    #         labels.append(label)
    #         origin_predictions.append(origin_pred)
    #         compress_predictions.append(compress_pred)
    #         pbar.update(1)
    #         org_acc = accuracy_score(labels, origin_predictions)
    #         org_f1 = f1_score(labels, origin_predictions, average="macro")
    #         compress_acc = accuracy_score(labels, compress_predictions)
    #         compress_f1 = f1_score(labels, compress_predictions, average="macro")
    #         pbar.set_postfix(origin_acc=org_acc, origin_f1=org_f1, compress_acc=compress_acc, compress_f1=compress_f1)
    #         output = copy.deepcopy(sample)
    #         output["origin_pred"] = origin_pred
    #         output["compress_pred"] = compress_pred
    #         output_data.append(output)
    #         if (i+1) % 10 == 0:
    #             with open(f'{args.root_path}/result/mmlu/answer_{args.gpt_model}.json', 'w') as fout:
    #                 json.dump(output_data, fout, indent=4)
