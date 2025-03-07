from configs.data_configs import *
from data_util.gpt_utils import get_response
from configs.templates import attr_extract_template
from configs.key import _API_KEY

from sklearn.metrics import accuracy_score, f1_score
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import numpy as np

import copy
from openai import OpenAI
import json
import argparse
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

def get_prediction(query, choices, client, args):
    prompt = (f"Question: {query}\n Please select one of the options, and output A-D only:\n"
                f"A: {choices[0]}\n B: {choices[1]}\n C: {choices[2]}\n D: {choices[3]}"
                "Remember to output only a single character from A to D!")
    # print(prompt)
    raw_pred = get_response(client, prompt, args)
    # print(raw_pred)
    pred = standard_ans(raw_pred, candidate_labels)
    return pred

def cluster_sentences(all_sentence_dicts, args):
    all_sentences = []
    for sent_dict in all_sentence_dicts:
        all_sentences.append(sent_dict["question"])
    embedder = SentenceTransformer('paraphrase-distilroberta-base-v1')
    corpus_embeddings = embedder.encode(all_sentences)
    clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=args.cluster_thd)
    
    # fit model and predict clusters
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    new_sentence_dicts = []
    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        this_sent_dict = copy.deepcopy(all_sentence_dicts[sentence_id])
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []
        clustered_sentences[cluster_id].append(this_sent_dict["question"])
        new_query = clustered_sentences[cluster_id][0]
        attrs = this_sent_dict["private attrs"]
        for attr in attrs:
            new_query = new_query.replace("#MASK", attr, 1)
        this_sent_dict["question"] = new_query
        new_sentence_dicts.append(this_sent_dict)

    return new_sentence_dicts, len(clustered_sentences)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default=root_path)
    parser.add_argument("--temperature", type=float, default=1,
        help = "temperature for text generation")
    parser.add_argument("--max_tokens", type=int, default=1000,
        help = "max new token for text generation")
    parser.add_argument("--gpt_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--cluster_thd", type=float, default=0.3)
    parser.add_argument("--n_fake", type=int, default=10)
    parser.add_argument("--ratio_retain", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    with open(f'{root_path}/data/mmlu/private_attrs_compress_ner.json') as fin:
        data = json.load(fin)
    # sample data with replacement
    proportions = np.random.dirichlet(np.repeat(args.alpha, len(data)))
    sampled_data = []
    sampled_indices = list(np.random.choice(len(data), len(data), p=proportions, replace=True))
    # print(sampled_indices)
    for idx in sampled_indices:
        sampled_data.append(data[idx])
    data = sampled_data
    candidate_labels = ["A", "B", "C", "D"]

    client = OpenAI(api_key=_API_KEY)
    # group questions
    org_sentence_dict = defaultdict(list)
    cpr_sentence_dict = defaultdict(list)
    # construct masked group by number of private attributes
    org_complexity, cpr_complexity = 0, 0
    for sample in data:
        org_sentence, org_attrs, cpr_sentence, cpr_attrs = sample["question"], sample["origin private attributes"], sample["compression"], sample["compress private attributes"]
        n_org, n_cpr = len(org_attrs), len(cpr_attrs)
        org_complexity += (args.n_fake ** n_org) * (args.ratio_retain ** max(n_org-1, 0))
        cpr_complexity += (args.n_fake ** n_cpr) * (args.ratio_retain ** max(n_cpr-1, 0))
        answer, choices = sample["answer"], sample["choices"]
        for attr in org_attrs:
            org_sentence = org_sentence.replace(attr, "#MASK")
        for attr in cpr_attrs:
            cpr_sentence = cpr_sentence.replace(attr, "#MASK")
        org_dict = {"question": org_sentence, "choices": choices, "answer": answer, "private attrs": org_attrs}
        cpr_dict = {"question": cpr_sentence, "choices": choices, "answer": answer, "private attrs": cpr_attrs}
        org_sentence_dict[len(org_attrs)].append(org_dict)
        cpr_sentence_dict[len(cpr_attrs)].append(cpr_dict)
    org_complexity /= len(data)
    cpr_complexity /= len(data)
    print("Complexity for original sentence before batching:", org_complexity)
    print("Complexity for compressed sentence before batching:", cpr_complexity)

    # cluster each sentence in the dictionary
    new_org_sentence_dict = []
    org_complexity = 0
    for n in org_sentence_dict:
        print("Cluster for original sentences:", n)
        all_sentence_dicts = org_sentence_dict[n]
        if len(all_sentence_dicts) == 1:
            new_sentence_dicts, n_sent = all_sentence_dicts, 1
        else:
            new_sentence_dicts, n_sent = cluster_sentences(all_sentence_dicts, args)
        org_complexity += n_sent * (args.n_fake ** n) * (args.ratio_retain ** max(n-1, 0))
        new_org_sentence_dict.extend(new_sentence_dicts)
    org_complexity /= len(data)
    print("Complexity for original sentence after batching:", org_complexity)
    
    new_cpr_sentence_dict = []
    cpr_complexity = 0
    for n in cpr_sentence_dict:
        print("Cluster for compressed sentences:", n)
        all_sentence_dicts = cpr_sentence_dict[n]
        if len(all_sentence_dicts) == 1:
            new_sentence_dicts, n_sent = all_sentence_dicts, 1
        else:
            new_sentence_dicts, n_sent = cluster_sentences(all_sentence_dicts, args)
        cpr_complexity += n_sent * (args.n_fake ** n) * (args.ratio_retain ** max(n-1, 0))
        new_cpr_sentence_dict.extend(new_sentence_dicts)
    cpr_complexity /= len(data)
    print("Complexity for compressed sentence after batching:", cpr_complexity)
    
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
