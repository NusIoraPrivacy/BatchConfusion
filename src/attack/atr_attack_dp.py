from transformers import default_data_collator, get_scheduler
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

from baselines.utils import paraphrase, custext_priv, load_para_model, load_emb_model

from tqdm import tqdm
import random
import json
import os
import argparse
import numpy as np
import logging

from attack.datasets import *
from attack.utils import get_model_tokenizer

current = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(current))

models = ["Qwen/Qwen2.5-1.5B-Instruct", "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.1-8B", "openai-community/gpt2"]
data_names = ["twitter"]
query_keys = ["text"]
attr_keys = ["gender"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default=root_path)
    parser.add_argument("--temperature", type=float, default=1,
        help = "temperature for text generation")
    parser.add_argument("--max_tokens", type=int, default=1000,
        help = "max new token for text generation")
    parser.add_argument("--token_len", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=10)
    parser.add_argument("--test_batch_size", type=int, default=20)
    parser.add_argument("--data_name", type=str, default=data_names[0])
    parser.add_argument("--in_file_name", type=str, default="gender_dataset.json")
    parser.add_argument("--query_key", type=str, default=query_keys[0])
    parser.add_argument("--attr_key", type=str, default=attr_keys[0])
    parser.add_argument("--model_name", type=str, default=models[0])
    parser.add_argument("--train_pct", type=float, default=0.8)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--max_word", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--para_model", type=str, default="eugenesiow/bart-paraphrase")
    parser.add_argument("--emb_model", type=str, default="openai-community/gpt2")
    parser.add_argument("--dp_method", type=str, default=["paraphrase", "custext", "none"][0])
    parser.add_argument("--use_emb", action='store_true')
    parser.add_argument("--epsilon", type=float, default=1)
    parser.add_argument("--sample_size", type=int, default=5000)
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()
    return args

def process_target(target):
    if target == "female":
        return 0
    elif target == "male":
        return 1
    else:
        return -1

if __name__ == "__main__":
    args = parse_args()
    print(args)
    log_root = f"{args.root_path}/result/{args.data_name}/logs"
    if not os.path.exists(log_root):
        os.makedirs(log_root)
    model_name = (args.model_name).split("/")[-1]
    file_name = f"atr-{model_name}-{args.dp_method}-eps-{args.epsilon}.log"
    file_path = f"{log_root}/{file_name}"
    logging.basicConfig(
        filename=file_path,
        filemode="w",  # use 'a' to append
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )

    model, tokenizer = get_model_tokenizer(args.model_name, num_labels=2, args=args)
    with open(f'{args.root_path}/data/{args.data_name}/{args.in_file_name}') as fin:
        data = json.load(fin)
    random.shuffle(data)
    if args.dp_method != "none":
        data = data[:args.sample_size]
    n_train = int(len(data) * 0.8)

    train_dataset = []
    test_dataset = []
    if args.dp_method == "paraphrase":
        para_tokenizer, para_model = load_para_model(args.para_model)
    elif args.dp_method == "custext":
        emb_tokenizer, emb_model = load_emb_model(args.emb_model)
    print("Constructing test data with dp")
    cnt = 0
    for sample in tqdm(data):
        cnt += 1
        query, target = sample[args.query_key], sample[args.attr_key]
        target = process_target(target)
        if target == -1:
            continue
        if args.dp_method == "paraphrase":
            query = paraphrase(para_tokenizer, para_model, query, args.para_model, args.epsilon, args)
        elif args.dp_method == "custext":
            query = custext_priv(emb_tokenizer, emb_model, query, args.emb_model, args.top_k, args.epsilon, args)
        new_sample = {"prompt": query, "label": target}
        if cnt <= n_train:
            train_dataset.append(new_sample)
        else:
            test_dataset.append(new_sample)
    
    # prepare dataloader
    train_dataset = AttackDataset(train_dataset, tokenizer, args.max_word)
    test_dataset = AttackDataset(test_dataset, tokenizer, args.max_word)
    train_loader = DataLoader(
            train_dataset, 
            batch_size=args.train_batch_size, 
            collate_fn=default_data_collator, 
            pin_memory=True,
            )
    test_loader = DataLoader(
            test_dataset, 
            batch_size=args.test_batch_size, 
            collate_fn=default_data_collator, 
            pin_memory=True,
            )
    
    # prepare optimizer and scheduler
    optimizer = optim.AdamW(
                    model.parameters(),
                    lr=args.lr,
                    weight_decay=0.0,
                )
    num_training_steps = args.epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
        )
    
    # train the model
    best_acc = 0
    for epoch in range(args.epochs):
        model.train()
        loss_list = []

        with tqdm(total=len(train_loader)) as pbar:
            for step, batch in enumerate(train_loader):
                for key in batch.keys():
                    batch[key] = batch[key].to(model.device)
                # print(batch)
                output = model(input_ids = batch["input_ids"], 
                            attention_mask = batch["attention_mask"],
                            labels = batch["labels"]) 
                loss = output.loss
                loss_list.append(loss.item())
                loss_avg = sum(loss_list)/len(loss_list)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                pbar.update(1)
                pbar.set_postfix(loss=loss_avg)
                # break
            print(f'[epoch: {epoch}] Loss: {np.mean(np.array(loss_list))}')
        
        labels = []
        predictions = []
        model.eval()
        with tqdm(total=len(test_loader)) as pbar:
            for i, batch in enumerate(test_loader):
                for key in batch:
                    batch[key] = batch[key].to(model.device)
                with torch.no_grad():
                    outputs = model(
                            input_ids = batch["input_ids"], 
                            attention_mask = batch["attention_mask"])
                logits = outputs.logits
                y_pred = torch.argmax(logits, -1)
                predictions += y_pred.tolist()
                labels += batch["labels"].tolist()
                acc = accuracy_score(labels, predictions)
                auc = roc_auc_score(labels, predictions)
                recall = recall_score(labels, predictions)
                precision = precision_score(labels, predictions)
                f1 = f1_score(labels, predictions)
                pbar.update(1)
                pbar.set_postfix(acc=acc, auc=auc, precision=precision, recall=recall, f1=f1)
        print(f"Accuracy for epoch {epoch}: {acc}")
        print(f"AUC for epoch {epoch}: {auc}")

        logging.info(
                f"Epoch {epoch+1}/{args.epochs} - "
                f"Accuracy {acc} - "
                f"AUC {auc} - "
                f"Precision {precision} - "
                f"Recall {recall} - "
                f"F1 {f1}"
            )