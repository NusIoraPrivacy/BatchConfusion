from transformers import default_data_collator, get_scheduler
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

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

models = ["Qwen/Qwen2.5-1.5B-Instruct", "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.2-3B", "FacebookAI/roberta-large"]
data_names = ["legal-qa-v1", "medical_o1_reasoning_SFT", "mmlu_fina", "twitter"]
fake_keys = ['fake attributes question', 'fake attributes compression', "fake attributes text"]
priv_keys = ["filtered private attributes question", "filtered private attributes compression", "filtered private attributes", "filtered private attributes text"]
query_keys = ["question", "compression", "text"]

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
    parser.add_argument("--data_name", type=str, default="mmlu_fina")
    parser.add_argument("--in_file_name", type=str, default="fake_cattr_random_0.5.json")
    parser.add_argument("--fake_key", type=str, default=fake_keys[-1])
    parser.add_argument("--priv_key", type=str, default=priv_keys[-1])
    parser.add_argument("--query_key", type=str, default=query_keys[-1])
    parser.add_argument("--model_name", type=str, default=models[1])
    parser.add_argument("--train_pct", type=float, default=0.8)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--max_word", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--sample_top_k", type=int, default=1)
    parser.add_argument("--n_negative", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--save_model", action='store_true')
    parser.add_argument("--use_peft", action='store_true')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print(args)
    log_root = f"{args.root_path}/result/{args.data_name}/logs"
    if not os.path.exists(log_root):
        os.makedirs(log_root)
    file_name = (args.in_file_name).replace(".json", "")
    model_name = (args.model_name).split("/")[-1]
    file_name = f"pjd-{file_name}-{model_name}-top-{args.sample_top_k}.log"
    file_path = f"{log_root}/{file_name}"
    logging.basicConfig(
        filename=file_path,
        filemode="w",  # use 'a' to append
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )

    model, tokenizer = get_model_tokenizer(args.model_name, num_labels=2, args=args)
    data_path = f'{args.root_path}/result/{args.data_name}/{args.in_file_name}'
    with open(data_path) as fin:
        raw_data = json.load(fin)
    # create classification dataset 
    process_data = []
    for sample in raw_data:
        priv_attrs, fake_attrs, query = sample[args.priv_key], sample[args.fake_key], sample[args.query_key]
        if len(fake_attrs) > 0:
            process_data.append({"prompt": query, "label": 1})
            top_k = max(args.sample_top_k, len(fake_attrs)-1)
            if top_k > 0:
                sample_fake_attrs = fake_attrs[(top_k-1):(top_k-1+args.n_negative)]
            else:
                sample_fake_attrs = random.sample(fake_attrs, args.n_negative)
            # print(sample_fake_attrs)
            for fake_attr_list in sample_fake_attrs:
                this_query = "" + query
                for priv_attr, fake_attr in zip(priv_attrs, fake_attr_list):
                    this_query = this_query.replace(priv_attr, fake_attr)
                process_data.append({"prompt": this_query, "label": 0})

    random.shuffle(process_data)
    n_train = int(len(process_data) * args.train_pct)
    train_data = process_data[:n_train]
    test_data = process_data[n_train:]
    train_dataset = AttackDataset(train_data, tokenizer, args.max_word)
    test_dataset = AttackDataset(test_data, tokenizer, args.max_word)
    # obtain dataloader
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
    
    # train models
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
                # break
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
    
    if args.save_model:
        model_name = "-".join((args.model_name).split("/"))
        save_path = f"{args.root_path}/model/{args.data_name}/{model_name}"
        if not os.path.exists(log_root):
            os.makedirs(log_root)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)