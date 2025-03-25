from transformers import default_data_collator, get_scheduler
from torch.utils.data import DataLoader
import torch.optim as optim
import torch

from train_local.dataset import CompressDataset
from train_local.model_util import get_model_tokenizer
from configs.templates import local_compress_template

from tqdm import tqdm
import random
import json
import os
import argparse
current = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(current))

models = ["Qwen/Qwen2.5-1.5B-Instruct", "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.1-8B", "princeton-nlp/gemma-2-9b-it-SimPO"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default=root_path)
    parser.add_argument("--temperature", type=float, default=1,
        help = "temperature for text generation")
    parser.add_argument("--max_tokens", type=int, default=1000,
        help = "max new token for text generation")
    parser.add_argument("--token_len", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--test_batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gpt_model", type=str, default="gpt-4o")
    parser.add_argument("--data_name", type=str, default="medical_o1_reasoning_SFT")
    parser.add_argument("--model_name", type=str, default=models[0])
    parser.add_argument("--test_only", type=bool, default=False)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    model, tokenizer = get_model_tokenizer(args.model_name, args)
    with open(f'{args.root_path}/data/{args.data_name}/compress_gpt.json') as fin:
        data = json.load(fin)
    random.shuffle(data)
    n_train = int(len(data) * 0.8)
    train_data = data[:n_train]
    test_data = data[n_train:]
    query2dict = {}
    for sample in test_data:
        question = sample["question"]
        question = tokenizer.decode(tokenizer.encode(question), skip_special_tokens=True)
        question = question.strip().strip("\n").strip()
        query2dict[question] = sample

    train_dataset = CompressDataset(train_data, tokenizer, args.token_len)
    test_dataset = CompressDataset(test_data, tokenizer, args.token_len)
    train_dataloader = DataLoader(
                train_dataset, 
                batch_size=args.train_batch_size, 
                collate_fn=default_data_collator, 
                pin_memory=True,
                )
    test_dataloader = DataLoader(
                test_dataset, 
                batch_size=args.test_batch_size, 
                collate_fn=default_data_collator, 
                pin_memory=True,
                )
    optimizer = optim.AdamW(
                    model.parameters(),
                    lr=args.lr,
                    weight_decay=0.0,
                )
    num_training_steps = args.epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
        )
    
    model_name = (args.model_name).split("/")[-1]
    output_dir = f'{args.root_path}/result/{args.data_name}/compression/{model_name}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.test_only:
        args.epochs = 1

    for epoch in range(args.epochs):
        model.train()
        loss_list = []
        if not args.test_only:
            with tqdm(total=len(train_dataloader), desc=f'Epoch {epoch + 1}/{args.epochs}', unit='batch') as pbar:
                for step, batch in enumerate(train_dataloader):
                    for key in batch.keys():
                        batch[key] = batch[key].to(model.device)

                    output = model(input_ids = batch["input_ids"], 
                                attention_mask = batch["attention_mask"],
                                labels = batch["labels"]) 
                    # print(batch["labels"][batch["labels"]>0])
                    loss = output.loss
                    loss_list.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    pbar.update(1)
                    avg_loss = sum(loss_list)/len(loss_list)
                    pbar.set_postfix(loss=avg_loss)
                    # break

        output_data = []
        with tqdm(total=len(test_dataloader)) as pbar:
            for step, batch in enumerate(test_dataloader):
                for key in batch.keys():
                    batch[key] = batch[key].to(model.device)
                
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids = batch["query_ids"], 
                        attention_mask = batch["query_attention_mask"], 
                        max_new_tokens=512, 
                        # repetition_penalty=0.1,
                        )
                
                queries = tokenizer.batch_decode(batch["query_ids"], skip_special_tokens=True)
                raw_y_preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                
                for query, pred in zip(queries, raw_y_preds):
                    question = query.replace(local_compress_template, "")
                    question = question.strip().strip("\n").strip()
                    try:
                        sample = query2dict[question]
                    except Exception as e:
                        continue
                    pred = pred.replace(query, "")
                    # print(pred)
                    # print("\n")
                    sample["predicted compression"] = pred
                    output_data.append(sample)
                pbar.update(1)

                if step % 10 == 0:
                    with open(f'{output_dir}/epoch_{epoch+1}.json', 'w') as fout:
                        json.dump(output_data, fout, indent=4)
                # break
        
        with open(f'{output_dir}/epoch_{epoch+1}.json', 'w') as fout:
            json.dump(output_data, fout, indent=4)