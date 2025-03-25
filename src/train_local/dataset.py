from datasets import Dataset
import torch
import copy
from configs.templates import local_compress_template

class CompressDataset(Dataset):
    def __init__(self, inputs, tokenizer, max_words=512, pad=True):
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.pad = pad
        self.inputs = inputs
    
    def __len__(self):
        return len(self.inputs)
    
    def pad_token(self, input_id):
        if self.pad:
            padding = self.max_words - input_id.shape[0]
            if padding > 0:
                input_id = torch.cat((input_id, torch.zeros(padding, dtype=torch.int64) - 1))
            elif padding < 0:
                input_id = input_id[: self.max_words]
        return input_id
    
    def __getitem__(self, index):
        IGNORE_INDEX = -100
        queries = []
        examples = []
        labels = []
        example_masks = []
        query_masks = []
        for i in index:
            sample = self.inputs[i]
            question = sample["question"]
            compression = sample["compression"]
            query = local_compress_template + f" {question} "
            # obtain the length of prefix id
            prefix_id = torch.tensor(
                self.tokenizer.encode(query), dtype=torch.int64
            )
            # print(prefix_id)
            # create input ids
            text = query + compression + self.tokenizer.eos_token
            # print(text)
            input_id = torch.tensor(
                self.tokenizer.encode(text), dtype=torch.int64
            )
            # print(input_id)
            if self.pad:
                input_id = self.pad_token(input_id)
            # create target ids
            label_id = copy.deepcopy(input_id)
            label_id[:(len(prefix_id)-1)] = -1
            label_mask = label_id.ge(0)
            label_id[~label_mask] = IGNORE_INDEX
            # print(label_id)

            if self.pad:
                prefix_id = self.pad_token(prefix_id)

            att_mask = input_id.ge(0)
            input_id[~att_mask] = self.tokenizer.pad_token_id
            att_mask = att_mask.float()
            query_att_mask = prefix_id.ge(0)
            prefix_id[~query_att_mask] = self.tokenizer.pad_token_id
            query_att_mask = query_att_mask.float()

            examples.append(input_id)
            labels.append(label_id)
            example_masks.append(att_mask)
            queries.append(prefix_id)
            query_masks.append(query_att_mask)

        return {
            "query_ids": queries,
            "input_ids": examples,
            "labels": labels,
            "attention_mask": example_masks,
            "query_attention_mask": query_masks,
        }


class CompressDatasetSeq2Seq(Dataset):
    def __init__(self, inputs, tokenizer, max_words=512, pad=True):
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.pad = pad
        self.inputs = inputs
    
    def __len__(self):
        return len(self.inputs)
    
    def pad_token(self, input_id):
        if self.pad:
            padding = self.max_words - input_id.shape[0]
            if padding > 0:
                input_id = torch.cat((input_id, torch.zeros(padding, dtype=torch.int64) - 1))
            elif padding < 0:
                input_id = input_id[: self.max_words]
        return input_id
    
    def __getitem__(self, index):
        IGNORE_INDEX = -100
        examples = []
        labels = []
        example_masks = []
        for i in index:
            sample = self.inputs[i]
            question = sample["question"]
            compression = sample["compression"]
            # obtain the length of prefix id
            input_id = torch.tensor(
                self.tokenizer.encode(question), dtype=torch.int64
            )
            # print(input_id)
            if self.pad:
                input_id = self.pad_token(input_id)
            att_mask = input_id.ge(0)
            input_id[~att_mask] = self.tokenizer.pad_token_id
            att_mask = att_mask.float()

            # create target ids
            label_id = torch.tensor(
                self.tokenizer.encode(compression), dtype=torch.int64
            )
            if self.pad:
                label_id = self.pad_token(label_id)
            label_mask = label_id.ge(0)
            label_id[~label_mask] = IGNORE_INDEX

            examples.append(input_id)
            labels.append(label_id)
            example_masks.append(att_mask)

        return {
            "input_ids": examples,
            "labels": labels,
            "attention_mask": example_masks,
        }