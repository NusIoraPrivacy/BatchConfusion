from datasets import Dataset
import torch

class AttackDataset(Dataset):
    def __init__(self, inputs, tokenizer, max_words=100, pad=True):
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
        examples = []
        labels = []
        example_masks = []
        for i in index:
            sample = self.inputs[i]
            sentence = sample["prompt"]
            label = sample["label"]
            
            # create input ids
            input_id = torch.tensor(
                self.tokenizer.encode(sentence), dtype=torch.int64
            )
            input_id = self.pad_token(input_id)
            
            att_mask = input_id.ge(0)
            input_id[~att_mask] = self.tokenizer.pad_token_id
            att_mask = att_mask.float()

            examples.append(input_id)
            labels.append(label)
            example_masks.append(att_mask)

        return {
            "input_ids": examples,
            "labels": labels,
            "attention_mask": example_masks,
        }