import os
import json
import pandas as pd
from tqdm import tqdm

import torch
import datasets
from transformers import AutoTokenizer


class FCDataset:
    
    def __init__(self, data_dir, tokenizer_path, unigram_prob_file=None, max_length=64):
        self.df = pd.concat(load_file(f"{data_dir}/{fn}") for fn in os.listdir(data_dir))
        self.tokenizer = load_tokenizer(tokenizer_path)
        self.unigram_prob_file = unigram_prob_file
        self.tokenized_datasets = self.tokenize(max_length)
        self.df["length"] = self.get_length_info()

    def tokenize(self, max_length):
        tokenized_datasets = preprocess_dataset(self.df, self.tokenizer, max_length)
        tokenized_datasets.set_format("torch")
        return tokenized_datasets

    def get_length_info(self):
        good_sent_len = self.tokenized_datasets["good"]["attention_mask"].sum(dim=1)
        bad_sent_len = self.tokenized_datasets["bad"]["attention_mask"].sum(dim=1)
        
        length_info = []
        for good_len, bad_len in zip(good_sent_len, bad_sent_len):
            if good_len == bad_len:
                length_info.append("equal length")
            elif good_len > bad_len:
                length_info.append("good is longer")
            else:
                length_info.append("bad is longer")
        return length_info

    @property
    def unigram_prob_dict(self):
        if self.unigram_prob_file is None:
            return None
        vocab_size = len(self.tokenizer)
        return preprocess_unigram(self.unigram_prob_file, vocab_size)


def load_file(fn):
    if fn.endswith(".csv"):
        df = pd.read_csv(fn)
    elif fn.endswith(".jsonl"):
        df = pd.read_json(fn, lines=True)
    else:
        raise ValueError("Invalid file format")
    return df

def whether_add_bos_token(tokenizer):
    return not tokenizer("Hello world").input_ids[0] == tokenizer.bos_token_id

def preprocess_dataset(df, tokenizer, max_length=64):
    # TODO: add field of "sentence_type"
    add_bos_token = whether_add_bos_token(tokenizer)    
    def tokenize_func(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=max_length
        )

    tokenized_datasets = { "good": None, "bad": None }

    for sent_type in ["good", "bad"]:
        sentences = df[f"sentence_{sent_type}"].tolist()
        if add_bos_token:
            sentences = [tokenizer.bos_token + s for s in sentences]

        dataset = datasets.Dataset.from_dict({"text": sentences})
        tokenized = dataset.map(
            tokenize_func, 
            remove_columns=["text"], 
            load_from_cache_file=False, 
            batched=True
        )
        tokenized = tokenized.select_columns(["input_ids", "attention_mask"])
        tokenized_datasets[sent_type] = tokenized

    return datasets.DatasetDict(tokenized_datasets)

def load_tokenizer(tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.bos_token:
        tokenizer.bos_token = tokenizer.eos_token
        tokenizer.bos_token_id = tokenizer.eos_token_id
    tokenizer.add_bos_token = False
    return tokenizer

def preprocess_unigram(unigram_prob_file, vocab_size, eps=None):
    if eps is None:
        eps = 1 / 3e+9
    unigram_prob = json.load(open(unigram_prob_file))
    unigram_prob = {int(k): v for k, v in unigram_prob.items()}
    return torch.log(torch.tensor(
        [
            unigram_prob[i] if i in unigram_prob else eps
            for i in range(vocab_size)
        ]
    ))