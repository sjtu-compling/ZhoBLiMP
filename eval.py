import os
import json
import time
import copy
import torch
import requests
import jsonlines
import pandas as pd

from tqdm import tqdm
from argparse import ArgumentParser
from collections import OrderedDict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", "-D", type=str, required=True)
    parser.add_argument("--model", "-M", type=str, required=True)
    parser.add_argument("--output_dir", "-O", type=str, default="results")
    parser.add_argument("--batch_size", "-B", type=int, default=400)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--unigram", type=str, default=None)
    parser.add_argument("--detail", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    print("Start evaluating %s" % args.model)

    if args.device:
        device = args.device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    except:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float16

    if str(device) == "cuda":
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch_dtype, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype, trust_remote_code=True).to(device)
    
    data, meta_info, average_metrics = eval(model, tokenizer, args.data_dir, args.batch_size, args.unigram)

    print(average_metrics)

    data_version = args.data_dir.replace("/", "-")
    output_dir = os.path.join(args.output_dir, data_version)

    model_name = os.path.basename(args.model)

    summary_dir = os.path.join(output_dir, "summary")
    all_outputs_dir = os.path.join(output_dir, "all_outputs")

    os.makedirs(summary_dir, exist_ok=True)
    meta_info.to_csv(os.path.join(summary_dir, model_name + ".csv"), index=False)

    if args.detail:
        os.makedirs(all_outputs_dir, exist_ok=True)
        data.to_csv(os.path.join(all_outputs_dir, model_name + ".csv"), index=False)


def load_data(data_dir, fields=None):
    
    if fields is None:
        fields = ["uid", "sentence_good", "sentence_bad"]
    
    def _load(fname):
        if ".csv" in fname:
            df = pd.read_csv(os.path.join(data_dir, fname))
            df = df[df.success]
            df["uid"] = fname.replace(".csv", "")
        elif ".jsonl" in fname:
            df = pd.read_json(os.path.join(data_dir, fname), lines=True)
            df["uid"] = df["UID"]
        else:
            print(fname)
            raise NotImplementedError
        df = df.loc[:,fields]
        example = df.sample(n=1, random_state=42)
        return df, example
    
    tmp = [_load(f) for f in sorted(os.listdir(data_dir))]
    data = pd.concat([d[0] for d in tmp])
    examples = pd.concat([d[1] for d in tmp])

    meta_info = pd.DataFrame()
    meta_info["uid"] = data.uid.value_counts().index
    meta_info["n_sample"] = data.uid.value_counts().values
    meta_info = meta_info.merge(examples, on="uid")
    return data, meta_info


def get_sentence_metrics(sents, model, tokenizer, unigram_prob_file=None):

    def preprocess_unigram(unigram_prob_file):
        unigram_prob = json.load(open(unigram_prob_file))
        unigram_prob = {int(k): v for k, v in unigram_prob.items()}
        return torch.log(torch.tensor(
            [
                unigram_prob[i] if i in unigram_prob else 1 / 3e9
                for i in range(max(unigram_prob.keys()) + 1)
            ]
        ))

    # if tokenizer.bos_token is None:
    #     sents = [tokenizer.pad_token + s for s in sents]

    inputs = tokenizer(sents, return_tensors="pt", padding=True).to(model.device)
    try:
        assert all(token_id in tokenizer.all_special_ids for token_id in inputs.input_ids[:,0])
    except AssertionError:
        sents = [tokenizer.pad_token + s for s in sents]
        inputs = tokenizer(sents, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logprobs = outputs.logits[:,:-1].log_softmax(dim=-1)

    labels = inputs.input_ids[:,1:]
    mask = inputs.attention_mask[:,1:]
    logprobs = logprobs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    logprobs = torch.sum(logprobs * mask, dim=1)
    mean_logprobs = logprobs / mask.sum(dim=-1)
    perplexity = torch.exp(-mean_logprobs)

    results = { "LP": logprobs.tolist(), "meanLP": mean_logprobs.tolist(), "PPL": perplexity.tolist() }

    slor = None
    if not unigram_prob_file is None:
        uni_logprob_dict = preprocess_unigram(unigram_prob_file).to(model.device)
        uni_logprobs = torch.sum(uni_logprob_dict[labels] * mask, dim=1)
        slor = (mean_logprobs - uni_logprobs) / mask.sum(dim=-1)

    if slor is not None:
        results.update({ "SLOR": slor.tolist() })
    return results


def eval_batch(sent_good, sent_bad, model, tokenizer, unigram_prob_file=None):
    results_good = get_sentence_metrics(sent_good, model, tokenizer, unigram_prob_file)
    results_bad = get_sentence_metrics(sent_bad, model, tokenizer, unigram_prob_file)
    return results_good, results_bad


def eval(model, tokenizer, data_dir, batch_size, unigram_prob_file=None):
    tokenizer = copy.deepcopy(tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    data, meta_info = load_data(data_dir, fields=["uid", "phenomenon", "sentence_good", "sentence_bad"])
    # data, meta_info = load_data(data_dir, fields=["uid", "sentence_good", "sentence_bad"])
    good_sent_list = data.sentence_good.tolist()
    bad_sent_list = data.sentence_bad.tolist()
    metrics = ["LP", "meanLP", "PPL"]
    if unigram_prob_file is not None:
        metrics.append("SLOR")
    results = {
        metric: { "good": [], "bad": [] } for metric in metrics
    }
    for i in tqdm(range(0, len(good_sent_list), batch_size)):
        good_results, bad_results = eval_batch(
            good_sent_list[i:i+batch_size], 
            bad_sent_list[i:i+batch_size],
            model,
            tokenizer,
            unigram_prob_file
        )
        for metric in metrics:
            results[metric]["good"].extend(good_results[metric])
            results[metric]["bad"].extend(bad_results[metric])
    
    average_metrics = {}
    ### compute acc by uid ###
    for metric in metrics:
        data[f"good_{metric}"] = results[metric]["good"]
        data[f"bad_{metric}"] = results[metric]["bad"]
        if metric == "PPL":
            average_metrics[f"eval/blimp_{metric}"] = data[f"good_{metric}"].mean()
            mean_ppl = data.groupby("uid")[f"good_{metric}"].mean()
            meta_info = meta_info.merge(mean_ppl, on="uid")
        else:
            data[f"{metric}_correct"] = data[f"good_{metric}"] > data[f"bad_{metric}"]
            average_metrics[f"eval/blimp_{metric}_acc"] = data[f"{metric}_correct"].mean()
            acc = data.groupby("uid")[f"{metric}_correct"].mean()
            meta_info = meta_info.merge(acc, on="uid")

    meta_info.sort_values("uid", ascending=True)
    return data, meta_info, average_metrics


if __name__ == "__main__":
    main()
