import os
import argparse
import pandas as pd
from copy import copy
from functools import partial

import metrics
from dataset import FCDataset
from inference import get_decoder_logprobs


def main(args):
    revisions = [args.model_name_or_path]
    if args.revision == "ALL":
        revisions.extend([f"{args.model_name_or_path}/{fn}" for fn in os.listdir(args.model_name_or_path) 
            if os.path.isdir(os.path.join(args.model_name_or_path, fn))])
    # elif args.revision == "main":
    #     revisions = [args.model_name_or_path]

    data_name = os.path.basename(args.data_dir)
    model_name = os.path.basename(args.model_name_or_path)
    output_dir = f"{args.output_dir}/{data_name}/{model_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # metrics_map = { "slln_lp": partial(metrics.pow_norm_lp, alpha=0.5) }

    metrics_map = {
        "lp": metrics.lp,
        "mean_lp": metrics.mean_lp,
    }

    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        metrics_map[f"pow{alpha}_norm_lp"] = partial(metrics.pow_norm_lp, alpha=alpha)

    # for beta in [1, 10, 100]:
    #     metrics_map[f"log{beta}_norm_lp"] = partial(metrics.log_norm_lp, beta=beta)

    if args.unigram_prob_file:
        metrics_map["slor"] = metrics.slor
        for beta in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            for gamma in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
                metrics_map[f"morcela_{beta}_{gamma}"] = partial(metrics.morcela, beta=beta, gamma=gamma)

    fc_dataset = FCDataset(
        data_dir=args.data_dir,
        tokenizer_path=revisions[0],
        unigram_prob_file=args.unigram_prob_file if args.unigram_prob_file else None,
        max_length=args.max_length
    )

    for revision in revisions:
        df = fc_dataset.df.copy()
        results = get_decoder_logprobs(
            model_name_or_path=revision,
            tokenized_datasets=fc_dataset.tokenized_datasets,
            batch_size=args.batch_size,
            device=args.device,
            unigram_prob_dict=fc_dataset.unigram_prob_dict
        )

        for metric_name, metric_func in metrics_map.items():
            for sent_type in ["good", "bad"]:
                kwargs = { k.replace(f"_{sent_type}", ""): v for k, v in results.items() if sent_type in k }
                acceptability = metric_func(**kwargs)
                df[f"{metric_name}_{sent_type}"] = acceptability.numpy()
            correct = df[f"{metric_name}_good"] > df[f"{metric_name}_bad"]
            print(f"{metric_name}: {correct.mean():.4f}")

        revision = os.path.basename(revision)
        df.to_json(f"{output_dir}/{revision}.jsonl", orient="records", lines=True, force_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--unigram_prob_file", type=str, default="")
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--batch_size", type=int, default=300)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()
    main(args)
