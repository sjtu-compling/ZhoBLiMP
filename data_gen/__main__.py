import os
import json
import random
import pandas as pd
from argparse import ArgumentParser

from data_gen import generate_pair

# TODO: main function of the module to be implemented

def main():
    parser = ArgumentParser()
    parser.add_argument("--vocab", "-V", type=str, default="assets/zh_vocab_saved.tsv")
    # parser.add_argument("--vocab", type=str, default="assets/zh_vocab")
    parser.add_argument("--phrase", type=str, default="assets/zh_phrase.json")
    parser.add_argument("--output_dir", "-O", type=str)
    parser.add_argument("--input_path", "-I", type=str)
    parser.add_argument("--num_target", "-N", type=int, default=200)
    parser.add_argument("--threshold", "-T", type=int, default=10)
    parser.add_argument("--debug", "-D", action="store_true")
    parser.add_argument("--seed", "-S", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)

    if os.path.isdir(args.input_path):
        paradigms = [load_paradigm(os.path.join(args.input_path, f)) for f in os.listdir(args.input_path)]
    else:
        paradigms = [load_paradigm(args.input_path)]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    not_enough = {}
    
    for paradigm in paradigms:
        print(f"\n\n**********Generating data for {paradigm['uid']}...")
        good_rule = paradigm["good_rule"]
        bad_rule = paradigm["bad_rule"]
        output_dir = paradigm.get("output_dir", args.output_dir)
        sep = paradigm.get("sep", "")
        strict_MP = paradigm.get("strict_MP", True)

        result = generate_pair(
            args.vocab, 
            good_rule, 
            bad_rule, 
            debug=args.debug,
            num_target=args.num_target, 
            threshold=args.threshold, 
            phrase_file=args.phrase, 
            sep=sep, 
            strict_MP=strict_MP
        )

        result["UID"] = [paradigm["uid"] for _ in range(len(result["good_sentence"]))]
        result["phenomenon"] = [paradigm["phenomenon"] for _ in range(len(result["good_sentence"]))]
        result["pairID"] = list(range(len(result["good_sentence"])))

        if len(result["good_sentence"]) < args.num_target:
            print(f"Warning: only {len(result['good_sentence'])} pairs generated for {paradigm['uid']}")
            not_enough[paradigm['uid']] = len(result["good_sentence"])
            continue
        
        df = pd.DataFrame(result, columns=["UID", "phenomenon", "good_sentence", "bad_sentence", "pairID"])
        df.rename(columns={"good_sentence": "sentence_good", "bad_sentence": "sentence_bad"}, inplace=True)
        # df.to_csv(os.path.join(output_dir, "result.csv"), index=False)
        df.to_json(os.path.join(output_dir, paradigm["uid"] + ".jsonl"), orient="records", force_ascii=False, lines=True)
    
    print("not enough: uid  n")
    for k, v in not_enough.items(): print(k, v)

def load_paradigm(filepath):
    print(f"Loading {filepath}")
    return json.loads(open(filepath, encoding='utf-8').read())


if __name__ == "__main__":
    main()
