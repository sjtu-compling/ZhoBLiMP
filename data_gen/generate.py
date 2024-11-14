import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data_gen.utils import *
from data_gen.api import Grammar, Rule, Generator, Vocabulary


def generate_pair(
        vocab_path_or_str, 
        rule_good, 
        rule_bad, 
        num_target=10, 
        threshold=2, 
        debug=False, 
        phrase_file=None, 
        sep="", 
        strict_MP=False
    ):
    """
    generate sentences
    """
    if os.path.isfile(vocab_path_or_str):
        vocab = Vocabulary.from_string(open(vocab_path_or_str, encoding='utf-8').read())
    else:
        vocab = Vocabulary.from_string(vocab_path_or_str)

    # load a phrase file and store the phrases in the instance as ``rule_maker``
    rule_maker = Rule(phrase_file)
    rules_good = string_to_rules(rule_good, rule_maker)
    # print(rule_good)
    replace_config = get_replace_config(rule_good, rule_bad, rule_maker)

    # only good grammar is needed, bad grammar will be modified according to the ``replace_config``
    grammar = Grammar(rules_good)
    config = {"sep": sep, "strict_MP": strict_MP}
    gen = Generator(vocab, grammar, replace_config, config=config)
    sents_good, sents_bad = [], []
    
    # generate sentences until the target number is reached
    # or the total number of generated sentences exceeds the threshold
    # for sometimes the generator may get stuck in a loop if the vocabulary size is too small
    total_gen_cnt = 0
    while len(sents_good) < num_target and total_gen_cnt < num_target * threshold:
        try:
            sent_good, sent_bad = gen.gen_pair()
            # print(sent_good, sent_bad)
            if sent_good in sents_good:
                raise AssertionError(f"{sent_good} already exists!")
            sents_good.append(sent_good)
            sents_bad.append(sent_bad)
        except AssertionError as e:
            if debug:
                print(e)
        total_gen_cnt += 1
        

    return { "good_sentence": sents_good, "bad_sentence": sents_bad }


def generate_sent(vocab, rule, n=10, threshold=2):
    """generate sentences according to the given rule"""
    pass


def get_replace_config(str_good, str_bad, rule_builder):
    if str_good == str_bad:
        raise AssertionError("The two rule strings are the same.")

    # if set(str_good) == set(str_bad):
    #     return {
    #         i: str_good.index(str_bad[i])
    #         for i in range(len(str_good))
    #         if str_good[i] != str_bad[i]
    #     }
    
    # else:
    #     return {
    #         i: [rule_builder.make_rule(r) for r in parse_rule(str_bad[i])] 
    #         for i in range(len(str_good))
    #         if str_good[i] != str_bad[i]
    #     }
    
    config = {}
    for i in range(len(str_good)):
        if str_good[i] != str_bad[i]:
            if str_bad[i] in str_good:
                config[i] = str_good.index(str_bad[i])
            else:
                config[i] = [rule_builder.make_rule(r) for r in parse_rule(str_bad[i])]
    return config


def string_to_rules(str_list, rule_builder):
    rules = [
        [rule_builder.make_rule(r) for r in parse_rule(rule)] 
        for rule in str_list
    ]
    return rules

