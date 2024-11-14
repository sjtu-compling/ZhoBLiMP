import copy
import random


class Generator:

    def __init__(self, vocab, grammar, replace=None, config=None):
        """
        replace: the difference between good and bad rules

            if the contrast is made by changing the order of rules,
            then ``replace`` is a dict such as {1: 2, 2: 1}, 
            denoting the 2nd rule in bad sentence is the 1st rule in good sentence

            if the contrast is made by changing the content of rules,
            then ``replace`` is a dict such as {1: [rule1, rule2]},
            denoting the 1st rule in bad sentence can be rule1 or rule2
        """
        self.vocab = vocab
        self.grammar = grammar
        self.replace = replace
        
        if config is None:
            self.config = {"sep": "", "strict_MP": False}
        else:
            self.config = config
    
    def get_rules(self):
        rules = copy.deepcopy(self.grammar.sample())
        return set_match_values(rules, self.vocab)

    def gen_word(self, rule):
        if rule.type == "Phrase":
            samples = [self.gen_sent(rule, to_str=True)]
        elif rule.type == "Direct":
            if not rule.wordlist:
                samples = [""]  # empty string
            else:
                samples = rule.wordlist
        else:
            samples = self.vocab.filter(rule, return_list=True)
        check_empty_list(rule, samples)
        return random.choice(samples)

    def gen_sent(self, rules=None, to_str=False):
        if rules is None:
            rules = self.get_rules()
        samples = [self.gen_word(rule) for rule in rules]
        check_repeat(samples, rules)
        sent = join(samples, self.config["sep"])
        return sent if to_str else samples
    
    def gen_pair(self):
        """
        generate a pair of good and bad sentences
            1. generate good sentence given the sampled rules
            2. copy the good rules and words
            3. modify the bad rules and words based on the ``replace``
        """
        rules_good = self.get_rules()
        rules_bad = copy.deepcopy(rules_good)
        
        words_good = self.gen_sent(rules_good)
        words_bad = copy.deepcopy(words_good)
        
        for i in self.replace:
            
            # the contrast is made by changing the order of rules
            if isinstance(self.replace[i], int):
                words_bad[i] = words_good[self.replace[i]]
                continue

            tmp_rule = copy.deepcopy(random.choice(self.replace[i]))
            # to accommodate the case where the rule is a list of posssible phrases
            # now the critical region can be occupied by a phrase item
            if isinstance(tmp_rule, list) and not hasattr(tmp_rule, "type"):
                tmp_rule = random.choice(tmp_rule)
            rules_bad[i] = tmp_rule
        rules_bad = set_match_values(rules_bad, self.vocab)
        
        for i in self.replace:
            rule_diff = rules_bad[i]
            words_bad[i] = self.gen_word(rule_diff)

        sentence_good = join(words_good, self.config["sep"])
        sentence_bad = join(words_bad, self.config["sep"])
        check_diff_word(sentence_good, sentence_bad, **self.config)

        return sentence_good, sentence_bad


def join(words, sep):
    """convert samples to full sentence"""
    return sep.join(words)


def set_match_values(rules, vocab):
    """set possible values for matched properties"""
    # rules = copy(rules)
    for i, rule in enumerate(rules):
        if rule.type == "Phrase":
            rules[i] = set_match_values(rule, vocab)
        elif rule.type == "Matched" and not rule.matched:
            rules = match_helper(i, rules, vocab)
    return rules


def match_helper(i, rules, vocab):
    """
    a helper function to set possible values for matched properties
        
        i: the index of the current rule
        rules: the list of rules that the current rule belongs to
        vocab: the vocabulary object

        return: the updated rules
    """
    rule = rules[i]
    if rule.mpos is not None:
        pros = rule.mpro if isinstance(rule.mpro, list) else [rule.mpro]
        for pro in pros:
            if pro != "expression":
                values = set(vocab.filter(rule).get_values(pro))
                values_ref = set(vocab.filter(rules[rule.mpos]).get_values(pro))
                shared_values = list(values.intersection(values_ref))
                if not shared_values:
                    rule_ref = rules[rule.mpos]
                    raise AssertionError(f"no shared value of ``{pro}`` between ``{rule}`` and ``{rule_ref}``")
                value = random.choice(list(set.intersection(values, values_ref)))
                rules[rule.mpos].update({pro: value})
            else:
                if isinstance(rules[rule.mpos], dict):
                    value = random.choice(vocab.filter(rules[rule.mpos]).get_values(pro))
                    rules[rule.mpos].update({pro: value})
                else:
                    value = random.choice(rules[rule.mpos].wordlist)
            rules[i].update({pro: value})
            rules[i].matched = True
    if rule.mmpos is not None:
        pros = rule.mmpro if isinstance(rule.mmpro, list) else [rule.mmpro]
        for pro in pros:
            ref_rule = copy.deepcopy(rules[rule.mmpos])
            if pro in ref_rule:
                value = ref_rule.pop(pro)
                values = vocab.filter(ref_rule).get_values(pro)
            else:
                values = vocab.filter(rules[rule.mmpos]).get_values(pro)
                value = random.choice(values)
            mm_values = list(set(values) - set([value]))
            if not mm_values:
                raise AssertionError(f"no mismatched value of ``{pro}`` between ``{rule}`` and ``{ref_rule}``")
            mm_value = random.choice(mm_values)
            rules[rule.mmpos].update({pro: value})
            rules[i].update({pro: mm_value})
            rules[i].matched = True
    return rules


### functions to check sentences or pairs generated
def check_empty_list(rule, expression_list):
    if len(expression_list) <= 0:
        raise AssertionError(f"no sample found: ``{rule}``")


def check_repeat(words, rules):

    def allow_repeat(rule):
        if rule.type == "Matched": return "expression" in rule.mpro
        if rule.type == "Lexical": return "expression" in rule
        if rule.type == "Direct": return True
        
    types = set()
    for i, word in enumerate(words):
        if word in types and not allow_repeat(rules[i]):
            raise AssertionError(f"word repeated error: ``{words}``")
        types.add(word)


def check_diff_word(sent_good, sent_bad, sep, strict_MP):
    if sent_good == sent_bad:
        raise AssertionError(f"same word error: ``{sent_good}`` and ``{sent_bad}``")

    get_len = lambda x: len(x) if sep == "" else lambda x: len(x.split(sep))

    if get_len(sent_good) != len(sent_bad) and strict_MP:
        raise AssertionError(f"length difference error: ``{sent_good}`` and ``{sent_bad}``")
