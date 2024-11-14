import copy
import random
from typing import List
from itertools import product

from data_gen.api.rule import Phrase


class Grammar(List):

    def __init__(self, rules):
        rules = [list(i) for i in product(*rules)]
        super().__init__(rules)

    def sample(self):
        rules = copy.deepcopy(random.choice(self))
        return random_choice_rules(rules)
    

def random_choice_rules(rules):
    """randomly choose one rule from the list recursively
    in case there is a nested phrase
    """
    rules = copy.deepcopy(rules)
    for i, rule in enumerate(rules):
        if isinstance(rule, Phrase):
            rules[i] = random_choice_rules(rule)
        elif isinstance(rule, list):
            rules[i] = random_choice_rules(random.choice(rule))
        else:
            rules[i] = rule
    return rules