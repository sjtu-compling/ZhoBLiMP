import json
from typing import List, Dict


class Rule:
    
    def __init__(self, phrase_file=None):
        self.phrases = {}
        if phrase_file:
            self.phrases = load_phrase(phrase_file)

    def make_rule(self, entry):
        return to_rule(entry, self.phrases)


class Lexical(Dict):

    def __init__(self, rule):
        super().__init__(rule)
        self._type = "Lexical"

    def __repr__(self):
        return super().__repr__()
    
    @property
    def type(self):
        return self._type
    

class Direct:
    
    def __init__(self, rule):
        self.wordlist = rule if isinstance(rule, list) else [rule]
        self._type = "Direct"
    
    def __repr__(self):
        return self.wordlist.__repr__()

    @property
    def type(self):
        return self._type
    

class Matched(Dict):

    def __init__(self, rule):
        super().__init__(rule)
        self._type = "Matched"
        self.matched = False

        self.mpos = eval(self.pop("matchedPosition", "None"))
        self.mpro = self.pop("matchedProperties", None)
        self.mmpos = eval(self.pop("mismatchedPosition", "None"))
        self.mmpro = self.pop("mismatchedProperties", None)
    
    def __repr__(self):
        return super().__repr__()
    
    @property
    def type(self):
        return self._type


class Phrase(List):

    def __init__(self, phrases, rules, additional_properties={}):
        rules = [to_rule(r, phrases) for r in rules]
        for idx, properties_dict in additional_properties.items():
            rules[int(idx)].update(properties_dict)
        
        super().__init__(rules)
        self._type = "Phrase"

    def __repr__(self):
        return super().__repr__()
    
    @property
    def type(self):
        return self._type
        

def load_phrase(phrase_file):
    with open(phrase_file, "r", encoding='utf-8') as f:
        return json.loads(f.read())


def to_phrase(name, phrases):
    if not phrases:
        raise ValueError("No phrases are loaded.")
    for phrase_name in phrases:
        if phrase_name == name:
            return phrases[name]
    raise ValueError(f"Phrase {name} not found.")


def to_rule(entry, phrases=None):
    assert "type" in entry and "rule" in entry, "Invalid entry."
    
    type = entry["type"]
    rule = entry["rule"]
    
    if type == "Lexical":
        return Lexical(rule)
    elif type == "Direct":
        return Direct(rule)
    elif type == "Matched":
        return Matched(rule)
    elif type == "Phrase":
        rules_list = to_phrase(rule["name"], phrases)
        add_props = rule.pop("additional_properties", {})
        return [Phrase(phrases, rules, add_props) for rules in rules_list]
    else:
        print(f"Invalid type {type}")
        raise NotImplementedError

