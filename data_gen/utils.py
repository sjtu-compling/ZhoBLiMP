import re
from collections import OrderedDict


def str_to_dict(kv_str):
    output_dict = {}
    for kv in kv_str.split():
        k, v = kv.split(":")
        if "/" in v:
            v = v.split("/")
        output_dict[k] = v
    return output_dict


def phrase_str_to_entry(phrase_str):
    tmp = OrderedDict()
    output_dict = {}
    
    additional_properties = re.findall(r"<([^>]+)>", phrase_str)
    for i, additional_property in enumerate(additional_properties):
        phrase_str = phrase_str.replace(f"<{additional_property}>", f"additional_property_{i}")
        additional_properties[i] = str_to_dict(additional_property)
    
    tmp.update(str_to_dict(phrase_str))
    output_dict["name"] = tmp.pop("phrase")
    output_dict["additional_properties"] = {}
    
    for i, k in enumerate(list(tmp.keys())):
        output_dict["additional_properties"][k] = additional_properties[i]
    return output_dict


def parse_rule(rule_str):
    str_dict_list = []
    rule_str = rule_str.replace("mmPos", "mismatchedPosition").replace("mPos", "matchedPosition").\
        replace("mmPro", "mismatchedProperties").replace("mPro", "matchedProperties")
    rule_str = rule_str.strip()
    for rule_choice in rule_str.split('|'):
        if not re.match(r"(\w+):(\w+)+", rule_choice.strip()):
            type_ = "Direct"
            rule = [i.strip() for i in rule_choice.split()]
        elif "phrase" in rule_choice:
            type_ = "Phrase"
            rule = phrase_str_to_entry(rule_choice.strip())
        else:
            type_ = "Lexical"
            rule = str_to_dict(rule_choice.strip())
            if "matchedPosition" in rule or "mismatchedPosition" in rule:
                type_ = "Matched"
        str_dict_list.append({"type": type_, "rule": rule})
    return str_dict_list
