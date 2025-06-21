import torch
from transformers import AutoModelForCausalLM

from tqdm import tqdm

@torch.no_grad()
def inference_decoder(model, unigram_prob_dict=None, **kwargs):
    
    input_ids = kwargs.pop("input_ids")
    attention_mask = kwargs.pop("attention_mask")

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logprobs = outputs.logits[:,:-1].log_softmax(dim=-1)
    labels = input_ids[:,1:]
    attention_mask = attention_mask[:,1:].bool()

    outputs = {}
    if unigram_prob_dict is not None:
        uni_logprobs = unigram_prob_dict[labels]
        uni_logprobs = uni_logprobs.masked_fill(~attention_mask, 0)
        outputs["uni_logprobs"] = uni_logprobs.cpu()

    lm_logprobs = logprobs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    lm_logprobs = lm_logprobs.masked_fill(~attention_mask, 0)
    outputs["lm_logprobs"] = lm_logprobs.cpu()
    return outputs


def get_decoder_logprobs(
    model_name_or_path, tokenized_datasets, batch_size=300, device="cuda", unigram_prob_dict=None):

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
    results = { "lm_logprobs_good": [], "lm_logprobs_bad": [] }
    
    if unigram_prob_dict is not None:
        unigram_prob_dict = unigram_prob_dict.to(model.device)
        results["uni_logprobs_good"] = []
        results["uni_logprobs_bad"] = []
    
    for sent_type, tokenized in tokenized_datasets.items():
    
        for i in tqdm(
            range(0, len(tokenized), batch_size), 
            desc=f"Getting logprobs of {sent_type} sentences"
        ):
            batch = tokenized[i:i+batch_size]
            batch = { k: v.to(model.device) for k, v in batch.items() }
            outputs = inference_decoder(model, **batch, unigram_prob_dict=unigram_prob_dict)
        
            for k, v in outputs.items():
                results[f"{k}_{sent_type}"].append(v)

    return { k: torch.concat(v) for k, v in results.items() }