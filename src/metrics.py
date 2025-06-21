import torch


def get_length(logprobs):
    length_matrix = torch.where(logprobs != 0, 1, 0)
    return length_matrix.sum(dim=1)

def lp(**kwargs):
    lm_logprobs = kwargs["lm_logprobs"]
    return lm_logprobs.sum(dim=1)

def mean_lp(**kwargs):
    lm_logprobs = kwargs["lm_logprobs"]
    length = get_length(lm_logprobs)
    return lm_logprobs.sum(dim=1) / length

def pow_norm_lp(alpha=None, **kwargs):
    # Sublinear length normalized log-probabilities (SLLN-LP)
    lm_logprobs = kwargs["lm_logprobs"]
    length = get_length(lm_logprobs)
    if alpha is None:
        alpha = 0.5
    return lm_logprobs.sum(dim=1) / torch.pow(length, alpha)

def log_norm_lp(beta=None, **kwargs):
    lm_logprobs = kwargs["lm_logprobs"]
    length = get_length(lm_logprobs)
    if beta is None:
        beta = 1
    return lm_logprobs.sum(dim=1) / torch.log(length + beta)

def slor(**kwargs):
    lm_logprobs = kwargs["lm_logprobs"]
    uni_logprobs = kwargs["uni_logprobs"]
    length = get_length(lm_logprobs)
    return (lm_logprobs - uni_logprobs).sum(dim=1) / length

def morcela(beta=None, gamma=None, **kwargs):
    lm_logprobs = kwargs["lm_logprobs"]
    uni_logprobs = kwargs["uni_logprobs"]
    if beta is None:
        beta = 1
    if gamma is None:
        gamma = 0
    length = get_length(lm_logprobs)
    return (lm_logprobs.sum(dim=1) - beta * uni_logprobs.sum(dim=1) + gamma) / length

