import torch
from einops import rearrange


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    # logits will have dimensions batch_size, vocab_len
    # targets will have dimensions batch_size; each element is the token id = col of

    max_el = torch.amax(logits, dim=-1, keepdim=True)
    logits = logits - max_el

    log_sum_exp = torch.log(torch.sum(torch.exp(logits), dim=-1))

    index = rearrange(targets, "batch -> batch 1")
    index = index.to(logits.device)
    target_logits = torch.gather(logits, dim=-1, index=index)
    target_logits = rearrange(target_logits, "batch 1 -> batch")

    loss = torch.mean(log_sum_exp - target_logits)
    return loss
