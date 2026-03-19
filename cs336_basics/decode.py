import torch
from einops import rearrange
from transformer import softmax


def decode(
    prompt: list[int], model, vocab, temp: float = 1, max_tokens: int = 256, p: float | None = None
) -> list[int]:
    prompt_tensor = torch.tensor(prompt)
    token = ""
    output = []
    count = 0

    while token != vocab["<|end_of_text|>"] or count == max_tokens:
        # run transformer on prompt which will give us, 1, seq_len, vocab
        output_tensor = model(prompt_tensor)
        output_tensor = rearrange(output_tensor, "1 seq_len vocab -> seq_len vocab")
        last_layer = output_tensor[-1]

        # get softmax for next token
        if temp == 0:
            temp = 1e-10
        softmaxxed = softmax(last_layer / temp, dim=-1)

        if p is not None:
            summed = 0
            sorted = torch.sort(softmaxxed, desc=True)
            idx = 0
            while summed < p:
                summed += sorted[idx]
                idx += 1
            cutoff = sorted[idx]
            softmaxxed *= softmaxxed >= cutoff
            softmaxxed /= torch.sum(softmaxxed)

        token = torch.multinomial(softmaxxed, num_samples=1).item()

        output.append(token)
        prompt_tensor = torch.append(prompt_tensor, token)
        count += 1

    return output
