import torch
from einops import rearrange
from transformer import softmax
from tokenizer import Tokenizer
from training_loop import load_checkpoint
from transformer import TransformerLM
from optimizer import AdamW


def decode(
    prompt: str, tokenizer, model, vocab, temp: float = 1, max_tokens: int = 256, p: float | None = None
) -> list[int]:
    
    int_prompt = tokenizer.encode(prompt)

    prompt_tensor = torch.tensor(int_prompt).unsqueeze(0)
    token = ""
    output = []
    count = 0
    reverse_vocab = {v: k for k, v in vocab.items()}

    while token != reverse_vocab[b'<|endoftext|>'] or count == max_tokens:
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
            sorted = torch.sort(softmaxxed, descending=True).values
            idx = 0
            while summed < p:
                summed += sorted[idx]
                idx += 1
            cutoff = sorted[idx]
            softmaxxed *= softmaxxed >= cutoff
            softmaxxed /= torch.sum(softmaxxed)

        token = torch.multinomial(softmaxxed, num_samples=1).item()

        output.append(token)
        int_prompt.append(token)
        prompt_tensor = torch.tensor(int_prompt).unsqueeze(0)
        count += 1

    tokenized_output = tokenizer.decode(output)


    return tokenized_output

if __name__ == "__main__":
    vocab_path = "tinystories_10000_vocab.txt"
    merges_path = "tinystories_10000_merges.txt"
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>"])
    model = TransformerLM(vocab_size=10000, context_length=256, d_model=512, num_layers=4, num_heads=16, d_ff=1344, theta=1000)
    optimizer = AdamW(params = model.parameters(), lr=1e-4, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    load_checkpoint("checkpoints/run-f9cda5bf4-checkpoint-999.pt", model=model, optimizer=optimizer)
    prompt = "There once was a cat that loved cookies. "


    output = decode(prompt, tokenizer, model, tokenizer.vocab, temp=0.8, max_tokens=512, p=0.9)
    print(output)


