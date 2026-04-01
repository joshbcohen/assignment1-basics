import torch
from einops import rearrange
from transformer import softmax
from tokenizer import Tokenizer
from training_loop import load_checkpoint
from transformer import TransformerLM
from optimizer import AdamW


def decode(
    prompt: str, tokenizer, model, vocab, temp: float = 1, max_tokens: int = 256, p: float | None = None, device: str = "cpu"
) -> list[int]:
    int_prompt = tokenizer.encode(prompt)

    prompt_tensor = torch.tensor(int_prompt, device=device).unsqueeze(0)
    token = ""
    output = []
    count = 0
    reverse_vocab = {v: k for k, v in vocab.items()}

    while token != reverse_vocab.get(b"<|endoftext|>") and count < max_tokens:
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
        prompt_tensor = torch.tensor(int_prompt, device=device).unsqueeze(0)
        count += 1

    tokenized_output = tokenizer.decode(output)

    return tokenized_output


if __name__ == "__main__":
    vocab_path = "32000_owt_vocab.txt"# "tinystories_10000_vocab.txt"
    merges_path = "32000_owt_merges.txt" # "tinystories_10000_merges.txt"
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>"])
    device = "mps"
    temp = 0.6
    p = 0.8
    model = TransformerLM(
        vocab_size=32000, context_length=256, d_model=512, num_layers=4, num_heads=16, d_ff=1344, theta=10000, device=device
    )
    optimizer = AdamW(params=model.parameters(), lr=1e-4, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    checkpoint = torch.load ("checkpoints/owt-run-9999.pt", map_location=device) # torch.load("checkpoints/run-2352bf0f7-checkpoint-9999.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    prompt = """#Recipe for Mac and Cheese\n\n"""
    output = decode(prompt, tokenizer, model, tokenizer.vocab, temp=temp, max_tokens=512, p=p, device=device)
    
    print(f"\ntemp: {temp}, p: {p}")
    print(f"PROMPT: {prompt}")
    print(f"\nOur Amazing Model Output:\n {output} \n")