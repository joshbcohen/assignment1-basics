# Quick overfit test — paste into a python script or run interactively
import numpy as np
from cs336_basics.transformer import TransformerLM
from cs336_basics.support import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_basics.training_loop import data_loading

dataset = np.load("data/TinyStoriesV2-GPT4-train.npy", mmap_mode="r")
device = "cuda"

model = TransformerLM(vocab_size=10000, context_length=256, d_model=512,
                    num_layers=4, num_heads=16, d_ff=1344, theta=10000, device=device)

optimizer = AdamW(params=model.parameters(), lr=1e-3)

# Get ONE batch and reuse it
inputs, targets = data_loading(dataset, batch_size=128, context_length=256, device=device)

for step in range(200):
    optimizer.zero_grad()
    logits = model(inputs)
    B, S, V = logits.shape
    loss = cross_entropy(logits.reshape(B*S, V), targets.reshape(B*S))
    loss.backward()
    optimizer.step()
    if step % 10 == 0:
        print(f"step {step}: loss={loss.item():.4f}")
