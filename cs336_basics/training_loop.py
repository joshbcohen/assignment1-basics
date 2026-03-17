import numpy.typing as npt
import numpy as np
import random
import torch
from jaxtyping import Int


def data_loading(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[Int[torch.Tensor, "batch_size context_length"]]:
    max_idx = min(len(dataset) - context_length - 1, batch_size * context_length)
    starts = [random.randint(0, max_idx) for _ in range(batch_size)]
    o1 = torch.tensor(np.array([dataset[s : s + context_length] for s in starts]))
    o2 = torch.tensor(np.array([dataset[s + 1 : s + 1 + context_length] for s in starts]))
    o1.to(device=device)
    o2.to(device=device)
    return (o1, o2)


if __name__ == "__main__":
    x = np.arange(0, 100)
    num_iters = 3
    for _ in range(num_iters):
        a, b = data_loading(
            dataset=x,
            batch_size=32,
            context_length=7,
            device="cpu",
        )
        print(a)
        print(b)
