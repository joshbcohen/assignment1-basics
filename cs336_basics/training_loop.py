import random
import os
import typing

import numpy.typing as npt
import numpy as np
import torch
from jaxtyping import Int


def data_loading(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[Int[torch.Tensor, "batch_size context_length"]]:
    max_idx = len(dataset) - context_length - 1
    starts = [random.randint(0, max_idx) for _ in range(batch_size)]
    o1 = torch.tensor(np.array([dataset[s : s + context_length] for s in starts])).to(device=device)
    o2 = torch.tensor(np.array([dataset[s + 1 : s + 1 + context_length] for s in starts])).to(device=device)
    return (o1, o2)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]


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
