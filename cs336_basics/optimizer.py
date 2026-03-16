from typing import Optional, Callable, Iterable
import math

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        """
        params: parameters to be optimized
        lr: alpha, learning rate
        b1: parameter that controls first moment estimate
        b2: parameter that controls second moment estimate
        eps: epsilon value for numerical stability when applying gradients
        decay: learning rate decay value
        """
        if lr < 0:
            raise ValueError(f"Invalid learning rate value: {lr}")
        if any(b < 0 for b in betas):
            raise ValueError(f"Invalid moment beta values: {betas}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid learning rate decay value: {weight_decay}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Get state associated with parameter
                state = self.state[p]
                # Get iteration number from state, or initial value of 0
                t = state.get("t", 1)

                # Get first and second moment vectors from state, or initialize to 0
                m = state.get("m", torch.zeros(p.data.shape))
                v = state.get("v", torch.zeros(p.data.shape))

                # Get gradient of loss with respect to p at current time step
                grad = p.grad.data

                # Calculate new moment estimates
                m = b1 * m + (1 - b1) * grad
                v = b2 * v + (1 - b2) * grad**2

                # Calculate adjusted learning rate for iteration t
                lr_t = lr * math.sqrt(1 - math.pow(b2, t)) / (1 - math.pow(b1, t))

                # Adjust parameters based on adjusted learning rate and moment calculation
                p.data -= lr_t * m / ((v ** (1 / 2)) + eps)
                p.data -= lr * weight_decay * p.data

                # Update values for time step and moment vectors
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss


def get_learning_rate_schedule(t: int, a_max: float, a_min: float, T_w: int, T_c: int) -> float:
    """
    Apply learning rate scheduling with cosine annealing
    t: current step
    a_max: maximum learning rate
    a_min: minimum learning rate
    T_w: number of warm-up iterations
    T_c: number of cosine annealing iterations
    returns: final learning rate to use for gradient update at step t
    """
    if t < T_w:
        return (t / T_w) * a_max
    elif T_w <= t <= T_c:
        return a_min + 0.5 * (1 + math.cos((t - T_w) / (T_c - T_w) * math.pi)) * (a_max - a_min)
    else:
        return a_min


def apply_gradient_clipping(params: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: int = 1e-6):
    grads = torch.flatten(torch.stack([param.grad for param in params if param.grad is not None]))
    l2_norm = torch.linalg.norm(grads, ord=2)
    scaling_factor = max_l2_norm / (l2_norm + eps)
    if l2_norm >= max_l2_norm:
        for param in params:
            if param.grad is None:
                continue
            param.grad *= scaling_factor
