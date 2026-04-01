"""
Ablation runs for CS336 Assignment 1.
Run with: uv run ablations_script.py
"""

import math
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
from einops import einsum
from tqdm import tqdm
import wandb

from cs336_basics.transformer import TransformerLM, Transformer, MultiheadSelfAttention
from cs336_basics.support import cross_entropy
from cs336_basics.optimizer import AdamW, get_learning_rate_schedule, apply_gradient_clipping
from cs336_basics.training_loop import data_loading, save_checkpoint


def evaluate_val_loss(val_dataset, model, loss_fn, args, num_batches=20, val_batch_size=64):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(num_batches):
            inputs, targets = data_loading(
                dataset=val_dataset,
                batch_size=val_batch_size,
                context_length=args.context_length,
                device=args.device,
            )
            logits = model(inputs)
            B, S, V = logits.shape
            loss = loss_fn(logits.reshape(B * S, V), targets.reshape(B * S))
            total_loss += loss.item()
    model.train()
    return total_loss / num_batches


# ---------------------------------------------------------------------------
# Variant Transformer blocks
# ---------------------------------------------------------------------------


class TransformerNoNorm(Transformer):
    """Transformer block with all RMSNorms removed."""

    def forward(self, x):
        seq_len = x.shape[-2]
        token_positions = torch.arange(seq_len, device=x.device)
        x = x + self.mha(x, token_positions=token_positions)
        x = x + self.swiglu(x)
        return x


class TransformerPostNorm(Transformer):
    """Transformer block with post-norm instead of pre-norm."""

    def forward(self, x):
        seq_len = x.shape[-2]
        token_positions = torch.arange(seq_len, device=x.device)
        x = self.rms_norm_attention(x + self.mha(x, token_positions=token_positions))
        x = self.rms_norm_swiglu(x + self.swiglu(x))
        return x


class TransformerNoRoPE(Transformer):
    """Transformer block with RoPE disabled."""

    def __init__(self, d_model, num_heads, d_ff, theta=None, max_seq_len=None, device=None, dtype=None):
        super().__init__(d_model, num_heads, d_ff, theta, max_seq_len, device, dtype)
        # Replace MHA with one that has apply_rope=False
        self.mha = MultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            apply_rope=False,
            device=device,
        )


class SiLU_FFN(torch.nn.Module):
    """FFN with SiLU activation (no gating). d_ff = 4 * d_model to match SwiGLU param count."""

    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        std = (2 / (d_ff + d_model)) ** 0.5
        w1 = torch.empty(d_ff, d_model, device=device, dtype=dtype)
        w2 = torch.empty(d_model, d_ff, device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(w1, mean=0, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(w2, mean=0, std=std, a=-3 * std, b=3 * std)
        self.W1 = torch.nn.Parameter(w1)
        self.W2 = torch.nn.Parameter(w2)

    def forward(self, x):
        from einops import einsum

        h = einsum(self.W1, x, "d_ff d_model, batch seq_len d_model -> batch seq_len d_ff")
        h = h * torch.sigmoid(h)  # SiLU
        return einsum(self.W2, h, "d_model d_ff, batch seq_len d_ff -> batch seq_len d_model")


class TransformerSiLU(Transformer):
    """Transformer block using SiLU FFN instead of SwiGLU."""

    def __init__(self, d_model, num_heads, d_ff, theta=None, max_seq_len=None, device=None, dtype=None):
        super().__init__(d_model, num_heads, d_ff, theta, max_seq_len, device, dtype)
        self.swiglu = SiLU_FFN(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Variant TransformerLM that accepts a custom block class + d_ff override
# ---------------------------------------------------------------------------


class TransformerLMVariant(TransformerLM):
    """TransformerLM that uses a custom Transformer block class."""

    def __init__(
        self,
        block_cls,
        skip_final_norm=False,
        d_ff_override=None,
        **kwargs,
    ):
        # Temporarily set d_ff to the override for super().__init__
        if d_ff_override is not None:
            kwargs["d_ff"] = d_ff_override
        super().__init__(**kwargs)

        d_ff = d_ff_override or kwargs["d_ff"]

        # Replace transformer blocks with the variant
        self.transformers = torch.nn.ModuleList(
            [
                block_cls(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    d_ff=d_ff,
                    theta=self.theta,
                    max_seq_len=self.context_length,
                    device=self.device,
                    dtype=self.dtype,
                )
                for _ in range(self.num_layers)
            ]
        )

        self._skip_final_norm = skip_final_norm

    def forward(self, x):
        embeddings = self.token_embedding(token_ids=x)
        for t in self.transformers:
            embeddings = t(embeddings)
        if not self._skip_final_norm:
            embeddings = self.rms_norm(x=embeddings)
        return self.linear(x=embeddings)


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

BASE_ARGS = dict(
    dataset="data/TinyStoriesV2-GPT4-train.npy",
    val_dataset="data/TinyStoriesV2-GPT4-valid.npy",
    device="cuda",
    batch_size=128,
    context_length=256,
    vocab_size=10000,
    d_model=512,
    num_layers=4,
    num_heads=16,
    d_ff=1344,
    theta=10000,
    a_max=0.003,
    a_min=0.00001,
    T_w=200,
    T_c=10000,
    iterations=7500,
    val_every=100,
    save_every=2000,
    max_l2_norm=1.0,
    eps=1e-6,
    weight_decay=0.01,
    lr=1e-4,
)


def run_ablation(name, model, args):
    dataset = np.load(args["dataset"], mmap_mode="r")
    val_dataset = np.load(args["val_dataset"], mmap_mode="r")
    loss_fn = cross_entropy

    optimizer = AdamW(params=model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])
    wandb_run = wandb.init(project="mini-llm", name=name, config=args, reinit=True)
    wandb_run.watch(model)

    start_time = time.time()

    for iteration in tqdm(range(args["iterations"]), desc=name):
        inputs, targets = data_loading(
            dataset=dataset,
            batch_size=args["batch_size"],
            context_length=args["context_length"],
            device=args["device"],
        )
        optimizer.zero_grad()
        logits = model(inputs)
        B, S, V = logits.shape
        loss = loss_fn(logits.reshape(B * S, V), targets.reshape(B * S))
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        apply_gradient_clipping(model.parameters(), max_l2_norm=args["max_l2_norm"], eps=args["eps"])

        new_lr = get_learning_rate_schedule(
            t=iteration, a_max=args["a_max"], a_min=args["a_min"], T_w=args["T_w"], T_c=args["T_c"]
        )
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr
        optimizer.step()

        wallclock = time.time() - start_time
        train_loss = loss.item()
        wandb_run.log(
            {"train/loss": train_loss, "train/lr": new_lr, "train/grad_norm": grad_norm.item(), "wallclock_time": wallclock},
            step=iteration,
        )

        if (iteration + 1) % args["val_every"] == 0:
            # Use a simple namespace so evaluate_val_loss works
            class Args:
                pass
            eval_args = Args()
            eval_args.batch_size = args["batch_size"]
            eval_args.context_length = args["context_length"]
            eval_args.device = args["device"]
            val_loss = evaluate_val_loss(val_dataset, model, loss_fn, eval_args)
            val_ppl = math.exp(val_loss)
            wandb_run.log({"val/loss": val_loss, "val/perplexity": val_ppl, "wallclock_time": wallclock}, step=iteration)
            tqdm.write(f"[{name}] step {iteration+1} | train={train_loss:.4f} | val={val_loss:.4f} | ppl={val_ppl:.2f}")

        if (iteration + 1) % args["save_every"] == 0:
            os.makedirs("checkpoints", exist_ok=True)
            save_checkpoint(model, optimizer, iteration, f"checkpoints/{name}-{iteration}.pt")

    wandb_run.finish()


# ---------------------------------------------------------------------------
# Define and run all ablations
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model_kwargs = dict(
        vocab_size=BASE_ARGS["vocab_size"],
        context_length=BASE_ARGS["context_length"],
        d_model=BASE_ARGS["d_model"],
        num_layers=BASE_ARGS["num_layers"],
        num_heads=BASE_ARGS["num_heads"],
        d_ff=BASE_ARGS["d_ff"],
        theta=BASE_ARGS["theta"],
        device=BASE_ARGS["device"],
    )

    ablations = [
        # 1) No RMSNorm, same LR
        (
            "ablation-no-rmsnorm-lr0.003",
            lambda: TransformerLMVariant(block_cls=TransformerNoNorm, skip_final_norm=True, **model_kwargs),
            {**BASE_ARGS, "a_max": 0.003},
        ),
        # 2) No RMSNorm, lower LR
        (
            "ablation-no-rmsnorm-lr0.0003",
            lambda: TransformerLMVariant(block_cls=TransformerNoNorm, skip_final_norm=True, **model_kwargs),
            {**BASE_ARGS, "a_max": 0.0003, "a_min": 0.00001},
        ),
        # 3) Post-norm
        (
            "ablation-post-norm",
            lambda: TransformerLMVariant(block_cls=TransformerPostNorm, **model_kwargs),
            {**BASE_ARGS},
        ),
        # 4) No position embeddings (NoPE)
        (
            "ablation-nope",
            lambda: TransformerLMVariant(block_cls=TransformerNoRoPE, **model_kwargs),
            {**BASE_ARGS},
        ),
        # 5) SiLU FFN instead of SwiGLU (d_ff = 4 * d_model = 2048)
        (
            "ablation-silu",
            lambda: TransformerLMVariant(block_cls=TransformerSiLU, d_ff_override=4 * 512, **model_kwargs),
            {**BASE_ARGS},
        ),
    ]

    for name, make_model, args in ablations:
        print(f"\n{'='*60}")
        print(f"Starting: {name}")
        print(f"{'='*60}\n")
        model = make_model()
        run_ablation(name, model, args)
        del model
        torch.cuda.empty_cache()
