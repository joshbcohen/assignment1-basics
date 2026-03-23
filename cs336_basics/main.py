
import argparse
import os
from uuid import uuid4
import logging
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from transformer import TransformerLM
from support import cross_entropy
from optimizer import AdamW, get_learning_rate_schedule, apply_gradient_clipping
from training_loop import data_loading, save_checkpoint, load_checkpoint
import wandb

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train transformer model.")

    parser.add_argument("--device", type=str, default="cpu", help='Device: "cpu", "mps" (Mac), or "cuda"')

    # data loading
    parser.add_argument("dataset", type=str, help="Path to training dataset. Should be pre-tokenized")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--context_length", type=int, default=256, help="Context length of LM.")

    # optimizer
    parser.add_argument("--lr", type=int, default=64, help="Learning rate")
    parser.add_argument("--weight_decay", type=int, default=256, help="Weight decay of optimizer.")

    # transformer
    parser.add_argument("--vocab_size", type=int, default=10000, help="Size of the token dataset.")
    parser.add_argument("--d_model", type=int, default=512, help="Dimensions of model.")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers for Transformer.")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of heads for Transformer.")
    parser.add_argument(
        "--d_ff", type=int, default=1344, help="Should usually be a multiple of 64, and around 3/8 * d_model."
    )
    parser.add_argument("--theta", type=int, default=1000, help="Theta value for RoPE.")

    # gradient_clipping
    parser.add_argument("--max_l2_norm", type=int, default=10000, help="Size of the token dataset.")
    parser.add_argument("--eps", type=int, default=512, help="Dimensions of model.")

    # learning rate schedulers
    parser.add_argument("--a_max", type=int, default=10000, help="Size of the token dataset.")
    parser.add_argument("--a_min", type=int, default=512, help="Dimensions of model.")
    parser.add_argument("--T_w", type=int, default=4, help="Number of layers for Transformer.")
    parser.add_argument("--T_c", type=int, default=16, help="Number of heads for Transformer.")

    # training loop
    parser.add_argument("--epochs", type=int, default=10000, help="Number of epochs to train")

    # save checkpoint args
    checkpoint_uuid = uuid4().hex[:9]
    logger.info(f"Using {checkpoint_uuid} as unique prefix if no checkpoint prefix is provided")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="directory to save checkpoints in")
    parser.add_argument("--checkpoint-prefix", type=str, default=f"run-{checkpoint_uuid}-checkpoint-")
    parser.add_argument("--save-every", type=int, default=1000, help="Save a checkpoint every N steps")

    # load checkpoint args
    parser.add_argument("--load-checkpoint-from", type=str, help="If provided, load checkpoint from this filepath")

    args = parser.parse_args()

    dataset = np.load(args.dataset, mmap_mode="r")

    training_loader = data_loading(
        dataset=dataset,
        batch_size=args.batch_size,
        context_length=args.context_length,
        device=args.device,
    )

    print(len(training_loader))
    print(len(training_loader[0]))

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta,
        device=args.device,
    )

    optimizer = AdamW(
        params=model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    loss_fn = cross_entropy

    if args.load_checkpoint_from is not None:
        last_epoch_index = load_checkpoint(args.load_checkpoint_from, model, optimizer)
    else:
        last_epoch_index = 0

    return training_loader, optimizer, model, loss_fn, last_epoch_index, args


def train_one_epoch(epoch_index, tb_writer, training_loader, optimizer, model, loss_fn, wandb_run, args):
    running_loss = 0.0
    last_loss = 0.0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    #for i, data in enumerate(training_loader):
    # Every data instance is an input + label pair
    inputs, targets = training_loader

    # Zero your gradients for every batch!
    optimizer.zero_grad()

    # Make predictions for this batch
    logits = model(inputs)

    # Compute the loss and its gradients
    # support.cross_entropy expects logits of shape (batch, vocab) and targets of shape (batch,)
    # the model returns (batch, seq, vocab) and targets are (batch, seq) — flatten the seq dim
    B, S, V = logits.shape
    loss = loss_fn(logits.reshape(B * S, V), targets.reshape(B * S))
    loss.backward()
    apply_gradient_clipping(model.parameters(), max_l2_norm=args.max_l2_norm, eps=args.eps)

    get_learning_rate_schedule(t=epoch_index, a_max=args.a_max, a_min=args.a_min, T_w=args.T_w, T_c=args.T_c)
    # Adjust learning weights
    optimizer.step()

    # Gather data and report
    running_loss += loss.item()
    wandb_run.log({"loss": loss})
    # if i % 1000 == 999:
    #     last_loss = running_loss / 1000  # loss per batch
    #     print("  batch {} loss: {}".format(i + 1, last_loss))
    #     tb_x = epoch_index * len(training_loader) + i + 1
    #     tb_writer.add_scalar("Loss/train", last_loss, tb_x)
    #     running_loss = 0.0

    if (epoch_index + 1) % args.save_every == 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            iteration=epoch_index,
            out=f"{args.checkpoint_dir}/{args.checkpoint_prefix}{epoch_index}.pt",
        )

    return last_loss


if __name__ == "__main__":
    training_loader, optimizer, model, loss_fn, last_epoch_index, args = main()
    tb_writer = SummaryWriter()
    wandb_run = wandb.init(project="mini-llm")
    wandb_run.watch(model)

    for epoch_index in tqdm(range(last_epoch_index, args.epochs), desc="Epochs"):
        train_one_epoch(epoch_index, tb_writer, training_loader, optimizer, model, loss_fn, wandb_run, args)
