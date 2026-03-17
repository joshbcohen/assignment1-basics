import argparse
from transformer import TransformerLM
from support import cross_entropy
from optimizer import AdamW, get_learning_rate_schedule, apply_gradient_clipping
from training_loop import data_loading, save_checkpoint, load_checkpoint

def main():

    parser = argparse.ArgumentParser(description="Train transformer model.")

    parser.add_argument("--device", type=str, default="cpu", help='Device: "cpu", "mps" (Mac), or "cuda"')

    # data loading
    parser.add_argument("dataset", type=str, help="Provide training dataset.")
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
    parser.add_argument("--d_ff", type=int, default=1344, help="Should usually be a multiple of 64, and around 3/8 * d_model.")
    parser.add_argument("--theta", type=int, default=1000, help="Theta value for RoPE.")

    # gradient_clipping
    parser.add_argument("--max_l2_norm", type=int, default=10000, help="Size of the token dataset.")
    parser.add_argument("--eps", type=int, default=512, help="Dimensions of model.")

    # learning rate schedulers
    parser.add_argument("--a_max", type=int, default=10000, help="Size of the token dataset.")
    parser.add_argument("--a_min", type=int, default=512, help="Dimensions of model.")
    parser.add_argument("--T_w", type=int, default=4, help="Number of layers for Transformer.")
    parser.add_argument("--T_c", type=int, default=16, help="Number of heads for Transformer.")

    args = parser.parse_args()


    training_loader = data_loading(
        dataset=args.dataset,
        batch_size=args.batch_size,
        context_length=args.context_length,
        device=args.device,
    )

    optimizer = AdamW(
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    model = TransformerLM(
        vocab_size=args.vocab_size, 
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta,
    )

    loss_fn = cross_entropy

    return training_loader, optimizer, model, loss_fn, args



def train_one_epoch(epoch_index, tb_writer, training_loader, optimizer, model, loss_fn, args):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, targets = data

        # Zero your gradients for every batch!
        optimizer.zero_grad() 

        # Make predictions for this batch
        logits = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(logits, targets)
        loss.backward()
        
        
        apply_gradient_clipping(model.parameters(max), max_l2_norm=args.max_l2_norm, eps=args.eps)

        get_learning_rate_schedule(t=i, a_max=args.a_max, a_min=args.a_min, T_w=args.T_w, T_c=args.T_c)
        # Adjust learning weights
        optimizer.step()
        

        # Save Checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            iteration=i,
            #todo: out=??? 
        )

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss



if __name__ == "__main__":
    training_loader, optimizer, model, loss_fn = main()
    train_one_epoch(epoch_index, tb_writer, training_loader, optimizer, model, loss_fn)
