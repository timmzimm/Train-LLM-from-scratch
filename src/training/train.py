import torch
import torch.optim as optim
from tqdm import tqdm
from src.utils.evaluation import evaluate
import math

def train(model, train_loader, val_loader, device, train_config):
    """
    Training loop for GPT-2 with linear warmup + cosine decay.
    
    Args:
        model: GPT-2 model
        train_loader: DataLoader for training
        val_loader: DataLoader for validation (can be empty)
        device: 'cuda' or 'cpu'
        train_config: dictionary with training parameters, e.g.:
            {
              "warmup_epochs": 1,
              "cosine_epochs": 5,
              "learning_rate": 0.0003,
              "eval_every_steps": 1000,
              "block_size": 1024,
              "batch_size": 2,
              ...
            }
    """
    warmup_epochs = train_config["warmup_epochs"]
    cosine_epochs = train_config["cosine_epochs"]
    lr = train_config["learning_rate"]
    eval_every_steps = train_config["eval_every_steps"]

    epochs = warmup_epochs + cosine_epochs
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    def get_lr_for_epoch(epoch):
        """
        Compute learning rate for the given epoch 
        based on linear warmup and cosine decay.
        """
        if epoch < warmup_epochs:
            # Linear warmup from 0 to lr
            return lr * float(epoch + 1) / float(warmup_epochs)
        else:
            # Cosine decay from lr down to 0
            progress = float(epoch - warmup_epochs) / float(cosine_epochs)
            return lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    global_step = 0

    for epoch in range(epochs):
        model.train()
        current_lr = get_lr_for_epoch(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} (lr={current_lr:.6f})", leave=True)
        for i, (x, y) in enumerate(pbar):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()

            global_step += 1

            # Evaluate only every 'eval_every_steps' steps
            if len(val_loader) > 0 and global_step % eval_every_steps == 0:
                val_loss = evaluate(model, val_loader, device)
                pbar.set_postfix({"train_loss": loss.item(), "val_loss": val_loss})

        # Optionally evaluate at the end of each epoch
        if len(val_loader) > 0:
            val_loss = evaluate(model, val_loader, device)
            print(f"End of epoch {epoch}, validation loss: {val_loss}")




