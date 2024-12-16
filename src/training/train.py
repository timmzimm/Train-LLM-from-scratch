import torch
import torch.optim as optim
from tqdm import tqdm
from src.utils.evaluation import evaluate

def train(model, train_loader, val_loader, device, epochs=1, lr=3e-4):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)
        for i, (x, y) in enumerate(pbar):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()

            if i % 100 == 0 and len(val_loader) > 0:
                val_loss = evaluate(model, val_loader, device)
                pbar.set_postfix({"train_loss": loss.item(), "val_loss": val_loss})
