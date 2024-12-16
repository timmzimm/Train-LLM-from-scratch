import torch
from tqdm import tqdm
import numpy as np

def evaluate(model, dataloader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Evaluating", leave=False):
            x = x.to(device)
            y = y.to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses))
