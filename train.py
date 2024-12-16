import os
import math
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from typing import List, Dict, Tuple
import numpy as np
from collections import Counter
from tqdm import tqdm

###############################################################################
# Full BPE Tokenizer from Scratch (Approximating GPT-2 style tokenization)
###############################################################################
"""
This implementation is more "complete" in the sense that:
- We will learn BPE merges on the entire training corpus.
- We'll store merges in a stable way.
- The approach is still simplified compared to a production GPT-2 tokenizer,
  but closer to the spirit of a full BPE approach.

Steps:
1. Collect a large corpus of text (train_texts).
2. Start with a vocab of characters + special tokens.
3. Perform a number of merges (like merges_count ~ 10000 for GPT-2 small).
4. Use the final vocab to encode/decode.
5. For simplicity, we won't segment by words, we treat everything as characters plus merges.

Note: Real GPT-2 uses Byte-level BPE. Here, we assume just chars. To be closer
to GPT-2, we should consider text as bytes. But let's assume chars for demonstration.

We will:
- Treat input texts as sequences of Unicode chars. 
- Implement a BPE training loop that merges pairs.
"""

class BPEVocabulary:
    def __init__(self, special_tokens: List[str] = ["<|endoftext|>"]):
        self.special_tokens = special_tokens
        self.token2id = {}
        self.id2token = {}
        self.merges = []
        self.vocab_size = 0

    def build_initial_vocab(self, texts: List[str]):
        char_counter = Counter()
        for text in texts:
            char_counter.update(list(text))
        chars = sorted(char_counter.keys())
        idx = 0
        for st in self.special_tokens:
            self.token2id[st] = idx
            idx += 1
        for ch in chars:
            if ch not in self.token2id:
                self.token2id[ch] = idx
                idx += 1
        self.id2token = {v:k for k,v in self.token2id.items()}
        self.vocab_size = len(self.id2token)

    def add_merge(self, new_token, t1, t2):
        if new_token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[new_token] = idx
            self.id2token[idx] = new_token
            self.vocab_size += 1
        self.merges.append((t1,t2))

    def encode_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.token2id[t] for t in tokens if t in self.token2id]

    def decode_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.id2token[i] for i in ids]

class BPETokenizer:
    def __init__(self, special_tokens=["<|endoftext|>"], vocab_size_limit=30000, merges_count=10000):
        self.vocab = BPEVocabulary(special_tokens)
        self.vocab_size_limit = vocab_size_limit
        self.merges_count = merges_count

    def _get_token_sequences(self, texts: List[str]) -> List[List[str]]:
        sequences = []
        for t in texts:
            seq = list(t)
            seq.append("<|endoftext|>")
            sequences.append(seq)
        return sequences

    def train(self, texts: List[str]):
        print("Training BPE tokenizer on the corpus...")
        self.vocab.build_initial_vocab(texts)
        sequences = self._get_token_sequences(texts)

        for i in tqdm(range(self.merges_count), desc="BPE merges"):
            pair_counts = Counter()
            for seq in sequences:
                for j in range(len(seq)-1):
                    pair = (seq[j], seq[j+1])
                    pair_counts[pair] += 1
            if not pair_counts:
                break
            best_pair, best_count = pair_counts.most_common(1)[0]
            new_token = best_pair[0] + best_pair[1]
            self.vocab.add_merge(new_token, best_pair[0], best_pair[1])

            # replace all occurrences
            new_sequences = []
            for seq in sequences:
                j = 0
                new_seq = []
                while j < len(seq):
                    if j < len(seq)-1 and (seq[j], seq[j+1]) == best_pair:
                        new_seq.append(new_token)
                        j += 2
                    else:
                        new_seq.append(seq[j])
                        j += 1
                new_sequences.append(new_seq)
            sequences = new_sequences

            if self.vocab.vocab_size >= self.vocab_size_limit:
                print("Reached vocab size limit.")
                break

        print("BPE training complete. Final vocab size:", self.vocab.vocab_size)

    def encode(self, text: str) -> List[int]:
        chars = list(text) + ["<|endoftext|>"]
        merges_set = set(self.vocab.merges)

        # repeatedly attempt merges until stable
        while True:
            merged = False
            new_seq = []
            j = 0
            while j < len(chars):
                if j < len(chars)-1 and (chars[j], chars[j+1]) in merges_set:
                    new_token = chars[j] + chars[j+1]
                    new_seq.append(new_token)
                    j += 2
                    merged = True
                else:
                    new_seq.append(chars[j])
                    j += 1
            chars = new_seq
            if not merged:
                break

        return self.vocab.encode_tokens_to_ids(chars)

    def decode(self, ids: List[int]) -> str:
        tokens = self.vocab.decode_ids_to_tokens(ids)
        if tokens and tokens[-1] == "<|endoftext|>":
            tokens = tokens[:-1]
        return "".join(tokens)


###############################################################################
# GPT-2 Model Architecture (GPT-2 small exact config)
###############################################################################
"""
GPT-2 small:
- n_ctx=1024
- n_embd=768
- n_layer=12
- n_head=12
We'll replicate these exactly.
"""

class GPT2Config:
    def __init__(self,
                 vocab_size: int,
                 n_ctx: int = 1024,
                 n_embd: int = 768,
                 n_layer: int = 12,
                 n_head: int = 12,
                 embedding_dropout: float = 0.1,
                 resid_dropout: float = 0.1,
                 attn_dropout: float = 0.1):
        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.embedding_dropout = embedding_dropout
        self.resid_dropout = resid_dropout
        self.attn_dropout = attn_dropout

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.d_head = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer("bias", torch.tril(torch.ones(config.n_ctx, config.n_ctx)).view(1,1,config.n_ctx,config.n_ctx))
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.resid_dropout = nn.Dropout(config.resid_dropout)

    def forward(self, x):
        B,T,C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B,T,self.n_head,self.d_head).transpose(1,2)
        k = k.view(B,T,self.n_head,self.d_head).transpose(1,2)
        v = v.view(B,T,self.n_head,self.d_head).transpose(1,2)

        att = (q @ k.transpose(-2,-1)) / math.sqrt(self.d_head)
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.resid_dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2Model(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_ctx, config.n_embd)
        self.drop = nn.Dropout(config.embedding_dropout)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B,T = idx.size()
        if T > self.config.n_ctx:
            raise ValueError("Sequence length exceeds model context length")
        pos = torch.arange(0,T,dtype=torch.long,device=idx.device).unsqueeze(0)
        x = self.wte(idx) + self.wpe(pos)
        x = self.drop(x)

        for block in self.h:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), targets.view(-1))

        return logits, loss


###############################################################################
# Dataset utilities
###############################################################################
class TextDataset(Dataset):
    def __init__(self, token_ids: List[int], block_size: int):
        self.block_size = block_size
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.num_sequences = max((len(self.data) - 1) // block_size, 0)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size
        x = self.data[start:end]
        y = self.data[start+1:end+1]
        return x, y


###############################################################################
# Extracting texts from messages
###############################################################################
def extract_texts(dataset, max_samples):
    texts = []
    count = 0
    for ex in dataset:
        if count >= max_samples:
            break
        if 'messages' in ex:
            try:
                messages = json.loads(ex['messages'])
                # Extract user and assistant
                instruction = next((m['content'] for m in messages if m.get('role')=='user'), "")
                output = next((m['content'] for m in messages if m.get('role')=='assistant'), "")
                text = f"{instruction} {output}".strip()
                if text:
                    texts.append(text)
                    count += 1
            except json.JSONDecodeError:
                # skip
                pass
    return texts


###############################################################################
# Training and evaluation
###############################################################################
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


###############################################################################
# Main script
###############################################################################

def main():
    block_size = 1024
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Loading dataset...")
    dataset = load_dataset("microsoft/orca-agentinstruct-1M-v1")
    # We'll pick some subsets
    ds_train = dataset['analytical_reasoning']
    ds_val = dataset['text_modification']

    print("Extracting training texts...")
    train_texts = extract_texts(ds_train, max_samples=10000)
    print(f"Collected {len(train_texts)} training samples.")

    print("Extracting validation texts...")
    val_texts = extract_texts(ds_val, max_samples=2000)
    print(f"Collected {len(val_texts)} validation samples.")

    if not train_texts:
        raise ValueError("No training data extracted. Check the dataset and extraction logic.")

    print("Training tokenizer...")
    tokenizer = BPETokenizer(merges_count=100)  # more merges for a fuller tokenizer
    tokenizer.train(train_texts)
    print("Tokenizer trained.")

    print("Encoding train set...")
    train_ids = []
    for text in tqdm(train_texts, desc="Encoding train"):
        train_ids.extend(tokenizer.encode(text))

    print("Encoding val set...")
    val_ids = []
    for text in tqdm(val_texts, desc="Encoding val"):
        val_ids.extend(tokenizer.encode(text))

    if not train_ids:
        raise ValueError("No tokens in training data after tokenization.")

    if not val_ids:
        print("No tokens in validation data. Will train without validation.")

    train_dataset = TextDataset(train_ids, block_size)
    val_dataset = TextDataset(val_ids, block_size) if val_ids else None

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)  # batch_size=2 for demonstration
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False) if val_dataset else []

    print(f"Train dataset sequences: {len(train_dataset)}, Val dataset sequences: {len(val_dataset) if val_dataset else 0}")

    # GPT-2 small config
    config = GPT2Config(
        vocab_size=tokenizer.vocab.vocab_size,
        n_ctx=1024,
        n_embd=768,
        n_layer=12,
        n_head=12
    )

    model = GPT2Model(config).to(device)
    print("Model created with", sum(p.numel() for p in model.parameters()), "parameters.")

    print("Starting training...")
    train(model, train_loader, val_loader, device, epochs=1, lr=3e-4)

    print("Saving model and tokenizer...")
    os.makedirs("model_checkpoint", exist_ok=True)
    torch.save(model.state_dict(), "model_checkpoint/model.pt")

    with open("model_checkpoint/tokenizer.json", "w") as f:
        json.dump({
            "special_tokens": tokenizer.vocab.special_tokens,
            "token2id": tokenizer.vocab.token2id,
            "merges": tokenizer.vocab.merges
        }, f)

    print("Training complete. Model and tokenizer saved.")


if __name__ == "__main__":
    main()

