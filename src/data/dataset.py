import json
import torch
from torch.utils.data import Dataset
from typing import List
from collections import Counter

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

def extract_texts(dataset, max_samples: int) -> List[str]:
    texts = []
    count = 0
    for ex in dataset:
        if count >= max_samples:
            break
        if 'messages' in ex:
            try:
                messages = json.loads(ex['messages'])
                instruction = next((m['content'] for m in messages if m.get('role')=='user'), "")
                output = next((m['content'] for m in messages if m.get('role')=='assistant'), "")
                text = f"{instruction} {output}".strip()
                if text:
                    texts.append(text)
                    count += 1
            except json.JSONDecodeError:
                pass
    return texts
