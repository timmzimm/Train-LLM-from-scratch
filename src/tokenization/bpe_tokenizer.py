import json
from typing import List
from collections import Counter
from tqdm import tqdm

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
