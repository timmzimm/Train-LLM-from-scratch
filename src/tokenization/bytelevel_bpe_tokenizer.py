import json
from typing import List
from collections import Counter
from tqdm import tqdm

class ByteLevelBPEVocabulary:
    """
    Maintains the vocabulary for Byte-Level BPE tokenization,
    including mappings from token to ID, ID to token, and merges.
    """
    def __init__(self, special_tokens: List[str] = ["<|endoftext|>"]):
        self.special_tokens = special_tokens
        self.token2id = {}
        self.id2token = {}
        self.merges = []
        self.vocab_size = 0

    def build_initial_vocab(self):
        """
        Build an initial byte-level vocab: 256 possible bytes plus special tokens.
        """
        idx = 0
        # Add special tokens first
        for st in self.special_tokens:
            self.token2id[st] = idx
            idx += 1

        # Add all 256 bytes
        for b in range(256):
            byte_char = f"<0x{b:02X}>"  # a representation for each byte
            self.token2id[byte_char] = idx
            idx += 1

        self.id2token = {v: k for k, v in self.token2id.items()}
        self.vocab_size = len(self.id2token)

    def add_merge(self, new_token: str, t1: str, t2: str):
        """
        Adds a new merged token to the vocabulary.
        """
        if new_token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[new_token] = idx
            self.id2token[idx] = new_token
            self.vocab_size += 1
        self.merges.append((t1, t2))

    def encode_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """
        Encodes a list of tokens into IDs.
        """
        return [self.token2id[t] for t in tokens if t in self.token2id]

    def decode_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """
        Decodes token IDs to the corresponding tokens.
        """
        return [self.id2token[i] for i in ids]

class ByteLevelBPETokenizer:
    """
    A Byte-Level BPE tokenizer:
    - Processes text at the byte level.
    - Learns merges similarly to a standard BPE approach.
    """
    def __init__(self, special_tokens=["<|endoftext|>"], vocab_size_limit=30000, merges_count=10000):
        self.vocab = ByteLevelBPEVocabulary(special_tokens)
        self.vocab_size_limit = vocab_size_limit
        self.merges_count = merges_count

    def _text_to_bytes(self, text: str) -> List[str]:
        """
        Converts a string into a list of byte-representations, plus end-of-text token.
        """
        byte_list = []
        for b in text.encode("utf-8"):
            byte_list.append(f"<0x{b:02X}>")
        byte_list.append("<|endoftext|>")
        return byte_list

    def train(self, texts: List[str]):
        """
        Train the byte-level BPE tokenizer on the given corpus.
        """
        print("Training Byte-Level BPE tokenizer on the corpus...")
        # Build the initial vocab (256 bytes + special tokens)
        self.vocab.build_initial_vocab()

        # Convert texts to sequences of byte-level tokens
        sequences = [self._text_to_bytes(t) for t in texts]

        for i in tqdm(range(self.merges_count), desc="Byte BPE merges"):
            pair_counts = Counter()
            # Count pair frequencies
            for seq in sequences:
                for j in range(len(seq)-1):
                    pair = (seq[j], seq[j+1])
                    pair_counts[pair] += 1

            if not pair_counts:
                break
            best_pair, best_count = pair_counts.most_common(1)[0]
            new_token = best_pair[0] + best_pair[1]
            self.vocab.add_merge(new_token, best_pair[0], best_pair[1])

            # Replace all occurrences of the best pair with the new token
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

        print("Byte-Level BPE training complete. Final vocab size:", self.vocab.vocab_size)

    def encode(self, text: str) -> List[int]:
        """
        Encodes a given text into a sequence of byte-level BPE token IDs.
        """
        tokens = self._text_to_bytes(text)
        merges_set = set(self.vocab.merges)

        while True:
            merged = False
            new_seq = []
            j = 0
            while j < len(tokens):
                if j < len(tokens)-1 and (tokens[j], tokens[j+1]) in merges_set:
                    new_token = tokens[j] + tokens[j+1]
                    new_seq.append(new_token)
                    j += 2
                    merged = True
                else:
                    new_seq.append(tokens[j])
                    j += 1
            tokens = new_seq
            if not merged:
                break

        return self.vocab.encode_tokens_to_ids(tokens)

    def decode(self, ids: List[int]) -> str:
        """
        Decodes a sequence of byte-level BPE IDs back into a text string.
        """
        tokens = self.vocab.decode_ids_to_tokens(ids)
        # Remove trailing end-of-text if present
        if tokens and tokens[-1] == "<|endoftext|>":
            tokens = tokens[:-1]

        # Reconstruct text from byte tokens
        byte_values = []
        for tok in tokens:
            if tok.startswith("<0x") and tok.endswith(">"):
                # convert hex to int then to bytes
                hex_val = tok[3:-1]  # e.g., "FF"
                byte_values.append(int(hex_val, 16))

        return bytearray(byte_values).decode("utf-8", errors="replace")
