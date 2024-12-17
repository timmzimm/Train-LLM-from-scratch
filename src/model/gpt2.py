import math
import torch
import torch.nn as nn

class GPT2Config:
    """
    Configuration class for the GPT-2 style model.
    Defines model hyperparameters.
    """
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
    """
    Causal self-attention mechanism for GPT-2 style models.
    Uses a lower-triangular mask to ensure causal (autoregressive) behavior.
    """
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
    """
    Multi-layer perceptron block in the GPT-2 model.
    Expands the embedding dimension, applies a non-linearity, and projects back.
    """
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
    """
    A single Transformer block consisting of:
    - LayerNorm
    - Causal self-attention
    - Another LayerNorm
    - MLP
    Residual connections are applied around both the attention and MLP sub-blocks.
    """
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
    """
    GPT-2 style model:
    - Token embedding + positional embedding
    - N Transformer blocks
    - Layer norm + linear head at the end
    """
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
        """
        Initialize weights in a manner similar to the original GPT-2 initialization.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        """
        Forward pass of the GPT-2 model.
        
        Args:
            idx: (B,T) tensor of token IDs
            targets: (B,T) tensor of token IDs to compute cross-entropy loss

        Returns:
            logits: (B,T,vocab_size) predictions of the next token
            loss: scalar cross-entropy loss if targets are provided
        """
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

