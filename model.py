
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class ArnoldActivation(nn.Module):
    def __init__(self, Omega=0.618, init_K=1.0):
        super().__init__()
        self.Omega = Omega
        self.K = nn.Parameter(torch.tensor([float(init_K)]))
        self.current_lyapunov = 0.0

    def forward(self, x):
        k_val = torch.abs(self.K)
        
        # Calculate Lyapunov Exponent: avg( ln |1 - K * cos(2*pi*x)| )
        with torch.no_grad():
            deriv = torch.abs(1 - k_val * torch.cos(2 * math.pi * x))
            self.current_lyapunov = torch.log(deriv + 1e-9).mean().item()
            
        out = x + self.Omega - (k_val / (2 * math.pi)) * torch.sin(2 * math.pi * x)
        return out % 1.0

class ArnoldAttention(nn.Module):
    def __init__(self, Omega=0.618, init_K=1.0):
        super().__init__()
        self.arnold = ArnoldActivation(Omega, init_K)

    def forward(self, q, k, v):
        # Apply phase-locking to the query and key
        q = self.arnold(q)
        k = self.arnold(k)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, block_size, dropout, attention_type='standard', init_K=1.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.attention_type = attention_type
        if attention_type == 'arnold':
            self.arnold_attention = ArnoldAttention(init_K=1.0)
        else:
            self.arnold_attention = None

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,16)
        q = self.query(x) # (B,T,16)
        v = self.value(x) # (B,T,16)

        if self.attention_type == 'standard':
            # compute attention scores ("affinities")
            wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, 16) @ (B, 16, T) -> (B, T, T)
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
            wei = F.softmax(wei, dim=-1) # (B, T, T)
            wei = self.dropout(wei)
            out = wei @ v # (B, T, T) @ (B, T, 16) -> (B, T, 16)
        else:
            out = self.arnold_attention(q, k, v)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout, attention_type='standard', init_K=1.0):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout, attention_type=attention_type, init_K=init_K) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.attention_type = attention_type

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout, activation_type='relu'):
        super().__init__()
        self.activation_type = activation_type
        layers = [
            nn.Linear(n_embd, 4 * n_embd),
        ]
        
        if activation_type == 'arnold':
            self.activation = ArnoldActivation()
            layers.append(self.activation)
        else:
            layers.append(nn.ReLU())
            
        layers.extend([
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        ])
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, block_size, dropout, activation_type='relu', attention_type='standard', init_K=1.0):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout, attention_type=attention_type, init_K=init_K)
        self.ffwd = FeedForward(n_embd, dropout, activation_type=activation_type)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attention_type = attention_type

    def forward(self, x):
        if self.attention_type == 'arnold':
            x = self.sa(x)
        else:
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):

    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, dropout, device, activation_types=None, attention_types=None, positional_encoding=None, init_K=1.0):
        super().__init__()
        self.device = device
        self.block_size = block_size
        if activation_types is None:
            activation_types = ['relu'] * n_layer
        self.activation_types = activation_types
        self.attention_types = attention_types
        self.positional_encoding = positional_encoding
        if len(activation_types) != n_layer:
            raise ValueError("activation_types must have length equal to n_layer")
            
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        if positional_encoding == 'arnold':
            self.positional_arnold = ArnoldActivation()
        else:
            self.positional_encoding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size, dropout, activation_type=activation_types[i], attention_type=attention_types[i], init_K=init_K) for i in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def get_max_lyapunov(self):
        max_lyap = -float('inf')
            
        for index, block in enumerate(self.blocks):
            if self.activation_types[index] == 'arnold':
                if hasattr(block.ffwd, 'activation'):
                    max_lyap = max(max_lyap, block.ffwd.activation.current_lyapunov)
            if self.attention_types is not None and self.attention_types[index] == 'arnold':
                if block.sa.attention_type == 'arnold':
                    for index, head in enumerate(block.sa.heads):
                        max_lyap = max(max_lyap, head.arnold_attention.arnold.current_lyapunov)
                    
        if max_lyap < -10:
            max_lyap = 0.0
        return max_lyap

    def get_min_max_K(self) -> tuple[list[float],list[float]]:
        k_act:list[float]=[]
        k_att:list[float]=[]
        for index, block in enumerate(self.blocks):
            if self.activation_types[index] == 'arnold':
                if hasattr(block.ffwd, 'activation'):
                    k_act.append(block.ffwd.activation.K.item())
            if self.attention_types is not None and self.attention_types[index] == 'arnold':
                if block.sa.attention_type == 'arnold':
                    for index, head in enumerate(block.sa.heads):
                        k_att.append(head.arnold_attention.arnold.K.item())                    
        return k_act, k_att

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        if self.positional_encoding == 'standard':
            pos_emb = self.positional_encoding_table(torch.arange(T, device=self.device)) # (T,C)
        else:
            pos_emb = self.positional_arnold(tok_emb)
        x = tok_emb + pos_emb # (B,T,C)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
