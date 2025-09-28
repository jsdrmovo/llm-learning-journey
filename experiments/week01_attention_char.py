import math, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F

# Tiny dataset: next-char LM on a small corpus
CORPUS = """
attention is all you need
transformers build on attention
hello world! llm theory first
"""
chars = sorted(list(set(CORPUS)))
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in stoi.items()}
vocab_size = len(chars)

def encode(s): return torch.tensor([stoi[c] for c in s], dtype=torch.long)
def get_batch(seq_len=32, batch_size=16):
    data = encode(CORPUS*50)  # repeat to have more samples
    ix = torch.randint(0, len(data)-seq_len-1, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x, y

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3*d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(1024,1024)).view(1,1,1024,1024))

    def forward(self, x):
        B,T,C = x.size()
        qkv = self.qkv(x).view(B,T,3,self.n_head,self.d_head)
        q,k,v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]  # (B,T,h,d)
        q = q.transpose(1,2)  # (B,h,T,d)
        k = k.transpose(1,2)
        v = v.transpose(1,2)
        att = (q @ k.transpose(-2,-1)) / math.sqrt(self.d_head)  # (B,h,T,T)
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B,h,T,d)
        y = y.transpose(1,2).contiguous().view(B,T,self.n_head*self.d_head)
        return self.out(y)

class TinyCharLM(nn.Module):
    def __init__(self, vocab, d_model=64, n_head=4, seq_len=32):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.attn = CausalSelfAttention(d_model, n_head)
        self.ff = nn.Sequential(nn.Linear(d_model, 4*d_model),
                                nn.GELU(),
                                nn.Linear(4*d_model, d_model))
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, idx):
        B,T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None, :, :]
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        logits = self.head(x)
        return logits

def train(steps=300, seq_len=32, lr=3e-3):
    device = 'cpu'
    torch.manual_seed(42)
    model = TinyCharLM(vocab_size, d_model=64, n_head=4, seq_len=seq_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for step in range(steps):
        x,y = get_batch(seq_len, 32)
        x,y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            print(f"step {step:03d} | loss {loss.item():.4f}")
    return model

if __name__ == "__main__":
    train()
    print("Done. This script demonstrates attention + tiny LM on CPU.")