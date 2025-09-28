import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model=128, n_head=4, max_len=512):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(max_len, max_len)).bool())

    def forward(self, x, past_k=None, past_v=None):
        # x: (B,T,C)
        B,T,C = x.shape
        q = self.q_proj(x).view(B,T,self.n_head,self.d_head).transpose(1,2)  # (B,h,T,d)
        k = self.k_proj(x).view(B,T,self.n_head,self.d_head).transpose(1,2)
        v = self.v_proj(x).view(B,T,self.n_head,self.d_head).transpose(1,2)

        if past_k is not None:
            k = torch.cat([past_k, k], dim=2)  # (B,h,T_total,d)
            v = torch.cat([past_v, v], dim=2)
            T_total = k.size(2)
            att = (q @ k.transpose(-2,-1)) / math.sqrt(self.d_head)  # (B,h,T,TT)
            att = att.masked_fill(self.mask[:T, :T_total].logical_not(), float('-inf'))
        else:
            att = (q @ k.transpose(-2,-1)) / math.sqrt(self.d_head)
            att = att.masked_fill(self.mask[:T, :T].logical_not(), float('-inf'))

        att = torch.softmax(att, dim=-1)
        y = att @ v  # (B,h,T,d)
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.o_proj(y)
        return y, (k, v)

class TinyDecoder(nn.Module):
    def __init__(self, vocab, d_model=128, n_head=4, max_len=512):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, max_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))
        self.head = nn.Linear(d_model, vocab)
        self.max_len = max_len

    def forward(self, idx, past_kv=None):
        B,T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None, :, :]
        x = self.ln1(x)
        y, kv = self.attn(x, *(past_kv or (None, None)))
        x = x + y
        x = x + self.ff(self.ln2(x))
        logits = self.head(x)
        return logits, kv

def generate(model, start_ids, steps=64, use_cache=True, device='cpu'):
    model.eval()
    idx = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1,T0)
    past_kv = None
    toks = idx.tolist()[0][:]
    with torch.no_grad():
        if use_cache:
            # prime prefix in one shot to build cache
            logits, past_kv = model(idx, past_kv=None)
            next_id = int(torch.argmax(logits[0, -1], -1))
            toks.append(next_id)
            cur = torch.tensor([[next_id]], dtype=torch.long, device=device)
            for _ in range(steps-1):
                logits, past_kv = model(cur, past_kv=past_kv)
                next_id = int(torch.argmax(logits[0, -1], -1))
                toks.append(next_id)
                cur = torch.tensor([[next_id]], dtype=torch.long, device=device)
        else:
            for _ in range(steps):
                logits, _ = model(idx, past_kv=None)
                next_id = int(torch.argmax(logits[0, -1], -1))
                toks.append(next_id)
                idx = torch.tensor([toks], dtype=torch.long, device=device)
    return toks

def bench(use_cache):
    vocab = 128
    model = TinyDecoder(vocab=vocab, d_model=128, n_head=4, max_len=1024)
    device = 'cpu'
    model.to(device)
    warmup = 10
    tokens = 128
    start = [1,2,3,4,5,6,7,8]

    # Warmup
    for _ in range(warmup):
        generate(model, start, steps=16, use_cache=use_cache, device=device)

    t0 = time.perf_counter()
    generate(model, start, steps=tokens, use_cache=use_cache, device=device)
    t1 = time.perf_counter()
    avg_ms = (t1 - t0) * 1000.0 / tokens
    print(f"use_cache={{use_cache}} | avg ms/token ~ {{avg_ms:.2f}}")

if __name__ == "__main__":
    bench(use_cache=False)
    bench(use_cache=True)