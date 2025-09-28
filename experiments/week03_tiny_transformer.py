import math, torch, torch.nn as nn, torch.nn.functional as F

CORPUS = ("theory first. small models, clear math. "
          "attention and transformers. practice minimal experiments. ")
chars = sorted(list(set(CORPUS)))
stoi = {c:i for i,c in enumerate(chars)}
itos = {i:c for c,i in stoi.items()}
def enc(s): return torch.tensor([stoi[c] for c in s], dtype=torch.long)

def get_batch(T=64, B=16):
    data = enc(CORPUS*100)
    ix = torch.randint(0, len(data)-T-1, (B,))
    x = torch.stack([data[i:i+T] for i in ix])
    y = torch.stack([data[i+1:i+T+1] for i in ix])
    return x,y

class Block(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_head, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.register_buffer("mask", torch.triu(torch.ones(1024,1024), diagonal=1).bool())
    def forward(self, x):
        x = self.ln1(x)
        attn_out, _ = self.attn(x, x, x, attn_mask=self.mask[:x.size(1), :x.size(1)])
        x = x + attn_out
        y = self.ln2(x)
        return x + self.ff(y)

class TinyTransformerLM(nn.Module):
    def __init__(self, vocab, T=64, d=96, n_head=4, n_layer=2):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(T, d)
        self.blocks = nn.Sequential(*[Block(d, n_head) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)
        self.T = T
    def forward(self, idx):
        B,T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        x = self.blocks(x)
        x = self.ln(x)
        return self.head(x)

def train(steps=400, T=64, lr=3e-3):
    torch.manual_seed(0)
    model = TinyTransformerLM(len(stoi), T=T)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for s in range(steps):
        x,y = get_batch(T, 32)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 50 == 0: print(f"step {s:03d} loss {loss.item():.4f}")
    # PPL on a tiny dev split
    with torch.no_grad():
        x_dev,y_dev = get_batch(T, 32)
        logits_dev = model(x_dev)
        ppl = math.exp(F.cross_entropy(logits_dev.view(-1, logits_dev.size(-1)), y_dev.view(-1)).item())
    print(f"Approx PPL: {ppl:.2f}")
    return model

if __name__ == "__main__":
    train()