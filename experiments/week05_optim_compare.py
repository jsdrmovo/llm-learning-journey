import torch, torch.nn as nn, torch.nn.functional as F

TEXT = "attention is all you need. minimal experiments show trends. "
chars = sorted(set(TEXT))
stoi = {c:i for i,c in enumerate(chars)}
def enc(s): return torch.tensor([stoi[c] for c in s], dtype=torch.long)

def batch(T=32, B=16):
    data = enc(TEXT*200)
    ix = torch.randint(0, len(data)-T-1, (B,))
    x = torch.stack([data[i:i+T] for i in ix])
    y = torch.stack([data[i+1:i+T+1] for i in ix])
    return x,y

class TinyLM(nn.Module):
    def __init__(self, vocab=64, d=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.lin = nn.Linear(d, vocab)
    def forward(self, x):
        h = self.emb(x)
        return self.lin(h)

def run(optimizer_name="adamw", steps=200, lr=3e-3):
    torch.manual_seed(123)
    model = TinyLM(vocab=len(stoi))
    if optimizer_name == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError("unknown optimizer")
    losses=[]
    for s in range(steps):
        x,y = batch(32,32)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return losses[-1]

if __name__ == "__main__":
    l_adamw = run("adamw", steps=300, lr=3e-3)
    l_sgd   = run("sgd",   steps=300, lr=1e-1)
    print(f"Final loss | AdamW: {l_adamw:.4f} | SGD: {l_sgd:.4f}")
