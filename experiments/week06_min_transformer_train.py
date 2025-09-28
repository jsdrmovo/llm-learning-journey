import os, math, json, time
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path

CONFIG = {
    "seq_len": 64,
    "d_model": 128,
    "n_head": 4,
    "n_layer": 2,
    "lr": 3e-3,
    "steps": 600,
    "batch_size": 32,
    "seed": 42,
    "save_dir": "outputs/ckpt",
}

CORPUS = ("minimal transformer training with config, checkpoint and ppl eval. "
          "this is a tiny demo for week six milestone.")

chars = sorted(list(set(CORPUS)))
stoi = {c:i for i,c in enumerate(chars)}
def enc(s): return torch.tensor([stoi[c] for c in s], dtype=torch.long)

def get_batch(T, B):
    data = enc(CORPUS*300)
    ix = torch.randint(0, len(data)-T-1, (B,))
    x = torch.stack([data[i:i+T] for i in ix])
    y = torch.stack([data[i+1:i+T+1] for i in ix])
    return x,y

class Block(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_head, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.register_buffer("mask", torch.triu(torch.ones(1024,1024),1).bool())
    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x),
                          attn_mask=self.mask[:x.size(1), :x.size(1)])[0]
        x = x + self.ff(self.ln2(x))
        return x

class LM(nn.Module):
    def __init__(self, vocab, cfg):
        super().__init__()
        self.tok = nn.Embedding(vocab, cfg["d_model"])
        self.pos = nn.Embedding(cfg["seq_len"], cfg["d_model"])
        self.blocks = nn.Sequential(*[Block(cfg["d_model"], cfg["n_head"]) for _ in range(cfg["n_layer"])])
        self.ln = nn.LayerNorm(cfg["d_model"])
        self.head = nn.Linear(cfg["d_model"], vocab)
        self.cfg = cfg
    def forward(self, idx):
        B,T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        x = self.blocks(x)
        x = self.ln(x)
        return self.head(x)

def evaluate(model, T=64):
    with torch.no_grad():
        x,y = get_batch(T, 64)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        ppl = math.exp(loss.item())
    return ppl

def main():
    cfg = CONFIG
    torch.manual_seed(cfg["seed"])
    Path(cfg["save_dir"]).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(cfg["save_dir"], "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    model = LM(vocab=len(stoi), cfg=cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    for step in range(cfg["steps"]):
        x,y = get_batch(cfg["seq_len"], cfg["batch_size"])
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            ppl = math.exp(loss.item())
            print(f"step {step:03d} | loss {loss.item():.4f} | ppl {ppl:.2f}")

    ckpt_path = os.path.join(cfg["save_dir"], "model.pt")
    torch.save({"model_state": model.state_dict(), "vocab": list(stoi.keys()), "config": cfg}, ckpt_path)
    final_ppl = evaluate(model, T=cfg["seq_len"])
    print(f"Saved checkpoint to {ckpt_path} | Eval PPL={final_ppl:.2f}")

if __name__ == "__main__":
    main()
