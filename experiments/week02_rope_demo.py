import math
import torch

def rotary_freqs(head_dim, T, base=10000.0, device='cpu'):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float()/head_dim))
    t = torch.arange(T, device=device).float()
    freqs = torch.einsum('t,f->t f', t, inv_freq)  # (T, head_dim/2)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rope(q, k):
    B,T,H,D = q.shape
    cos, sin = rotary_freqs(D, T, device=q.device)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    def rope(x):
        d = x.size(-1)
        x1, x2 = x[..., :d//2], x[..., d//2:]
        return torch.cat([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)
    return rope(q), rope(k)

def demo():
    torch.manual_seed(0)
    B,T,H,D = 1, 16, 2, 32
    q = torch.randn(B,T,H,D)
    k = torch.randn(B,T,H,D)
    pos = 8
    score_no_rope = (q[:,pos] @ k.transpose(-2,-1)).mean().item()
    q_r, k_r = apply_rope(q.clone(), k.clone())
    score_rope = (q_r[:,pos] @ k_r.transpose(-2,-1)).mean().item()
    print(f"Dot score at position {pos}: no RoPE={score_no_rope:.4f} | with RoPE={score_rope:.4f}")
    print("This demo shows how RoPE transforms q/k with position-dependent rotation.")

if __name__ == "__main__":
    demo()
