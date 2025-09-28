import math, torch, torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=8, bias=False):
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=bias)
        for p in self.base.parameters(): p.requires_grad = False
        self.r = r
        self.alpha = alpha
        if r > 0:
            self.A = nn.Linear(in_features, r, bias=False)
            self.B = nn.Linear(r, out_features, bias=False)
            nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.B.weight)
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

    def forward(self, x):
        y = self.base(x)
        if self.r > 0:
            y = y + (self.alpha / self.r) * self.B(self.A(x))
        return y

def count_trainable(m: nn.Module):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    torch.manual_seed(0)
    in_f, out_f = 64, 32
    r = 4
    # Construct a "new task" true mapping
    W_true = torch.randn(out_f, in_f)
    # Build LoRA-adaptable layer with a different frozen base
    layer = LoRALinear(in_f, out_f, r=r, alpha=8, bias=False)
    with torch.no_grad():
        layer.base.weight.copy_(torch.randn_like(layer.base.weight))  # base is mismatched

    print(f"Trainable params (LoRA): {count_trainable(layer)} vs full: {in_f*out_f}")

    # Toy data: y = W_true x
    N = 2048
    X = torch.randn(N, in_f)
    Y = (X @ W_true.T)

    opt = torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad], lr=5e-3)
    loss_fn = nn.MSELoss()

    for step in range(400):
        idx = torch.randint(0, N, (64,))
        x, y = X[idx], Y[idx]
        y_hat = layer(x)
        loss = loss_fn(y_hat, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 80 == 0:
            with torch.no_grad():
                mse = loss_fn(layer(X), Y).item()
            print(f"step {step:03d} | batch_loss {loss.item():.4f} | full_MSE {mse:.4f}")

    with torch.no_grad():
        final_mse = loss_fn(layer(X), Y).item()
    print(f"Final MSE with LoRA (r={r}): {final_mse:.4f}")

if __name__ == "__main__":
    main()