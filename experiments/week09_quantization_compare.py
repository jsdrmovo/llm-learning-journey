import io, time, torch, torch.nn as nn
from torch.ao.quantization import quantize_dynamic

class TinyMLP(nn.Module):
    def __init__(self, d_in=256, d_h=256, d_out=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_h), nn.ReLU(),
            nn.Linear(d_h, d_h), nn.ReLU(),
            nn.Linear(d_h, d_out)
        )
    def forward(self, x): return self.net(x)

def model_bytes(m: nn.Module):
    buf = io.BytesIO()
    torch.save(m.state_dict(), buf)
    return len(buf.getvalue())

def bench(model, iters=50, B=512, d_in=256):
    x = torch.randn(B, d_in)
    # warmup
    with torch.no_grad():
        for _ in range(10): _ = model(x)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters): _ = model(x)
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1000.0

if __name__ == "__main__":
    torch.manual_seed(0)
    fp32 = TinyMLP()
    ms_fp32 = bench(fp32)
    size_fp32 = model_bytes(fp32)

    qdyn = quantize_dynamic(fp32, {nn.Linear}, dtype=torch.qint8)
    ms_int8 = bench(qdyn)
    size_int8 = model_bytes(qdyn)

    print(f"Latency (ms/batch): FP32={{ms_fp32:.2f}} | INT8(dynamic)={{ms_int8:.2f}}")
    print(f"State size (bytes): FP32={{size_fp32}} | INT8(dynamic)={{size_int8}}")
    # Check numerical deviation
    x = torch.randn(1024, 256)
    with torch.no_grad():
        err = (fp32(x) - qdyn(x)).abs().mean().item()
    print(f"Avg |FP32-INT8|: {{err:.6f}}")
