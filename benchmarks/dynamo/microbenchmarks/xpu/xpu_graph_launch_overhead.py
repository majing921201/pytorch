import argparse
import time
import torch
import torch.nn as nn
import torch.autograd.profiler as profiler


"""
This benchmark demonstrates CPU kernel launch overhead
dominating small-batch XPU workloads and the benefit
of XPUGraph replay.

Expected:
- Speedup increases with depth
- Speedup decreases with width
- Batch size fixed to 1

VTune:
This benchmark emits ITT ranges via torch.autograd.profiler.emit_itt()
to enable CPU-side launch overhead analysis in Intel VTune.
"""

class TinyKernelStorm(nn.Module):
    def __init__(self, depth, width):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(width, width),
                nn.ReLU(),
                nn.Linear(width, width),
                nn.ReLU(),
            )
            for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def run_eager(model, x, iters, fine_grain_itt):
    for _ in range(50):
        model(x)
    torch.xpu.synchronize()

    with profiler.emit_itt(), profiler.record_function("eager"):
        start = time.perf_counter()
        for _ in range(iters):
            if fine_grain_itt:
                with profiler.record_function("eager_iter"):
                    model(x)
            else:
                model(x)
        torch.xpu.synchronize()
        return (time.perf_counter() - start) / iters

def run_xpu_graph(model, x, iters, fine_grain_itt):
    g = torch.xpu.XPUGraph()
    static_x = x.clone()

    for _ in range(10):
        model(static_x)
    torch.xpu.synchronize()

    with torch.xpu.graph(g):
        static_y = model(static_x)

    torch.xpu.synchronize()
    with profiler.emit_itt(), profiler.record_function("xpugraph"):
        start = time.perf_counter()
        for _ in range(iters):
            if fine_grain_itt:
                with profiler.record_function("xpugraph_iter"):
                    static_x.copy_(x)
                    g.replay()
            else:
                static_x.copy_(x)
                g.replay()
        torch.xpu.synchronize()
        end = time.perf_counter()

    return (end - start) / iters

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=200)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--fine-grain-itt", action="store_true")
    args = parser.parse_args()

    assert torch.xpu.is_available()

    device = "xpu"
    model = TinyKernelStorm(args.depth, args.width).to(device).eval()
    x = torch.randn(1, args.width, device=device)

    eager_t = run_eager(model, x, args.iters, args.fine_grain_itt)
    graph_t = run_xpu_graph(model, x, args.iters, args.fine_grain_itt)

    print(f"Eager:     {eager_t * 1000:.3f} ms")
    print(f"XPUGraph:  {graph_t * 1000:.3f} ms")
    print(f"Speedup:   {eager_t / graph_t:.2f}x")

if __name__ == "__main__":
    main()
