import os
import time

import torch
from torch.utils.benchmark import Timer

from sbcd import _backends


def main():
    torch.manual_seed(1234)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = 32
    sizes = torch.cat(
        (
            torch.randint(2, 100, (batch_size // 2,)),
            torch.randint(100, 1000, (batch_size // 2,)),
        )
    ).to(device)
    ptr = torch.cat((sizes.new_zeros(1, dtype=torch.long), sizes.cumsum(0)))
    x = torch.randn((int(ptr[-1].item()), 320), device=device)
    # batch = torch.arange(x.size(0), device=device).repeat_interleave(sizes)

    kwargs = dict()
    if not x.is_cuda:
        kwargs["num_threads"] = (os.cpu_count() or 2) // 2
        n_iter = 100
    else:
        n_iter = 10000

    print("Benchmarking...")

    for name, fn in [
        ("segment", _backends._stacked_batch_chamfer_distance_scatter),
        ("slicing", _backends._stacked_batch_chamfer_distance_slicing),
        ("cuda", _backends._stacked_batch_chamfer_distance),
    ]:
        start = time.perf_counter()
        timer = Timer(
            stmt="fn(x, ptr)",
            globals={"fn": fn, "x": x, "ptr": ptr},
            label=name,
            description=f"SBCD using {name}",
            **kwargs,
        )
        print(timer.timeit(n_iter))
        duration = time.perf_counter() - start
        print(f"Total time: {duration:.3f}s")


if __name__ == "__main__":
    main()
