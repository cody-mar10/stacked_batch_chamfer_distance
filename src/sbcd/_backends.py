from pathlib import Path
from typing import Callable, Tuple

import torch
from torch.utils.cpp_extension import load
from torch_scatter import segment_mean_csr, segment_min_csr

_srcdir = Path(__file__).parent

_stacked_batch_chamfer_distance_cpu = load(
    name="stacked_batch_chamfer_distance_cpu",
    sources=[_srcdir.joinpath("_C/sbcd.cpp").as_posix()],
    verbose=True,
    extra_cflags=["-O2"],
)

if torch.cuda.is_available():
    _stacked_batch_chamfer_distance_cuda = load(
        name="stacked_batch_chamfer_distance_cuda",
        sources=[
            _srcdir.joinpath("_C/sbcd_cuda.cpp").as_posix(),
            _srcdir.joinpath("_C/sbcd_cuda.cu").as_posix(),
        ],
        verbose=True,
        extra_cflags=["-O2"],
    )

PairTensor = Tuple[torch.Tensor, torch.Tensor]
KernelType = Callable[[torch.Tensor, torch.Tensor], PairTensor]


def _stacked_batch_chamfer_distance_scatter(
    x: torch.Tensor, ptr: torch.Tensor
) -> PairTensor:
    all_dist = torch.cdist(x, x, p=2.0).square().fill_diagonal_(0.0)

    min_dists, min_idx = segment_min_csr(all_dist, ptr)
    mean_dists = segment_mean_csr(min_dists.t(), ptr)

    chamfer_distance = mean_dists + mean_dists.t()

    return chamfer_distance, min_idx.t()


def _stacked_batch_chamfer_distance_slicing(
    x: torch.Tensor, ptr: torch.Tensor
) -> PairTensor:
    all_dist = torch.cdist(x, x, p=2.0).square().fill_diagonal_(0.0)
    batch_size = ptr.numel() - 1
    n_nodes = int(x.size(0))

    chamfer_distance = x.new_zeros((batch_size, batch_size))
    min_idx = torch.arange(n_nodes, device=x.device).unsqueeze(-1).repeat(1, batch_size)

    for i in range(batch_size):
        x_start, x_end = ptr[i], ptr[i + 1]
        for j in range(i + 1, batch_size):
            y_start, y_end = ptr[j], ptr[j + 1]

            block = all_dist[x_start:x_end, y_start:y_end]

            x_min: torch.Tensor
            x_idx: torch.Tensor
            y_min: torch.Tensor
            y_idx: torch.Tensor
            x_min, x_idx = block.min(dim=1)
            y_min, y_idx = block.min(dim=0)

            dist = x_min.mean() + y_min.mean()

            chamfer_distance[i, j] = dist
            chamfer_distance[j, i] = dist

            min_idx[x_start:x_end, j] = y_start + x_idx
            min_idx[y_start:y_end, i] = x_start + y_idx

    return chamfer_distance, min_idx


def _stacked_batch_chamfer_distance(x: torch.Tensor, ptr: torch.Tensor) -> PairTensor:
    if x.is_cuda:
        fn = _stacked_batch_chamfer_distance_cuda.stacked_batch_chamfer_distance_cuda  # type: ignore # noqa: E501
    else:
        fn = _stacked_batch_chamfer_distance_cpu.stacked_batch_chamfer_distance_cpu  # type: ignore # noqa: E501

    min_dist: torch.Tensor
    min_idx: torch.Tensor
    min_dist, min_idx = fn(x, ptr)
    mean_dist = segment_mean_csr(min_dist, ptr)
    chamfer_distance = mean_dist + mean_dist.t()
    return chamfer_distance, min_idx
