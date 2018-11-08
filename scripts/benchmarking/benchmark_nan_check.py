#!/usr/bin/env python3

import timeit
from typing import Tuple

import torch


REPLICATIONS = 10000


def safe_topk(tensor: torch.Tensor, k, **kwargs) -> Tuple[torch.Tensor, torch.LongTensor]:
    if torch.isnan(tensor).any():
        raise ValueError("nan encountered")
    return torch.topk(tensor, k, **kwargs)


def main():
    print(f"Running benchmarks with {REPLICATIONS} replications")

    x_cpu = torch.zeros(10000, dtype=torch.float32)
    print("CPU results:")
    result = timeit.timeit("safe_topk(x_cpu, 5)", number=REPLICATIONS, globals={**globals(), **locals()})
    print(f"safe_topk:  {result:.8f}")
    result = timeit.timeit("torch.topk(x_cpu, 5)", number=REPLICATIONS, globals={**globals(), **locals()})
    print(f"torch.topk: {result:.8f}")

    if torch.cuda.is_available():
        x_gpu = x_cpu.cuda()  # pylint: disable=unused-variable
        print("GPU results:")
        result = timeit.timeit("safe_topk(x_gpu, 5)", number=REPLICATIONS, globals={**globals(), **locals()})
        print(f"safe_topk:  {result:0.8f}")
        result = timeit.timeit("torch.topk(x_gpu, 5)", number=REPLICATIONS, globals={**globals(), **locals()})
        print(f"torch.topk: {result:0.8f}")


if __name__ == "__main__":
    main()
