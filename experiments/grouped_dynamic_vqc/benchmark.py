"""Measure complete differentiable CUDA train steps, never forward-only timing."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import torch

from .model import GroupedDynamicVQC
from .objectives import loss_function
from .runtime import atomic_json, cuda_runtime, report, utc_now


def benchmark_batch(batch_size: int, repeats: int, front_layers: int) -> dict:
    model = GroupedDynamicVQC(front_layers=front_layers).cuda()
    angles = torch.rand(batch_size, 10, 4, device="cuda") * torch.pi
    concepts = torch.stack(
        (
            torch.randint(3, (batch_size,), device="cuda"),
            torch.randint(6, (batch_size,), device="cuda"),
        ),
        dim=1,
    )
    labels = ((concepts[:, 0] == 2) ^ (concepts[:, 1] > 2)).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def step() -> float:
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(angles), concepts, labels)[0]
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    for _ in range(2):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    final_loss = 0.0
    for _ in range(repeats):
        final_loss = step()
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - started) / repeats
    return {
        "batch_size": batch_size,
        "seconds_per_step": seconds,
        "samples_per_second": batch_size / seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "final_loss": final_loss,
        "parameter_device": str(next(model.parameters()).device),
        "front_layers": front_layers,
        "forward_backward_optimizer": True,
    }


def run_benchmark(candidates: list[int], repeats: int, front_layers: int) -> dict:
    runtime = cuda_runtime(0)
    records = []
    for batch_size in candidates:
        gc.collect()
        torch.cuda.empty_cache()
        try:
            record = benchmark_batch(batch_size, repeats, front_layers)
        except torch.cuda.OutOfMemoryError:
            record = {"batch_size": batch_size, "out_of_memory": True}
        records.append(record)
        report(f"CUDA batch benchmark: {record}")
        if record.get("out_of_memory"):
            break
    successful = [item for item in records if "samples_per_second" in item]
    if not successful:
        raise RuntimeError("No candidate batch size fits CUDA memory")
    # Keep 20% device memory headroom; throughput alone does not establish
    # optimization quality, so the training default is capped at 1024.
    eligible = [
        item
        for item in successful
        if item["peak_reserved_gib"] < 0.8 * runtime["total_memory_gib"]
        and item["batch_size"] <= 1024
    ]
    if not eligible:
        eligible = successful[:1]
    chosen = max(eligible, key=lambda item: item["samples_per_second"])
    return {
        "created_at": utc_now(),
        "runtime": runtime,
        "records": records,
        "recommended_batch_size": chosen["batch_size"],
        "selection_rule": "max train throughput, <=1024 samples, <80% memory",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--batches", type=int, nargs="+", default=[64, 128, 256, 512, 1024]
    )
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--front-layers", type=int, default=4)
    args = parser.parse_args()
    if args.repeats < 1 or min(args.batches) < 1:
        parser.error("Positive batch sizes and repeats required")
    result = run_benchmark(args.batches, args.repeats, args.front_layers)
    atomic_json(args.out, result)


if __name__ == "__main__":
    main()
