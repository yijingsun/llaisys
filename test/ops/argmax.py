from calendar import c
import sys
import os
import time

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import llaisys
import torch
from test_utils import random_tensor, check_equal, benchmark, zero_tensor


def torch_argmax(max_idx, max_val, vals):
    torch.max(vals, keepdim=True, dim=-1, out=(max_val, max_idx))


def test_op_argmax(
    shape,
    dtype_name="f32",
    device_name="cpu",
    profile=False,
):
    print(f"   shape {shape} dtype <{dtype_name}>")
    vals, vals_ = random_tensor(shape, dtype_name, device_name)
    max_idx, max_idx_ = zero_tensor((1,), "i64", device_name)  # dtype of max_idx is i64
    max_val, max_val_ = zero_tensor((1,), dtype_name, device_name)

    start_time = time.time()
    torch_argmax(max_idx, max_val, vals)
    end_time = time.time()
    print(f"    Torch time elapsed: {(end_time - start_time):.2f}s")
    start_time = time.time()
    llaisys.Ops.argmax(max_idx_, max_val_, vals_)
    end_time = time.time()
    print(f"    Llaisys time elapsed: {(end_time - start_time):.2f}s\n")

    assert check_equal(max_val_, max_val, strict=True) or check_equal(
        max_idx_, max_idx, strict=True
    )

    if profile:
        benchmark(
            lambda: torch_argmax(max_idx, max_val, vals),
            lambda: llaisys.Ops.argmax(max_idx_, max_val_, vals_),
            device_name,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    testShapes = [(4,), (4096,)]
    testDtype = ["f32", "f16", "bf16"]
    print(f"Testing Ops.argmax on {args.device}")
    for shape in testShapes:
        for dtype_name in testDtype:
            test_op_argmax(shape, dtype_name, args.device, args.profile)

    print("\033[92mTest passed!\033[0m\n")
