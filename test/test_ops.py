#!/usr/bin/env python3
import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
parser.add_argument("--profile", action="store_true")
args = parser.parse_args()

# # run all ops tests
tests = [
    "test/ops/add.py",
    "test/ops/argmax.py",
    "test/ops/embedding.py",
    "test/ops/linear.py",
    "test/ops/rms_norm.py",
    "test/ops/rope.py",
    "test/ops/swiglu.py",
    "test/ops/self_attention.py",
]

extra_args = f"--device {args.device}"
if args.profile:
    extra_args += " --profile"

print(f"Running {len(tests)} tests on {args.device}...")
print()

for test in tests:
    print(f"Running {test}...")
    if os.system(f"python {test} {extra_args}") != 0:
        print(f"Test failed: {test}")
        sys.exit(1)
    print()

print("\033[92mAll ops tests passed!\033[0m\n")