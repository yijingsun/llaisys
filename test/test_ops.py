#!/usr/bin/env python3
import os
import sys

# # run all ops tests
tests = [
    "test/ops/argmax.py",
    "test/ops/embedding.py",
    "test/ops/linear.py",
    "test/ops/rms_norm.py",
    "test/ops/rope.py",
    "test/ops/swiglu.py",
    "test/ops/self_attention.py",
]

print(f"Running {len(tests)} tests...")
print()

for test in tests:
    print(f"Running {test}...")
    if os.system(f"python {test}") != 0:
        print(f"Test failed: {test}")
        sys.exit(1)
    print()

print("\033[92mAll ops tests passed!\033[0m\n")
