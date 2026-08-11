#!/usr/bin/env python3
"""
NCU profiling of the optimized flash_attn kernel.
Profiles key shapes and collects:
  - Memory bandwidth (DRAM, L2, L1/SMEM)
  - Compute utilization (SM, Tensor Core)
  - Stall reasons (barrier, memory, math, dependency)
  - Occupancy (register pressure, shared memory pressure)

Usage:
    python3 profile_flash_attn.py          # profile all shapes
    python3 profile_flash_attn.py --quick  # profile 1 shape only
    python3 profile_flash_attn.py --roofline  # roofline analysis on 1 shape
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import torch
from extension_cpp.ops import flash_attn

# ── Config ────────────────────────────────────────────────────────────

# Representative shapes: (B, H, N, D)
PROFILE_SHAPES = [
    # (name, B, H, N, D) — pick shapes that exercise different code paths
    ("d64_s512",  1, 8, 512,  64),   # most common: HeadDim=64, medium seq
    ("d128_s256", 1, 8, 256, 128),  # larger head dim
    ("d256_s128", 1, 4, 128, 256),  # largest head dim, small seq
    ("d64_s4096", 1, 8, 4096, 64),  # long seq (compute-bound)
    ("d64_b4",    4, 8, 256,  64),   # batched, L2 reuse test
]

# Metrics we care about (source metric names used by ncu)
# Use --set basic for quick overview, --set full for deep dive
METRIC_SETS = {
    "quick": "--set basic",
    "detailed": "--set detailed",
    "full": "--set full",
}

# ── Helpers ────────────────────────────────────────────────────────────

def _run_ncu(shape_name: str, b: int, h: int, n: int, d: int,
             metric_set: str, extra_args: str = "",
             roofline: bool = False) -> str:
    """Profile one shape with ncu, return raw output string."""
    # Build a self-contained script that runs the kernel
    script = f"""
import torch
from extension_cpp.ops import flash_attn

q = torch.randn({b}, {h}, {n}, {d}, dtype=torch.float16, device="cuda")
k = torch.randn({b}, {h}, {n}, {d}, dtype=torch.float16, device="cuda")
v = torch.randn({b}, {h}, {n}, {d}, dtype=torch.float16, device="cuda")

# Warmup
for _ in range(3):
    flash_attn(q, k, v)
torch.cuda.synchronize()

# Timed run
flash_attn(q, k, v)
torch.cuda.synchronize()
"""
    script_path = f"/tmp/ncu_script_{shape_name}.py"
    with open(script_path, "w") as f:
        f.write(script)

    ncu_args = [
        "ncu",
        metric_set,
        "--kernel-name", "regex:flash_attn_cute_kernel",
        "--launch-skip", "0",
        "--launch-count", "1",
        "--target-processes", "all",
        "-o", f"/tmp/ncu_{shape_name}",
    ]
    if extra_args:
        ncu_args.extend(extra_args.split())
    if roofline:
        ncu_args.extend(["--set", "full", "--roofline", "on"])
    ncu_args.extend([sys.executable, script_path])

    print(f"\n{'='*80}")
    print(f"  Profiling: {shape_name} (B={b}, H={h}, N={n}, D={d})")
    print(f"  Metric set: {metric_set}")
    print(f"{'='*80}")

    result = subprocess.run(ncu_args, capture_output=True, text=True, timeout=120)
    return result.stdout + "\n" + result.stderr


def _cleanup():
    for f in Path("/tmp").glob("ncu_script_*.py"):
        f.unlink(missing_ok=True)


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Profile 1 shape only")
    parser.add_argument("--roofline", action="store_true", help="Roofline analysis on 1 shape")
    parser.add_argument("--shape", type=str, default="", help="Specific shape name to profile")
    args = parser.parse_args()

    shapes = PROFILE_SHAPES
    if args.quick or args.roofline:
        shapes = [shapes[0]]  # d64_s512 only
    if args.shape:
        shapes = [s for s in shapes if s[0] == args.shape]
        if not shapes:
            print(f"Unknown shape: {args.shape}. Available: {[s[0] for s in PROFILE_SHAPES]}")
            sys.exit(1)

    all_outputs = {}

    for name, b, h, n, d in shapes:
        if args.roofline:
            out = _run_ncu(name, b, h, n, d, METRIC_SETS["full"], roofline=True)
        else:
            # First, quick overview
            out = _run_ncu(name, b, h, n, d, METRIC_SETS["detailed"])
        all_outputs[name] = out

    # ── Parse and summarize ─────────────────────────────────────────────
    print("\n\n" + "="*80)
    print("  SUMMARY — Key Performance Metrics")
    print("="*80)

    for name, out in all_outputs.items():
        print(f"\n── {name} ──")
        # Extract key metrics from ncu text output
        # These patterns match ncu's CLI text output format
        patterns = {
            "GPU Frequency (MHz)": r"sys__clk\.avg\.value\s+(\S+)\s+(\S+)",
            "DRAM Throughput (%)": r"dram__throughput\.avg\.pct_of_peak_sustained_elapsed\s+(\S+)\s+(\S+)",
            "L2 Throughput (%)": r"lts__throughput\.avg\.pct_of_peak_sustained_elapsed\s+(\S+)\s+(\S+)",
            "SM Throughput (%)": r"sm__throughput\.avg\.pct_of_peak_sustained_elapsed\s+(\S+)\s+(\S+)",
            "Tensor Active (%)": r"sm__inst_executed\.pipe_tensor\.avg\.pct_of_peak_sustained_elapsed\s+(\S+)\s+(\S+)",
            "FP32 Active (%)": r"sm__pipe_fma_cycles_active\.avg\.pct_of_peak_sustained_elapsed\s+(\S+)\s+(\S+)",
            "Warp Occupancy (%)": r"sm__warps_active\.avg\.pct_of_peak_sustained_elapsed\s+(\S+)\s+(\S+)",
            "Registers/Thread": r"launch__registers_per_thread\s+(\S+)\s+(\S+)",
            "Shared Mem (KB)": r"launch__shared_mem_per_block\s+(\S+)\s+(\S+)",
            "Block Limit (SM)": r"launch__blocks_per_sm_limit\s+(\S+)\s+(\S+)",
        }

        for label, pattern in patterns.items():
            m = re.search(pattern, out, re.MULTILINE)
            if m:
                val = m.group(2) if m.lastindex >= 2 else m.group(1)
                print(f"  {label:<30s}: {val}")

        # Stall reasons (key bottleneck indicators)
        stall_patterns = {
            "Memory Stall (%)": r"smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active\.ratio\s+(\S+)\s+(\S+)",
            "Math Stall (%)": r"smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active\.ratio\s+(\S+)\s+(\S+)",
            "Barrier Stall (%)": r"smsp__average_warps_issue_stalled_barrier_per_issue_active\.ratio\s+(\S+)\s+(\S+)",
            "Wait Stall (%)": r"smsp__average_warps_issue_stalled_wait_per_issue_active\.ratio\s+(\S+)\s+(\S+)",
            "Not Selected (%)": r"smsp__average_warps_issue_stalled_not_selected_per_issue_active\.ratio\s+(\S+)\s+(\S+)",
        }

        print(f"  ── Stall Breakdown ──")
        for label, pattern in stall_patterns.items():
            m = re.search(pattern, out, re.MULTILINE)
            if m:
                val = m.group(2) if m.lastindex >= 2 else m.group(1)
                try:
                    pct = float(val) * 100
                    bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                    print(f"  {label:<30s}: {pct:5.1f}%  {bar}")
                except ValueError:
                    print(f"  {label:<30s}: {val}")

        # Duration
        m = re.search(r"gpu__time_duration\.avg\.value\s+(\S+)\s+(\S+)", out, re.MULTILINE)
        if m:
            dur = m.group(2)
            print(f"\n  Kernel Duration: {dur} µs")

    _cleanup()
    print("\nDone. NCU report files saved to /tmp/ncu_*.ncu-rep")


if __name__ == "__main__":
    main()
