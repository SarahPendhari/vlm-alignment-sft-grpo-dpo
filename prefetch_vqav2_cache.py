#!/usr/bin/env python3
"""
Prefetch VQAv2 into the local HF cache, but stop early if disk
free space drops below a threshold.

This script is meant to run in parallel with train_GRPO2.py, without
modifying it. It only pre-downloads/caches; you can safely stop it
without affecting training, *as long as training is not relying on it
for current reads*.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import sys
import time
from pathlib import Path


_STOP = False


def _handle_sig(_signum: int, _frame) -> None:  # type: ignore[no-untyped-def]
    global _STOP
    _STOP = True


def free_gib(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-home", required=True, help="HF_HOME dir to use (same as training).")
    ap.add_argument("--min-free-gib", type=float, default=5.0, help="Stop if free space < this.")
    ap.add_argument("--split", default="validation", help="Dataset split to cache.")
    ap.add_argument("--sleep-s", type=float, default=2.0, help="Polling interval for stop flag.")
    args = ap.parse_args()

    hf_home = Path(args.hf_home).expanduser().resolve()
    hf_home.mkdir(parents=True, exist_ok=True)

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    free0 = free_gib(hf_home)
    print(f"[prefetch] HF_HOME={hf_home}")
    print(f"[prefetch] free space: {free0:.2f} GiB (min allowed {args.min_free_gib:.2f} GiB)")
    if free0 < args.min_free_gib:
        print("[prefetch] Not enough free space. Exiting.")
        return 2

    # Make sure both hub and datasets caches live under the same root.
    os.environ["HF_HOME"] = str(hf_home)
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "hub"))

    # Import after env vars, so caches are honored.
    from datasets import load_dataset  # noqa: WPS433

    if _STOP:
        print("[prefetch] Stopped before starting.")
        return 130

    # Start the download/caching. This may take time and disk.
    print(f"[prefetch] Loading lmms-lab/VQAv2 split={args.split} (streaming=False) ...")
    ds = load_dataset("lmms-lab/VQAv2", split=args.split, streaming=False)
    print(f"[prefetch] Loaded dataset object: {ds}")

    # Touch a small number of items to force some materialization, but don't fully iterate.
    # If the dataset is fully downloaded by load_dataset already, this is cheap.
    try:
        n_touch = min(50, len(ds)) if hasattr(ds, "__len__") else 50
    except Exception:
        n_touch = 50
    print(f"[prefetch] Touching first {n_touch} examples to warm cache ...")
    for i in range(n_touch):
        if _STOP:
            print("[prefetch] Stop requested. Exiting.")
            return 130
        # Accessing one field is enough to trigger lazy downloads in some builders.
        _ = ds[i]
        if i % 10 == 0:
            free_now = free_gib(hf_home)
            if free_now < args.min_free_gib:
                print(f"[prefetch] Free space low ({free_now:.2f} GiB). Stopping prefetch.")
                return 3
        time.sleep(args.sleep_s * 0.0)

    free1 = free_gib(hf_home)
    print(f"[prefetch] Done. free space now: {free1:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

