#!/usr/bin/env python3
"""
Create a small local VQAv2 subset for offline training (e.g. 15k examples).

It:
- streams `lmms-lab/VQAv2` (so it works even if full non-streaming prep is huge)
- writes images to disk as JPEGs
- writes a JSONL manifest with: question_id, question, answers, image_path

Example:
  source /etc/profile
  .env/bin/python make_vqav2_local_subset.py --out-dir /mnt/data/vqav2_15k --n 15000 --split validation
"""

from __future__ import annotations

import argparse
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def _to_pil(img_obj: Any) -> Image.Image:
    if isinstance(img_obj, Image.Image):
        return img_obj.convert("RGB")
    if isinstance(img_obj, dict):
        if img_obj.get("bytes") is not None:
            return Image.open(BytesIO(img_obj["bytes"])).convert("RGB")
        if img_obj.get("path") is not None:
            return Image.open(img_obj["path"]).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(img_obj)}")


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="Output directory for subset.")
    ap.add_argument("--n", type=int, default=15000, help="Number of examples to export.")
    ap.add_argument("--split", default="validation", help="Split to stream from.")
    ap.add_argument("--jpeg-quality", type=int, default=90, help="JPEG quality for saved images.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    img_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "dataset.jsonl"
    meta_path = out_dir / "meta.json"

    ds = load_dataset("lmms-lab/VQAv2", split=args.split, streaming=True)

    n = max(0, int(args.n))
    jpeg_q = min(100, max(1, int(args.jpeg_quality)))

    wrote = 0
    with open(manifest_path, "w") as f:
        for ex in tqdm(ds, total=n, desc=f"Export VQAv2 {args.split}"):
            if wrote >= n:
                break

            # Basic fields
            qid = ex.get("question_id", wrote)
            q = ex["question"]
            answers = [a["answer"] for a in ex.get("answers", [])]

            # Save image
            pil = _to_pil(ex["image"])
            qid_int = _safe_int(qid, wrote)
            img_rel = Path("images") / f"{qid_int:08d}.jpg"
            img_abs = out_dir / img_rel
            pil.save(img_abs, format="JPEG", quality=jpeg_q, optimize=True)

            row: Dict[str, Any] = {
                "question_id": qid,
                "question": q,
                "answers": answers,
                "image_path": str(img_rel),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            wrote += 1

    with open(meta_path, "w") as f:
        json.dump(
            {
                "source": "lmms-lab/VQAv2",
                "split": args.split,
                "n": wrote,
                "out_dir": str(out_dir),
                "manifest": str(manifest_path),
                "images_dir": str(img_dir),
                "jpeg_quality": jpeg_q,
                "note": "Use dataset.jsonl + images/ for offline training.",
            },
            f,
            indent=2,
            sort_keys=True,
        )

    print(f"[subset] wrote {wrote} examples")
    print(f"[subset] manifest: {manifest_path}")
    print(f"[subset] images:   {img_dir}")
    print(f"[subset] meta:     {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

