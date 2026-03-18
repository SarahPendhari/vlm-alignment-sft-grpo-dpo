# VLM Alignment with GRPO/DPO

This project investigates post-training alignment strategies for vision-language models, focusing on:

* Supervised Fine-Tuning (SFT)
* Direct Preference Optimization (DPO)
* Group Relative Policy Optimization (GRPO)

## Goal

Understand when and why GRPO outperforms SFT and DPO on visual reasoning tasks.

## Setup

```bash
pip install -r requirements.txt
```

Note: this codebase expects a working PyTorch install. On macOS, PyTorch wheels may not be available for very recent Python versions (e.g. Python 3.13). If you hit a Torch `dlopen` / missing `libtorch_cpu.dylib` error, use Python 3.10–3.12 in a fresh environment and reinstall dependencies.

## Training (SFT)

```bash
python scripts/train_sft.py
```

## Training (GRPO, on top of SFT)

1) Run SFT first (or point `--sft_dir` to an existing SFT checkpoint).

```bash
python scripts/train_grpo.py --sft_dir ./outputs/qwen_sft --output_dir ./outputs/qwen_grpo
```

## Structure

* `src/` → core logic
* `scripts/` → training scripts
* `configs/` → experiment configs

## Status

In progress (SFT baseline)
