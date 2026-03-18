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

## Training (SFT)

```bash
python scripts/train_sft.py
```

## Structure

* `src/` → core logic
* `scripts/` → training scripts
* `configs/` → experiment configs

## Status

In progress (SFT baseline)
