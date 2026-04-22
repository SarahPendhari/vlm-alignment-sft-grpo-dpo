"""
Self-contained GRPO training loop script (extracted from GRPO_tests.ipynb).

Goal: be able to SSH into a VM and run:
  python train_GRPO2.py

It will:
- load Qwen2-VL base + processor
- load SFT LoRA adapter and merge it into the base (reference policy)
- attach a fresh LoRA adapter for GRPO training (trainable policy)
- run a GRPO/PPO-style loop on streaming VQAv2 (`lmms-lab/VQAv2`, validation split)
- periodically save checkpoints and append metrics to a TSV file

Configuration is controlled via env vars (see `env_*` functions below).
"""

from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from peft import LoraConfig, PeftModel, get_peft_model

from src.metrics import vqa_accuracy


def env_int(name: str, default: int) -> int:
    v = os.environ.get(name, None)
    return default if v is None or v == "" else int(v)


def env_float(name: str, default: float) -> float:
    v = os.environ.get(name, None)
    return default if v is None or v == "" else float(v)


def env_str(name: str, default: str) -> str:
    v = os.environ.get(name, None)
    return default if v is None or v == "" else v


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_pil(img_obj: Any) -> Image.Image:
    if isinstance(img_obj, Image.Image):
        return img_obj.convert("RGB")
    if isinstance(img_obj, dict):
        if img_obj.get("bytes") is not None:
            return Image.open(BytesIO(img_obj["bytes"])).convert("RGB")
        if img_obj.get("path") is not None:
            return Image.open(img_obj["path"]).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(img_obj)}")


# -----------------------
# GRPO helpers (inlined)
# -----------------------
def build_prompt_messages(question: str, instruction: Optional[str]) -> List[Dict[str, Any]]:
    user_text = f"Question: {question}"
    if instruction:
        user_text = instruction.strip() + "\n" + user_text
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        }
    ]


def make_text_prompt(processor: AutoProcessor, messages: List[Dict[str, Any]]) -> str:
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def compute_rewards(raw_generations: List[str], answers: List[str]) -> torch.Tensor:
    rewards: List[float] = []
    for gen in raw_generations:
        rewards.append(float(vqa_accuracy(gen, answers)))
    return torch.tensor(rewards, dtype=torch.float32)


def group_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r_mean = rewards.mean()
    r_std = rewards.std(unbiased=False)
    return (rewards - r_mean) / (r_std + eps)


def logprobs_of_generated_tokens(
    model: Qwen2VLForConditionalGeneration,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """
    Returns sum log-prob of generated tokens for each sequence.
    `input_ids` include prompt+generated.
    We score next-token logits and sum log p(x_t) over generated region.
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits  # (B, T, V)
    B, T, _V = logits.shape
    if prompt_len >= T:
        raise ValueError(f"prompt_len={prompt_len} must be < sequence length T={T}")

    logits_slice = logits[:, prompt_len - 1 : T - 1, :]  # (B, gen_T, V)
    targets = input_ids[:, prompt_len:T]  # (B, gen_T)

    logp = torch.log_softmax(logits_slice, dim=-1)
    token_logp = torch.gather(logp, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # (B, gen_T)

    gen_attn = attention_mask[:, prompt_len:T].to(token_logp.dtype)
    token_logp = token_logp * gen_attn
    return token_logp.sum(dim=-1)  # (B,)


def maybe_remap_lora_safetensors(adapter_dir: Path) -> None:
    """
    Notebook had a one-off remap where checkpoint contained `adapter_model2.safetensors`
    with keys like `...language_model.layers...`, while PEFT expects `adapter_model.safetensors`
    with `...layers...`.

    If adapter_model.safetensors is missing but adapter_model2.safetensors exists, remap+write.
    """
    dst = adapter_dir / "adapter_model.safetensors"
    src = adapter_dir / "adapter_model2.safetensors"
    if dst.exists() or not src.exists():
        return

    from safetensors.torch import load_file, save_file

    sd = load_file(str(src))
    remapped = {k.replace(".language_model.layers.", ".layers."): v for k, v in sd.items()}
    save_file(remapped, str(dst))


@dataclass(frozen=True)
class Config:
    project_dir: Path
    model_name: str
    cache_dir: str
    sft_adapter_dir: Path
    grpo_adapter_dir: Path
    instruction: Optional[str]
    seed: int

    # rollout/gen
    group_size: int
    max_new_tokens: int
    temperature: float
    top_p: float

    # optim
    learning_rate: float
    gradient_accumulation_steps: int
    max_grad_norm: float

    # GRPO
    beta_kl: float

    # runtime
    smoke_n: int
    log_every: int
    save_every: int
    log_table_every: int
    out_dir: Path
    log_path: Path


def write_run_config(cfg: Config) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "project_dir": str(cfg.project_dir),
        "model_name": cfg.model_name,
        "cache_dir": cfg.cache_dir,
        "sft_adapter_dir": str(cfg.sft_adapter_dir),
        "grpo_adapter_dir": str(cfg.grpo_adapter_dir),
        "instruction": cfg.instruction,
        "seed": cfg.seed,
        "group_size": cfg.group_size,
        "max_new_tokens": cfg.max_new_tokens,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "learning_rate": cfg.learning_rate,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "max_grad_norm": cfg.max_grad_norm,
        "beta_kl": cfg.beta_kl,
        "smoke_n": cfg.smoke_n,
        "log_every": cfg.log_every,
        "save_every": cfg.save_every,
        "log_table_every": cfg.log_table_every,
        "out_dir": str(cfg.out_dir),
        "log_path": str(cfg.log_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    path = cfg.out_dir / "run_config.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print("RUN_CONFIG:", str(path))


def load_config() -> Config:
    project_dir = Path(env_str("PROJECT_DIR", str(Path(__file__).resolve().parent))).resolve()

    model_name = env_str("MODEL_NAME", "Qwen/Qwen2-VL-2B-Instruct")
    cache_dir = env_str("HF_HOME", str(project_dir / "cache"))

    # Adapters
    sft_adapter_dir = Path(
        env_str(
            "SFT_ADAPTER_DIR",
            str(project_dir / "outputs" / "qwen_sft" ),
        )
    ).resolve()
    grpo_adapter_dir = Path(env_str("GRPO_ADAPTER_DIR", str(project_dir / "outputs" / "qwen_grpo"))).resolve()

    instruction_raw = os.environ.get("INSTRUCTION", "ONE WORD WITHOUT BALISE, NO <answer> OR <thinking> ")
    instruction = instruction_raw.strip() if isinstance(instruction_raw, str) and instruction_raw.strip() else None

    # Output dir: if user doesn't specify, make it unique per run to avoid overwriting.
    out_dir_env = os.environ.get("GRPO_SMOKE_OUT", "").strip()
    if out_dir_env:
        out_dir = Path(out_dir_env).expanduser().resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = (project_dir / "outputs" / "grpo_runs" / f"run_{ts}").resolve()

    log_path_env = os.environ.get("GRPO_LOG_TABLE_PATH", "").strip()
    if log_path_env:
        log_path = Path(log_path_env).expanduser().resolve()
    else:
        log_path = (out_dir / "grpo_metrics.tsv").resolve()

    return Config(
        project_dir=project_dir,
        model_name=model_name,
        cache_dir=cache_dir,
        sft_adapter_dir=sft_adapter_dir,
        grpo_adapter_dir=grpo_adapter_dir,
        instruction=instruction,
        seed=env_int("SEED", 0),
        smoke_n=env_int("GRPO_SMOKE_N", 15000),
        group_size=env_int("GRPO_GROUP_SIZE", 8),
        max_new_tokens=env_int("GRPO_MAX_NEW_TOKENS", 32),
        temperature=env_float("GRPO_TEMPERATURE", 0.8),
        top_p=env_float("GRPO_TOP_P", 0.9),
        learning_rate=env_float("GRPO_LR", 1e-4),
        gradient_accumulation_steps=env_int("GRPO_GRAD_ACCUM", 8),
        max_grad_norm=env_float("GRPO_MAX_GRAD_NORM", 1.0),
        beta_kl=env_float("GRPO_BETA_KL", 0.05),
        log_every=env_int("GRPO_LOG_EVERY", 2),
        save_every=env_int("GRPO_SAVE_EVERY", 500),
        log_table_every=env_int("GRPO_LOG_TABLE_EVERY", 100),
        out_dir=out_dir,
        log_path=log_path,
    )


def ensure_metrics_tsv(log_path: Path) -> List[str]:
    header = [
        "step",
        "opt_step",
        "n",
        # reward stats
        "reward_mean",
        "reward_std",
        "reward_min",
        "reward_max",
        # approx KL stats
        "kl_mean",
        "kl_std",
        # losses
        "policy_loss_mean",
        "total_loss_mean",
        # advantages
        "adv_mean",
        "adv_std",
        "adv_min",
        "adv_max",
        # generation proxies
        "gen_len_mean",
        "unique_frac_mean",
        "nll_per_token_mean",
    ]
    if not log_path.exists():
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(header)
    return header


def main() -> None:
    cfg = load_config()
    seed_everything(cfg.seed)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(cfg)
    log_header = ensure_metrics_tsv(cfg.log_path)

    device_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print("PROJECT_DIR:", str(cfg.project_dir))
    print("MODEL_NAME:", cfg.model_name)
    print("CACHE_DIR:", cfg.cache_dir)
    print("SFT_ADAPTER_DIR:", str(cfg.sft_adapter_dir), "exists=", cfg.sft_adapter_dir.exists())
    print("GRPO_ADAPTER_DIR:", str(cfg.grpo_adapter_dir), "exists=", cfg.grpo_adapter_dir.exists())
    print("OUT:", str(cfg.out_dir))
    print("LOG:", str(cfg.log_path))

    processor = AutoProcessor.from_pretrained(cfg.model_name, cache_dir=cfg.cache_dir)

    # Some checkpoints in this project used adapter_model2.safetensors + key remap.
    if cfg.sft_adapter_dir.exists():
        maybe_remap_lora_safetensors(cfg.sft_adapter_dir)

    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        cfg.model_name,
        torch_dtype=device_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        cache_dir=cfg.cache_dir,
    )

    print("Loading SFT adapter and merging into base (reference policy)...")
    sft_merged = PeftModel.from_pretrained(base_model, str(cfg.sft_adapter_dir))
    sft_merged = sft_merged.merge_and_unload()

    try:
        processor_sft = AutoProcessor.from_pretrained(str(cfg.sft_adapter_dir))
        print("Loaded processor from SFT adapter dir.")
    except Exception as e:
        processor_sft = processor
        print("Falling back to base processor (could not load from SFT dir):", repr(e))

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(sft_merged, lora_config)
    model.train()
    # Gradient checkpointing disabled (uses more VRAM, but avoids checkpoint warnings).
    model.config.use_cache = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"PARAMS trainable/total: {trainable}/{total}")
    if trainable == 0:
        raise RuntimeError("No trainable parameters found (all params have requires_grad=False).")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    stream_ds = load_dataset(
        "lmms-lab/VQAv2",
        split="validation",
        streaming=True,
    )

    global_step = 0
    opt_step = 0

    # Rolling window for table logging
    win_rewards: List[float] = []
    win_kls: List[float] = []
    win_policy_losses: List[float] = []
    win_total_losses: List[float] = []
    win_adv: List[torch.Tensor] = []
    win_gen_lens: List[float] = []
    win_unique_frac: List[float] = []
    win_nll_tok: List[float] = []

    running_reward = 0.0
    running_kl = 0.0
    running_loss = 0.0
    running_vqa = 0.0

    pbar = tqdm(total=cfg.smoke_n, desc="GRPO train (script)")

    for ex in islice(stream_ds, cfg.smoke_n):
        question = ex["question"]
        answers = [a["answer"] for a in ex["answers"]]
        image = _to_pil(ex["image"])

        messages = build_prompt_messages(question, cfg.instruction)
        text_prompt = make_text_prompt(processor_sft, messages)
        inputs = processor_sft(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        dev = next(model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=True,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                num_return_sequences=cfg.group_size,
            )

        seq_lens = torch.tensor([seq.shape[0] for seq in gen_ids], device=gen_ids.device)
        max_len = int(seq_lens.max().item())

        pad_id = processor_sft.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = processor_sft.tokenizer.eos_token_id

        padded = torch.full((cfg.group_size, max_len), pad_id, dtype=gen_ids.dtype, device=gen_ids.device)
        attn = torch.zeros((cfg.group_size, max_len), dtype=torch.long, device=gen_ids.device)
        for i, seq in enumerate(gen_ids):
            L = seq.shape[0]
            padded[i, :L] = seq
            attn[i, :L] = 1

        trimmed: List[torch.Tensor] = []
        for i in range(cfg.group_size):
            out_ids = padded[i]
            L = int(attn[i].sum().item())
            out_ids = out_ids[:L]
            gen_part = out_ids[prompt_len:]
            trimmed.append(gen_part)

        raw_generations = processor_sft.batch_decode(trimmed, skip_special_tokens=True)

        rewards = compute_rewards(raw_generations=raw_generations, answers=answers).to(padded.device)
        advantages = group_advantages(rewards)

        with torch.no_grad():
            vqa_scores = torch.tensor([vqa_accuracy(g, answers) for g in raw_generations], device=padded.device)

        logp_new = logprobs_of_generated_tokens(model, padded, attn, prompt_len=prompt_len)

        with torch.no_grad():
            with model.disable_adapter():
                logp_ref = logprobs_of_generated_tokens(model, padded, attn, prompt_len=prompt_len)

        # GRPO (no PPO): policy gradient weighted by group-normalized advantages.
        policy_loss = -(advantages * logp_new).mean()

        kl = logp_new - logp_ref
        kl_mean = kl.mean()
        kl_loss = cfg.beta_kl * kl_mean

        loss = policy_loss + kl_loss

        # --- diagnostics to log ---
        with torch.no_grad():
            gen_lens = attn[:, prompt_len:].sum(dim=-1).to(torch.float32).clamp_min(1.0)
            gen_len_mean = gen_lens.mean()
            uniq = len(set([s.strip() for s in raw_generations]))
            unique_frac = float(uniq) / float(len(raw_generations))
            nll_per_tok = (-logp_new / gen_lens).mean()

        (loss / cfg.gradient_accumulation_steps).backward()

        global_step += 1
        if global_step % cfg.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            opt_step += 1

        running_reward += float(rewards.mean().item())
        running_kl += float(kl_mean.item())
        running_loss += float(loss.item())
        running_vqa += float(vqa_scores.mean().item())

        # rolling window table
        win_rewards.append(float(rewards.mean().item()))
        win_kls.append(float(kl_mean.item()))
        win_policy_losses.append(float(policy_loss.item()))
        win_total_losses.append(float((policy_loss + kl_loss).item()))
        win_adv.append(advantages.detach().float().cpu())
        win_gen_lens.append(float(gen_len_mean.item()))
        win_unique_frac.append(float(unique_frac))
        win_nll_tok.append(float(nll_per_tok.item()))

        if cfg.log_table_every > 0 and global_step % cfg.log_table_every == 0:
            r = torch.tensor(win_rewards)
            k = torch.tensor(win_kls)
            adv_all = torch.cat(win_adv, dim=0)
            row = {
                "step": global_step,
                "opt_step": opt_step,
                "n": len(win_rewards),
                "reward_mean": float(r.mean().item()),
                "reward_std": float(r.std(unbiased=False).item()) if len(win_rewards) > 1 else 0.0,
                "reward_min": float(r.min().item()),
                "reward_max": float(r.max().item()),
                "kl_mean": float(k.mean().item()),
                "kl_std": float(k.std(unbiased=False).item()) if len(win_kls) > 1 else 0.0,
                "policy_loss_mean": float(sum(win_policy_losses) / len(win_policy_losses)),
                "total_loss_mean": float(sum(win_total_losses) / len(win_total_losses)),
                "adv_mean": float(adv_all.mean().item()),
                "adv_std": float(adv_all.std(unbiased=False).item()) if adv_all.numel() > 1 else 0.0,
                "adv_min": float(adv_all.min().item()),
                "adv_max": float(adv_all.max().item()),
                "gen_len_mean": float(sum(win_gen_lens) / len(win_gen_lens)),
                "unique_frac_mean": float(sum(win_unique_frac) / len(win_unique_frac)),
                "nll_per_token_mean": float(sum(win_nll_tok) / len(win_nll_tok)),
            }
            with open(cfg.log_path, "a", newline="") as f:
                w = csv.writer(f, delimiter="\t")
                w.writerow([row[k] for k in log_header])

            win_rewards.clear()
            win_kls.clear()
            win_policy_losses.clear()
            win_total_losses.clear()
            win_adv.clear()
            win_gen_lens.clear()
            win_unique_frac.clear()
            win_nll_tok.clear()

        if global_step % cfg.log_every == 0:
            d = float(cfg.log_every)
            print(
                f"[iter {global_step}] loss={running_loss/d:.4f} reward={running_reward/d:.4f} "
                f"vqa={running_vqa/d:.4f} kl={running_kl/d:.4f} opt_steps={opt_step}"
            )
            running_reward = 0.0
            running_kl = 0.0
            running_loss = 0.0
            running_vqa = 0.0

        pbar.update(1)

        if cfg.save_every > 0 and global_step > 0 and global_step % cfg.save_every == 0:
            ck = cfg.out_dir / f"checkpoint-{global_step}"
            ck.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ck)
            processor_sft.save_pretrained(ck)
            print(f"Checkpoint: {ck}")

    pbar.close()
    print("GRPO run done. opt_steps:", opt_step, "examples:", cfg.smoke_n)
    print("Metrics TSV:", str(cfg.log_path))

    model.save_pretrained(cfg.out_dir)
    processor_sft.save_pretrained(cfg.out_dir)
    print("Final adapter saved under:", str(cfg.out_dir))


if __name__ == "__main__":
    main()

