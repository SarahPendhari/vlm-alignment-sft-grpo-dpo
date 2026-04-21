import os
import math
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import LoraConfig, PeftModel, get_peft_model

from src.metrics import vqa_accuracy, extract_answer


@dataclass
class GrpoBatch:
    input_ids: torch.Tensor          # (G, T)
    attention_mask: torch.Tensor     # (G, T)
    prompt_len: int


def build_prompt_messages(question: str, instruction: str | None) -> List[Dict[str, Any]]:
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


def compute_rewards(
    raw_generations: List[str],
    answers: List[str],
) -> torch.Tensor:
    """
    Reward = VQA soft accuracy only (no format constraint, no length penalty).
    - VQA accuracy uses `extract_answer` internally (see src/metrics.py).
    """
    rewards: List[float] = []
    for gen in raw_generations:
        rewards.append(float(vqa_accuracy(gen, answers)))

    return torch.tensor(rewards, dtype=torch.float32)


def group_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute per-group relative advantages (normalize within group).
    """
    r = rewards
    r_mean = r.mean()
    r_std = r.std(unbiased=False)
    return (r - r_mean) / (r_std + eps)


def logprobs_of_generated_tokens(
    model: Qwen2VLForConditionalGeneration,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """
    Returns sum log-prob of generated tokens for each sequence.

    input_ids include prompt+generated.
    We compute log p(x_{prompt_len:} | x_{<prompt_len}) by scoring next-token logits.
    """
    # logits: (B, T, V)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits

    # next-token prediction: logits[:, t-1] predicts token at t
    # We want positions t in [prompt_len, T-1] (generated tokens).
    # So we use logits positions [prompt_len-1, T-2] and target tokens [prompt_len, T-1]
    B, T, V = logits.shape
    if prompt_len >= T:
        raise ValueError(f"prompt_len={prompt_len} must be < sequence length T={T}")

    logits_slice = logits[:, prompt_len - 1 : T - 1, :]  # (B, gen_T, V)
    targets = input_ids[:, prompt_len:T]                 # (B, gen_T)

    logp = torch.log_softmax(logits_slice, dim=-1)
    token_logp = torch.gather(logp, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # (B, gen_T)

    # Mask out padding tokens if any exist in generated region
    gen_attn = attention_mask[:, prompt_len:T].to(token_logp.dtype)
    token_logp = token_logp * gen_attn

    return token_logp.sum(dim=-1)  # (B,)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GRPO after SFT for Qwen2-VL (LoRA).")

    # Model / paths
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--sft_model_dir", type=str, default="./outputs/qwen_sft")
    parser.add_argument("--output_dir", type=str, default="./outputs/qwen_grpo")
    parser.add_argument("--cache_dir", type=str, default=os.environ.get("HF_HOME", "./cache"))

    # Data
    parser.add_argument("--train_subset_size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=55, help="Seed for shuffling GRPO subset (use != SFT seed).")

    # Generation / rollouts
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)

    # Optimization
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # GRPO / PPO-style
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--beta_kl", type=float, default=0.02)

    # Reward shaping
    # (disabled by design for this run; keep reward = VQA soft accuracy)

    # Prompt format
    parser.add_argument(
        "--instruction",
        type=str,
        default="REPONDS EN UN SEUL MOT SANS BALISES.",
        help="Optional instruction prepended to the user question. "
             "Example: 'REPONDS EN UN SEUL MOT SANS BALISES.'",
    )

    parser.add_argument("--log_every", type=int, default=25)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # =========================
    # Load base model
    # =========================
    print("1) Loading base model...")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        cache_dir=args.cache_dir,
    )

    # =========================
    # Load SFT adapter and merge -> this becomes the fixed reference policy
    # =========================
    print("2) Loading SFT LoRA adapter and merging into base...")
    sft_merged = PeftModel.from_pretrained(base_model, args.sft_model_dir)
    sft_merged = sft_merged.merge_and_unload()

    # Processor from SFT dir (ensures consistent chat_template/tokenizer)
    processor = AutoProcessor.from_pretrained(args.sft_model_dir)

    # =========================
    # Apply a NEW LoRA adapter for GRPO training
    #   - ref policy: SFT merged base (adapter disabled)
    #   - train policy: SFT merged base + trainable GRPO adapter
    # =========================
    print("3) Applying new LoRA adapter for GRPO...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(sft_merged, lora_config)
    model.print_trainable_parameters()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # =========================
    # Load dataset subset
    # =========================
    print("4) Loading VQAv2 train split...")
    ds = load_dataset(
        "HuggingFaceM4/VQAv2",
        split="train",
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )

    subset_size = min(args.train_subset_size, len(ds))
    ds = ds.shuffle(seed=args.seed).select(range(subset_size))
    print(f"GRPO subset size: {len(ds)}")

    # =========================
    # Training loop
    # =========================
    instruction = args.instruction.strip() or None
    scaler = None  # placeholder if you later want torch.cuda.amp

    global_step = 0
    opt_step = 0

    running_reward = 0.0
    running_kl = 0.0
    running_loss = 0.0
    running_vqa = 0.0

    total_iters = args.num_epochs * len(ds)
    pbar = tqdm(total=total_iters, desc="GRPO training (iters)")

    for epoch in range(args.num_epochs):
        for ex in ds:
            question = ex["question"]
            answers = [a["answer"] for a in ex["answers"]]
            image = ex["image"]

            # Build prompt
            messages = build_prompt_messages(question, instruction)
            text_prompt = make_text_prompt(processor, messages)

            inputs = processor(
                text=[text_prompt],
                images=[image],
                padding=True,
                return_tensors="pt",
            )

            # Move to model device (peft model may be sharded across GPUs)
            # For generate(), passing CPU tensors usually works with device_map="auto",
            # but moving explicitly is safer when not sharded.
            if torch.cuda.is_available() and hasattr(model, "device"):
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

            prompt_len = inputs["input_ids"].shape[1]

            # Generate G samples in ONE call (usually avoids re-encoding vision G times)
            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_return_sequences=args.group_size,
                )

            # Build attention mask for full sequences
            # input attention mask is shape (1, prompt_len); repeat for G and then append 1s for generated tokens.
            # Safer: re-tokenize by building full attention mask as non-pad positions.
            # Here, generated_ids are not padded (each row may have different lengths if EOS happens early).
            # We'll pad to max length within the group to score in one forward pass.
            seq_lens = torch.tensor([seq.shape[0] for seq in gen_ids], device=gen_ids.device)
            max_len = int(seq_lens.max().item())

            # Pad sequences
            pad_id = processor.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = processor.tokenizer.eos_token_id

            padded = torch.full((args.group_size, max_len), pad_id, dtype=gen_ids.dtype, device=gen_ids.device)
            attn = torch.zeros((args.group_size, max_len), dtype=torch.long, device=gen_ids.device)
            for i, seq in enumerate(gen_ids):
                L = seq.shape[0]
                padded[i, :L] = seq
                attn[i, :L] = 1

            # Decode only the generated part for reward
            # Trim prompt from each sequence using prompt_len of the original input.
            trimmed = []
            for i in range(args.group_size):
                out_ids = padded[i]
                # find effective length
                L = int(attn[i].sum().item())
                out_ids = out_ids[:L]
                gen_part = out_ids[prompt_len:]
                trimmed.append(gen_part)

            raw_generations = processor.batch_decode(trimmed, skip_special_tokens=True)

            # Compute rewards and advantages
            rewards = compute_rewards(
                raw_generations=raw_generations,
                answers=answers,
            ).to(padded.device)

            advantages = group_advantages(rewards)  # (G,)

            # Metrics for logging
            with torch.no_grad():
                vqa_scores = torch.tensor([vqa_accuracy(g, answers) for g in raw_generations], device=padded.device)

            # Compute logprobs under current policy (with grad)
            logp_new = logprobs_of_generated_tokens(model, padded, attn, prompt_len=prompt_len)  # (G,)
            logp_old = logp_new.detach()

            # Reference logprobs (SFT policy): disable adapter on current peft model
            with torch.no_grad():
                with model.disable_adapter():
                    logp_ref = logprobs_of_generated_tokens(model, padded, attn, prompt_len=prompt_len)  # (G,)

            # PPO-style ratio
            ratio = torch.exp(logp_new - logp_old)  # starts at 1.0 but has gradient
            clipped = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps)
            pg = torch.minimum(ratio * advantages, clipped * advantages)
            policy_loss = -pg.mean()

            # KL penalty (approx): KL(new || ref) over sampled actions ~ (logp_new - logp_ref)
            kl = (logp_new - logp_ref)
            kl_mean = kl.mean()
            kl_loss = args.beta_kl * kl_mean

            loss = policy_loss + kl_loss
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            # Gradient accumulation
            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1

            # Update running logs
            running_reward += float(rewards.mean().item())
            running_kl += float(kl_mean.item())
            running_loss += float(loss.item() * args.gradient_accumulation_steps)
            running_vqa += float(vqa_scores.mean().item())

            if global_step % args.log_every == 0:
                denom = float(args.log_every)
                print(
                    f"[epoch {epoch+1}/{args.num_epochs} | iter {global_step}] "
                    f"loss={running_loss/denom:.4f} "
                    f"reward={running_reward/denom:.4f} "
                    f"vqa={running_vqa/denom:.4f} "
                    f"kl={running_kl/denom:.4f} "
                    f"opt_steps={opt_step}"
                )
                running_reward = running_kl = running_loss = running_vqa = 0.0

            pbar.update(1)

        # Save adapter checkpoint per epoch
        ckpt_dir = os.path.join(args.output_dir, f"epoch_{epoch+1}")
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        processor.save_pretrained(ckpt_dir)
        print(f"Saved GRPO adapter checkpoint to: {ckpt_dir}")

    pbar.close()

    # Save final
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"GRPO training complete. Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

