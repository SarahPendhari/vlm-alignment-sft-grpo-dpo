import argparse
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor


ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def normalize_answer(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


def extract_answer(text: str) -> str:
    m = ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    # fallback: take last non-empty line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def format_grpo_example(example: Dict[str, Any]) -> Dict[str, Any]:
    question = example["question"]
    answers = [a["answer"] for a in example["answers"]]
    answer = max(set(answers), key=answers.count)
    prompt = f"Question: {question}\n<think> </think>\n<answer>"
    return {
        "image": example["image"],
        "prompt": prompt,
        "target_answer": answer,
    }


@dataclass
class Batch:
    pixel_values: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    target_answers: List[str]


def collate_grpo(batch: List[Dict[str, Any]], processor: AutoProcessor) -> Batch:
    prompts = [b["prompt"] for b in batch]
    images = [b["image"] for b in batch]
    targets = [b["target_answer"] for b in batch]
    enc = processor(text=prompts, images=images, padding=True, return_tensors="pt")
    return Batch(
        pixel_values=enc["pixel_values"],
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        target_answers=targets,
    )


@torch.no_grad()
def generate_group(
    model: torch.nn.Module,
    processor: AutoProcessor,
    batch: Batch,
    num_generations: int,
    max_new_tokens: int,
    temperature: float,
) -> Tuple[List[List[str]], List[List[torch.Tensor]]]:
    """
    Returns:
      - decoded_texts[b][g]
      - sequences[b][g]  (token ids, shape [seq_len])
    """
    device = next(model.parameters()).device
    pixel_values = batch.pixel_values.to(device)
    input_ids = batch.input_ids.to(device)
    attention_mask = batch.attention_mask.to(device)

    decoded: List[List[str]] = [[] for _ in range(input_ids.size(0))]
    seqs: List[List[torch.Tensor]] = [[] for _ in range(input_ids.size(0))]

    for _ in range(num_generations):
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
        texts = processor.tokenizer.batch_decode(out, skip_special_tokens=True)
        for i, t in enumerate(texts):
            decoded[i].append(t)
            seqs[i].append(out[i].detach().cpu())

    return decoded, seqs


def sequence_logprob(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
    seq: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """
    Sum log-prob of generated tokens (excluding prompt tokens).
    """
    device = next(model.parameters()).device
    seq = seq.to(device)
    pixel_values = pixel_values.to(device)

    attn = torch.ones_like(seq, device=device)
    out = model(input_ids=seq.unsqueeze(0), attention_mask=attn.unsqueeze(0), pixel_values=pixel_values.unsqueeze(0))
    logits = out.logits[0]  # [T, V]
    logp = torch.log_softmax(logits, dim=-1)

    # token t predicts token t+1
    next_tokens = seq[1:]
    token_logp = logp[:-1].gather(dim=-1, index=next_tokens.unsqueeze(-1)).squeeze(-1)  # [T-1]

    gen_start = max(prompt_len - 1, 0)
    if gen_start >= token_logp.numel():
        return token_logp.sum() * 0.0
    return token_logp[gen_start:].sum()


def reward_exact_match(generated_text: str, target_answer: str) -> float:
    pred = normalize_answer(extract_answer(generated_text))
    tgt = normalize_answer(target_answer)
    return 1.0 if pred == tgt else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--sft_dir", type=str, default="./outputs/qwen_sft")
    parser.add_argument("--output_dir", type=str, default="./outputs/qwen_grpo")
    parser.add_argument("--data_split", type=str, default="train[:2%]")
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beta_kl", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_name)

    print("Loading base model...")
    base_model = AutoModelForVision2Seq.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
    )

    print("Loading SFT adapter (if present)...")
    try:
        policy_model: torch.nn.Module = PeftModel.from_pretrained(base_model, args.sft_dir)
        policy_model = policy_model.merge_and_unload()
    except Exception:
        policy_model = base_model

    print("Applying LoRA for GRPO...")
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    policy_model = get_peft_model(policy_model, lora_config)
    policy_model.print_trainable_parameters()

    print("Creating frozen reference model (SFT)...")
    ref_model = AutoModelForVision2Seq.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
    )
    try:
        ref_model = PeftModel.from_pretrained(ref_model, args.sft_dir)
        ref_model = ref_model.merge_and_unload()
    except Exception:
        pass
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_model.to(device)
    ref_model.to(device)

    print("Loading dataset...")
    ds = load_dataset("HuggingFaceM4/VQAv2", split=args.data_split)
    ds = ds.map(format_grpo_example, remove_columns=ds.column_names)

    dl = DataLoader(
        ds,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_grpo(b, processor),
    )

    optimizer = AdamW([p for p in policy_model.parameters() if p.requires_grad], lr=args.lr)
    policy_model.train()

    running_loss = 0.0
    step = 0
    accum = 0

    it = iter(dl)
    while step < args.num_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)

        with torch.no_grad():
            decoded, seqs = generate_group(
                model=policy_model,
                processor=processor,
                batch=batch,
                num_generations=args.num_generations,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

        # compute losses
        bsz = len(batch.target_answers)
        device = next(policy_model.parameters()).device
        pixel_values = batch.pixel_values.to(device)
        prompt_lens = batch.attention_mask.sum(dim=-1).tolist()

        losses: List[torch.Tensor] = []
        rewards_all: List[float] = []

        for i in range(bsz):
            rewards = torch.tensor(
                [reward_exact_match(decoded[i][g], batch.target_answers[i]) for g in range(args.num_generations)],
                device=device,
                dtype=torch.float32,
            )
            rewards_all.extend([float(r.item()) for r in rewards])
            baseline = rewards.mean()
            adv = rewards - baseline  # group-relative baseline

            for g in range(args.num_generations):
                seq = seqs[i][g].to(device)
                logp_pi = sequence_logprob(policy_model, pixel_values[i], seq, prompt_len=int(prompt_lens[i]))
                with torch.no_grad():
                    logp_ref = sequence_logprob(ref_model, pixel_values[i], seq, prompt_len=int(prompt_lens[i]))

                kl = (logp_pi - logp_ref)
                loss = -(adv[g].detach() * logp_pi) + args.beta_kl * kl
                losses.append(loss)

        loss = torch.stack(losses).mean() / args.grad_accum_steps
        loss.backward()
        running_loss += float(loss.item())
        accum += 1

        if accum >= args.grad_accum_steps:
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            accum = 0

            avg_reward = sum(rewards_all) / max(len(rewards_all), 1)
            print(
                f"step={step:04d} loss={running_loss:.4f} avg_reward={avg_reward:.3f} "
                f"(num_gen={args.num_generations}, beta_kl={args.beta_kl})"
            )
            running_loss = 0.0

    print("Saving GRPO adapter...")
    policy_model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("GRPO training complete!")


if __name__ == "__main__":
    main()
