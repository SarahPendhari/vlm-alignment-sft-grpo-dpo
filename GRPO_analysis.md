## Repo context (SFT already done)

### Dataset: `HuggingFaceM4/VQAv2` (VQAv2)
- **Train**: 443,757 questions, 82,783 images, 4,437,570 answers (10 annotations per question)
- **Validation**: 214,354 questions, 40,504 images, 2,143,540 answers
- **Test**: 447,793 questions, 81,434 images
- **Testdev**: 107,394 questions, 36,807 images (not explicitly listed in the card, but stats are provided)

### Model and LoRA (SFT)
- **Base model**: `MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"`
- **SFT data** (`train_sft.py`): the `train` split is shuffled (`seed=42`) and we keep **20%** → **≈ 88,751** questions (0.2 × 443,757). Validation uses **10%** of the `validation` split.
- **LoRA config** (from `train_sft.py`):
  - `r=16`
  - `lora_alpha=32`
  - `target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`
  - `lora_dropout=0.05`
  - `bias="none"`
  - `task_type="CAUSAL_LM"`

---

## Questions → decisions (GRPO after SFT, <24h and <$100)

### 1) Should GRPO be trained on exactly the same data as SFT?
- **Short answer**: **no, not necessarily “exactly”**, but **staying in-domain** helps a lot.
- **Practical recommendation here**:
  - Run GRPO on the `train` split (like SFT), but ideally on a **different subset** than the one used for SFT (or at least a different shuffle/seed) to avoid over-optimizing the same examples.
  - If you do not have the budget/time for careful splitting: GRPO on a **small subset** of `train` is still OK (standard RL practice to start small).

### 2) For validation, should we use the same data as SFT?
- **Yes**: to compare “SFT vs SFT+GRPO” fairly, you want the **same evaluation procedure** on a **fixed validation set**.
- **Recommendation**:
  - Keep the `validation` split for final evaluation.
  - Use a **fixed subset** (seed + fixed indices) to track progress during GRPO (otherwise you add measurement noise).

### 3) Which AWS GPU should I pick (budget $100 / 24h)?
- GRPO cost is dominated by **generation** (rollouts) + some LoRA backward.
- **Default recommendation**: **1× A10G 24GB** (AWS `g5` family) = great price/perf ratio for 2B + vision + LoRA.
  - In most regions, 24h on-demand A10G typically fits within a $100 budget.
- Alternatives:
  - **L4 (g6 family)**: often good $/perf, depends on region availability.
  - **T4**: possible but often too slow for GRPO (you burn your budget in hours without enough updates).
  - **A100**: great but can exceed budget on-demand (unless Spot / favorable pricing).

### 4) Are images explicitly resized in this codebase?
- **No explicit resize** in the repo (no `Resize(...)`, no manual transforms).
- In SFT, the call is `processor(text=..., images=..., padding=True, return_tensors="pt")`:
  - Qwen2-VL’s `AutoProcessor` typically performs the vision preprocessing (including resize/normalization) as required by the model.
- **Conclusion**: resize is very likely done **implicitly by the processor**, not by custom code.

---

## Recommended hyperparameters (goal: “it learns” without blowing up compute)

### Generation parameters (rollouts)
Goal: VQA answers are short → limit generated tokens, otherwise cost + verbosity.

- **`max_new_tokens`**: **32** (if you need to reduce further: 16)
  - Why: cost ∝ generated tokens; VQAv2 rarely needs > 1–5 words.
- **Rollout / group size `G`**: **4** (minimum viable); go to **8** if it is fast enough
  - Why: too small (2) → very noisy intra-group baseline → weak/unstable GRPO signal.
- **Sampling**:
  - **`temperature`**: **0.8**
  - **`top_p`**: **0.9**
  - Why: you need diversity within the group; greedy decoding makes the `G` samples too similar → little signal.

### Prompt batch and accumulation
- **Prompt batch size (per GPU)**: **1** (VLM + generation is memory-heavy)
- **Gradient accumulation**: **8** (starting point)
  - Why: keep a reasonable effective batch size without OOM.
  - Adjust based on VRAM and throughput.

### KL (stability vs progress)
- The KL “strength” is controlled by a coefficient **β** (or by a **KL target** with adaptive β).
- **Recommendation**: use a KL target if implemented; otherwise start with:
  - **β = 0.02** (start) and monitor observed KL
  - If you see drift/verbosity: increase β
  - If you see no gains: decrease β

### Other practical hyperparams (to finalize during implementation)
- **`max_grad_norm`**: 1.0 (same as SFT)
- **LoRA learning rate**: start low (e.g. 5e-5 to 1e-4) because RL is more unstable than SFT

### Number of RL prompts (sanity check vs real training)
Reference: SFT saw **≈ 88,751** questions (20% of train). One **RL prompt** = one dataset row (image + question + `answers` for reward).

| Phase | Order of magnitude | Purpose |
|------|---------------------|---------|
| **Minimal sanity check** | **256–512** | Check the pipeline runs (no crash, no NaNs, reward/KL logs look sane). Too small for statistical conclusions. |
| **Solid sanity check** | **1,000–2,000** | Enough to see whether mean reward moves in the right direction over 1–2 passes and KL stays reasonable. Moderate cost with `G=4`. |
| **Real training (budget / 24h)** | **8,000–20,000** | Good cost/diversity trade-off: thousands of unique questions, potentially multiple epochs on this subset. |
| **More ambitious** | **25,000–50,000** | If machine and budget allow: better train coverage without full pass. |
| **Comparable scale to SFT** | **up to ~89,000** | Same order as SFT examples, ideally using different indices/seed than SFT to reduce overfitting to the same pairs. |

**Cost order of magnitude**: with `G=4`, 10,000 prompts ≈ **40,000 generations** per pass on that subset; adjust `#prompts` and `G` based on remaining GPU time.

### Compute optimization: reuse image encoding across rollouts (`G` times same image + same question)

**Idea**: for a given RL prompt (same image, same question), the `G` outputs differ only by **sampling** (seed/noise). The **vision encoder** (image → visual embeddings fused with text) is identical across rollouts. A naive loop that calls `generate` **G** times from scratch may re-encode the image **G** times → wasted compute.

**Reusing** means computing the vision inputs (or vision tower output) once, and re-running only the autoregressive **decoding** with different random seeds, leveraging the **KV cache** for the shared prefix (visual tokens + prompt text).

#### Does it reduce performance (model quality / rollout distribution)?
**No**, if implemented correctly: the inputs and randomness match what you would get by recomputing everything. The distribution of the `G` responses should not change.

Potential regressions only if:
- implementation bugs (wrong dtype/device, partially reused tensors when prompt text changed, cache leakage between rollouts);
- using a **stale** cache across weight updates without invalidation (rare if you generate the `G` samples *before* the optimizer step).

#### Downsides
- **Code complexity**: you must align with the `transformers` / Qwen2-VL API (what is exposed as `pixel_values`, `past_key_values`, etc.).
- **Memory**: keeping prefix activations / KV can increase VRAM during the `G` decodes (usually acceptable for a single prompt at a time).
- **Maintenance**: HF versions / model internals can change → fragile if you rely on low-level details without tests.

#### Difficulty
- **Naive loop** (`G`× full `processor` + full `generate`) is **easy** and a common starting point.
- **One vision encode + `G` decodes** is **medium** difficulty (hours to ~1 day depending on experience), and should be validated against the naive baseline on a few examples.

---

## Recommended reward (RLAIF, no humans)

### Key idea
The reward should reflect the final objective (VQA accuracy) and prevent degenerate behaviors (format/verbosity hacks).

### “Simple and robust” reward recommendation for VQAv2
For each rollout (image+question → generated text):
1) Extract the predicted answer using the same parsing as evaluation (e.g. `extract_answer`)
2) Compute VQA “soft” accuracy (e.g. `vqa_accuracy(pred, answers)`), \( \in [0,1] \)
3) Optionally add format term and a mild length penalty

Proposal:
- **Task reward**: `r_task = vqa_accuracy(pred, answers)` (weight 1.0)
- **Format reward**: +0.1 if `<answer>...</answer>` parses, otherwise -0.1
- **Length penalty**: `-0.001 * (#generated_tokens)` (small, just to reduce verbosity)

### Link to the SFT loss
- In SFT, the loss is (implicitly) token-level cross-entropy on the target response.
- In GRPO, you no longer optimize cross-entropy directly; you optimize a reward.
- The key junction points are:
  - the **format** learned during SFT (e.g. `<answer> ... </answer>`)
  - the **metric** you want to improve (VQA accuracy)
So the reward must be compatible with the output format and the evaluation metric, rather than “copying” the SFT loss.

