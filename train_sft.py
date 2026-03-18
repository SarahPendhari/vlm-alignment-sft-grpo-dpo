import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    get_cosine_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from src.data import format_example
from src.collate import collate_fn

# =========================
# Config
# =========================
MODEL_NAME         = "Qwen/Qwen2-VL-2B-Instruct"  # swap to 7B on cloud
DATA_SPLIT         = "validation[:50]"              # 50 samples to verify on Mac
OUTPUT_DIR         = "./outputs/qwen_sft"
GRAD_ACCUM_STEPS   = 8
BATCH_SIZE         = 1
NUM_EPOCHS         = 1
LEARNING_RATE      = 2e-5
MAX_LENGTH         = 512                            # reduce for speed on Mac

USE_MPS  = torch.backends.mps.is_available()
USE_CUDA = torch.cuda.is_available()

# =========================
# Device setup
# =========================
if USE_CUDA:
    dtype  = torch.bfloat16
    device = "cuda"
elif USE_MPS:
    dtype  = torch.float16
    device = "mps"
else:
    dtype  = torch.float32
    device = "cpu"

print(f"Running on: {device} | dtype: {dtype}")

# =========================
# Model + Processor
# =========================
print("Loading model...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    dtype=dtype,
    device_map="auto" if USE_CUDA else None,
    trust_remote_code=True
)

if not USE_CUDA:
    model = model.to(device)

processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
processor.tokenizer.padding_side = "right"

# Quick sanity check
print("Running forward pass sanity check...")
dummy_input = torch.zeros(1, 1, dtype=torch.long).to(device)
try:
    with torch.no_grad():
        model(input_ids=dummy_input)
    print(f"Forward pass OK on {device}")
except Exception as e:
    print(f"Forward pass failed: {e}")

# =========================
# LoRA
# =========================
print("Applying LoRA...")
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.enable_input_require_grads()
model.print_trainable_parameters()

# =========================
# Dataset
# =========================
print("Loading dataset...")
dataset = load_dataset("lmms-lab/VQAv2", split=DATA_SPLIT)
print(f"Raw columns: {dataset.column_names}")

dataset = dataset.map(
    format_example,
    remove_columns=dataset.column_names,
    load_from_cache_file=False
)
print(f"Formatted columns: {dataset.column_names}")

# Sanity check formatted data
sample = dataset[0]
user_content = sample["messages"][1]["content"]
assert isinstance(user_content, list), "format_example failed — content is not a list"
assert sample["image"] is not None,    " image is missing from sample"
print(f"Data format correct — {len(dataset)} samples ready")

# =========================
# Dataloader
# =========================
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,          # must be 0 on MPS — avoids Trainer deadlock
    collate_fn=lambda batch: collate_fn(batch, processor)
)

# =========================
# Optimizer + Scheduler
# =========================
total_steps = (len(dataloader) * NUM_EPOCHS)
warmup_steps = max(1, int(0.05 * total_steps))

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# =========================
# Training Loop
# =========================
print(f"\nStarting training...")
print(f"  Samples:       {len(dataset)}")
print(f"  Total steps:   {total_steps}")
print(f"  Grad accum:    {GRAD_ACCUM_STEPS} (optimizer steps every {GRAD_ACCUM_STEPS} batches)")
print(f"  Warmup steps:  {warmup_steps}")
print(f"  LR:            {LEARNING_RATE}\n")

os.makedirs(OUTPUT_DIR, exist_ok=True)

model.train()
optimizer.zero_grad()

running_loss = 0.0

for epoch in range(NUM_EPOCHS):
    print(f"--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")

    for step, batch in enumerate(dataloader):
        # Move all tensors to device
        inputs = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        outputs = model(**inputs)
        loss = outputs.loss / GRAD_ACCUM_STEPS      # normalize for accumulation
        running_loss += loss.item() * GRAD_ACCUM_STEPS

        # Backward pass
        loss.backward()

        # Optimizer step every GRAD_ACCUM_STEPS batches (or at end of epoch)
        is_accum_step = (step + 1) % GRAD_ACCUM_STEPS == 0
        is_last_step  = (step + 1) == len(dataloader)

        if is_accum_step or is_last_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            opt_step   = (step + 1) // GRAD_ACCUM_STEPS
            avg_loss   = running_loss / GRAD_ACCUM_STEPS
            running_loss = 0.0
            print(f"  step {step+1:>3}/{len(dataloader)} "
                  f"| opt step {opt_step} "
                  f"| loss {avg_loss:.4f} "
                  f"| lr {scheduler.get_last_lr()[0]:.2e}")

print("\nTraining complete!")

# =========================
# Save
# =========================
print("Saving model...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"Done. πref checkpoint saved to {OUTPUT_DIR}")