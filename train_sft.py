import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.optim import AdamW
import json
import math
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    get_cosine_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from src.data import format_example
from src.collate import collate_fn
from torch.utils.data import DataLoader, IterableDataset
from itertools import islice

# =========================
# Config
# =========================
MODEL_NAME        = "Qwen/Qwen2-VL-2B-Instruct"
TRAIN_SPLIT       = "train"
VAL_SPLIT         = "validation"
TRAIN_EXAMPLES    = 200
VAL_EXAMPLES      = 50
OUTPUT_DIR        = "./outputs/qwen_sft"
GRAD_ACCUM_STEPS  = 8
BATCH_SIZE        = 1
NUM_EPOCHS        = 1
LEARNING_RATE     = 2e-5

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
    torch_dtype=dtype,
    device_map="auto" if USE_CUDA else None,
    trust_remote_code=True
)
if not USE_CUDA:
    model = model.to(device)

processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
processor.tokenizer.padding_side = "right"

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

def limit_stream(stream, max_examples):
    return islice(stream, max_examples)


class StreamingDataset(IterableDataset):
    def __init__(self, iterator, transform=None):
        super().__init__()
        self.iterator = iterator
        self.transform = transform

    def __iter__(self):
        for example in self.iterator:
            if self.transform:
                example = self.transform(example)
            yield example


def make_train_loader():
    train_stream = load_dataset("lmms-lab/VQAv2", split=TRAIN_SPLIT, streaming=True)
    train_stream = limit_stream(train_stream, TRAIN_EXAMPLES)
    train_dataset = StreamingDataset(train_stream, transform=format_example)
    return DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=lambda x: collate_fn(x, processor),
    )


def make_val_loader():
    val_stream = load_dataset("lmms-lab/VQAv2", split=VAL_SPLIT, streaming=True)
    val_stream = limit_stream(val_stream, VAL_EXAMPLES)
    val_dataset = StreamingDataset(val_stream, transform=format_example)
    return DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=lambda x: collate_fn(x, processor),
    )


# Streaming sanity check (consume 1 example only)
_sanity_stream = load_dataset("lmms-lab/VQAv2", split=TRAIN_SPLIT, streaming=True)
_sanity_example = format_example(next(iter(_sanity_stream)))
assert isinstance(_sanity_example["messages"][1]["content"], list), "format_example failed"
assert _sanity_example["image"] is not None, "image missing"
print(f"✅ Streaming data format correct — train_examples={TRAIN_EXAMPLES} / val_examples={VAL_EXAMPLES}")

# =========================
# Optimizer + Scheduler
# =========================
steps_per_epoch = math.ceil(TRAIN_EXAMPLES / BATCH_SIZE)
total_steps  = steps_per_epoch * NUM_EPOCHS
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
print(f"  Train examples/epoch: {TRAIN_EXAMPLES}")
print(f"  Val examples/epoch:   {VAL_EXAMPLES}")
print(f"  Steps/epoch:          {steps_per_epoch}")
print(f"  Total steps:  {total_steps}")
print(f"  Grad accum:   every {GRAD_ACCUM_STEPS} steps")
print(f"  Warmup steps: {warmup_steps}")
print(f"  LR:           {LEARNING_RATE}\n")

os.makedirs(OUTPUT_DIR, exist_ok=True)
model.train()
optimizer.zero_grad()
running_loss = 0.0

train_losses = []
val_losses = []

for epoch in range(NUM_EPOCHS):
    print(f"--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")

    train_loader = make_train_loader()
    val_loader = make_val_loader()

    epoch_train_loss_sum = 0.0
    for step, batch in enumerate(train_loader):
        if step >= steps_per_epoch:
            break
        inputs = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**inputs)
        loss = outputs.loss / GRAD_ACCUM_STEPS
        running_loss += loss.item() * GRAD_ACCUM_STEPS
        epoch_train_loss_sum += loss.item() * GRAD_ACCUM_STEPS
        loss.backward()

        is_accum_step = (step + 1) % GRAD_ACCUM_STEPS == 0
        is_last_step  = (step + 1) == steps_per_epoch

        if is_accum_step or is_last_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            opt_step     = (step + 1) // GRAD_ACCUM_STEPS
            avg_loss     = running_loss / GRAD_ACCUM_STEPS
            running_loss = 0.0
            print(f"  step {step+1:>3}/{steps_per_epoch}"
                  f" | opt step {opt_step}"
                  f" | loss {avg_loss:.4f}"
                  f" | lr {scheduler.get_last_lr()[0]:.2e}")

    epoch_train_loss = epoch_train_loss_sum / max(steps_per_epoch, 1)
    model.eval()
    val_loss_sum = 0.0
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            if step >= math.ceil(VAL_EXAMPLES / BATCH_SIZE):
                break
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            val_loss_sum += outputs.loss.item()

    val_steps = math.ceil(VAL_EXAMPLES / BATCH_SIZE)
    epoch_val_loss = val_loss_sum / max(val_steps, 1)
    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)
    print(f"  -> epoch train loss: {epoch_train_loss:.4f} | val loss: {epoch_val_loss:.4f}")
    model.train()

    # Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        ckpt_dir = os.path.join(OUTPUT_DIR, f"checkpoint-epoch-{epoch+1}")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Saving checkpoint: {ckpt_dir}")
        model.save_pretrained(ckpt_dir)
        processor.save_pretrained(ckpt_dir)

print("\nTraining complete!")

# =========================
# Save
# =========================
print("Saving model...")
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"✅ Done. Checkpoint saved to {OUTPUT_DIR}")

# =========================
# Plot loss curve
# =========================
losses_payload = {"train_loss": train_losses, "val_loss": val_losses}
losses_path = os.path.join(OUTPUT_DIR, "losses.json")
with open(losses_path, "w") as f:
    json.dump(losses_payload, f, indent=2)

try:
    import matplotlib.pyplot as plt

    epochs = list(range(1, NUM_EPOCHS + 1))
    plt.figure()
    plt.plot(epochs, train_losses, label="train_loss")
    plt.plot(epochs, val_losses, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("SFT training/validation loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot_path = os.path.join(OUTPUT_DIR, "loss_curve.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"✅ Loss curve saved to {plot_path}")
except Exception as e:
    print(f"⚠️ Could not plot loss curve (matplotlib missing?). Saved {losses_path}. Error: {e}")