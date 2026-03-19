import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.optim import AdamW
import json
import math
import hashlib
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

# DataLoader perf knobs
NUM_WORKERS = 2
PIN_MEMORY = torch.cuda.is_available()

# Optional (advanced) speed boost
USE_TORCH_COMPILE = False

# =========================
# Config
# =========================
MODEL_NAME        = "Qwen/Qwen2-VL-2B-Instruct"
TRAIN_SPLIT       = "validation"
VAL_SPLIT         = "testdev"
TRAIN_EXAMPLES    = 200
VAL_EXAMPLES      = 50
SEED              = 0
# Buffered shuffle for streaming (approximate shuffle).
# On a T4, 2k–10k is usually a good tradeoff (CPU RAM vs randomness).
SHUFFLE_BUFFER_SIZE = 4096
# If TRAIN_SPLIT == VAL_SPLIT, deterministically partition that split
# into disjoint train/val streams using a stable hash.
VAL_FRACTION      = 0.2
OUTPUT_DIR        = "./outputs/qwen_sft"
GRAD_ACCUM_STEPS  = 8
BATCH_SIZE        = 1
NUM_EPOCHS        = 4
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
# Training perf settings
# =========================
# Important for training memory/speed
try:
    model.config.use_cache = False
except Exception:
    pass

# Saves VRAM; compute increases slightly
try:
    model.gradient_checkpointing_enable()
except Exception:
    pass

# Optional advanced speed boost (mostly useful on CUDA)
if USE_TORCH_COMPILE and torch.cuda.is_available():
    try:
        model = torch.compile(model)
        print("torch.compile enabled")
    except Exception as e:
        print(f"torch.compile failed (continuing without): {e}")

# =========================
# LoRA
# =========================
print("Applying LoRA...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
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

def _stable_id(example: dict) -> str:
    for k in ("question_id", "id", "example_id", "image_id"):
        if k in example and example[k] is not None:
            return str(example[k])
    q = str(example.get("question", ""))
    img = str(example.get("image_id", ""))
    return f"{q}|{img}"

def _in_val(example: dict, val_fraction: float) -> bool:
    sid = _stable_id(example)
    h = hashlib.md5(sid.encode("utf-8")).hexdigest()
    # map to [0,1)
    bucket = int(h[:8], 16) / 0x100000000
    return bucket < val_fraction


class StreamingDataset(IterableDataset):
    def __init__(self, iterator, transform=None):
        super().__init__()
        self.iterator = iterator
        self.transform = transform

    def __iter__(self):
        # Make IterableDataset safe with num_workers>0 (avoid duplicated samples).
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        for idx, example in enumerate(self.iterator):
            if (idx % num_workers) != worker_id:
                continue
            if self.transform:
                example = self.transform(example)
            yield example


def make_train_loader(epoch: int):
    train_stream = load_dataset("lmms-lab/VQAv2", split=TRAIN_SPLIT, streaming=True)
    if TRAIN_SPLIT == VAL_SPLIT:
        train_stream = train_stream.filter(lambda ex: not _in_val(ex, VAL_FRACTION))
    # Resample each epoch via epoch-dependent seed.
    train_stream = train_stream.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE, seed=SEED + epoch)
    train_stream = limit_stream(train_stream, TRAIN_EXAMPLES)
    train_dataset = StreamingDataset(train_stream, transform=format_example)
    return DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=lambda x: collate_fn(x, processor),
    )


def make_val_loader():
    val_stream = load_dataset("lmms-lab/VQAv2", split=VAL_SPLIT, streaming=True)
    if TRAIN_SPLIT == VAL_SPLIT:
        val_stream = val_stream.filter(lambda ex: _in_val(ex, VAL_FRACTION))
    val_stream = limit_stream(val_stream, VAL_EXAMPLES)
    val_dataset = StreamingDataset(val_stream, transform=format_example)
    return DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
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

# Keep loss as tensors to avoid per-step GPU sync (.item())
running_loss_t = torch.zeros((), device=device)

train_losses = []
val_losses = []

for epoch in range(NUM_EPOCHS):
    print(f"--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")

    train_loader = make_train_loader(epoch)
    val_loader = make_val_loader()

    epoch_train_loss_sum_t = torch.zeros((), device=device)
    for step, batch in enumerate(train_loader):
        if step >= steps_per_epoch:
            break
        inputs = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**inputs)
        loss = outputs.loss / GRAD_ACCUM_STEPS
        loss_unscaled = outputs.loss.detach()
        running_loss_t += loss_unscaled
        epoch_train_loss_sum_t += loss_unscaled
        loss.backward()

        is_accum_step = (step + 1) % GRAD_ACCUM_STEPS == 0
        is_last_step  = (step + 1) == steps_per_epoch

        if is_accum_step or is_last_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            opt_step     = (step + 1) // GRAD_ACCUM_STEPS
            avg_loss     = (running_loss_t / GRAD_ACCUM_STEPS).item()
            running_loss_t.zero_()
            print(f"  step {step+1:>3}/{steps_per_epoch}"
                  f" | opt step {opt_step}"
                  f" | loss {avg_loss:.4f}"
                  f" | lr {scheduler.get_last_lr()[0]:.2e}")

    epoch_train_loss = (epoch_train_loss_sum_t / max(steps_per_epoch, 1)).item()
    model.eval()
    val_loss_sum_t = torch.zeros((), device=device)
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            if step >= math.ceil(VAL_EXAMPLES / BATCH_SIZE):
                break
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            val_loss_sum_t += outputs.loss.detach()

    val_steps = math.ceil(VAL_EXAMPLES / BATCH_SIZE)
    epoch_val_loss = (val_loss_sum_t / max(val_steps, 1)).item()
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