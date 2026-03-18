import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data import format_example
from src.collate import collate_fn

import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from src.data import format_example
from src.collate import collate_fn

# =========================
# Config
# =========================
MODEL_NAME  = "Qwen/Qwen2-VL-2B-Instruct"
DATA_SPLIT  = "validation[:50]"          # 50 samples to verify pipeline on Mac
OUTPUT_DIR  = "./outputs/qwen_sft"
USE_MPS     = torch.backends.mps.is_available()
USE_CUDA    = torch.cuda.is_available()

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
    dtype=dtype,                              # fixed deprecation warning too
    device_map="auto" if USE_CUDA else None,
    trust_remote_code=True
)

if not USE_CUDA:
    model = model.to(device)

processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
processor.tokenizer.padding_side = "right"

print("Running GPU sanity check...")
dummy_input = torch.zeros(1, 1, dtype=torch.long).to(device)
try:
    with torch.no_grad():
        out = model(input_ids=dummy_input)
    print(f"Forward pass OK on {device}")
except Exception as e:
    print(f" Forward pass failed: {e}")

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
print(f"Columns: {dataset.column_names}")
dataset = dataset.map(format_example, remove_columns=dataset.column_names, load_from_cache_file=False)

print(f"Formatted columns: {dataset.column_names}")
print(f"Formatted sample: {dataset[0]['messages']}")

sample = dataset[0]
user_content = sample["messages"][1]["content"]
print(f"Content type: {type(user_content)}")
print(f"User content: {user_content}")
# Now correctly prints: [{'type': 'text', 'text': 'Where is he looking?'}]
# Image is in sample["image"] separately
assert isinstance(user_content, list), "format_example failed"
assert "image" in sample and sample["image"] is not None, "image missing"
print(" Data format correct, proceeding to training...")

# =========================
# Training Arguments
# =========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-5,
    warmup_steps=1,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    bf16=USE_CUDA,
    fp16=False,
    gradient_checkpointing=USE_CUDA,
    use_cpu=not USE_CUDA and not USE_MPS, 
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=0,        #  fixes MPS deadlock
    dataloader_pin_memory=False,
)

# =========================
# Trainer
# =========================
print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=lambda batch: collate_fn(batch, processor)
)

# =========================
# Train
# =========================
print("Starting training...")
trainer.train()

# =========================
# Save
# =========================
print("Saving model...")
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"Done. Checkpoint saved to {OUTPUT_DIR}")