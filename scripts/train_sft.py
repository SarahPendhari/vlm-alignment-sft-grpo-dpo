import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model

from src.data import format_example
from src.collate import collate_fn


# =========================
# 1. Config
# =========================
MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
DATA_SPLIT = "train[:2%]"   # small subset for debugging
OUTPUT_DIR = "./outputs/qwen_sft"


# =========================
# 2. Load Model + Processor
# =========================
print("🔹 Loading model...")
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(MODEL_NAME)


# =========================
# 3. Apply LoRA
# =========================
print("🔹 Applying LoRA...")
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# =========================
# 4. Load Dataset
# =========================
print("🔹 Loading dataset...")
dataset = load_dataset("HuggingFaceM4/VQAv2", split=DATA_SPLIT)

print("🔹 Formatting dataset...")
dataset = dataset.map(format_example)


# =========================
# 5. Training Arguments
# =========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=200,
    fp16=True,
    report_to="none",
    remove_unused_columns=False
)


# =========================
# 6. Trainer
# =========================
print("🔹 Initializing Trainer...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=lambda batch: collate_fn(batch, processor)
)


# =========================
# 7. Train
# =========================
print("🚀 Starting training...")
trainer.train()


# =========================
# 8. Save Model
# =========================
print("💾 Saving model...")
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print("SFT training complete!")