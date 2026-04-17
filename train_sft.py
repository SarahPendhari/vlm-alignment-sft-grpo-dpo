import os
import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from src.data import format_example

# =========================
# Collate Function (UNCHANGED)
# =========================
def collate_fn(batch, processor):
    texts = [
        processor.apply_chat_template(
            item["messages"], 
            tokenize=False, 
            add_generation_prompt=False
        )
        for item in batch
    ]
    images = [item["image"] for item in batch]

    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt"
    )

    labels = inputs["input_ids"].clone()
    tokenizer = processor.tokenizer
    
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100

    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    
    for i in range(labels.size(0)):
        matches = (labels[i] == im_start_id).nonzero(as_tuple=True)[0]
        
        if len(matches) > 0:
            assistant_start_idx = matches[-1].item() + 3
            labels[i, :assistant_start_idx] = -100

    inputs["labels"] = labels
    return inputs

# =========================
# Config
# =========================
DEBUG = os.environ.get("DEBUG", "0") == "1"

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

OUTPUT_DIR = "./outputs/qwen_sft"
CACHE_DIR = os.environ.get("HF_HOME", "./cache")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# =========================
# Load Model + Processor
# =========================
print("Loading model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir=CACHE_DIR
)
print("Model loaded")

processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR
)
print("Processor loaded")

model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# =========================
# Apply LoRA (FASTER)
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
model.print_trainable_parameters()

# =========================
# Load Dataset (20% SHUFFLED)
# =========================
print("Loading dataset...")

train_dataset = load_dataset(
    "HuggingFaceM4/VQAv2",
    split="train",
    cache_dir=CACHE_DIR,
    trust_remote_code=True
)

# Shuffle BEFORE selecting
train_dataset = train_dataset.shuffle(seed=42).select(
    range(int(0.2 * len(train_dataset)))
).map(format_example, num_proc=2, load_from_cache_file=False)

print("Train dataset size:", len(train_dataset))

val_dataset = load_dataset(
    "HuggingFaceM4/VQAv2",
    split="validation",
    cache_dir=CACHE_DIR,
    trust_remote_code=True
)

val_dataset = val_dataset.shuffle(seed=42).select(
    range(int(0.1 * len(val_dataset)))  # 10% val is enough
).map(format_example, num_proc=2, load_from_cache_file=False)

print("Val dataset size:", len(val_dataset))

# =========================
# Training Arguments (FASTER)
# =========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,  # ↓ reduced from 8 → faster
    num_train_epochs=1,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=50,
    save_steps=1000,
    eval_strategy="steps",
    eval_steps=1000,
    fp16=True,
    bf16=False,
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=2,
    max_grad_norm=1.0
)

# =========================
# Trainer
# =========================
print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=lambda batch: collate_fn(batch, processor),
)
print("Trainer ready")

# =========================
# Train
# =========================
print("Starting SFT training...")
trainer.train()

# =========================
# Save Model
# =========================
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print("SFT training complete!")