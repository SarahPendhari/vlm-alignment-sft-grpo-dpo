# train_dpo.py
import os
import torch
from datasets import load_from_disk
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import LoraConfig, PeftModel, get_peft_model
from trl import DPOTrainer, DPOConfig

SFT_MODEL_DIR = "./outputs/qwen_sft"
DPO_DATASET_DIR = "./outputs/dpo_pairs.json"
OUTPUT_DIR = "./outputs/qwen_dpo"
CACHE_DIR = os.environ.get("HF_HOME", "/opt/dlami/nvme/huggingface_cache")

print("1. Loading base model...")
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir=CACHE_DIR
)

print("2. Merging SFT weights into base model...")
model = PeftModel.from_pretrained(base_model, SFT_MODEL_DIR)
model = model.merge_and_unload() # The SFT model is now the new "Base" model!

print("3. Applying new LoRA adapter for DPO...")
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

processor = AutoProcessor.from_pretrained(SFT_MODEL_DIR)

print("4. Loading generated DPO dataset...")
train_dataset = load_from_disk(DPO_DATASET_DIR)

print("5. Initializing DPO Trainer...")
dpo_config = DPOConfig(
    output_dir=OUTPUT_DIR,
    beta=0.1, # From your Midway Report
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=5e-6, # DPO learning rates should be 10x smaller than SFT
    fp16=True,
    bf16=False,
    logging_steps=10,
    save_steps=100,
    report_to="none",
    remove_unused_columns=False, # Keep the image column
    dataset_num_proc=2
)

# Because we are using PEFT, we DO NOT need to pass a ref_model. 
# DPOTrainer automatically disables the LoRA adapter to compute reference probabilities!
trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=train_dataset,
    processing_class=processor,
)

print("Starting Offline DPO training...")
trainer.train()

trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print("DPO training complete!")