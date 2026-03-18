import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText
from src.data import format_example
from src.collate import collate_fn

print("=== Step 1: Load processor ===")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
processor.tokenizer.padding_side = "right"
print("OK")

print("=== Step 2: Load 3 samples ===")
dataset = load_dataset("lmms-lab/VQAv2", split="validation[:3]")
dataset = dataset.map(format_example, remove_columns=dataset.column_names, load_from_cache_file=False)
print("OK")

print("=== Step 3: Run collate manually ===")
batch = [dataset[0], dataset[1], dataset[2]]
encoded = collate_fn(batch, processor)
print(f"Keys: {encoded.keys()}")
print(f"input_ids shape: {encoded['input_ids'].shape}")
print(f"labels shape: {encoded['labels'].shape}")
print("OK")

print("=== Step 4: Load model ===")
model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    dtype=torch.float16,
    trust_remote_code=True
).to("mps")
print("OK")

print("=== Step 5: Single forward pass ===")
inputs = {k: v.to("mps") for k, v in encoded.items()}
with torch.no_grad():
    out = model(**inputs)
print(f"Loss: {out.loss}")
print("ALL STEPS PASSED — hang is inside Trainer, not your code")