# generate_dpo_pairs.py
import os
import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel
from src.metrics import vqa_accuracy, extract_answer

SFT_MODEL_DIR = "./outputs/qwen_sft"
CACHE_DIR = os.environ.get("HF_HOME", "/opt/dlami/nvme/huggingface_cache")
OUTPUT_FILE = "./outputs/dpo_pairs.json"

print("Loading base model...")
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir=CACHE_DIR
)

print("Applying SFT LoRA adapter...")
model = PeftModel.from_pretrained(base_model, SFT_MODEL_DIR)
model.eval()

processor = AutoProcessor.from_pretrained(SFT_MODEL_DIR)

print("Loading dataset...")
# Using the train split since this is for training the DPO model
dataset = load_dataset(
    "HuggingFaceM4/VQAv2",
    split="train", 
    cache_dir=CACHE_DIR,
    trust_remote_code=True
)

# We will sample 2,000 examples to try and get ~500 good pairs (since ties are discarded)
generation_subset = dataset.select(range(2000))
dpo_pairs = []

print("Generating responses...")
for example in tqdm(generation_subset):
    question = example["question"]
    answers = [a["answer"] for a in example["answers"]]
    image = example["image"]

    prompt_messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"Question: {question}"}]}
    ]
    
    text_prompt = processor.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True 
    )

    inputs = processor(
        text=[text_prompt], images=[image], padding=True, return_tensors="pt"
    ).to(model.device)

    # Generate 2 answers using probabilistic sampling!
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,          # Turn off greedy decoding
            temperature=0.8,         # Inject diversity
            top_p=0.90,              # Prevent total gibberish
            num_return_sequences=2   # Generate two at once!
        )

    # Trim input prompts
    trimmed_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    raw_preds = processor.batch_decode(trimmed_ids, skip_special_tokens=True)
    
    # Parse and score
    pred_1, pred_2 = extract_answer(raw_preds[0]), extract_answer(raw_preds[1])
    score_1, score_2 = vqa_accuracy(pred_1, answers), vqa_accuracy(pred_2, answers)

    # Create the pair ONLY if there is a clear winner
    if score_1 != score_2:
        chosen_raw = raw_preds[0] if score_1 > score_2 else raw_preds[1]
        rejected_raw = raw_preds[1] if score_1 > score_2 else raw_preds[0]

        dpo_pairs.append({
            "prompt": prompt_messages,
            "chosen": [{"role": "assistant", "content": [{"type": "text", "text": chosen_raw}]}],
            "rejected": [{"role": "assistant", "content": [{"type": "text", "text": rejected_raw}]}],
            "image": image # Keep the PIL image in the row for the collator
        })

print(f"Generated {len(dpo_pairs)} valid preference pairs from 2000 examples.")

# Save to disk
import datasets
from datasets import Dataset
dpo_dataset = Dataset.from_list(dpo_pairs)
dpo_dataset.save_to_disk(OUTPUT_FILE)