# evaluate_vqa.py
import os
import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel

from src.metrics import vqa_accuracy, extract_answer

MODEL_DIR = "./outputs/qwen_sft"
CACHE_DIR = os.environ.get("HF_HOME", "/opt/dlami/nvme/huggingface_cache")
RESULTS_FILE = os.path.join(MODEL_DIR, "vqa_comparison_results.json")

# Load model
print("Loading base model...")
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir=CACHE_DIR
)

print("Applying LoRA adapter...")
model = PeftModel.from_pretrained(base_model, MODEL_DIR)
model.eval()

processor = AutoProcessor.from_pretrained(MODEL_DIR)

# Load dataset
print("Loading dataset...")
dataset = load_dataset(
    "HuggingFaceM4/VQAv2",
    split="validation",
    cache_dir=CACHE_DIR,
    trust_remote_code=True
)

results = []
base_scores = []
sft_scores = []

# Evaluate on a 500-example subset
evaluation_subset = dataset.select(range(500))

print(f"Running side-by-side evaluation on {len(evaluation_subset)} examples...")
for example in tqdm(evaluation_subset):
    question = example["question"]
    answers = [a["answer"] for a in example["answers"]]
    image = example["image"]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"Question: {question}"}
            ]
        }
    ]
    
    text_prompt = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True 
    )

    inputs = processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    # ==========================================
    # 1. EVALUATE BASE MODEL (LoRA Disabled)
    # ==========================================
    with model.disable_adapter():
        with torch.no_grad():
            base_generated_ids = model.generate(**inputs, max_new_tokens=50)
            
    # ==========================================
    # 2. EVALUATE SFT MODEL (LoRA Enabled)
    # ==========================================
    with torch.no_grad():
        sft_generated_ids = model.generate(**inputs, max_new_tokens=50)

    # Trim prompts
    base_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, base_generated_ids)]
    sft_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, sft_generated_ids)]

    # Decode
    base_raw = processor.batch_decode(base_trimmed, skip_special_tokens=True)[0]
    sft_raw = processor.batch_decode(sft_trimmed, skip_special_tokens=True)[0]

    # Parse using your metric functions
    base_pred = extract_answer(base_raw)
    sft_pred = extract_answer(sft_raw)

    # Score
    base_score = vqa_accuracy(base_pred, answers)
    sft_score = vqa_accuracy(sft_pred, answers)
    
    base_scores.append(base_score)
    sft_scores.append(sft_score)

    results.append({
        "question_id": example["question_id"],
        "base_prediction": base_pred,
        "sft_prediction": sft_pred,
        "answers": answers,
        "base_score": base_score,
        "sft_score": sft_score
    })

# Save results
with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "="*40)
print("🏆 EVALUATION RESULTS")
print("="*40)
print(f"Base Model Accuracy: {sum(base_scores)/len(base_scores):.4f}")
print(f"SFT Model Accuracy:  {sum(sft_scores)/len(sft_scores):.4f}")
improvement = (sum(sft_scores)/len(sft_scores)) - (sum(base_scores)/len(base_scores))
print(f"Net Improvement:     {improvement:+.4f}")
print("="*40)