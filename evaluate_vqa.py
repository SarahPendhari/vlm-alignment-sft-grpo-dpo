# evaluate_vqa.py
import os
import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
from src.metrics import vqa_accuracy, normalize_answer

MODEL_DIR = "./outputs/qwen_sft"
CACHE_DIR = "./cache"
RESULTS_FILE = os.path.join(MODEL_DIR, "vqa_results.json")

# Load model
base_model = AutoModelForVision2Seq.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, MODEL_DIR)
processor = AutoProcessor.from_pretrained(MODEL_DIR)

# Load dataset
dataset = load_dataset(
    "HuggingFaceM4/VQAv2",
    split="validation",
    cache_dir=CACHE_DIR
)

results = []
scores = []

print("Running evaluation...")
for example in tqdm(dataset):
    question = example["question"]
    answers = [a["answer"] for a in example["answers"]]
    image = example["image"]

    prompt = f"Question: {question}\n<answer>"

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=20
        )

    prediction = processor.decode(
        generated_ids[0],
        skip_special_tokens=True
    ).split("<answer>")[-1].strip()

    score = vqa_accuracy(prediction, answers)
    scores.append(score)

    results.append({
        "question_id": example["question_id"],
        "prediction": prediction,
        "answers": answers,
        "score": score
    })

# Save results
with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"Mean VQA Accuracy: {sum(scores)/len(scores):.4f}")