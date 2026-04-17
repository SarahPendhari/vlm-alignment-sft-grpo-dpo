# src/metrics.py
from collections import Counter
import re
# Optional: for semantic similarity if you implement it later
# from sentence_transformers import SentenceTransformer, util 

def normalize_answer(ans):
    """Normalize answers for fair comparison."""
    if not isinstance(ans, str):
        return ""
    ans = ans.lower().strip()
    ans = re.sub(r"[^\w\s]", "", ans)
    return ans

def extract_answer(text):
    """
    Extracts the actual answer from within the <answer> tags.
    Falls back to the raw text if tags are missing.
    """
    match = re.search(r'<answer>(.*?)</answer>', text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def format_compliance(prediction):
    """
    Checks if the model strictly followed the <think> and <answer> format.
    Returns 1.0 if compliant, 0.0 otherwise.
    """
    has_think = bool(re.search(r'<think>.*?</think>', prediction, re.IGNORECASE | re.DOTALL))
    has_answer = bool(re.search(r'<answer>.*?</answer>', prediction, re.IGNORECASE | re.DOTALL))
    
    return 1.0 if (has_think and has_answer) else 0.0

def vqa_accuracy(raw_prediction, ground_truths):
    """
    VQA accuracy: score = min(1, (# humans that provided the answer) / 3)
    Now parses the raw prediction first to ignore formatting artifacts.
    """
    # 1. Isolate the intended answer from the structural tags
    parsed_prediction = extract_answer(raw_prediction)
    
    # 2. Normalize
    prediction_norm = normalize_answer(parsed_prediction)
    ground_truths_norm = [normalize_answer(a) for a in ground_truths]
    
    # 3. Calculate score
    count = ground_truths_norm.count(prediction_norm)
    return min(1.0, count / 3.0)

# ==========================================
# Future Implementation: Semantic Similarity
# ==========================================
# model_encoder = SentenceTransformer('all-MiniLM-L6-v2')
#
# def semantic_similarity(raw_prediction, ground_truths):
#     """
#     Computes cosine similarity between the prediction and the best ground truth.
#     """
#     parsed_prediction = extract_answer(raw_prediction)
#     if not parsed_prediction:
#         return 0.0
#        
#     pred_emb = model_encoder.encode(parsed_prediction, convert_to_tensor=True)
#     gt_embs = model_encoder.encode(ground_truths, convert_to_tensor=True)
#    
#     cosine_scores = util.cos_sim(pred_emb, gt_embs)
#     return cosine_scores.max().item()