# src/data.py
from collections import Counter

def format_example(example):
    """
    Convert a VQAv2 example into a conversational format suitable for Qwen2-VL.
    The most frequent answer among annotators is used as the target.
    """
    question = example["question"]
    answers = [a["answer"] for a in example["answers"]]
    answer = Counter(answers).most_common(1)[0][0]

    # Use the conversational messages format required by Qwen2-VL
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"Question: {question}"}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"<think> </think>\n<answer> {answer} </answer>"}
            ]
        }
    ]

    return {
        "image": example["image"],
        "messages": messages, # <--- This is the key the collator is looking for!
        "answer": answer,
        "answers": answers,
        "question_id": example["question_id"]
    }