SYSTEM_PROMPT = (
    "You are a visual reasoning assistant. "
    "Think step by step inside <think>...</think>, "
    "then give your final answer inside <answer>...</answer>."
)

def format_example(example):
    answer = example.get("multiple_choice_answer") or example.get("answer", "")
    return {
        "image": example["image"],       # keep image separate — NOT inside messages
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": example["question"]}]
                # no image here — collator will inject it from the "image" field
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": f"<think></think><answer>{answer}</answer>"}]
            }
        ],
        "answer": answer
    }