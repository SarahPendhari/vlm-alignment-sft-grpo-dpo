def format_example(example):
    question = example["question"]
    answers = [a["answer"] for a in example["answers"]]
    answer = max(set(answers), key=answers.count)

    text = f"""Question: {question}
<think> </think>
<answer> {answer} </answer>"""

    return {
        "image": example["image"],
        "text": text
    }