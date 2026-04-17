import torch

def collate_fn(batch, processor):
    # Inject <image> token
    texts = [f"<image>\n{item['text']}" for item in batch]
    images = [item["image"] for item in batch]

    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt"
    )

    labels = inputs["input_ids"].clone()

    tokenizer = processor.tokenizer

    # Optional masking (safe fallback)
    answer_token = "<answer>"
    if answer_token in tokenizer.get_vocab():
        answer_token_id = tokenizer.convert_tokens_to_ids(answer_token)
    else:
        answer_token_id = None

    for i in range(labels.size(0)):
        row = labels[i]

        if answer_token_id is not None:
            matches = (row == answer_token_id).nonzero(as_tuple=True)

            if len(matches[0]) > 0:
                start_idx = matches[0][0]
                labels[i, :start_idx] = -100

    inputs["labels"] = labels
    return inputs