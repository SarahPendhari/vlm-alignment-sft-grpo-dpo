def collate_fn(batch, processor):
    texts, images = [], []

    for item in batch:
        # Inject image into the user message at collation time
        messages_with_image = []
        for msg in item["messages"]:
            if msg["role"] == "user":
                messages_with_image.append({
                    "role": "user",
                    "content": [
                        {"type": "image"},              # placeholder — processor fills this in
                        *msg["content"]                 # existing text content
                    ]
                })
            else:
                messages_with_image.append(msg)

        text = processor.apply_chat_template(
            messages_with_image,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
        images.append([item["image"]])                  # PIL image passed separately

    encoded = processor(
        text=texts,
        images=images,
        padding=True,
        truncation=True,
        max_length=1024,
        return_tensors="pt"
    )

    # Prompt masking — only compute loss on assistant tokens
    labels = encoded["input_ids"].clone()
    for i, item in enumerate(batch):
        # Rebuild prompt-only messages with image placeholder for length calculation
        prompt_messages = []
        for msg in item["messages"][:-1]:              # exclude assistant turn
            if msg["role"] == "user":
                prompt_messages.append({
                    "role": "user",
                    "content": [{"type": "image"}, *msg["content"]]
                })
            else:
                prompt_messages.append(msg)

        prompt_only = processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompt_len = len(processor.tokenizer(
            prompt_only,
            add_special_tokens=False
        )["input_ids"])
        labels[i, :prompt_len] = -100

    labels[labels == processor.tokenizer.pad_token_id] = -100
    encoded["labels"] = labels
    return encoded