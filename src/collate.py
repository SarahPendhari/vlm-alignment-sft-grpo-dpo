def collate_fn(batch, processor):
    texts = []
    images = []
    prompt_texts = []

    for item in batch:
        messages_with_image = []
        prompt_messages = []

        for msg in item["messages"]:
            if msg["role"] == "user":
                user_msg = {
                    "role": "user",
                    "content": [{"type": "image"}, *msg["content"]],
                }
                messages_with_image.append(user_msg)
                prompt_messages.append(user_msg)
            else:
                messages_with_image.append(msg)
                if msg is not item["messages"][-1]:
                    prompt_messages.append(msg)

        text = processor.apply_chat_template(
            messages_with_image,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_text = processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        texts.append(text)
        prompt_texts.append(prompt_text)
        images.append([item["image"]])

    encoded = processor(
        text=texts,
        images=images,
        padding=True,
        truncation=True,
        max_length=1024,
        return_tensors="pt",
    )

    prompt_ids = processor.tokenizer(
        prompt_texts,
        add_special_tokens=False,
        padding=False,
        truncation=True,
        max_length=1024,
    )["input_ids"]
    prompt_lens = [len(ids) for ids in prompt_ids]

    labels = encoded["input_ids"].clone()
    for i, plen in enumerate(prompt_lens):
        labels[i, :plen] = -100

    labels[labels == processor.tokenizer.pad_token_id] = -100
    encoded["labels"] = labels
    return encoded