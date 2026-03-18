def collate_fn(batch, processor):
    texts = [item["text"] for item in batch]
    images = [item["image"] for item in batch]

    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt"
    )

    inputs["labels"] = inputs["input_ids"].clone()
    return inputs