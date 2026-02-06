correct = 0
total = len(test_samples)

for sample in test_samples:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a news classifier. Classify the following news"
                " snippet into exactly one category: business,"
                " entertainment, politics, sport, tech. Reply with only"
                " the category name, nothing else."
            ),
        },
        {"role": "user", "content": sample["text"]},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True
    )
    input_ids = inputs["input_ids"]
    output = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    response = tokenizer.decode(
        output[0][input_ids.shape[1]:], skip_special_tokens=True
    ).strip().lower()
    predicted = response.split()[0] if response else ""
    is_correct = predicted == sample["label"]
    correct += int(is_correct)
    print(f"Text: {sample['text'][:60]}...")
    print(
        f"  True: {sample['label']}, Predicted: {predicted}"
        f" {'correct' if is_correct else 'wrong'}"
    )

accuracy = correct / total
print(f"\nAccuracy: {accuracy:.1%} ({correct}/{total})")
print(f"Random baseline: {1/5:.1%}")
