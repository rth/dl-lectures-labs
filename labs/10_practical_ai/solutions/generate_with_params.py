import time

prompt = "The future of artificial intelligence is"
messages = [{"role": "user", "content": prompt}]
inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt", return_dict=True
)
input_ids = inputs["input_ids"]

temperatures = [0.1, 0.7, 1.5]
for temp in temperatures:
    start = time.time()
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=temp,
        top_p=0.95,
    )
    elapsed = time.time() - start
    text = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\n--- Temperature {temp} ({elapsed:.1f}s) ---")
    print(text)
