def rag_answer(question, k=3):
    """Answer a question using RAG: retrieve passages then generate."""
    # Step 1: Retrieve relevant passages
    results = search(question, k=k)

    # Step 2: Build context from retrieved passages
    context = "\n\n".join(
        f"[Passage {i+1}]: {r['passage']}" for i, r in enumerate(results)
    )

    # Step 3: Build prompt
    messages = [
        {"role": "system", "content": (
            "Answer the question based on the provided context. "
            "Be concise and specific. If the context doesn't contain "
            "the answer, say so."
        )},
        {"role": "user", "content": (
            f"Context:\n{context}\n\nQuestion: {question}"
        )},
    ]

    # Step 4: Generate answer
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True
    )
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        output = gen_model.generate(
            **inputs, max_new_tokens=100, do_sample=False
        )
    answer = tokenizer.decode(
        output[0][input_ids.shape[1]:], skip_special_tokens=True
    )

    return {"answer": answer, "passages": results}

# Test on sample questions from the QA set
for i in range(5):
    question = qa[i]["question"]
    ground_truth = qa[i]["answer"]
    result = rag_answer(question)
    print(f"\nQ: {question}")
    print(f"Ground truth: {ground_truth}")
    print(f"RAG answer: {result['answer']}")
