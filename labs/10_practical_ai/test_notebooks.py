#!/usr/bin/env python
"""
Test script for Chapter 10: Practical AI with Large Language Models.

Tests Notebook 1 (Local LLMs) and Notebook 3 (RAG) solutions.
Notebook 2 (LLM APIs) is skipped because it requires an API key.

Usage:
    python test_notebooks.py
    python test_notebooks.py --with-api   # also test notebook 2 (needs OPENAI_API_KEY)
"""
import sys
import time
import os

print("=" * 60)
print("Chapter 10: Practical AI - Test Script")
print("=" * 60)

# ============================================================
# Notebook 1: Local LLMs with HuggingFace
# ============================================================
print("\n" + "=" * 60)
print("NOTEBOOK 1: Local LLMs with HuggingFace")
print("=" * 60)

# --- Tokenization ---
print("\n--- Testing tokenization ---")
from transformers import AutoTokenizer

model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"Tokenizer loaded: vocab size = {tokenizer.vocab_size:,}")

text = "Hello, how are you doing today?"
token_ids = tokenizer.encode(text)
decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
print(f"Encode/decode round-trip: '{text}' -> {token_ids} -> '{decoded}'")
assert decoded == text, f"Round-trip failed: got '{decoded}'"
print("Tokenization: OK")

# --- Model loading ---
print("\n--- Testing model loading ---")
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    model_name, dtype=torch.float16, device_map="cpu"
)
model.eval()
print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

# --- Text generation ---
print("\n--- Testing text generation ---")
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt")
start = time.time()
output = model.generate(**inputs, max_new_tokens=20)
elapsed = time.time() - start
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
n_new_tokens = output.shape[1] - inputs["input_ids"].shape[1]
print(f"Prompt: '{prompt}'")
print(f"Generated: '{generated_text}'")
print(f"Speed: {n_new_tokens} tokens in {elapsed:.1f}s ({n_new_tokens/elapsed:.1f} tok/s)")
assert len(generated_text) > len(prompt), "Generation produced no new text"
print("Text generation: OK")

# --- Chat template ---
print("\n--- Testing chat template ---")
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
]
inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
)
input_ids = inputs["input_ids"]
output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
response = tokenizer.decode(
    output[0][input_ids.shape[1]:], skip_special_tokens=True
)
print(f"Chat response: '{response}'")
assert len(response) > 0, "Chat generation produced empty response"
print("Chat template: OK")

# --- SOLUTION: generate_with_params ---
print("\n--- Testing solution: generate_with_params ---")
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
        max_new_tokens=50,
        do_sample=True,
        temperature=temp,
        top_p=0.95,
    )
    elapsed = time.time() - start
    text = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"  Temperature {temp} ({elapsed:.1f}s): {text[:80]}...")
    assert len(text) > 0, f"Empty generation at temperature {temp}"
print("generate_with_params: OK")

# --- SOLUTION: chat_classification ---
print("\n--- Testing solution: chat_classification ---")
test_samples = [
    {"text": "The stock market surged today as investors reacted positively to the latest earnings reports from major tech companies.", "label": "business"},
    {"text": "The tennis champion won her fifth Grand Slam title after a thrilling three-set final.", "label": "sport"},
    {"text": "The company unveiled its latest smartphone featuring an AI-powered camera at the tech conference.", "label": "tech"},
]

correct = 0
total = len(test_samples)
for sample in test_samples:
    msgs = [
        {"role": "system", "content": "You are a news classifier. Classify the following news snippet into exactly one category: business, entertainment, politics, sport, tech. Reply with only the category name, nothing else."},
        {"role": "user", "content": sample["text"]},
    ]
    chat_inputs = tokenizer.apply_chat_template(
        msgs, return_tensors="pt", return_dict=True
    )
    chat_ids = chat_inputs["input_ids"]
    out = model.generate(**chat_inputs, max_new_tokens=10, do_sample=False)
    resp = tokenizer.decode(out[0][chat_ids.shape[1]:], skip_special_tokens=True).strip().lower()
    predicted = resp.split()[0] if resp else ""
    is_correct = predicted == sample["label"]
    correct += int(is_correct)
    print(f"  True: {sample['label']}, Predicted: {predicted} {'correct' if is_correct else 'wrong'}")

print(f"  Accuracy: {correct}/{total}")
print("chat_classification: OK (solution runs)")

# Clean up notebook 1
del model
print("\nNotebook 1: ALL TESTS PASSED")

# ============================================================
# Notebook 2: LLM APIs (optional, requires API key)
# ============================================================
run_api_tests = "--with-api" in sys.argv and os.environ.get("OPENAI_API_KEY")

if run_api_tests:
    print("\n" + "=" * 60)
    print("NOTEBOOK 2: LLM APIs (optional)")
    print("=" * 60)

    from openai import OpenAI
    client = OpenAI()
    MODEL = "gpt-4o-mini"

    print("\n--- Testing basic API call ---")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Say hello in one word."}],
        max_tokens=10,
    )
    content = response.choices[0].message.content
    print(f"API response: '{content}'")
    assert len(content) > 0, "Empty API response"
    print("Basic API call: OK")

    print("\n--- Testing structured output ---")
    from pydantic import BaseModel
    from typing import Literal

    class SimpleExtraction(BaseModel):
        topic: str
        sentiment: Literal["positive", "negative", "neutral"]

    response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Extract topic and sentiment."},
            {"role": "user", "content": "I love sunny weather!"},
        ],
        response_format=SimpleExtraction,
    )
    parsed = response.choices[0].message.parsed
    print(f"Parsed: topic='{parsed.topic}', sentiment='{parsed.sentiment}'")
    assert parsed.sentiment == "positive"
    print("Structured output: OK")

    print("\nNotebook 2: ALL TESTS PASSED")
else:
    print("\n" + "=" * 60)
    print("NOTEBOOK 2: SKIPPED (run with --with-api and OPENAI_API_KEY set)")
    print("=" * 60)

# ============================================================
# Notebook 3: Retrieval-Augmented Generation
# ============================================================
print("\n" + "=" * 60)
print("NOTEBOOK 3: Retrieval-Augmented Generation")
print("=" * 60)

# --- Dataset loading ---
print("\n--- Testing dataset loading ---")
from datasets import load_dataset

corpus = load_dataset(
    "rag-datasets/rag-mini-wikipedia", "text-corpus"
)["passages"]
qa = load_dataset(
    "rag-datasets/rag-mini-wikipedia", "question-answer"
)["test"]
print(f"Corpus: {len(corpus)} passages")
print(f"QA set: {len(qa)} question-answer pairs")
assert len(corpus) > 100, "Corpus too small"
assert len(qa) > 100, "QA set too small"
print("Dataset loading: OK")

# --- Embedding model ---
print("\n--- Testing embedding model ---")
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
test_embedding = embed_model.encode(["test sentence"])
print(f"Embedding shape: {test_embedding.shape}")
assert test_embedding.shape[1] == 384, f"Expected 384 dims, got {test_embedding.shape[1]}"
print("Embedding model: OK")

# --- SOLUTION: embed_and_index ---
print("\n--- Testing solution: embed_and_index ---")
import numpy as np
import faiss

passages = corpus["passage"]
print(f"Embedding {len(passages)} passages...")
start = time.time()
embeddings = embed_model.encode(
    passages, show_progress_bar=True, batch_size=64, normalize_embeddings=True
)
embeddings = np.array(embeddings, dtype="float32")
elapsed = time.time() - start
print(f"Embeddings shape: {embeddings.shape} (took {elapsed:.1f}s)")

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)
print(f"FAISS index built with {index.ntotal} vectors")


def search(query, k=5):
    query_embedding = embed_model.encode(
        [query], normalize_embeddings=True
    ).astype("float32")
    scores, indices = index.search(query_embedding, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "passage": passages[idx],
            "score": float(score),
            "index": int(idx),
        })
    return results


test_questions = [
    "What is the capital of France?",
    "Who invented the telephone?",
    "What causes earthquakes?",
]
for question in test_questions:
    results = search(question, k=3)
    print(f"\n  Q: {question}")
    print(f"  Top result (score={results[0]['score']:.3f}): {results[0]['passage'][:80]}...")
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    assert results[0]["score"] > 0, "Score should be positive"
print("\nembed_and_index: OK")

# --- TF-IDF baseline ---
print("\n--- Testing TF-IDF baseline ---")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
tfidf_matrix = tfidf_vectorizer.fit_transform(passages)
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")


def tfidf_search(query, k=5):
    query_vec = tfidf_vectorizer.transform([query])
    similarities = sklearn_cosine(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[::-1][:k]
    results = []
    for idx in top_indices:
        results.append({
            "passage": passages[idx],
            "score": float(similarities[idx]),
            "index": int(idx),
        })
    return results


tfidf_results = tfidf_search("Who invented the telephone?", k=3)
assert len(tfidf_results) == 3
print("TF-IDF baseline: OK")

# --- SOLUTION: rag_query ---
print("\n--- Testing solution: rag_query ---")
from transformers import AutoModelForCausalLM
import torch

gen_model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
gen_model = AutoModelForCausalLM.from_pretrained(
    gen_model_name, dtype=torch.float16, device_map="cpu"
)
gen_model.eval()
print(f"Generation model loaded: {gen_model_name}")

# Use gen_tokenizer as 'tokenizer' for the solution code
tokenizer_nb3 = gen_tokenizer


def rag_answer(question, k=3):
    results = search(question, k=k)
    context = "\n\n".join(
        f"[Passage {i+1}]: {r['passage']}" for i, r in enumerate(results)
    )
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
    rag_inputs = tokenizer_nb3.apply_chat_template(
        messages, return_tensors="pt", return_dict=True
    )
    input_ids = rag_inputs["input_ids"]
    with torch.no_grad():
        output = gen_model.generate(
            **rag_inputs, max_new_tokens=100, do_sample=False
        )
    answer = tokenizer_nb3.decode(
        output[0][input_ids.shape[1]:], skip_special_tokens=True
    )
    return {"answer": answer, "passages": results}


for i in range(3):
    question = qa[i]["question"]
    ground_truth = qa[i]["answer"]
    result = rag_answer(question)
    print(f"\n  Q: {question}")
    print(f"  Ground truth: {ground_truth}")
    print(f"  RAG answer: {result['answer'][:100]}")
    assert len(result["answer"]) > 0, "Empty RAG answer"
    assert len(result["passages"]) == 3, "Expected 3 passages"
print("\nrag_query: OK")

# --- SOLUTION: evaluate_rag (quick version with 10 samples) ---
print("\n--- Testing solution: evaluate_rag (10 samples) ---")
n_eval = 10
recall_at_5_dense = 0
recall_at_5_tfidf = 0

for i in range(n_eval):
    question = qa[i]["question"]
    ground_truth = qa[i]["answer"]

    dense_results = search(question, k=5)
    dense_passages = [r["passage"] for r in dense_results]

    tfidf_results = tfidf_search(question, k=5)
    tfidf_passages = [r["passage"] for r in tfidf_results]

    gt_terms = set(ground_truth.lower().split())
    gt_terms -= {"the", "a", "an", "is", "was", "of", "in", "to", "and", "for", "on", "it"}

    dense_text = " ".join(dense_passages).lower()
    tfidf_text = " ".join(tfidf_passages).lower()

    if gt_terms and sum(t in dense_text for t in gt_terms) / len(gt_terms) > 0.5:
        recall_at_5_dense += 1
    if gt_terms and sum(t in tfidf_text for t in gt_terms) / len(gt_terms) > 0.5:
        recall_at_5_tfidf += 1

print(f"  Dense retrieval Recall@5: {recall_at_5_dense/n_eval:.1%}")
print(f"  TF-IDF retrieval Recall@5: {recall_at_5_tfidf/n_eval:.1%}")
print("evaluate_rag: OK")

print("\nNotebook 3: ALL TESTS PASSED")

# ============================================================
print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
