import numpy as np

# Embed all passages
passages = corpus["passage"]
print(f"Embedding {len(passages)} passages...")
embeddings = embed_model.encode(
    passages, show_progress_bar=True, batch_size=64, normalize_embeddings=True
)
embeddings = np.array(embeddings, dtype="float32")
print(f"Embeddings shape: {embeddings.shape}")

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)
print(f"FAISS index built with {index.ntotal} vectors")

def search(query, k=5):
    """Search the FAISS index and return top-k passages with scores."""
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

# Test on sample questions
test_questions = [
    "What is the capital of France?",
    "Who invented the telephone?",
    "What causes earthquakes?",
]

for question in test_questions:
    print(f"\nQuestion: {question}")
    results = search(question, k=3)
    for i, r in enumerate(results):
        print(f"  {i+1}. (score={r['score']:.3f}) {r['passage'][:100]}...")
