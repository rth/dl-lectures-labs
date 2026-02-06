import numpy as np

n_eval = 50
recall_at_5_dense = 0
recall_at_5_tfidf = 0
answer_has_key_terms = 0

for i in range(n_eval):
    question = qa[i]["question"]
    ground_truth = qa[i]["answer"]

    # Dense retrieval (FAISS)
    dense_results = search(question, k=5)
    dense_passages = [r["passage"] for r in dense_results]

    # TF-IDF retrieval
    tfidf_results = tfidf_search(question, k=5)
    tfidf_passages = [r["passage"] for r in tfidf_results]

    # Check if any ground-truth relevant info appears in retrieved passages
    gt_terms = set(ground_truth.lower().split())
    # Remove very common words for matching
    gt_terms -= {"the", "a", "an", "is", "was", "of", "in", "to", "and", "for", "on", "it"}

    # Recall: check if key answer terms appear in top-k passages
    dense_text = " ".join(dense_passages).lower()
    tfidf_text = " ".join(tfidf_passages).lower()

    if gt_terms and sum(t in dense_text for t in gt_terms) / len(gt_terms) > 0.5:
        recall_at_5_dense += 1
    if gt_terms and sum(t in tfidf_text for t in gt_terms) / len(gt_terms) > 0.5:
        recall_at_5_tfidf += 1

    # Generation evaluation: check if key terms from ground truth appear in answer
    result = rag_answer(question, k=3)
    answer_lower = result["answer"].lower()
    if gt_terms and sum(t in answer_lower for t in gt_terms) / len(gt_terms) > 0.3:
        answer_has_key_terms += 1

print(f"Evaluation over {n_eval} questions:")
print(f"  Dense retrieval Recall@5: {recall_at_5_dense/n_eval:.1%}")
print(f"  TF-IDF retrieval Recall@5: {recall_at_5_tfidf/n_eval:.1%}")
print(f"  Answer key-term match:     {answer_has_key_terms/n_eval:.1%}")
