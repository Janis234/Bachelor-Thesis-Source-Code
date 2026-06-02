import numpy as np


def precision_at_k(recommended: list, relevant: list, k: int) -> float:
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / k


def recall_at_k(recommended: list, relevant: list, k: int) -> float:
    if not relevant:
        return 0.0
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(relevant))
    return hits / len(relevant)


def average_precision(recommended: list, relevant: list) -> float:
    relevant_set = set(relevant)
    hits = 0
    precision_sum = 0.0

    for i, item in enumerate(recommended):
        if item in relevant_set:
            hits += 1
            precision_sum += hits / (i + 1)

    if not relevant_set:
        return 0.0

    return precision_sum / len(relevant_set)


def dcg_at_k(recommended: list, relevant: list, k: int) -> float:
    relevant_set = set(relevant)
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant_set:
            dcg += 1.0 / np.log2(i + 2)
    return dcg


def ndcg_at_k(recommended: list, relevant: list, k: int) -> float:
    ideal = sorted([1 if item in set(relevant) else 0
                    for item in recommended], reverse=True)
    ideal_dcg = dcg_at_k(relevant[:k], relevant, k)

    if ideal_dcg == 0:
        return 0.0

    return dcg_at_k(recommended, relevant, k) / ideal_dcg


def intra_list_diversity(recommended_indices: list, embeddings: np.ndarray) -> float:
    if len(recommended_indices) < 2:
        return 0.0
    vecs = embeddings[recommended_indices]
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    normalized = vecs / (norms + 1e-8)
    sim_matrix = normalized @ normalized.T
    n = len(recommended_indices)
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1.0 - float(sim_matrix[i, j])
            count += 1
    return total / count if count > 0 else 0.0


def novelty(recommended_indices: list, popularity: np.ndarray) -> float:
    scores = []
    for idx in recommended_indices:
        p = float(popularity[idx])
        if p > 0:
            scores.append(-np.log2(p))
    return float(np.mean(scores)) if scores else 0.0


def gini_index(recommendation_counts: np.ndarray) -> float:
    counts = np.sort(recommendation_counts.astype(float))
    n = len(counts)
    if n == 0 or counts.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * counts) / (n * counts.sum())) - (n + 1) / n)


def evaluate_all(recommended: list, relevant: list, k: int = 10,
                 embeddings: np.ndarray = None, popularity: np.ndarray = None) -> dict:
    results = {
        f"precision@{k}":  round(precision_at_k(recommended, relevant, k), 4),
        f"recall@{k}":     round(recall_at_k(recommended, relevant, k), 4),
        "AP":              round(average_precision(recommended, relevant), 4),
        f"NDCG@{k}":       round(ndcg_at_k(recommended, relevant, k), 4),
    }
    if embeddings is not None:
        results["ILD"] = round(intra_list_diversity(recommended[:k], embeddings), 4)
    if popularity is not None:
        results["Novelty"] = round(novelty(recommended[:k], popularity), 4)
    return results
