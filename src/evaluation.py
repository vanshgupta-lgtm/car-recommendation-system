"""
Evaluation Module for Car Recommendation System.
Computes offline ranking and recommendation metrics:
- Precision@K
- Recall@K
- NDCG@K (Normalized Discounted Cumulative Gain)
- Catalog Coverage
- Intra-List Diversity (Pairwise Cosine Distance)
Produces comprehensive comparison benchmarks across all models.
"""

from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_distances


def precision_at_k(recommended_ids: List[str], relevant_ids: Set[str], k: int = 5) -> float:
    """
    Computes Precision@K:
    Fraction of recommended items in top-K that are relevant.
    """
    if k <= 0 or not recommended_ids:
        return 0.0
    top_k = recommended_ids[:k]
    num_relevant = sum(1 for cid in top_k if cid in relevant_ids)
    return num_relevant / float(k)


def recall_at_k(recommended_ids: List[str], relevant_ids: Set[str], k: int = 5) -> float:
    """
    Computes Recall@K:
    Fraction of all relevant items retrieved in the top-K recommendations.
    """
    if not relevant_ids or k <= 0 or not recommended_ids:
        return 0.0
    top_k = recommended_ids[:k]
    num_relevant = sum(1 for cid in top_k if cid in relevant_ids)
    return num_relevant / float(len(relevant_ids))


def ndcg_at_k(recommended_ids: List[str], relevance_dict: Dict[str, float], k: int = 5) -> float:
    """
    Computes Normalized Discounted Cumulative Gain at K (NDCG@K).
    Accounts for position-dependent discounting of recommendations.
    """
    if k <= 0 or not recommended_ids:
        return 0.0

    top_k = recommended_ids[:k]

    # Calculate Discounted Cumulative Gain (DCG)
    dcg = 0.0
    for i, cid in enumerate(top_k):
        rel = relevance_dict.get(cid, 0.0)
        dcg += rel / np.log2(i + 2)

    # Calculate Ideal DCG (IDCG)
    ideal_rels = sorted(relevance_dict.values(), reverse=True)[:k]
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))

    if idcg <= 0.0:
        return 0.0

    return float(dcg / idcg)


def catalog_coverage(all_recommendations: List[List[str]], catalog_car_ids: Set[str]) -> float:
    """
    Computes Catalog Coverage:
    Proportion of distinct catalog items recommended at least once across all test queries.
    """
    if not catalog_car_ids:
        return 0.0
    unique_recommended = set()
    for rec_list in all_recommendations:
        unique_recommended.update(rec_list)
    return len(unique_recommended) / float(len(catalog_car_ids))


def intra_list_diversity(recommended_indices: List[int], feature_matrix: csr_matrix) -> float:
    """
    Computes Intra-List Diversity:
    Mean pairwise cosine distance among recommended items. Higher indicates more diverse suggestions.
    """
    if len(recommended_indices) <= 1:
        return 0.0
    vectors = feature_matrix[recommended_indices]
    dist_matrix = cosine_distances(vectors)
    n = len(recommended_indices)
    # Upper triangle mean (excluding diagonal 0s)
    pairwise_dists = dist_matrix[np.triu_indices(n, k=1)]
    return float(np.mean(pairwise_dists)) if len(pairwise_dists) > 0 else 0.0


def run_benchmark_evaluation(
    cars_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    preprocessor: Any,
    content_rec: Any,
    knn_rec: Any,
    hybrid_rec: Any,
    cf_rec: Any,
    k: int = 5,
    num_queries: int = 100,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Runs an end-to-end benchmark across all 4 recommender paradigms using
    a synthesized test suite of diverse user personas and queries.
    """
    np.random.seed(random_seed)

    # 1. Define distinct buyer personas for test queries
    test_queries = []
    catalog_ids = set(cars_df["car_id"])

    # Sample reference cars for item-similarity tests
    sampled_ref_cars = cars_df.sample(num_queries, random_state=random_seed)

    for idx, (_, ref_car) in enumerate(sampled_ref_cars.iterrows()):
        query = {
            "query_id": f"Q{idx+1:03d}",
            "reference_car_id": ref_car["car_id"],
            "max_budget": float(ref_car["price"] * np.random.uniform(1.05, 1.25)),
            "min_budget": float(ref_car["price"] * np.random.uniform(0.70, 0.95)),
            "body_type": ref_car["body_type"],
            "fuel_type": ref_car["fuel_type"],
            "transmission": ref_car["transmission"],
            "min_seats": int(ref_car["seats"]),
            "min_mileage": float(max(10.0, ref_car["mileage_kmpl"] * 0.9)),
            "min_safety": int(max(2, ref_car["safety_rating"] - 1)),
            "features": [f.strip() for f in str(ref_car["features"]).split(",")[:2] if f.strip()],
            "user_id": f"U{(idx % 100) + 1:04d}",
        }

        # Ground truth relevance: cars in catalog that match the body_type and fall within budget ± 15%
        rel_cond = (
            (cars_df["body_type"] == query["body_type"])
            & (cars_df["price"] <= query["max_budget"] * 1.1)
            & (cars_df["price"] >= query["min_budget"] * 0.9)
            & (cars_df["car_id"] != query["reference_car_id"])
        )
        relevant_df = cars_df[rel_cond]
        relevant_ids = set(relevant_df["car_id"])

        # Continuous relevance score for NDCG (1.0 for perfect spec match, down to 0.2 for partial)
        relevance_dict = {}
        for _, car in relevant_df.iterrows():
            rel_score = 1.0
            if car["fuel_type"] == query["fuel_type"]:
                rel_score += 0.5
            if car["transmission"] == query["transmission"]:
                rel_score += 0.5
            relevance_dict[car["car_id"]] = rel_score

        query["relevant_ids"] = relevant_ids
        query["relevance_dict"] = relevance_dict
        test_queries.append(query)

    # 2. Benchmark each model
    results = {
        "Content-Based": {"p_at_k": [], "r_at_k": [], "ndcg": [], "diversity": [], "all_recs": []},
        "KNN Recommender": {"p_at_k": [], "r_at_k": [], "ndcg": [], "diversity": [], "all_recs": []},
        "Hybrid Engine": {"p_at_k": [], "r_at_k": [], "ndcg": [], "diversity": [], "all_recs": []},
        "Collaborative Filtering": {"p_at_k": [], "r_at_k": [], "ndcg": [], "diversity": [], "all_recs": []},
    }

    feature_matrix = content_rec.feature_matrix

    for q in test_queries:
        ref_id = q["reference_car_id"]
        rel_ids = q["relevant_ids"]
        rel_dict = q["relevance_dict"]
        user_id = q["user_id"]

        if not rel_ids:
            continue

        # A. Content-Based
        cb_df = content_rec.get_similar_cars(ref_id, top_k=k)
        cb_recs = cb_df["car_id"].tolist()
        cb_indices = [content_rec.car_id_to_idx[cid] for cid in cb_recs if cid in content_rec.car_id_to_idx]
        results["Content-Based"]["p_at_k"].append(precision_at_k(cb_recs, rel_ids, k))
        results["Content-Based"]["r_at_k"].append(recall_at_k(cb_recs, rel_ids, k))
        results["Content-Based"]["ndcg"].append(ndcg_at_k(cb_recs, rel_dict, k))
        results["Content-Based"]["diversity"].append(intra_list_diversity(cb_indices, feature_matrix))
        results["Content-Based"]["all_recs"].append(cb_recs)

        # B. KNN Recommender
        knn_df = knn_rec.get_similar_cars(ref_id, top_k=k)
        knn_recs = knn_df["car_id"].tolist()
        knn_indices = [knn_rec.car_id_to_idx[cid] for cid in knn_recs if cid in knn_rec.car_id_to_idx]
        results["KNN Recommender"]["p_at_k"].append(precision_at_k(knn_recs, rel_ids, k))
        results["KNN Recommender"]["r_at_k"].append(recall_at_k(knn_recs, rel_ids, k))
        results["KNN Recommender"]["ndcg"].append(ndcg_at_k(knn_recs, rel_dict, k))
        results["KNN Recommender"]["diversity"].append(intra_list_diversity(knn_indices, feature_matrix))
        results["KNN Recommender"]["all_recs"].append(knn_recs)

        # C. Hybrid Engine
        hyb_df = hybrid_rec.recommend(q, reference_car_id=ref_id, user_id=user_id, top_k=k)
        hyb_recs = hyb_df["car_id"].tolist()
        hyb_indices = [content_rec.car_id_to_idx[cid] for cid in hyb_recs if cid in content_rec.car_id_to_idx]
        results["Hybrid Engine"]["p_at_k"].append(precision_at_k(hyb_recs, rel_ids, k))
        results["Hybrid Engine"]["r_at_k"].append(recall_at_k(hyb_recs, rel_ids, k))
        results["Hybrid Engine"]["ndcg"].append(ndcg_at_k(hyb_recs, rel_dict, k))
        results["Hybrid Engine"]["diversity"].append(intra_list_diversity(hyb_indices, feature_matrix))
        results["Hybrid Engine"]["all_recs"].append(hyb_recs)

        # D. Collaborative Filtering
        if cf_rec is not None:
            cf_df = cf_rec.recommend_for_user(user_id, top_k=k)
            cf_recs = cf_df["car_id"].tolist()
            cf_indices = [content_rec.car_id_to_idx[cid] for cid in cf_recs if cid in content_rec.car_id_to_idx]
            results["Collaborative Filtering"]["p_at_k"].append(precision_at_k(cf_recs, rel_ids, k))
            results["Collaborative Filtering"]["r_at_k"].append(recall_at_k(cf_recs, rel_ids, k))
            results["Collaborative Filtering"]["ndcg"].append(ndcg_at_k(cf_recs, rel_dict, k))
            results["Collaborative Filtering"]["diversity"].append(intra_list_diversity(cf_indices, feature_matrix))
            results["Collaborative Filtering"]["all_recs"].append(cf_recs)

    # 3. Aggregate into benchmark table
    rows = []
    for model_name, metrics in results.items():
        if not metrics["p_at_k"]:
            continue
        coverage = catalog_coverage(metrics["all_recs"], catalog_ids)
        rows.append({
            "Model": model_name,
            f"Precision@{k}": np.round(np.mean(metrics["p_at_k"]), 4),
            f"Recall@{k}": np.round(np.mean(metrics["r_at_k"]), 4),
            f"NDCG@{k}": np.round(np.mean(metrics["ndcg"]), 4),
            "Catalog Coverage": np.round(coverage * 100, 2),  # Percentage
            "Intra-List Diversity": np.round(np.mean(metrics["diversity"]), 4),
        })

    benchmark_df = pd.DataFrame(rows)
    return benchmark_df


if __name__ == "__main__":
    print("Testing evaluation module...")
    p = precision_at_k(["C1", "C2", "C3"], {"C2", "C4"}, k=3)
    r = recall_at_k(["C1", "C2", "C3"], {"C2", "C4"}, k=3)
    n = ndcg_at_k(["C1", "C2"], {"C2": 1.0, "C1": 0.5}, k=2)
    print(f"Sample test: P@3={p:.3f}, R@3={r:.3f}, NDCG@2={n:.3f}")
