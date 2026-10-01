"""
Unit tests for evaluation metrics:
Precision@K, Recall@K, NDCG@K, Catalog Coverage, Intra-List Diversity.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from src.evaluation import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    catalog_coverage,
    intra_list_diversity,
)


def test_precision_at_k():
    """Verifies Precision@K calculation under normal and edge conditions."""
    recommended = ["C1", "C2", "C3", "C4", "C5"]
    relevant = {"C2", "C4", "C7"}

    # In top 3: C1 (no), C2 (yes), C3 (no) -> 1/3
    assert pytest.approx(precision_at_k(recommended, relevant, k=3)) == 1.0 / 3.0
    # In top 5: C2, C4 -> 2/5
    assert pytest.approx(precision_at_k(recommended, relevant, k=5)) == 2.0 / 5.0
    # Empty recommended
    assert precision_at_k([], relevant, k=5) == 0.0
    # Zero relevant
    assert precision_at_k(recommended, set(), k=5) == 0.0


def test_recall_at_k():
    """Verifies Recall@K calculation."""
    recommended = ["C1", "C2", "C3", "C4", "C5"]
    relevant = {"C2", "C4", "C7", "C8"}  # 4 relevant in total

    # In top 3: only C2 -> 1/4
    assert pytest.approx(recall_at_k(recommended, relevant, k=3)) == 1.0 / 4.0
    # In top 5: C2 and C4 -> 2/4
    assert pytest.approx(recall_at_k(recommended, relevant, k=5)) == 2.0 / 4.0
    # Empty relevant
    assert recall_at_k(recommended, set(), k=5) == 0.0


def test_ndcg_at_k():
    """Verifies NDCG@K calculation with graded relevance."""
    recommended = ["C1", "C2", "C3"]
    # Ideal order is C2 (rel=2.0), C1 (rel=1.0), C3 (rel=0.0)
    relevance = {"C1": 1.0, "C2": 2.0, "C3": 0.0}

    # Recommended order has C1 (1.0) at rank 1, C2 (2.0) at rank 2
    # DCG = 1.0/log2(2) + 2.0/log2(3) = 1.0 + 1.2619 = 2.2619
    # IDCG = 2.0/log2(2) + 1.0/log2(3) = 2.0 + 0.6309 = 2.6309
    # NDCG = 2.2619 / 2.6309 ~ 0.8597
    score = ndcg_at_k(recommended, relevance, k=2)
    assert 0.80 <= score <= 0.90

    # Perfect ranking
    perfect = ["C2", "C1", "C3"]
    assert pytest.approx(ndcg_at_k(perfect, relevance, k=3)) == 1.0

    # Zero relevance
    assert ndcg_at_k(recommended, {}, k=3) == 0.0


def test_catalog_coverage():
    """Verifies catalog coverage computation."""
    all_recs = [["C1", "C2"], ["C2", "C3"], ["C3", "C4"]]
    catalog = {"C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"}
    # Unique recs: C1, C2, C3, C4 = 4 / 8 = 0.5
    assert pytest.approx(catalog_coverage(all_recs, catalog)) == 0.5


def test_intra_list_diversity():
    """Verifies diversity calculation across feature vectors."""
    # 3 distinct orthogonal vectors
    matrix = csr_matrix([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    # Pairwise cosine distances between orthogonal vectors is 1.0
    diversity = intra_list_diversity([0, 1, 2], matrix)
    assert pytest.approx(diversity) == 1.0

    # Single item has 0 diversity
    assert intra_list_diversity([0], matrix) == 0.0
