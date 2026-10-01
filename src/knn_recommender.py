"""
KNN Recommender Module.
Uses NearestNeighbors algorithm in vector space to retrieve nearest vehicles
with sub-millisecond latency.
"""

from typing import Dict, List, Optional, Tuple, Any
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from src.preprocessing import CarDataPreprocessor


class KNNRecommender:
    """
    Unsupervised Nearest Neighbors recommender operating over high-dimensional
    engineered automotive feature space.
    """

    def __init__(self, preprocessor: CarDataPreprocessor, n_neighbors: int = 25) -> None:
        self.preprocessor = preprocessor
        self.n_neighbors = n_neighbors
        self.model = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
        self.cars_df: Optional[pd.DataFrame] = None
        self.feature_matrix: Optional[csr_matrix] = None
        self.car_id_to_idx: Dict[str, int] = {}
        self.idx_to_car_id: Dict[int, str] = {}
        self.is_fitted = False

    def fit(self, cars_df: pd.DataFrame) -> "KNNRecommender":
        """Fits NearestNeighbors on transformed sparse feature matrix."""
        self.cars_df = cars_df.reset_index(drop=True)
        self.feature_matrix, _ = self.preprocessor.transform(self.cars_df)
        self.model.fit(self.feature_matrix)

        self.car_id_to_idx = {cid: idx for idx, cid in enumerate(self.cars_df["car_id"])}
        self.idx_to_car_id = {idx: cid for idx, cid in enumerate(self.cars_df["car_id"])}
        self.is_fitted = True
        return self

    def get_similar_cars(self, car_id: str, top_k: int = 5) -> pd.DataFrame:
        """
        Queries top_k closest cars by cosine distance. Excludes the queried car.
        """
        if not self.is_fitted or self.cars_df is None or self.feature_matrix is None:
            raise ValueError("KNNRecommender must be fitted before querying.")

        if car_id not in self.car_id_to_idx:
            raise ValueError(f"Car ID '{car_id}' not found in catalog.")

        target_idx = self.car_id_to_idx[car_id]
        target_vec = self.feature_matrix[target_idx]

        # Query neighbors with ample buffer to discover distinct models
        query_k = min(max(self.n_neighbors, top_k * 10), len(self.cars_df))
        distances, indices = self.model.kneighbors(target_vec, n_neighbors=query_k)

        flat_dists = distances.flatten()
        flat_indices = indices.flatten()

        target_model = self.cars_df.iloc[target_idx]["model"]
        seen_models = {target_model}

        filtered_indices = []
        filtered_dists = []

        # Pass 1: Select distinct competitor vehicle models
        for idx, dist in zip(flat_indices, flat_dists):
            if idx == target_idx:
                continue
            m = self.cars_df.iloc[idx]["model"]
            if m not in seen_models:
                seen_models.add(m)
                filtered_indices.append(idx)
                filtered_dists.append(dist)
            if len(filtered_indices) == top_k:
                break

        # Pass 2: Fill remaining slots if distinct models are fewer than top_k
        if len(filtered_indices) < top_k:
            for idx, dist in zip(flat_indices, flat_dists):
                if idx == target_idx or idx in filtered_indices:
                    continue
                filtered_indices.append(idx)
                filtered_dists.append(dist)
                if len(filtered_indices) == top_k:
                    break

        results = self.cars_df.iloc[filtered_indices].copy()
        results["knn_distance"] = np.round(filtered_dists, 4)
        # Cosine similarity = 1 - cosine distance
        results["similarity_score"] = np.round(np.clip(1.0 - np.array(filtered_dists), 0.0, 1.0), 4)
        results["match_percentage"] = np.round(results["similarity_score"] * 100, 1)

        return results

    def recommend_from_vector(self, query_matrix: csr_matrix, top_k: int = 5) -> pd.DataFrame:
        """Queries nearest neighbors from an arbitrary feature vector."""
        if not self.is_fitted or self.cars_df is None:
            raise ValueError("KNNRecommender must be fitted before querying.")

        distances, indices = self.model.kneighbors(query_matrix, n_neighbors=top_k)
        flat_dists = distances.flatten()
        flat_indices = indices.flatten()

        results = self.cars_df.iloc[flat_indices].copy()
        results["knn_distance"] = np.round(flat_dists, 4)
        results["similarity_score"] = np.round(np.clip(1.0 - np.array(flat_dists), 0.0, 1.0), 4)
        results["match_percentage"] = np.round(results["similarity_score"] * 100, 1)
        return results

    def save(self, filepath: str) -> None:
        """Saves fitted KNN model state."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "KNNRecommender":
        """Loads fitted KNN model state."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
