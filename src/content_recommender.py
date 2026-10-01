"""
Content-Based Recommender Module.
Computes cosine similarity over dense engineered features and TF-IDF textual feature tags
to identify similar vehicles and score preference profiles.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessing import CarDataPreprocessor


class ContentBasedRecommender:
    """
    Content-Based Filtering using Cosine Similarity on feature matrices and TF-IDF vectors.
    """

    def __init__(self, preprocessor: CarDataPreprocessor) -> None:
        self.preprocessor = preprocessor
        self.cars_df: Optional[pd.DataFrame] = None
        self.feature_matrix: Optional[csr_matrix] = None
        self.car_id_to_idx: Dict[str, int] = {}
        self.idx_to_car_id: Dict[int, str] = {}

    def fit(self, cars_df: pd.DataFrame) -> "ContentBasedRecommender":
        """
        Fits recommender by computing feature matrix and building indexing maps.
        """
        self.cars_df = cars_df.reset_index(drop=True)
        self.feature_matrix, _ = self.preprocessor.transform(self.cars_df)
        self.car_id_to_idx = {cid: idx for idx, cid in enumerate(self.cars_df["car_id"])}
        self.idx_to_car_id = {idx: cid for idx, cid in enumerate(self.cars_df["car_id"])}
        return self

    def get_similar_cars(self, car_id: str, top_k: int = 5) -> pd.DataFrame:
        """
        Retrieves top_k most similar cars to a target car_id using cosine similarity.
        Excludes the target car itself.
        """
        if self.feature_matrix is None or self.cars_df is None:
            raise ValueError("Recommender must be fitted before finding similar cars.")

        if car_id not in self.car_id_to_idx:
            raise ValueError(f"Car ID '{car_id}' not found in catalog.")

        target_idx = self.car_id_to_idx[car_id]
        target_vec = self.feature_matrix[target_idx]

        # Compute cosine similarity between target car and all cars in catalog
        sim_scores = cosine_similarity(target_vec, self.feature_matrix).flatten()

        # Sort all candidates by cosine similarity in descending order
        sorted_indices = np.argsort(sim_scores)[::-1]

        target_model = self.cars_df.iloc[target_idx]["model"]
        seen_models = {target_model}

        distinct_indices = []

        # Pass 1: Select distinct competitor vehicle models
        for idx in sorted_indices:
            if idx == target_idx or sim_scores[idx] <= 0:
                continue
            m = self.cars_df.iloc[idx]["model"]
            if m not in seen_models:
                distinct_indices.append(idx)
                seen_models.add(m)
            if len(distinct_indices) == top_k:
                break

        # Pass 2: Fill remaining slots if distinct models are fewer than top_k
        if len(distinct_indices) < top_k:
            for idx in sorted_indices:
                if idx == target_idx or idx in distinct_indices:
                    continue
                distinct_indices.append(idx)
                if len(distinct_indices) == top_k:
                    break

        results = self.cars_df.iloc[distinct_indices].copy()
        results["similarity_score"] = np.round(sim_scores[distinct_indices], 4)
        results["match_percentage"] = np.round(results["similarity_score"] * 100, 1)

        return results

    def recommend_by_profile(self, preference_text: str, top_k: int = 5) -> pd.DataFrame:
        """
        Finds cars whose textual features best match a natural language query or preference tags.
        """
        if self.feature_matrix is None or self.cars_df is None:
            raise ValueError("Recommender must be fitted before querying.")

        query_vec = self.preprocessor.tfidf.transform([preference_text.lower()])
        # Compute similarity against the TF-IDF slice of the feature matrix
        tfidf_start_idx = len(self.preprocessor.NUMERIC_COLS) + len(
            self.preprocessor.encoder.get_feature_names_out(self.preprocessor.CATEGORICAL_COLS)
        )
        tfidf_features = self.feature_matrix[:, tfidf_start_idx:]

        sim_scores = cosine_similarity(query_vec, tfidf_features).flatten()
        top_indices = np.argsort(sim_scores)[::-1][:top_k]

        results = self.cars_df.iloc[top_indices].copy()
        results["text_similarity"] = np.round(sim_scores[top_indices], 4)
        results["match_percentage"] = np.round(np.clip(results["text_similarity"] * 100, 0, 100), 1)
        return results
