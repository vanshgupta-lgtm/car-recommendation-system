"""
Collaborative Filtering Module.
Implements Matrix Factorization via TruncatedSVD on the User-Car interaction matrix
to capture latent taste dimensions (e.g. sporty performance vs family utility).
"""

from typing import Dict, List, Optional, Tuple, Any
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


class CollaborativeFilteringRecommender:
    """
    Matrix Factorization Collaborative Filtering using TruncatedSVD.
    Predicts user affinity for cars based on collective interaction patterns.
    """

    def __init__(self, n_factors: int = 20, random_state: int = 42) -> None:
        self.n_factors = n_factors
        self.random_state = random_state
        self.svd = TruncatedSVD(n_components=n_factors, random_state=random_state)
        self.cars_df: Optional[pd.DataFrame] = None
        self.interactions_df: Optional[pd.DataFrame] = None
        self.user_to_idx: Dict[str, int] = {}
        self.idx_to_user: Dict[int, str] = {}
        self.car_to_idx: Dict[str, int] = {}
        self.idx_to_car: Dict[int, str] = {}
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.global_mean_rating: float = 3.5
        self.car_popularity_: Optional[pd.Series] = None
        self.is_fitted = False

    def fit(self, cars_df: pd.DataFrame, interactions_df: pd.DataFrame) -> "CollaborativeFilteringRecommender":
        """
        Builds interaction matrix and fits TruncatedSVD to derive latent factors.
        """
        self.cars_df = cars_df.reset_index(drop=True)
        self.interactions_df = interactions_df.copy()

        # Map IDs to continuous indices
        unique_users = sorted(self.interactions_df["user_id"].unique())
        unique_cars = sorted(self.cars_df["car_id"].unique())

        self.user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
        self.idx_to_user = {i: uid for i, uid in enumerate(unique_users)}
        self.car_to_idx = {cid: i for i, cid in enumerate(unique_cars)}
        self.idx_to_car = {i: cid for i, cid in enumerate(unique_cars)}

        num_users = len(unique_users)
        num_cars = len(unique_cars)

        # Filter to interactions where both user and car exist in catalog
        valid_mask = self.interactions_df["car_id"].isin(self.car_to_idx) & self.interactions_df["user_id"].isin(self.user_to_idx)
        valid_inter = self.interactions_df[valid_mask]

        row_indices = valid_inter["user_id"].map(self.user_to_idx).astype(int).values
        col_indices = valid_inter["car_id"].map(self.car_to_idx).astype(int).values
        ratings = valid_inter["rating"].values.astype(float)

        interaction_matrix = csr_matrix((ratings, (row_indices, col_indices)), shape=(num_users, num_cars))
        self.global_mean_rating = float(np.mean(ratings)) if len(ratings) > 0 else 3.5

        # Fit SVD
        # Cap factors if catalog or users are small
        actual_factors = min(self.n_factors, num_users - 1, num_cars - 1)
        self.svd.n_components = actual_factors
        self.user_factors = self.svd.fit_transform(interaction_matrix)  # Shape: (num_users, factors)
        self.item_factors = self.svd.components_.T  # Shape: (num_cars, factors)

        # Calculate popular baseline for cold start
        rating_agg = self.interactions_df.groupby("car_id")["rating"].agg(["mean", "count"])
        # Bayesian weighted rating
        C = rating_agg["count"].median() if len(rating_agg) > 0 else 10.0
        m = self.global_mean_rating
        rating_agg["bayesian_score"] = (rating_agg["count"] * rating_agg["mean"] + C * m) / (rating_agg["count"] + C)
        self.car_popularity_ = rating_agg["bayesian_score"]

        self.is_fitted = True
        return self

    def predict_user_car_affinity(self, user_id: str) -> np.ndarray:
        """
        Returns predicted rating vector of length num_cars for the given user.
        If user is unknown, returns popular baseline scores.
        """
        if not self.is_fitted:
            raise ValueError("CollaborativeFilteringRecommender must be fitted first.")

        if user_id in self.user_to_idx and self.user_factors is not None and self.item_factors is not None:
            user_idx = self.user_to_idx[user_id]
            u_vec = self.user_factors[user_idx]  # shape: (factors,)
            pred_scores = np.dot(self.item_factors, u_vec)
            # Normalize to 0-5 scale
            pred_scores = np.clip(pred_scores + self.global_mean_rating, 1.0, 5.0)
            return pred_scores
        else:
            # Cold-start fallback: Bayesian average score mapped to car index
            scores = np.full(len(self.car_to_idx), self.global_mean_rating)
            if self.car_popularity_ is not None:
                for cid, score in self.car_popularity_.items():
                    if cid in self.car_to_idx:
                        scores[self.car_to_idx[cid]] = score
            return scores

    def recommend_for_user(self, user_id: str, top_k: int = 5, exclude_known: bool = True) -> pd.DataFrame:
        """
        Returns top_k personalized recommendations for a user.
        """
        if not self.is_fitted or self.cars_df is None:
            raise ValueError("Recommender not fitted.")

        predicted_ratings = self.predict_user_car_affinity(user_id)
        scores_series = pd.Series(predicted_ratings, index=[self.idx_to_car[i] for i in range(len(predicted_ratings))])

        if exclude_known and user_id in self.user_to_idx and self.interactions_df is not None:
            known_cars = set(self.interactions_df[self.interactions_df["user_id"] == user_id]["car_id"])
            scores_series = scores_series.drop(labels=known_cars, errors="ignore")

        top_car_ids = scores_series.nlargest(top_k).index.tolist()
        results = self.cars_df[self.cars_df["car_id"].isin(top_car_ids)].copy()
        results["predicted_rating"] = results["car_id"].map(scores_series).round(2)
        results["match_percentage"] = (results["predicted_rating"] / 5.0 * 100).round(1)

        # Sort in order of score
        results = results.sort_values(by="predicted_rating", ascending=False)
        return results

    def save(self, filepath: str) -> None:
        """Saves CF model state."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "CollaborativeFilteringRecommender":
        """Loads CF model state."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
