"""
Unit tests for recommender algorithms:
ContentBasedRecommender, KNNRecommender, CollaborativeFilteringRecommender, HybridCarRecommender.
"""

import pandas as pd
import numpy as np
import pytest
from src.preprocessing import CarDataPreprocessor
from src.content_recommender import ContentBasedRecommender
from src.knn_recommender import KNNRecommender
from src.collaborative import CollaborativeFilteringRecommender
from src.hybrid_recommender import HybridCarRecommender


@pytest.fixture(scope="module")
def full_test_dataset():
    """Loads 5k dataset and interaction dataset for recommender tests."""
    df_cars = pd.read_csv("data/processed/cars_synthesized_5k.csv").head(200)
    df_inter = pd.read_csv("data/processed/user_interactions.csv").head(800)
    prep = CarDataPreprocessor().fit(df_cars)
    return df_cars, df_inter, prep


def test_content_recommender(full_test_dataset):
    """Verifies content-based similarity retrieval and bounds."""
    df_cars, _, prep = full_test_dataset
    c_rec = ContentBasedRecommender(prep).fit(df_cars)

    target_id = df_cars.iloc[0]["car_id"]
    sim_cars = c_rec.get_similar_cars(target_id, top_k=5)

    assert len(sim_cars) == 5
    assert target_id not in sim_cars["car_id"].values
    assert (sim_cars["similarity_score"] >= 0.0).all()
    assert (sim_cars["similarity_score"] <= 1.0).all()
    assert (sim_cars["match_percentage"] >= 0.0).all()


def test_content_recommender_text_search(full_test_dataset):
    """Verifies natural language profile matching."""
    df_cars, _, prep = full_test_dataset
    c_rec = ContentBasedRecommender(prep).fit(df_cars)

    results = c_rec.recommend_by_profile("Sunroof Petrol Automatic", top_k=3)
    assert len(results) == 3
    assert "text_similarity" in results.columns


def test_knn_recommender(full_test_dataset):
    """Verifies KNN vector retrieval and distance calculations."""
    df_cars, _, prep = full_test_dataset
    knn_rec = KNNRecommender(prep, n_neighbors=10).fit(df_cars)

    target_id = df_cars.iloc[0]["car_id"]
    neighbors = knn_rec.get_similar_cars(target_id, top_k=4)

    assert len(neighbors) == 4
    assert target_id not in neighbors["car_id"].values
    assert (neighbors["knn_distance"] >= 0.0).all()
    assert (neighbors["similarity_score"] >= 0.0).all()


def test_collaborative_filtering(full_test_dataset):
    """Verifies Matrix Factorization collaborative recommendations and cold start."""
    df_cars, df_inter, _ = full_test_dataset
    cf_rec = CollaborativeFilteringRecommender(n_factors=5).fit(df_cars, df_inter)

    # Known user
    user_id = df_inter.iloc[0]["user_id"]
    recs = cf_rec.recommend_for_user(user_id, top_k=3)
    assert len(recs) == 3
    assert "predicted_rating" in recs.columns

    # Cold start user (unknown)
    cold_user = "U9999_UNKNOWN"
    cold_recs = cf_rec.recommend_for_user(cold_user, top_k=3)
    assert len(cold_recs) == 3
    assert (cold_recs["predicted_rating"] >= 1.0).all()


def test_hybrid_recommender_cold_start(full_test_dataset):
    """Verifies hybrid recommendations for a cold-start user (preferences only)."""
    df_cars, df_inter, prep = full_test_dataset
    c_rec = ContentBasedRecommender(prep).fit(df_cars)
    knn_rec = KNNRecommender(prep).fit(df_cars)
    cf_rec = CollaborativeFilteringRecommender(n_factors=5).fit(df_cars, df_inter)
    hybrid_rec = HybridCarRecommender(prep, c_rec, knn_rec, cf_rec).fit(df_cars, df_inter)

    query = {
        "max_budget": 18.0,
        "body_type": "SUV",
        "fuel_type": "Diesel",
        "min_seats": 5,
        "min_mileage": 15.0,
        "features": ["Sunroof"]
    }

    recs = hybrid_rec.recommend(query, top_k=5)
    assert len(recs) == 5
    assert "match_score" in recs.columns
    assert "why_recommended" in recs.columns
    assert (recs["match_percentage"] <= 100.0).all()
    # Check that explanation text is generated
    assert len(recs.iloc[0]["why_recommended"]) > 10


def test_hybrid_recommender_with_reference_car(full_test_dataset):
    """Verifies hybrid recommendations when user provides a reference car they like."""
    df_cars, df_inter, prep = full_test_dataset
    c_rec = ContentBasedRecommender(prep).fit(df_cars)
    knn_rec = KNNRecommender(prep).fit(df_cars)
    hybrid_rec = HybridCarRecommender(prep, c_rec, knn_rec).fit(df_cars)

    ref_car_id = df_cars.iloc[0]["car_id"]
    query = {"max_budget": 25.0}

    recs = hybrid_rec.recommend(query, reference_car_id=ref_car_id, top_k=5)
    assert len(recs) == 5
    assert ref_car_id not in recs["car_id"].values
    assert "sim_score" in recs.columns
