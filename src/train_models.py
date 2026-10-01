"""
Script to train all recommenders and serialize model artifacts into models/
for low-latency serving in Streamlit.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import pandas as pd
from src.preprocessing import CarDataPreprocessor
from src.content_recommender import ContentBasedRecommender
from src.knn_recommender import KNNRecommender
from src.collaborative import CollaborativeFilteringRecommender
from src.hybrid_recommender import HybridCarRecommender


def train_and_export_all():
    print("Loading data...")
    cars_path = "data/processed/cars_synthesized_5k.csv"
    inter_path = "data/processed/user_interactions.csv"

    df_cars = pd.read_csv(cars_path)
    df_inter = pd.read_csv(inter_path)

    os.makedirs("models", exist_ok=True)

    print("Fitting Preprocessor...")
    preprocessor = CarDataPreprocessor()
    preprocessor.fit(df_cars)
    preprocessor.save("models/preprocessor.pkl")

    print("Fitting Content-Based Recommender...")
    content_rec = ContentBasedRecommender(preprocessor).fit(df_cars)

    print("Fitting KNN Recommender...")
    knn_rec = KNNRecommender(preprocessor, n_neighbors=25).fit(df_cars)
    knn_rec.save("models/knn_model.pkl")

    print("Fitting Collaborative Filtering...")
    cf_rec = CollaborativeFilteringRecommender(n_factors=20).fit(df_cars, df_inter)
    cf_rec.save("models/cf_model.pkl")

    print("All artifacts successfully trained and saved into models/")


if __name__ == "__main__":
    train_and_export_all()
