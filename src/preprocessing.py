"""
Preprocessing and Feature Engineering Module for Brand New Car Recommendation System.
Provides cleaning, feature engineering, categorical encoding, numeric scaling,
text TF-IDF vectorization, and model persistence for brand new showroom cars.
"""

from typing import Dict, List, Optional, Tuple, Any
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer


class CarDataPreprocessor:
    """
    Robust pipeline for transforming brand-new car catalog data into
    engineered feature representations for similarity and recommendation models.
    """

    NUMERIC_COLS = [
        "price_lakh",
        "power_bhp",
        "engine_cc",
        "mileage_kmpl",
        "seats",
        "safety_rating",
        "price_per_bhp",
        "power_to_displacement",
        "feature_count",
    ]

    CATEGORICAL_COLS = [
        "brand",
        "body_type",
        "fuel_type",
        "transmission",
    ]

    def __init__(self, tfidf_max_features: int = 80) -> None:
        self.scaler = RobustScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        self.tfidf = TfidfVectorizer(
            token_pattern=r"(?u)\b[\w\-\+]+\b",
            ngram_range=(1, 2),
            max_features=tfidf_max_features,
        )
        self.is_fitted = False
        self.feature_names_: List[str] = []

    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans data types, ensures price columns exist, and handles potential missing values.
        """
        df_clean = df.copy()

        # Handle price columns (both price_lakh and price)
        if "price_lakh" not in df_clean.columns and "price" in df_clean.columns:
            df_clean["price_lakh"] = pd.to_numeric(df_clean["price"], errors="coerce")
        elif "price_lakh" in df_clean.columns:
            df_clean["price_lakh"] = pd.to_numeric(df_clean["price_lakh"], errors="coerce")

        if "price" not in df_clean.columns and "price_lakh" in df_clean.columns:
            df_clean["price"] = df_clean["price_lakh"]

        df_clean["price_lakh"] = df_clean["price_lakh"].fillna(df_clean["price_lakh"].median() if not df_clean["price_lakh"].dropna().empty else 15.0)
        df_clean["price"] = df_clean["price_lakh"]

        if "price_cr" not in df_clean.columns:
            df_clean["price_cr"] = df_clean["price_lakh"] / 100.0

        if "mileage_kmpl" in df_clean.columns:
            df_clean["mileage_kmpl"] = pd.to_numeric(df_clean["mileage_kmpl"], errors="coerce").fillna(18.0)

        if "range_km" in df_clean.columns:
            df_clean["range_km"] = pd.to_numeric(df_clean["range_km"], errors="coerce").fillna(0)
        else:
            df_clean["range_km"] = 0

        if "power_bhp" in df_clean.columns:
            df_clean["power_bhp"] = pd.to_numeric(df_clean["power_bhp"], errors="coerce").fillna(120.0)

        if "engine_cc" in df_clean.columns:
            df_clean["engine_cc"] = pd.to_numeric(df_clean["engine_cc"], errors="coerce").fillna(0)

        if "safety_rating" in df_clean.columns:
            df_clean["safety_rating"] = pd.to_numeric(df_clean["safety_rating"], errors="coerce").fillna(5)

        if "features" in df_clean.columns:
            df_clean["features"] = df_clean["features"].fillna("").astype(str)

        return df_clean

    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes domain-specific engineered features for brand-new cars:
        - price_per_bhp: price in lakhs / power_bhp (spend per horsepower)
        - power_to_displacement: power_bhp / (engine_cc + 100) (specific output)
        - feature_count: count of premium equipment
        - feature_soup: combined textual representation for TF-IDF
        """
        df_eng = df.copy()

        df_eng["price_per_bhp"] = df_eng["price_lakh"] / np.maximum(df_eng["power_bhp"], 10.0)
        df_eng["power_to_displacement"] = np.where(
            df_eng["engine_cc"] > 0,
            df_eng["power_bhp"] / (df_eng["engine_cc"] + 1e-5),
            df_eng["power_bhp"] / 1000.0,  # EV surrogate
        )
        df_eng["feature_count"] = df_eng["features"].apply(
            lambda f: len([x.strip() for x in f.split(",") if x.strip()])
        )

        df_eng["feature_soup"] = (
            df_eng["brand"].astype(str)
            + " "
            + df_eng["model"].astype(str)
            + " "
            + df_eng["body_type"].astype(str)
            + " "
            + df_eng["fuel_type"].astype(str)
            + " "
            + df_eng["transmission"].astype(str)
            + " "
            + df_eng["features"].str.replace(",", " ")
        ).str.lower()

        return df_eng

    def fit(self, df: pd.DataFrame) -> "CarDataPreprocessor":
        """Fits numerical scalers, categorical encoders, and TF-IDF vectorizer."""
        df_clean = self.clean_data(df)
        df_eng = self.engineer_features(df_clean)

        self.scaler.fit(df_eng[self.NUMERIC_COLS])
        self.encoder.fit(df_eng[self.CATEGORICAL_COLS])
        self.tfidf.fit(df_eng["feature_soup"])

        num_names = self.NUMERIC_COLS
        cat_names = list(self.encoder.get_feature_names_out(self.CATEGORICAL_COLS))
        tfidf_names = [f"tfidf_{w}" for w in self.tfidf.get_feature_names_out()]
        self.feature_names_ = num_names + cat_names + tfidf_names
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[csr_matrix, pd.DataFrame]:
        """
        Transforms raw data into a combined sparse matrix (scaled numeric + one-hot + TF-IDF)
        and returns the engineered DataFrame.
        """
        if not self.is_fitted:
            raise ValueError("CarDataPreprocessor must be fitted before calling transform().")

        df_clean = self.clean_data(df)
        df_eng = self.engineer_features(df_clean)

        num_scaled = self.scaler.transform(df_eng[self.NUMERIC_COLS])
        num_sparse = csr_matrix(num_scaled)
        cat_sparse = self.encoder.transform(df_eng[self.CATEGORICAL_COLS])
        text_sparse = self.tfidf.transform(df_eng["feature_soup"])

        combined_matrix = hstack([num_sparse, cat_sparse, text_sparse]).tocsr()
        return combined_matrix, df_eng

    def fit_transform(self, df: pd.DataFrame) -> Tuple[csr_matrix, pd.DataFrame]:
        """Fits transformers and produces combined sparse matrix."""
        return self.fit(df).transform(df)

    def save(self, filepath: str) -> None:
        """Serializes fitted preprocessor to disk."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "CarDataPreprocessor":
        """Loads fitted preprocessor from disk."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
