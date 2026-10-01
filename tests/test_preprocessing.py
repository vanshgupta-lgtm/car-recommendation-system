"""
Unit tests for brand new car data preprocessing and feature engineering.
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from src.preprocessing import CarDataPreprocessor


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    """Provides a realistic sample brand-new car DataFrame for testing."""
    return pd.DataFrame([
        {
            "car_id": "CAR0001",
            "brand": "Hyundai",
            "model": "Creta",
            "year": 2025,
            "condition": "Brand New",
            "price_lakh": 14.5,
            "price_cr": 0.145,
            "price": 14.5,
            "fuel_type": "Petrol",
            "transmission": "Automatic",
            "body_type": "SUV",
            "engine_cc": 1497,
            "power_bhp": 113,
            "mileage_kmpl": 17.0,
            "seats": 5,
            "safety_rating": 4,
            "features": "Panoramic Sunroof, Level 2 ADAS, Wireless Apple CarPlay"
        },
        {
            "car_id": "CAR0002",
            "brand": "Tata",
            "model": "Nexon",
            "year": 2025,
            "condition": "Brand New",
            "price_lakh": 11.2,
            "price_cr": 0.112,
            "price": 11.2,
            "fuel_type": "Diesel",
            "transmission": "Manual",
            "body_type": "SUV",
            "engine_cc": 1497,
            "power_bhp": 118,
            "mileage_kmpl": 21.0,
            "seats": 5,
            "safety_rating": 5,
            "features": "Panoramic Sunroof, 360 3D Camera, Electronic Stability Program (ESP)"
        },
        {
            "car_id": "CAR0003",
            "brand": "Rolls-Royce",
            "model": "Phantom VIII Extended",
            "year": 2025,
            "condition": "Brand New",
            "price_lakh": 1150.0,
            "price_cr": 11.5,
            "price": 1150.0,
            "fuel_type": "Petrol",
            "transmission": "Automatic",
            "body_type": "Sedan",
            "engine_cc": 6749,
            "power_bhp": 563,
            "mileage_kmpl": 6.7,
            "seats": 4,
            "safety_rating": 5,
            "features": "Bespoke Leather Upholstery, Burmester 4D Audio, Massage Seats"
        }
    ])


def test_clean_data(sample_raw_data: pd.DataFrame):
    """Verifies that missing values are cleanly imputed and types are sound."""
    dirty_df = sample_raw_data.copy()
    dirty_df.loc[0, "price_lakh"] = np.nan
    dirty_df.loc[1, "mileage_kmpl"] = np.nan

    cleaned = CarDataPreprocessor.clean_data(dirty_df)
    assert cleaned["price_lakh"].isnull().sum() == 0
    assert cleaned["mileage_kmpl"].isnull().sum() == 0
    assert len(cleaned) == len(sample_raw_data)


def test_feature_engineering(sample_raw_data: pd.DataFrame):
    """Verifies that all domain-specific engineered features are created."""
    cleaned = CarDataPreprocessor.clean_data(sample_raw_data)
    eng = CarDataPreprocessor.engineer_features(cleaned)

    expected_cols = [
        "price_per_bhp", "power_to_displacement", "feature_count", "feature_soup"
    ]
    for col in expected_cols:
        assert col in eng.columns, f"Missing engineered column {col}"

    assert eng.loc[0, "feature_count"] == 3
    assert "creta" in eng.loc[0, "feature_soup"]
    # Check Rolls-Royce pricing in Lakhs (11.5 Cr = 1150 Lakhs)
    assert eng.loc[2, "price_lakh"] == 1150.0


def test_preprocessor_fit_transform(sample_raw_data: pd.DataFrame):
    """Verifies that the preprocessor builds a valid sparse feature matrix."""
    prep = CarDataPreprocessor(tfidf_max_features=20)
    matrix, df_eng = prep.fit_transform(sample_raw_data)

    assert prep.is_fitted
    assert matrix.shape[0] == len(sample_raw_data)
    assert matrix.shape[1] > len(prep.NUMERIC_COLS)
    assert len(prep.feature_names_) == matrix.shape[1]


def test_preprocessor_persistence(sample_raw_data: pd.DataFrame):
    """Verifies saving and loading preprocessor preserves state."""
    prep = CarDataPreprocessor()
    prep.fit(sample_raw_data)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        prep.save(tmp_path)
        loaded = CarDataPreprocessor.load(tmp_path)
        assert loaded.is_fitted
        assert loaded.feature_names_ == prep.feature_names_
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
