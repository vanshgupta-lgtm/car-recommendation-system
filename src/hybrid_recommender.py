"""
Hybrid Recommender Module for Brand New Cars.
Combines user preference matching, content/KNN similarity, quality metrics,
and collaborative filtering into a unified scoring engine with cold-start resilience
and support for budgets up to ₹15 Crore.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from src.preprocessing import CarDataPreprocessor
from src.content_recommender import ContentBasedRecommender
from src.knn_recommender import KNNRecommender
from src.collaborative import CollaborativeFilteringRecommender
from src.explainability import RecommendationExplainer, format_currency


class HybridCarRecommender:
    """
    Production-grade Hybrid Recommendation Engine for Brand New Vehicles.
    Handles cold start (new users with no history), item-to-item similarity,
    and multi-objective constraint optimization up to ₹15.0 Crore.
    """

    def __init__(
        self,
        preprocessor: CarDataPreprocessor,
        content_rec: Optional[ContentBasedRecommender] = None,
        knn_rec: Optional[KNNRecommender] = None,
        cf_rec: Optional[CollaborativeFilteringRecommender] = None,
    ) -> None:
        self.preprocessor = preprocessor
        self.content_rec = content_rec or ContentBasedRecommender(preprocessor)
        self.knn_rec = knn_rec or KNNRecommender(preprocessor)
        self.cf_rec = cf_rec
        self.explainer = RecommendationExplainer()
        self.cars_df: Optional[pd.DataFrame] = None
        self.is_fitted = False

    def fit(self, cars_df: pd.DataFrame, interactions_df: Optional[pd.DataFrame] = None) -> "HybridCarRecommender":
        """
        Fits all constituent recommenders on the catalog and interaction logs.
        """
        self.cars_df = cars_df.reset_index(drop=True)
        self.content_rec.fit(self.cars_df)
        self.knn_rec.fit(self.cars_df)

        if interactions_df is not None:
            self.cf_rec = CollaborativeFilteringRecommender()
            self.cf_rec.fit(self.cars_df, interactions_df)

        self.is_fitted = True
        return self

    def _compute_preference_score(self, car: pd.Series, pref: Dict[str, Any]) -> float:
        """
        Scores how well an individual brand-new car satisfies the user's explicit filter preferences.
        Returns a score in [0.0, 1.0].
        """
        scores = []
        weights = []

        # 1. Budget constraint (in Lakhs) - Critical user purchasing parameter
        max_budget = float(pref.get("max_budget", 0.0))
        min_budget = float(pref.get("min_budget", 0.0))
        price = float(car.get("price_lakh", car.get("price", 0.0)))

        if max_budget > 0:
            if price <= max_budget:
                # Cars within budget get rewarded; cars utilizing budget efficiently score highest
                budget_score = 1.0 - 0.12 * max(0.0, (max_budget - price) / max_budget)
            else:
                # Sharp penalty for exceeding user budget cap
                overshoot = (price - max_budget) / max_budget
                budget_score = max(0.0, 1.0 - 4.5 * overshoot)
            scores.append(budget_score)
            weights.append(5.0)

        if min_budget > 0:
            if price >= min_budget:
                scores.append(1.0)
                weights.append(2.0)
            else:
                undershoot = (min_budget - price) / min_budget
                scores.append(max(0.0, 1.0 - 3.0 * undershoot))
                weights.append(2.5)

        # 2. Body type constraint
        body_pref = pref.get("body_type")
        if body_pref and body_pref != "All":
            body_match = 1.0 if str(car.get("body_type", "")).lower() == body_pref.lower() else 0.0
            scores.append(body_match)
            weights.append(2.2)

        # 3. Fuel type constraint
        fuel_pref = pref.get("fuel_type")
        if fuel_pref and fuel_pref != "All":
            fuel_match = 1.0 if str(car.get("fuel_type", "")).lower() == fuel_pref.lower() else 0.0
            scores.append(fuel_match)
            weights.append(2.0)

        # 4. Transmission constraint
        trans_pref = pref.get("transmission")
        if trans_pref and trans_pref != "All":
            trans_match = 1.0 if str(car.get("transmission", "")).lower() == trans_pref.lower() else 0.0
            scores.append(trans_match)
            weights.append(1.8)

        # 5. Seating capacity
        min_seats = int(pref.get("min_seats", 0))
        if min_seats > 0:
            seats = int(car.get("seats", 5))
            seat_score = 1.0 if seats >= min_seats else max(0.0, 1.0 - 0.4 * (min_seats - seats))
            scores.append(seat_score)
            weights.append(1.5)

        # 6. Minimum mileage / Efficiency
        min_mileage = float(pref.get("min_mileage", 0.0))
        if min_mileage > 0:
            if car.get("fuel_type") == "Electric":
                # Electric vehicles inherently satisfy fuel economy requirements
                mileage_score = 1.0
            else:
                mileage = float(car.get("mileage_kmpl", 0.0))
                if mileage >= min_mileage:
                    mileage_score = min(1.0, 0.85 + 0.15 * ((mileage - min_mileage) / 10.0))
                else:
                    deficit = (min_mileage - mileage) / min_mileage
                    mileage_score = max(0.0, 1.0 - 2.0 * deficit)
            scores.append(mileage_score)
            weights.append(1.2)

        # 7. Brands
        selected_brands = pref.get("brands", [])
        if selected_brands and "All" not in selected_brands:
            brand_match = 1.0 if car.get("brand") in selected_brands else 0.0
            scores.append(brand_match)
            weights.append(2.0)

        # 8. Desired features overlap
        user_features = pref.get("features", [])
        if user_features:
            car_feats = [f.strip().lower() for f in str(car.get("features", "")).split(",") if f.strip()]
            matched = sum(1 for uf in user_features if any(uf.lower() in cf for cf in car_feats))
            feat_score = matched / len(user_features)
            scores.append(feat_score)
            weights.append(2.0)

        # 9. Minimum safety rating
        min_safety = int(pref.get("min_safety", 0))
        if min_safety > 0:
            safety = int(car.get("safety_rating", 3))
            safety_score = 1.0 if safety >= min_safety else max(0.0, 1.0 - 0.35 * (min_safety - safety))
            scores.append(safety_score)
            weights.append(1.5)

        if not scores:
            return 0.8

        return float(np.average(scores, weights=weights))

    def recommend(
        self,
        preferences: Dict[str, Any],
        reference_car_id: Optional[str] = None,
        user_id: Optional[str] = None,
        top_k: int = 10,
        weights: Optional[Dict[str, float]] = None,
        strict_filters: bool = False,
    ) -> pd.DataFrame:
        """
        Executes hybrid recommendation pipeline for brand-new cars.
        """
        if not self.is_fitted or self.cars_df is None:
            raise ValueError("Hybrid recommender not fitted.")

        df = self.cars_df.copy()

        # Budget Envelope Enforcement: ensure recommended vehicles strictly adhere to user budget
        price_col = "price_lakh" if "price_lakh" in df.columns else "price"
        max_b = float(preferences.get("max_budget", 0.0))
        min_b = float(preferences.get("min_budget", 0.0))

        budget_filtered = df.copy()
        if max_b > 0:
            upper_tol = 1.05 if strict_filters else 1.15
            budget_filtered = budget_filtered[budget_filtered[price_col] <= max_b * upper_tol]
        if min_b > 0:
            lower_tol = 0.95 if strict_filters else 0.80
            budget_filtered = budget_filtered[budget_filtered[price_col] >= min_b * lower_tol]

        # Keep budget-filtered subset whenever sufficient candidates exist
        if len(budget_filtered) >= max(3, min(top_k, 5)):
            df = budget_filtered

        # Apply strict categorical filters if requested
        if strict_filters:
            if preferences.get("body_type") and preferences["body_type"] != "All":
                matched = df[df["body_type"].str.lower() == preferences["body_type"].lower()]
                if len(matched) > 0:
                    df = matched
            if preferences.get("fuel_type") and preferences["fuel_type"] != "All":
                matched = df[df["fuel_type"].str.lower() == preferences["fuel_type"].lower()]
                if len(matched) > 0:
                    df = matched
            if preferences.get("transmission") and preferences["transmission"] != "All":
                matched = df[df["transmission"].str.lower() == preferences["transmission"].lower()]
                if len(matched) > 0:
                    df = matched
            if preferences.get("brands") and "All" not in preferences["brands"]:
                matched = df[df["brand"].isin(preferences["brands"])]
                if len(matched) > 0:
                    df = matched
            if preferences.get("min_seats"):
                matched = df[df["seats"] >= int(preferences["min_seats"])]
                if len(matched) > 0:
                    df = matched

        if len(df) == 0:
            df = self.cars_df.copy()

        # Component weights
        w_pref = 0.50
        w_sim = 0.25 if reference_car_id else 0.0
        w_quality = 0.25
        w_cf = 0.0

        if weights:
            w_pref = weights.get("pref", w_pref)
            w_sim = weights.get("sim", w_sim)
            w_quality = weights.get("quality", w_quality)
            w_cf = weights.get("cf", w_cf)

        # 1. Preference score
        pref_scores = df.apply(lambda row: self._compute_preference_score(row, preferences), axis=1).values

        # 2. Similarity score
        if reference_car_id and reference_car_id in self.content_rec.car_id_to_idx:
            ref_idx = self.content_rec.car_id_to_idx[reference_car_id]
            ref_vec = self.content_rec.feature_matrix[ref_idx]
            from sklearn.metrics.pairwise import cosine_similarity
            all_sim = cosine_similarity(ref_vec, self.content_rec.feature_matrix).flatten()
            sim_scores = np.array([all_sim[self.content_rec.car_id_to_idx[cid]] for cid in df["car_id"]])
        else:
            sim_scores = np.full(len(df), 0.5)

        # 3. Quality score: Safety (50%) + Power tier (30%) + Efficiency / Range (20%)
        safety_norm = df["safety_rating"].values / 5.0
        power_norm = np.clip(df["power_bhp"].values / 800.0, 0.0, 1.0)
        
        is_electric = (df["fuel_type"] == "Electric").values
        eff_norm = np.zeros(len(df))
        if "range_km" in df.columns:
            eff_norm[is_electric] = np.clip(df.loc[is_electric, "range_km"].values / 600.0, 0.0, 1.0)
            eff_norm[~is_electric] = np.clip(df.loc[~is_electric, "mileage_kmpl"].values / 28.0, 0.0, 1.0)
        else:
            eff_norm[is_electric] = 1.0
            eff_norm[~is_electric] = np.clip(df.loc[~is_electric, "mileage_kmpl"].values / 28.0, 0.0, 1.0)
            
        quality_scores = 0.50 * safety_norm + 0.30 * power_norm + 0.20 * eff_norm

        # 4. Collaborative filtering
        if user_id and self.cf_rec is not None and w_cf > 0:
            all_cf_scores = self.cf_rec.predict_user_car_affinity(user_id)
            cf_scores = np.array([
                all_cf_scores[self.cf_rec.car_to_idx[cid]] / 5.0 if cid in self.cf_rec.car_to_idx else 0.7
                for cid in df["car_id"]
            ])
        else:
            cf_scores = np.full(len(df), 0.5)

        total_weight = w_pref + w_sim + w_quality + w_cf
        if total_weight <= 0:
            total_weight = 1.0

        hybrid_scores = (
            w_pref * pref_scores +
            w_sim * sim_scores +
            w_quality * quality_scores +
            w_cf * cf_scores
        ) / total_weight

        df["match_score"] = np.round(hybrid_scores, 4)
        df["hybrid_score"] = df["match_score"]
        df["match_percentage"] = np.round(np.clip(hybrid_scores * 100, 10.0, 99.5), 1)
        df["pref_score"] = np.round(pref_scores, 3)
        df["sim_score"] = np.round(sim_scores, 3)

        if reference_car_id:
            df = df[df["car_id"] != reference_car_id]

        # Model Diversity: Pick the best-scoring variant for each distinct vehicle model
        # Avoids overwhelming the user with 5 variants of the same single car
        df_sorted = df.sort_values(by="match_score", ascending=False)
        distinct_models = df_sorted.drop_duplicates(subset=["brand", "model"], keep="first")

        # In focused/niche segments (e.g., under-15L EVs), ensure all distinct vehicle choices
        # (like MG Comet EV, MG Windsor EV, Tiago EV, etc.) are included with zero duplicates.
        if len(distinct_models) <= 10:
            top_df = distinct_models.copy()
        elif len(distinct_models) >= top_k:
            top_df = distinct_models.head(top_k).copy()
        else:
            remaining_needed = top_k - len(distinct_models)
            used_ids = set(distinct_models["car_id"])
            remaining_df = df_sorted[~df_sorted["car_id"].isin(used_ids)].head(remaining_needed)
            top_df = pd.concat([distinct_models, remaining_df]).head(top_k).copy()

        # Attach Explainability
        explanations = []
        for _, row in top_df.iterrows():
            explanation = self.explainer.explain_preference_match(row, preferences)
            explanations.append(explanation)
        top_df["why_recommended"] = explanations

        return top_df
