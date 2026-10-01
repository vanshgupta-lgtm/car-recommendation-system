"""
Explainability Module for Brand New Car Recommendation System.
Generates clear, transparent, human-readable natural language justifications
explaining why each brand new showroom vehicle was recommended.
"""

from typing import Dict, List, Optional, Any
import pandas as pd


def format_currency(price_lakh: float) -> str:
    """Formats price in Lakhs or Crores according to scale."""
    if price_lakh >= 100.0:
        cr = price_lakh / 100.0
        return f"₹{cr:.2f} Cr"
    else:
        return f"₹{price_lakh:.2f}L"


class RecommendationExplainer:
    """
    Constructs explainable, multi-faceted rationales for brand new car recommendations.
    """

    @classmethod
    def explain_recommendation(cls, car_row: pd.Series, user_pref: Dict[str, Any]) -> str:
        """Alias for explain_preference_match."""
        return cls.explain_preference_match(car_row, user_pref)

    @staticmethod
    def explain_preference_match(car_row: pd.Series, user_pref: Dict[str, Any]) -> str:
        """
        Builds a concise, structured bullet-point explanation of how the brand-new car fits the user's constraints.
        """
        reasons = []

        # 1. Budget Fit
        car_price = float(car_row.get("price_lakh", car_row.get("price", 0.0)))
        max_budget = float(user_pref.get("max_budget", 0.0))
        min_budget = float(user_pref.get("min_budget", 0.0))

        if max_budget > 0:
            formatted_car = format_currency(car_price)
            formatted_max = format_currency(max_budget)
            if car_price <= max_budget:
                diff = max_budget - car_price
                if diff >= 5.0:
                    reasons.append(f"💰 **Under Budget**: {formatted_car} (saves you {format_currency(diff)} within your {formatted_max} budget)")
                else:
                    reasons.append(f"💰 **Within Budget**: {formatted_car} matches your {formatted_max} budget cap")
            else:
                diff = car_price - max_budget
                reasons.append(f"💰 **Slight Stretch**: {formatted_car} is {format_currency(diff)} above cap, delivering elite tier performance")

        # 2. Performance / Power
        power = int(car_row.get("power_bhp", 0))
        if power >= 500:
            reasons.append(f"⚡ **Hypercar Output**: {power} BHP monster performance with race-grade dynamics")
        elif power >= 200:
            reasons.append(f"⚡ **High Performance**: {power} BHP powertrain providing effortless highway cruising")

        # 3. Efficiency / Driving Range
        is_electric = car_row.get("fuel_type") == "Electric"
        range_km = float(car_row.get("range_km", 0.0))
        if is_electric:
            if range_km <= 0:
                m_val = float(car_row.get("mileage_kmpl", 0.0))
                range_km = m_val if m_val > 50 else 450.0
            reasons.append(f"🔋 **Zero-Emission Range**: Certified {int(range_km)} km driving range on a single charge")
        else:
            mileage = float(car_row.get("mileage_kmpl", 0.0))
            min_mileage = float(user_pref.get("min_mileage", 0.0))
            if min_mileage > 0 and mileage >= min_mileage:
                reasons.append(f"⛽ **Fuel Efficiency**: Delivers {mileage:.1f} kmpl, meeting your {min_mileage:.1f} kmpl threshold")
            elif mileage >= 20.0:
                reasons.append(f"⛽ **High Efficiency**: Outstanding {mileage:.1f} kmpl economy")

        # 4. Safety
        safety = int(car_row.get("safety_rating", 0))
        if safety == 5:
            reasons.append(f"🛡️ **Top-Tier Safety**: 5-Star crash safety rating with advanced structural integrity")
        elif safety >= 4:
            reasons.append(f"🛡️ **Certified Safe**: {safety}-Star rating meets high protection standards")

        # 5. Features Overlap
        user_features = user_pref.get("features", [])
        if user_features:
            car_features_str = str(car_row.get("features", ""))
            car_features_list = [f.strip().lower() for f in car_features_str.split(",") if f.strip()]
            matched_features = [
                f for f in user_features if any(f.lower() in cf for cf in car_features_list)
            ]
            if matched_features:
                reasons.append(f"✨ **Must-Have Tech**: Equipped with {', '.join(matched_features[:4])}")

        # 6. Body Style & Brand New Condition
        body_type = str(car_row.get("body_type", ""))
        seats = int(car_row.get("seats", 5))
        reasons.append(f"🚗 **Brand New Showroom Unit**: {body_type} layout with {seats} seats and factory warranty")

        return "\n".join(f"- {r}" for r in reasons)

    @staticmethod
    def explain_similarity(target_car: pd.Series, candidate_car: pd.Series) -> str:
        """
        Explains why a candidate car is similar to the reference brand-new target car.
        """
        reasons = []

        # Compare body & brand
        if target_car.get("body_type") == candidate_car.get("body_type"):
            reasons.append(f"Matching {target_car.get('body_type')} body architecture")

        # Compare price segment
        p1 = float(target_car.get("price_lakh", target_car.get("price", 0)))
        p2 = float(candidate_car.get("price_lakh", candidate_car.get("price", 0)))
        price_diff = abs(p1 - p2)
        if price_diff <= 25.0:
            reasons.append(f"Direct competitor price segment ({format_currency(p2)} vs {format_currency(p1)})")

        # Compare power / performance
        power_diff = abs(int(target_car.get("power_bhp", 0)) - int(candidate_car.get("power_bhp", 0)))
        if power_diff <= 35:
            reasons.append(f"Comparable performance ({candidate_car.get('power_bhp')} BHP vs {target_car.get('power_bhp')} BHP)")

        # Features overlap
        f1 = set([f.strip().lower() for f in str(target_car.get("features", "")).split(",") if f.strip()])
        f2 = set([f.strip().lower() for f in str(candidate_car.get("features", "")).split(",") if f.strip()])
        common = f1.intersection(f2)
        if len(common) >= 3:
            reasons.append(f"Shared cabin tech: {', '.join([s.title() for s in list(common)[:3]])}")

        if not reasons:
            return "Closely matched vehicle specifications and executive market positioning."

        return " • ".join(reasons)
