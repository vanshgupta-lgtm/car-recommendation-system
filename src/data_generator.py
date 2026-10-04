"""
Data generation module for Brand New Car Recommendation System.
Generates 5,200+ realistic BRAND NEW car listings across the full price spectrum
(from ₹3.5 Lakhs entry hatchbacks up to ₹15.0 Crore ultra-luxury hypercars),
grounded in real Indian ex-showroom automotive market specifications.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# Comprehensive seed specifications for brand-new cars (Ex-Showroom INR)
# Prices in Lakhs (100 Lakhs = 1 Crore; 1500 Lakhs = 15 Crore)
BRAND_NEW_CAR_SEEDS = [
    # --- MASS MARKET ENTRY & BUDGET (₹3.5L – ₹12L) ---
    {"brand": "Maruti Suzuki", "model": "Alto K10", "body_type": "Hatchback", "fuel_types": ["Petrol", "CNG"], "transmissions": ["Manual", "Automatic"], "price_min_lakh": 3.99, "price_max_lakh": 5.96, "engine_cc": 998, "power_bhp": 66, "mileage_kmpl": 24.9, "seats": 5, "safety_stars": 2, "tier": "entry"},
    {"brand": "Maruti Suzuki", "model": "WagonR", "body_type": "Hatchback", "fuel_types": ["Petrol", "CNG"], "transmissions": ["Manual", "Automatic"], "price_min_lakh": 5.54, "price_max_lakh": 7.42, "engine_cc": 1197, "power_bhp": 88, "mileage_kmpl": 24.3, "seats": 5, "safety_stars": 2, "tier": "entry"},
    {"brand": "Maruti Suzuki", "model": "Swift", "body_type": "Hatchback", "fuel_types": ["Petrol", "CNG"], "transmissions": ["Manual", "Automatic"], "price_min_lakh": 6.49, "price_max_lakh": 9.64, "engine_cc": 1197, "power_bhp": 82, "mileage_kmpl": 24.8, "seats": 5, "safety_stars": 3, "tier": "budget"},
    {"brand": "Maruti Suzuki", "model": "Baleno", "body_type": "Hatchback", "fuel_types": ["Petrol", "CNG"], "transmissions": ["Manual", "Automatic"], "price_min_lakh": 6.66, "price_max_lakh": 9.88, "engine_cc": 1197, "power_bhp": 88, "mileage_kmpl": 22.4, "seats": 5, "safety_stars": 3, "tier": "budget"},
    {"brand": "Maruti Suzuki", "model": "Dzire", "body_type": "Sedan", "fuel_types": ["Petrol", "CNG"], "transmissions": ["Manual", "Automatic"], "price_min_lakh": 6.79, "price_max_lakh": 10.14, "engine_cc": 1197, "power_bhp": 82, "mileage_kmpl": 25.7, "seats": 5, "safety_stars": 5, "tier": "budget"},
    {"brand": "Tata", "model": "Tiago", "body_type": "Hatchback", "fuel_types": ["Petrol", "CNG", "Electric"], "transmissions": ["Manual", "Automatic"], "price_min_lakh": 5.65, "price_max_lakh": 8.90, "engine_cc": 1199, "power_bhp": 85, "mileage_kmpl": 20.0, "seats": 5, "safety_stars": 4, "tier": "budget"},
    {"brand": "Tata", "model": "Punch", "body_type": "SUV", "fuel_types": ["Petrol", "CNG", "Electric"], "transmissions": ["Manual", "Automatic"], "price_min_lakh": 6.13, "price_max_lakh": 10.20, "engine_cc": 1199, "power_bhp": 86, "mileage_kmpl": 18.8, "seats": 5, "safety_stars": 5, "tier": "budget"},
    {"brand": "Hyundai", "model": "Grand i10 Nios", "body_type": "Hatchback", "fuel_types": ["Petrol", "CNG"], "transmissions": ["Manual", "Automatic"], "price_min_lakh": 5.92, "price_max_lakh": 8.56, "engine_cc": 1197, "power_bhp": 82, "mileage_kmpl": 20.7, "seats": 5, "safety_stars": 3, "tier": "budget"},
    {"brand": "Hyundai", "model": "Exter", "body_type": "SUV", "fuel_types": ["Petrol", "CNG"], "transmissions": ["Manual", "Automatic"], "price_min_lakh": 6.13, "price_max_lakh": 10.43, "engine_cc": 1197, "power_bhp": 82, "mileage_kmpl": 19.4, "seats": 5, "safety_stars": 4, "tier": "budget"},

    # --- MID-SIZE & FAMILY (₹10L – ₹25L) ---
    {"brand": "Maruti Suzuki", "model": "Brezza", "body_type": "SUV", "fuel_types": ["Petrol", "CNG"], "transmissions": ["Manual", "Automatic"], "price_min_lakh": 8.34, "price_max_lakh": 14.14, "engine_cc": 1462, "power_bhp": 102, "mileage_kmpl": 19.8, "seats": 5, "safety_stars": 4, "tier": "mid"},
    {"brand": "Maruti Suzuki", "model": "Grand Vitara", "body_type": "SUV", "fuel_types": ["Petrol", "Hybrid", "CNG"], "transmissions": ["Manual", "Automatic"], "price_min_lakh": 10.80, "price_max_lakh": 20.09, "engine_cc": 1490, "power_bhp": 114, "mileage_kmpl": 27.9, "seats": 5, "safety_stars": 4, "tier": "mid_premium"},
    {"brand": "Tata", "model": "Nexon", "body_type": "SUV", "fuel_types": ["Petrol", "Diesel", "Electric", "CNG"], "transmissions": ["Manual", "Automatic"], "price_min_lakh": 8.15, "price_max_lakh": 15.80, "engine_cc": 1199, "power_bhp": 118, "mileage_kmpl": 17.4, "seats": 5, "safety_stars": 5, "tier": "mid"},
    {"brand": "Tata", "model": "Harrier", "body_type": "SUV", "fuel_types": ["Diesel"], "transmissions": ["Manual", "Automatic"], "price_min_lakh": 15.49, "price_max_lakh": 26.44, "engine_cc": 1956, "power_bhp": 168, "mileage_kmpl": 16.2, "seats": 5, "safety_stars": 5, "tier": "mid_premium"},
    {"brand": "Tata", "model": "Safari", "body_type": "SUV", "fuel_types": ["Diesel"], "transmissions": ["Manual", "Automatic"], "price_min_lakh": 16.19, "price_max_lakh": 27.34, "engine_cc": 1956, "power_bhp": 168, "mileage_kmpl": 15.8, "seats": 7, "safety_stars": 5, "tier": "mid_premium"},
    {"brand": "Hyundai", "model": "Creta", "body_type": "SUV", "fuel_types": ["Petrol", "Diesel"], "transmissions": ["Manual", "Automatic"], "price_min_lakh": 11.00, "price_max_lakh": 20.30, "engine_cc": 1497, "power_bhp": 113, "mileage_kmpl": 17.5, "seats": 5, "safety_stars": 4, "tier": "mid_premium"},
    {"brand": "Hyundai", "model": "Verna", "body_type": "Sedan", "fuel_types": ["Petrol"], "transmissions": ["Manual", "Automatic"], "price_min_lakh": 11.00, "price_max_lakh": 17.42, "engine_cc": 1498, "power_bhp": 158, "mileage_kmpl": 18.6, "seats": 5, "safety_stars": 5, "tier": "mid_premium"},
    {"brand": "Mahindra", "model": "XUV 7XO", "body_type": "SUV", "fuel_types": ["Petrol", "Diesel"], "transmissions": ["Manual", "Automatic"], "price_min_lakh": 13.99, "price_max_lakh": 26.99, "engine_cc": 2184, "power_bhp": 197, "mileage_kmpl": 15.2, "seats": 7, "safety_stars": 5, "tier": "mid_premium"},
    {"brand": "Mahindra", "model": "XUV 3XO", "body_type": "SUV", "fuel_types": ["Petrol", "Diesel"], "transmissions": ["Manual", "Automatic"], "price_min_lakh": 7.79, "price_max_lakh": 15.49, "engine_cc": 1197, "power_bhp": 129, "mileage_kmpl": 20.1, "seats": 5, "safety_stars": 5, "tier": "mid"},
    {"brand": "Mahindra", "model": "Scorpio-N", "body_type": "SUV", "fuel_types": ["Petrol", "Diesel"], "transmissions": ["Manual", "Automatic"], "price_min_lakh": 13.85, "price_max_lakh": 24.54, "engine_cc": 2184, "power_bhp": 172, "mileage_kmpl": 14.0, "seats": 7, "safety_stars": 5, "tier": "mid_premium"},
    {"brand": "Mahindra", "model": "Thar Roxx", "body_type": "SUV", "fuel_types": ["Petrol", "Diesel"], "transmissions": ["Manual", "Automatic"], "price_min_lakh": 12.99, "price_max_lakh": 22.49, "engine_cc": 2184, "power_bhp": 174, "mileage_kmpl": 14.0, "seats": 5, "safety_stars": 5, "tier": "mid_premium"},
    {"brand": "Kia", "model": "Seltos", "body_type": "SUV", "fuel_types": ["Petrol", "Diesel"], "transmissions": ["Manual", "Automatic"], "price_min_lakh": 10.90, "price_max_lakh": 20.35, "engine_cc": 1497, "power_bhp": 113, "mileage_kmpl": 17.7, "seats": 5, "safety_stars": 3, "tier": "mid_premium"},
    {"brand": "Honda", "model": "City", "body_type": "Sedan", "fuel_types": ["Petrol", "Hybrid"], "transmissions": ["Manual", "Automatic"], "price_min_lakh": 12.08, "price_max_lakh": 20.55, "engine_cc": 1498, "power_bhp": 119, "mileage_kmpl": 18.4, "seats": 5, "safety_stars": 5, "tier": "mid_premium"},
    {"brand": "Volkswagen", "model": "Virtus", "body_type": "Sedan", "fuel_types": ["Petrol"], "transmissions": ["Manual", "Automatic"], "price_min_lakh": 11.56, "price_max_lakh": 19.41, "engine_cc": 1498, "power_bhp": 148, "mileage_kmpl": 18.7, "seats": 5, "safety_stars": 5, "tier": "mid_premium"},
    {"brand": "Volkswagen", "model": "Taigun", "body_type": "SUV", "fuel_types": ["Petrol"], "transmissions": ["Manual", "Automatic"], "price_min_lakh": 11.70, "price_max_lakh": 19.74, "engine_cc": 1498, "power_bhp": 148, "mileage_kmpl": 18.6, "seats": 5, "safety_stars": 5, "tier": "mid_premium"},
    {"brand": "Skoda", "model": "Slavia", "body_type": "Sedan", "fuel_types": ["Petrol"], "transmissions": ["Manual", "Automatic"], "price_min_lakh": 11.63, "price_max_lakh": 19.12, "engine_cc": 1498, "power_bhp": 148, "mileage_kmpl": 18.8, "seats": 5, "safety_stars": 5, "tier": "mid_premium"},

    # --- PREMIUM & EXECUTIVE (₹25L – ₹70L) ---
    {"brand": "Toyota", "model": "Innova Hycross", "body_type": "MUV", "fuel_types": ["Petrol", "Hybrid"], "transmissions": ["Automatic"], "price_min_lakh": 19.77, "price_max_lakh": 30.98, "engine_cc": 1987, "power_bhp": 172, "mileage_kmpl": 23.2, "seats": 7, "safety_stars": 5, "tier": "premium"},
    {"brand": "Maruti Suzuki", "model": "Invicto", "body_type": "MUV", "fuel_types": ["Petrol", "Hybrid"], "transmissions": ["Automatic"], "price_min_lakh": 25.21, "price_max_lakh": 28.92, "engine_cc": 1987, "power_bhp": 184, "mileage_kmpl": 21.2, "seats": 7, "safety_stars": 5, "tier": "premium"},
    {"brand": "Kia", "model": "Sorento", "body_type": "SUV", "fuel_types": ["Petrol", "Diesel", "Hybrid"], "transmissions": ["Automatic"], "price_min_lakh": 38.00, "price_max_lakh": 45.00, "engine_cc": 2151, "power_bhp": 227, "mileage_kmpl": 16.5, "seats": 7, "safety_stars": 5, "tier": "premium"},
    {"brand": "Toyota", "model": "Fortuner", "body_type": "SUV", "fuel_types": ["Petrol", "Diesel"], "transmissions": ["Manual", "Automatic"], "price_min_lakh": 33.43, "price_max_lakh": 51.44, "engine_cc": 2755, "power_bhp": 201, "mileage_kmpl": 12.0, "seats": 7, "safety_stars": 5, "tier": "premium"},
    {"brand": "Volkswagen", "model": "Tiguan", "body_type": "SUV", "fuel_types": ["Petrol"], "transmissions": ["Automatic"], "price_min_lakh": 35.17, "price_max_lakh": 38.50, "engine_cc": 1984, "power_bhp": 187, "mileage_kmpl": 12.6, "seats": 5, "safety_stars": 5, "tier": "premium"},
    {"brand": "Skoda", "model": "Kodiaq", "body_type": "SUV", "fuel_types": ["Petrol"], "transmissions": ["Automatic"], "price_min_lakh": 38.50, "price_max_lakh": 41.99, "engine_cc": 1984, "power_bhp": 188, "mileage_kmpl": 13.3, "seats": 7, "safety_stars": 5, "tier": "premium"},
    {"brand": "Toyota", "model": "Camry", "body_type": "Sedan", "fuel_types": ["Hybrid", "Petrol"], "transmissions": ["Automatic"], "price_min_lakh": 46.17, "price_max_lakh": 48.50, "engine_cc": 2487, "power_bhp": 215, "mileage_kmpl": 23.2, "seats": 5, "safety_stars": 5, "tier": "premium"},
    {"brand": "Kia", "model": "Carnival", "body_type": "MUV", "fuel_types": ["Diesel"], "transmissions": ["Automatic"], "price_min_lakh": 63.90, "price_max_lakh": 69.90, "engine_cc": 2151, "power_bhp": 197, "mileage_kmpl": 14.8, "seats": 7, "safety_stars": 5, "tier": "luxury"},
    {"brand": "BMW", "model": "3 Series Gran Limousine", "body_type": "Sedan", "fuel_types": ["Petrol", "Diesel"], "transmissions": ["Automatic"], "price_min_lakh": 60.60, "price_max_lakh": 62.00, "engine_cc": 1998, "power_bhp": 255, "mileage_kmpl": 15.3, "seats": 5, "safety_stars": 5, "tier": "luxury"},
    {"brand": "BMW", "model": "X1", "body_type": "SUV", "fuel_types": ["Petrol", "Diesel"], "transmissions": ["Automatic"], "price_min_lakh": 49.50, "price_max_lakh": 52.50, "engine_cc": 1995, "power_bhp": 148, "mileage_kmpl": 16.3, "seats": 5, "safety_stars": 5, "tier": "luxury"},
    {"brand": "Mercedes-Benz", "model": "C-Class", "body_type": "Sedan", "fuel_types": ["Petrol", "Diesel"], "transmissions": ["Automatic"], "price_min_lakh": 61.85, "price_max_lakh": 69.00, "engine_cc": 1993, "power_bhp": 201, "mileage_kmpl": 16.9, "seats": 5, "safety_stars": 5, "tier": "luxury"},
    {"brand": "Mercedes-Benz", "model": "GLA", "body_type": "SUV", "fuel_types": ["Petrol", "Diesel"], "transmissions": ["Automatic"], "price_min_lakh": 51.75, "price_max_lakh": 58.15, "engine_cc": 1950, "power_bhp": 188, "mileage_kmpl": 17.4, "seats": 5, "safety_stars": 5, "tier": "luxury"},
    {"brand": "Audi", "model": "A4", "body_type": "Sedan", "fuel_types": ["Petrol"], "transmissions": ["Automatic"], "price_min_lakh": 45.34, "price_max_lakh": 53.50, "engine_cc": 1984, "power_bhp": 201, "mileage_kmpl": 17.4, "seats": 5, "safety_stars": 5, "tier": "luxury"},

    # --- ELECTRIC VEHICLES (EVs) (₹6.99L – ₹3.5 Cr) ---
    {"brand": "MG", "model": "Comet EV", "body_type": "Hatchback", "fuel_types": ["Electric"], "transmissions": ["Automatic"], "price_min_lakh": 6.99, "price_max_lakh": 9.53, "engine_cc": 0, "power_bhp": 42, "range_km": 230, "mileage_kmpl": 0.0, "seats": 4, "safety_stars": 3, "tier": "budget"},
    {"brand": "Citroen", "model": "eC3", "body_type": "Hatchback", "fuel_types": ["Electric"], "transmissions": ["Automatic"], "price_min_lakh": 11.61, "price_max_lakh": 13.50, "engine_cc": 0, "power_bhp": 57, "range_km": 320, "mileage_kmpl": 0.0, "seats": 5, "safety_stars": 3, "tier": "budget"},
    {"brand": "Tata", "model": "Tigor EV", "body_type": "Sedan", "fuel_types": ["Electric"], "transmissions": ["Automatic"], "price_min_lakh": 12.49, "price_max_lakh": 13.75, "engine_cc": 0, "power_bhp": 74, "range_km": 315, "mileage_kmpl": 0.0, "seats": 5, "safety_stars": 4, "tier": "mid"},
    {"brand": "MG", "model": "Windsor EV", "body_type": "SUV", "fuel_types": ["Electric"], "transmissions": ["Automatic"], "price_min_lakh": 13.50, "price_max_lakh": 15.50, "engine_cc": 0, "power_bhp": 136, "range_km": 331, "mileage_kmpl": 0.0, "seats": 5, "safety_stars": 5, "tier": "mid"},
    {"brand": "Mahindra", "model": "XUV400 EV", "body_type": "SUV", "fuel_types": ["Electric"], "transmissions": ["Automatic"], "price_min_lakh": 15.49, "price_max_lakh": 17.69, "engine_cc": 0, "power_bhp": 150, "range_km": 456, "mileage_kmpl": 0.0, "seats": 5, "safety_stars": 5, "tier": "mid_premium"},
    {"brand": "Mahindra", "model": "BE6", "body_type": "SUV", "fuel_types": ["Electric"], "transmissions": ["Automatic"], "price_min_lakh": 18.90, "price_max_lakh": 26.90, "engine_cc": 0, "power_bhp": 282, "range_km": 683, "mileage_kmpl": 0.0, "seats": 5, "safety_stars": 5, "tier": "mid_premium"},
    {"brand": "Mahindra", "model": "XEV 9e", "body_type": "SUV", "fuel_types": ["Electric"], "transmissions": ["Automatic"], "price_min_lakh": 21.90, "price_max_lakh": 30.50, "engine_cc": 0, "power_bhp": 282, "range_km": 656, "mileage_kmpl": 0.0, "seats": 5, "safety_stars": 5, "tier": "premium"},
    {"brand": "Mahindra", "model": "XEV 9S", "body_type": "SUV", "fuel_types": ["Electric"], "transmissions": ["Automatic"], "price_min_lakh": 19.95, "price_max_lakh": 30.90, "engine_cc": 0, "power_bhp": 282, "range_km": 679, "mileage_kmpl": 0.0, "seats": 7, "safety_stars": 5, "tier": "premium"},
    {"brand": "Tata", "model": "Curvv EV", "body_type": "SUV", "fuel_types": ["Electric"], "transmissions": ["Automatic"], "price_min_lakh": 17.49, "price_max_lakh": 21.99, "engine_cc": 0, "power_bhp": 165, "range_km": 502, "mileage_kmpl": 0.0, "seats": 5, "safety_stars": 5, "tier": "mid_premium"},
    {"brand": "BYD", "model": "Atto 3", "body_type": "SUV", "fuel_types": ["Electric"], "transmissions": ["Automatic"], "price_min_lakh": 24.99, "price_max_lakh": 33.99, "engine_cc": 0, "power_bhp": 201, "range_km": 521, "mileage_kmpl": 0.0, "seats": 5, "safety_stars": 5, "tier": "premium"},
    {"brand": "BYD", "model": "Seal", "body_type": "Sedan", "fuel_types": ["Electric"], "transmissions": ["Automatic"], "price_min_lakh": 41.00, "price_max_lakh": 53.00, "engine_cc": 0, "power_bhp": 523, "range_km": 650, "mileage_kmpl": 0.0, "seats": 5, "safety_stars": 5, "tier": "luxury"},
    {"brand": "Hyundai", "model": "Ioniq 5", "body_type": "SUV", "fuel_types": ["Electric"], "transmissions": ["Automatic"], "price_min_lakh": 46.05, "price_max_lakh": 48.50, "engine_cc": 0, "power_bhp": 214, "range_km": 631, "mileage_kmpl": 0.0, "seats": 5, "safety_stars": 5, "tier": "luxury"},
    {"brand": "Kia", "model": "EV6", "body_type": "SUV", "fuel_types": ["Electric"], "transmissions": ["Automatic"], "price_min_lakh": 60.95, "price_max_lakh": 65.95, "engine_cc": 0, "power_bhp": 320, "range_km": 708, "mileage_kmpl": 0.0, "seats": 5, "safety_stars": 5, "tier": "luxury"},
    {"brand": "MG", "model": "Cyberster", "body_type": "Coupe", "fuel_types": ["Electric"], "transmissions": ["Automatic"], "price_min_lakh": 65.00, "price_max_lakh": 75.00, "engine_cc": 0, "power_bhp": 503, "range_km": 500, "mileage_kmpl": 0.0, "seats": 2, "safety_stars": 5, "tier": "luxury"},
    {"brand": "Volvo", "model": "EX40", "body_type": "SUV", "fuel_types": ["Electric"], "transmissions": ["Automatic"], "price_min_lakh": 56.10, "price_max_lakh": 57.90, "engine_cc": 0, "power_bhp": 408, "range_km": 530, "mileage_kmpl": 0.0, "seats": 5, "safety_stars": 5, "tier": "luxury"},
    {"brand": "Tesla", "model": "Model Y", "body_type": "SUV", "fuel_types": ["Electric"], "transmissions": ["Automatic"], "price_min_lakh": 59.90, "price_max_lakh": 69.90, "engine_cc": 0, "power_bhp": 456, "range_km": 533, "mileage_kmpl": 0.0, "seats": 5, "safety_stars": 5, "tier": "luxury"},
    {"brand": "BMW", "model": "iX1", "body_type": "SUV", "fuel_types": ["Electric"], "transmissions": ["Automatic"], "price_min_lakh": 66.90, "price_max_lakh": 69.90, "engine_cc": 0, "power_bhp": 313, "range_km": 440, "mileage_kmpl": 0.0, "seats": 5, "safety_stars": 5, "tier": "luxury"},
    {"brand": "Mercedes-Benz", "model": "G 580 EV", "body_type": "SUV", "fuel_types": ["Electric"], "transmissions": ["Automatic"], "price_min_lakh": 300.00, "price_max_lakh": 350.00, "engine_cc": 0, "power_bhp": 579, "range_km": 473, "mileage_kmpl": 0.0, "seats": 5, "safety_stars": 5, "tier": "supercar"},

    # --- LUXURY & SUPERCARS (₹70L – ₹3.5 Cr) ---
    {"brand": "BMW", "model": "X3", "body_type": "SUV", "fuel_types": ["Petrol", "Diesel"], "transmissions": ["Automatic"], "price_min_lakh": 72.50, "price_max_lakh": 74.90, "engine_cc": 1995, "power_bhp": 188, "mileage_kmpl": 16.5, "seats": 5, "safety_stars": 5, "tier": "luxury"},
    {"brand": "BMW", "model": "6 Series GT", "body_type": "Sedan", "fuel_types": ["Petrol", "Diesel"], "transmissions": ["Automatic"], "price_min_lakh": 73.50, "price_max_lakh": 78.90, "engine_cc": 1995, "power_bhp": 188, "mileage_kmpl": 15.8, "seats": 5, "safety_stars": 5, "tier": "luxury"},
    {"brand": "Mercedes-Benz", "model": "E-Class", "body_type": "Sedan", "fuel_types": ["Petrol", "Diesel"], "transmissions": ["Automatic"], "price_min_lakh": 78.50, "price_max_lakh": 92.50, "engine_cc": 1993, "power_bhp": 197, "mileage_kmpl": 16.0, "seats": 5, "safety_stars": 5, "tier": "luxury"},
    {"brand": "Land Rover", "model": "Range Rover Velar", "body_type": "SUV", "fuel_types": ["Petrol", "Diesel"], "transmissions": ["Automatic"], "price_min_lakh": 87.90, "price_max_lakh": 95.00, "engine_cc": 1997, "power_bhp": 201, "mileage_kmpl": 13.1, "seats": 5, "safety_stars": 5, "tier": "luxury"},
    {"brand": "Audi", "model": "Q7", "body_type": "SUV", "fuel_types": ["Petrol"], "transmissions": ["Automatic"], "price_min_lakh": 88.66, "price_max_lakh": 97.84, "engine_cc": 2995, "power_bhp": 335, "mileage_kmpl": 11.2, "seats": 7, "safety_stars": 5, "tier": "luxury"},
    {"brand": "BMW", "model": "X5", "body_type": "SUV", "fuel_types": ["Petrol", "Diesel"], "transmissions": ["Automatic"], "price_min_lakh": 97.00, "price_max_lakh": 115.00, "engine_cc": 2993, "power_bhp": 282, "mileage_kmpl": 12.0, "seats": 5, "safety_stars": 5, "tier": "luxury"},
    {"brand": "Mercedes-Benz", "model": "GLE", "body_type": "SUV", "fuel_types": ["Petrol", "Diesel"], "transmissions": ["Automatic"], "price_min_lakh": 97.00, "price_max_lakh": 118.00, "engine_cc": 2989, "power_bhp": 362, "mileage_kmpl": 12.5, "seats": 5, "safety_stars": 5, "tier": "luxury"},
    {"brand": "Land Rover", "model": "Defender 110", "body_type": "SUV", "fuel_types": ["Petrol", "Diesel"], "transmissions": ["Automatic"], "price_min_lakh": 104.00, "price_max_lakh": 157.00, "engine_cc": 2996, "power_bhp": 395, "mileage_kmpl": 10.5, "seats": 5, "safety_stars": 5, "tier": "ultra_luxury"},
    {"brand": "Land Rover", "model": "Range Rover Sport", "body_type": "SUV", "fuel_types": ["Petrol", "Diesel"], "transmissions": ["Automatic"], "price_min_lakh": 145.00, "price_max_lakh": 185.00, "engine_cc": 2997, "power_bhp": 345, "mileage_kmpl": 10.8, "seats": 5, "safety_stars": 5, "tier": "ultra_luxury"},
    {"brand": "BMW", "model": "7 Series", "body_type": "Sedan", "fuel_types": ["Petrol", "Diesel", "Electric"], "transmissions": ["Automatic"], "price_min_lakh": 181.50, "price_max_lakh": 213.00, "engine_cc": 2998, "power_bhp": 375, "mileage_kmpl": 12.6, "seats": 5, "safety_stars": 5, "tier": "ultra_luxury"},
    {"brand": "Mercedes-Benz", "model": "S-Class", "body_type": "Sedan", "fuel_types": ["Petrol", "Diesel"], "transmissions": ["Automatic"], "price_min_lakh": 176.50, "price_max_lakh": 217.00, "engine_cc": 2999, "power_bhp": 362, "mileage_kmpl": 12.8, "seats": 5, "safety_stars": 5, "tier": "ultra_luxury"},
    {"brand": "Toyota", "model": "Land Cruiser", "body_type": "SUV", "fuel_types": ["Diesel"], "transmissions": ["Automatic"], "price_min_lakh": 210.00, "price_max_lakh": 225.00, "engine_cc": 3346, "power_bhp": 304, "mileage_kmpl": 10.2, "seats": 5, "safety_stars": 5, "tier": "ultra_luxury"},
    {"brand": "Mercedes-Benz", "model": "V-Class", "body_type": "MUV", "fuel_types": ["Diesel", "Petrol"], "transmissions": ["Automatic"], "price_min_lakh": 71.10, "price_max_lakh": 146.00, "engine_cc": 1950, "power_bhp": 236, "mileage_kmpl": 16.0, "seats": 7, "safety_stars": 5, "tier": "ultra_luxury"},
    {"brand": "Toyota", "model": "Vellfire", "body_type": "MUV", "fuel_types": ["Hybrid"], "transmissions": ["Automatic"], "price_min_lakh": 122.30, "price_max_lakh": 132.50, "engine_cc": 2487, "power_bhp": 190, "mileage_kmpl": 19.3, "seats": 7, "safety_stars": 5, "tier": "ultra_luxury"},
    {"brand": "Porsche", "model": "911 Carrera", "body_type": "Coupe", "fuel_types": ["Petrol"], "transmissions": ["Automatic"], "price_min_lakh": 198.00, "price_max_lakh": 275.00, "engine_cc": 2981, "power_bhp": 388, "mileage_kmpl": 10.2, "seats": 4, "safety_stars": 5, "tier": "supercar"},
    {"brand": "Porsche", "model": "Taycan Turbo", "body_type": "Sedan", "fuel_types": ["Electric"], "transmissions": ["Automatic"], "price_min_lakh": 189.00, "price_max_lakh": 252.00, "engine_cc": 0, "power_bhp": 670, "range_km": 505, "mileage_kmpl": 0.0, "seats": 4, "safety_stars": 5, "tier": "supercar"},
    {"brand": "Mercedes-AMG", "model": "G 63", "body_type": "SUV", "fuel_types": ["Petrol"], "transmissions": ["Automatic"], "price_min_lakh": 360.00, "price_max_lakh": 400.00, "engine_cc": 3982, "power_bhp": 577, "mileage_kmpl": 6.8, "seats": 5, "safety_stars": 5, "tier": "supercar"},

    # --- ULTRA-LUXURY & EXOTIC (₹3.5 Cr – ₹8 Cr) ---
    {"brand": "Mercedes-Maybach", "model": "GLS Maybach", "body_type": "SUV", "fuel_types": ["Petrol"], "transmissions": ["Automatic"], "price_min_lakh": 335.00, "price_max_lakh": 380.00, "engine_cc": 3982, "power_bhp": 550, "mileage_kmpl": 8.5, "seats": 4, "safety_stars": 5, "tier": "exotic"},
    {"brand": "Mercedes-Maybach", "model": "S 680", "body_type": "Sedan", "fuel_types": ["Petrol"], "transmissions": ["Automatic"], "price_min_lakh": 343.00, "price_max_lakh": 410.00, "engine_cc": 5980, "power_bhp": 603, "mileage_kmpl": 7.5, "seats": 4, "safety_stars": 5, "tier": "exotic"},
    {"brand": "Land Rover", "model": "Range Rover SV", "body_type": "SUV", "fuel_types": ["Petrol", "Diesel"], "transmissions": ["Automatic"], "price_min_lakh": 417.00, "price_max_lakh": 515.00, "engine_cc": 4395, "power_bhp": 523, "mileage_kmpl": 8.7, "seats": 4, "safety_stars": 5, "tier": "exotic"},
    {"brand": "Lamborghini", "model": "Urus Performante", "body_type": "SUV", "fuel_types": ["Petrol"], "transmissions": ["Automatic"], "price_min_lakh": 422.00, "price_max_lakh": 465.00, "engine_cc": 3996, "power_bhp": 657, "mileage_kmpl": 7.8, "seats": 5, "safety_stars": 5, "tier": "exotic"},
    {"brand": "Ferrari", "model": "296 GTB", "body_type": "Coupe", "fuel_types": ["Hybrid"], "transmissions": ["Automatic"], "price_min_lakh": 540.00, "price_max_lakh": 620.00, "engine_cc": 2992, "power_bhp": 818, "mileage_kmpl": 14.0, "seats": 2, "safety_stars": 5, "tier": "exotic"},
    {"brand": "Ferrari", "model": "Purosangue", "body_type": "SUV", "fuel_types": ["Petrol"], "transmissions": ["Automatic"], "price_min_lakh": 1050.00, "price_max_lakh": 1180.00, "engine_cc": 6496, "power_bhp": 715, "mileage_kmpl": 5.8, "seats": 4, "safety_stars": 5, "tier": "hypercar"},
    {"brand": "Bentley", "model": "Continental GT Speed", "body_type": "Coupe", "fuel_types": ["Petrol", "Hybrid"], "transmissions": ["Automatic"], "price_min_lakh": 523.00, "price_max_lakh": 680.00, "engine_cc": 3996, "power_bhp": 771, "mileage_kmpl": 8.2, "seats": 4, "safety_stars": 5, "tier": "exotic"},
    {"brand": "Bentley", "model": "Flying Spur Mulliner", "body_type": "Sedan", "fuel_types": ["Petrol", "Hybrid"], "transmissions": ["Automatic"], "price_min_lakh": 525.00, "price_max_lakh": 650.00, "engine_cc": 3996, "power_bhp": 771, "mileage_kmpl": 8.5, "seats": 4, "safety_stars": 5, "tier": "exotic"},
    {"brand": "Rolls-Royce", "model": "Ghost Series II", "body_type": "Sedan", "fuel_types": ["Petrol"], "transmissions": ["Automatic"], "price_min_lakh": 695.00, "price_max_lakh": 850.00, "engine_cc": 6749, "power_bhp": 563, "mileage_kmpl": 7.1, "seats": 5, "safety_stars": 5, "tier": "exotic"},
    {"brand": "Rolls-Royce", "model": "Cullinan Series II", "body_type": "SUV", "fuel_types": ["Petrol"], "transmissions": ["Automatic"], "price_min_lakh": 750.00, "price_max_lakh": 920.00, "engine_cc": 6749, "power_bhp": 592, "mileage_kmpl": 6.6, "seats": 5, "safety_stars": 5, "tier": "exotic"},
    {"brand": "Rolls-Royce", "model": "Spectre", "body_type": "Coupe", "fuel_types": ["Electric"], "transmissions": ["Automatic"], "price_min_lakh": 750.00, "price_max_lakh": 880.00, "engine_cc": 0, "power_bhp": 577, "range_km": 530, "mileage_kmpl": 0.0, "seats": 4, "safety_stars": 5, "tier": "exotic"},

    # --- TOP TIER BESPOKE & HYPERCARS (₹8 Cr – ₹15.0 Cr) ---
    {"brand": "Lamborghini", "model": "Revuelto", "body_type": "Coupe", "fuel_types": ["Hybrid"], "transmissions": ["Automatic"], "price_min_lakh": 889.00, "price_max_lakh": 1050.00, "engine_cc": 6498, "power_bhp": 1001, "mileage_kmpl": 9.5, "seats": 2, "safety_stars": 5, "tier": "hypercar"},
    {"brand": "Ferrari", "model": "SF90 Stradale", "body_type": "Coupe", "fuel_types": ["Hybrid"], "transmissions": ["Automatic"], "price_min_lakh": 750.00, "price_max_lakh": 920.00, "engine_cc": 3990, "power_bhp": 986, "mileage_kmpl": 12.0, "seats": 2, "safety_stars": 5, "tier": "hypercar"},
    {"brand": "Rolls-Royce", "model": "Phantom VIII Extended", "body_type": "Sedan", "fuel_types": ["Petrol"], "transmissions": ["Automatic"], "price_min_lakh": 1040.00, "price_max_lakh": 1280.00, "engine_cc": 6749, "power_bhp": 563, "mileage_kmpl": 6.7, "seats": 4, "safety_stars": 5, "tier": "hypercar"},
    {"brand": "Aston Martin", "model": "Valkyrie", "body_type": "Coupe", "fuel_types": ["Hybrid"], "transmissions": ["Automatic"], "price_min_lakh": 1350.00, "price_max_lakh": 1500.00, "engine_cc": 6500, "power_bhp": 1160, "mileage_kmpl": 7.0, "seats": 2, "safety_stars": 5, "tier": "hypercar"},
    {"brand": "Bugatti", "model": "Chiron Super Sport", "body_type": "Coupe", "fuel_types": ["Petrol"], "transmissions": ["Automatic"], "price_min_lakh": 1400.00, "price_max_lakh": 1500.00, "engine_cc": 7993, "power_bhp": 1578, "mileage_kmpl": 4.5, "seats": 2, "safety_stars": 5, "tier": "hypercar"},
]

COMMON_FEATURES = [
    "Panoramic Sunroof", "Level 2 ADAS", "360 3D Camera", "Ventilated Front & Rear Seats",
    "Massage Seats", "Head-Up Display", "Wireless Apple CarPlay", "Wireless Android Auto",
    "Burmester 4D Audio", "Bespoke Leather Upholstery", "Alloy Wheels", "Matrix LED Headlamps",
    "Air Suspension", "Rear Seat Entertainment Screens", "Adaptive Cruise Control",
    "Carbon Ceramic Brakes", "Soft-Close Doors", "Keyless Entry & Push Button Start",
    "Multi-Zone Climate Control", "Electronic Stability Program (ESP)"
]

# Authentic, manufacturer-correct variants for all 52 models
MODEL_VARIANTS = {
    # Maruti Suzuki
    "Alto K10": ["Std", "LXi", "VXi", "VXi+"],
    "WagonR": ["LXi", "VXi", "ZXi", "ZXi+"],
    "Swift": ["LXi", "VXi", "ZXi", "ZXi+"],
    "Dzire": ["LXi", "VXi", "ZXi", "ZXi+"],
    "Brezza": ["LXi", "VXi", "ZXi", "ZXi+"],
    "Baleno": ["Sigma", "Delta", "Zeta", "Alpha"],
    "Grand Vitara": ["Sigma", "Delta", "Zeta", "Alpha"],
    "Invicto": ["Zeta+ 7-Seater", "Zeta+ 8-Seater", "Alpha+ 7-Seater"],

    # Tata Motors
    "Tiago": ["XE", "XM", "XT", "XZ+", "XE EV", "XT EV", "XZ+ Tech Lux EV"],
    "Punch": ["Pure", "Adventure", "Accomplished", "Creative", "Smart EV", "Adventure EV", "Empowered EV", "Empowered+ EV"],
    "Nexon": ["Smart", "Pure", "Creative", "Fearless", "Creative EV", "Fearless EV", "Empowered+ EV"],
    "Harrier": ["Smart", "Pure", "Adventure", "Fearless+"],
    "Safari": ["Smart", "Pure", "Adventure+", "Accomplished+"],

    # Hyundai
    "Grand i10 Nios": ["Era", "Magna", "Sportz", "Asta"],
    "Exter": ["EX", "S", "SX", "SX(O)"],
    "Creta": ["E", "EX", "S(O)", "SX", "SX(O)"],
    "Verna": ["EX", "S", "SX", "SX(O) Turbo"],

    # Mahindra
    "Thar Roxx": ["MX1", "MX3", "MX5", "AX5L", "AX7L"],
    "XUV 7XO": ["MX", "AX3", "AX5", "AX7", "AX7 Luxury"],
    "XUV 3XO": ["MX1", "MX2", "MX3", "AX5", "AX7 Luxury"],
    "XUV700": ["MX", "AX3", "AX5", "AX7", "AX7 Luxury"],
    "Scorpio-N": ["Z2", "Z4", "Z6", "Z8", "Z8L"],
    "BE6": ["Pack One 59kWh", "Pack Two 59kWh", "Pack Two 79kWh", "Pack Three 79kWh"],
    "BE 6": ["Pack One 59kWh", "Pack Two 59kWh", "Pack Two 79kWh", "Pack Three 79kWh"],
    "BE 6e": ["Pack One 59kWh", "Pack Two 59kWh", "Pack Two 79kWh", "Pack Three 79kWh"],
    "XEV 9e": ["Pack One 59kWh", "Pack Two 59kWh", "Pack Two 79kWh", "Pack Three 79kWh"],
    "XEV 9S": ["Pack One 59kWh", "Pack Two 70kWh", "Pack Three 79kWh"],

    # Kia & Honda
    "Seltos": ["HTE", "HTK+", "HTX", "GTX+", "X-Line"],
    "Sorento": ["Premium 7-Seater", "Prestige AWD", "Limousine Plus Hybrid"],
    "Carnival": ["Limousine 7-Seater", "Limousine Plus 7-Seater"],
    "City": ["SV", "V", "VX", "ZX"],

    # Volkswagen & Skoda
    "Virtus": ["Comfortline", "Highline", "Topline", "GT Plus"],
    "Taigun": ["Comfortline", "Highline", "Topline", "GT Plus", "GT Plus Edge"],
    "Tiguan": ["Elegance", "Exclusive Edition", "R-Line"],
    "Slavia": ["Active", "Ambition", "Style", "Monte Carlo"],
    "Kodiaq": ["Style", "Sportline", "L&K"],

    # Toyota
    "Innova Hycross": ["GX", "VX", "ZX", "ZX(O)"],
    "Fortuner": ["Standard 4x2", "4x4", "Legender", "GR-S"],
    "Camry": ["2.5 Hybrid", "Elegance Hybrid", "ZX Hybrid"],
    "Vellfire": ["Hi Grade Executive Lounge", "VIP Executive Lounge"],
    "Land Cruiser": ["ZX (LC300)", "GR-S (LC300)"],

    # Audi, BMW, Mercedes-Benz
    "A4": ["Premium", "Premium Plus", "Technology"],
    "Q7": ["Premium Plus 55 TFSI", "Technology 55 TFSI"],
    "X1": ["sDrive18i xLine", "sDrive18d M Sport", "xDrive20d M Sport"],
    "X3": ["xDrive20d M Sport", "xDrive20d Shadow Edition", "xDrive20i SportX"],
    "X5": ["xDrive30d xLine", "xDrive30d M Sport", "xDrive40i M Sport"],
    "3 Series Gran Limousine": ["330Li M Sport", "320Ld M Sport", "330Li Iconic Edition"],
    "6 Series GT": ["620d Luxury Line", "630d M Sport", "630i M Sport"],
    "7 Series": ["740i M Sport", "740d M Sport", "i7 xDrive60"],
    "C-Class": ["C 200", "C 220d", "C 300 AMG Line"],
    "GLA": ["GLA 200", "GLA 220d 4MATIC", "AMG GLA 35 4MATIC"],
    "E-Class": ["E 200 Exclusive", "E 220d AMG Line", "E 450 4MATIC"],
    "GLE": ["GLE 300d 4MATIC", "GLE 450d 4MATIC", "GLE 450 4MATIC"],
    "S-Class": ["S 350d", "S 450 4MATIC", "S 580e"],
    "V-Class": ["Expression", "Exclusive", "Elite", "Marco Polo Horizon"],

    # Land Rover
    "Defender 110": ["S", "SE", "HSE", "X-Dynamic", "V8 Carpathian"],
    "Range Rover Velar": ["Dynamic HSE 2.0L Petrol", "Dynamic HSE 2.0L Diesel"],
    "Range Rover Sport": ["Dynamic SE D350", "Dynamic HSE D350", "Autobiography"],
    "Range Rover SV": ["Autobiography", "SV Serenity", "SV Intrepid"],

    # Porsche
    "Taycan Turbo": ["Taycan 4S", "Taycan Turbo", "Taycan Turbo S"],
    "911 Carrera": ["Carrera", "Carrera S", "Carrera GTS", "GT3"],

    # Mercedes High Performance & Bespoke
    "G 63": ["Speedshift 4MATIC+", "Grand Edition", "Magno Edition"],
    "S 680": ["V12 4MATIC", "Haute Voiture", "Night Series"],
    "GLS Maybach": ["GLS 600 4MATIC", "Night Series Edition", "First Class Lounge Specification"],

    # Ferrari
    "296 GTB": ["Assetto Fiorano", "Atelier Bespoke", "Tailor Made"],
    "SF90 Stradale": ["Assetto Fiorano", "XX Stradale", "Tailor Made"],
    "Purosangue": ["V12 Atmosferico", "Atelier", "Tailor Made"],

    # Lamborghini
    "Urus Performante": ["Performante", "Pearl Capsule", "Ad Personam"],
    "Revuelto": ["V12 Hybrid HPEV", "Ad Personam Bespoke"],

    # Bentley
    "Continental GT Speed": ["Speed Edition 12", "Mulliner", "First Edition"],
    "Flying Spur Mulliner": ["V8 Mulliner", "Speed Mulliner", "W12 Mulliner"],

    # Rolls-Royce
    "Ghost Series II": ["Series II", "Black Badge", "Extended Wheelbase"],
    "Spectre": ["Bespoke Commission", "Black Badge", "Coachbuild"],
    "Cullinan Series II": ["Series II", "Black Badge", "Bespoke"],
    "Phantom VIII Extended": ["Extended Wheelbase", "Privacy Suite", "Bespoke Commission"],

    # Hypercars
    "Bugatti Chiron Super Sport": ["Super Sport 300+", "Pur Sport", "Les Légendes du Ciel"],
    "Chiron Super Sport": ["Super Sport 300+", "Pur Sport", "Les Légendes du Ciel"],
    "Valkyrie": ["Track Pack", "AMR Pro Specification", "Q by Aston Martin"],

    # Electric Vehicles (EVs)
    "Comet EV": ["Pace", "Play", "Exclusive", "100-Year Edition"],
    "eC3": ["Live", "Feel", "Shine", "Shine Dual Tone"],
    "Tigor EV": ["XE", "XT", "XZ+", "XZ+ LUX"],
    "Windsor EV": ["Excite 38kWh", "Exclusive 38kWh", "Essence 38kWh"],
    "XUV400 EV": ["EC Pro 34.5kWh", "EL Pro 34.5kWh", "EL Pro 39.4kWh"],
    "Curvv EV": ["Creative 45", "Accomplished 45", "Accomplished+ 55", "Empowered+ 55"],
    "Atto 3": ["Dynamic 50", "Premium 60", "Superior 60"],
    "Seal": ["Dynamic Range", "Premium Range", "Performance AWD"],
    "Ioniq 5": ["Standard RWD", "Long Range RWD", "N-Line Edition"],
    "EV6": ["GT-Line RWD", "GT-Line AWD"],
    "Cyberster": ["Single Motor Trophy", "Dual Motor GT"],
    "Model Y": ["Rear-Wheel Drive", "Long Range AWD", "Performance AWD"],
    "EX40": ["Plus Single Motor", "Ultimate Twin Motor"],
    "iX1": ["xDrive30 M Sport"],
    "G 580 EV": ["Standard Edition", "Edition One", "AMG Line Edition"]
}


# Authentic ARAI/WLTP certified driving range (in km on single full charge) for EV models
EV_CERTIFIED_RANGE = {
    "Comet EV": 230,
    "eC3": 320,
    "Tiago": 315,
    "Tigor EV": 315,
    "Punch": 421,
    "Windsor EV": 331,
    "Nexon": 465,
    "XUV400 EV": 456,
    "BE6": 683,
    "BE 6": 683,
    "BE 6e": 683,
    "XEV 9e": 656,
    "XEV 9S": 679,
    "Curvv EV": 502,
    "Atto 3": 521,
    "Seal": 650,
    "Ioniq 5": 631,
    "EV6": 708,
    "Cyberster": 500,
    "EX40": 530,
    "Model Y": 533,
    "iX1": 440,
    "7 Series": 625,  # i7 xDrive60
    "Taycan Turbo": 505,
    "G 580 EV": 473,
    "Spectre": 530,
}


def generate_brand_new_car_dataset(num_rows: int = 5200, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic catalog of 5,200 BRAND NEW cars (ex-showroom 2024–2025 models)
    ranging from ₹3.5 Lakhs to ₹15.0 Crore with manufacturer-correct variants.
    """
    np.random.seed(random_seed)
    cars = []

    # Probability weights favor mass market & mid-range (65%), executive/luxury (25%), supercars/hypercars (10%)
    seed_weights = []
    for s in BRAND_NEW_CAR_SEEDS:
        tier = s["tier"]
        if tier in ["entry", "budget"]:
            w = 5.0
        elif tier in ["mid", "mid_premium"]:
            w = 4.0
        elif tier in ["premium", "luxury"]:
            w = 2.0
        elif tier in ["ultra_luxury", "supercar"]:
            w = 1.0
        else: # exotic, hypercar
            w = 0.5
        seed_weights.append(w)
    seed_weights = np.array(seed_weights) / sum(seed_weights)

    for i in range(num_rows):
        seed_idx = np.random.choice(len(BRAND_NEW_CAR_SEEDS), p=seed_weights)
        seed = BRAND_NEW_CAR_SEEDS[seed_idx]
        brand = seed["brand"]
        model = seed["model"]
        body_type = seed["body_type"]
        tier = seed["tier"]

        # All cars are BRAND NEW current showroom units (2024–2025)
        year = int(np.random.choice([2024, 2025], p=[0.45, 0.55]))
        fuel_type = np.random.choice(seed["fuel_types"])
        transmission = np.random.choice(seed["transmissions"])

        # Engine CC, BHP, Mileage & EV Driving Range
        if fuel_type == "Electric":
            engine_cc = 0
            power_bhp = int(np.round(seed["power_bhp"] * np.random.uniform(0.98, 1.05)))
            base_range = EV_CERTIFIED_RANGE.get(model, seed.get("range_km", 450))
            range_km = int(np.round(base_range * np.random.uniform(0.97, 1.03)))
            mileage_kmpl = 0.0
        else:
            engine_cc = int(seed["engine_cc"] * np.random.uniform(0.99, 1.01))
            power_bhp = int(np.round(seed["power_bhp"] * np.random.uniform(0.97, 1.06)))
            mileage_kmpl = float(np.round(seed["mileage_kmpl"] * np.random.uniform(0.94, 1.06), 1))
            range_km = 0

        seats = seed["seats"]
        safety_stars = seed["safety_stars"]

        # Variant & Brand New Ex-Showroom Price calculation
        variant_list = MODEL_VARIANTS.get(model, ["Standard", "Executive", "Luxury"])

        # Intelligently match powertrain-specific variants (e.g., 30d vs 40i, 220d vs 200, i7 vs 740d)
        matching_variants = []
        if fuel_type == "Diesel":
            matching_variants = [v for v in variant_list if any(tag in v.lower() for tag in ["d ", "d", "diesel", "d350", "220d", "300d", "450d", "320ld", "740d", "350d", "20d", "30d", "18d", "z8", "fearless", "adventure", "legender", "carpathian"]) and not any(ptag in v.lower() for ptag in ["40i", "630i", "20i", "330li", "55 tfsi", "i7", "petrol", "580e", "450 4matic"])]
        elif fuel_type == "Petrol":
            matching_variants = [v for v in variant_list if not any(dtag in v.lower() for dtag in ["20d", "30d", "18d", "220d", "300d", "450d", "320ld", "740d", "350d", "d350", "diesel", "i7", " ev"])]
        elif fuel_type == "Electric":
            if set(seed.get("fuel_types", [])) == {"Electric"}:
                matching_variants = variant_list
            else:
                matching_variants = [v for v in variant_list if any(etag in v.lower() for etag in ["i7", "ev", "electric", "range", "ludicrous", "creative", "empowered", "superior", "dynamic", "gt-line", "trophy", "rwd", "awd"])]

        if matching_variants:
            variant = str(np.random.choice(matching_variants))
        else:
            variant = str(np.random.choice(variant_list))

        variant_idx = variant_list.index(variant)
        num_variants = len(variant_list)

        # Monotonic price scaling across variant hierarchy:
        # Base variant (index 0) starts near price_min_lakh; top variant (index -1) reaches price_max_lakh
        t_var = variant_idx / max(num_variants - 1, 1) if num_variants > 1 else 0.5
        
        # Real-world EV pricing adjustments for multi-fuel models
        if fuel_type == "Electric" and model == "Tiago":
            nominal_price = 7.99 + t_var * (11.89 - 7.99)
        elif fuel_type == "Electric" and model == "Punch":
            nominal_price = 9.99 + t_var * (14.29 - 9.99)
        elif fuel_type == "Electric" and model == "Nexon":
            nominal_price = 12.49 + t_var * (16.99 - 12.49)
        else:
            nominal_price = seed["price_min_lakh"] + t_var * (seed["price_max_lakh"] - seed["price_min_lakh"])

        # Transmission delta: Automatic commands realistic ex-showroom premium (~4-6%)
        if transmission == "Automatic" and tier not in ["supercar", "hypercar", "exotic"]:
            nominal_price *= 1.05

        # Fuel delta: Diesel commands small realistic premium (~3%) if model offers both
        if fuel_type == "Diesel" and "Petrol" in seed.get("fuel_types", []):
            nominal_price *= 1.03
        elif fuel_type == "CNG":
            nominal_price *= 0.98

        # Subtle realistic market jitter (±1%)
        jitter = np.random.uniform(0.99, 1.01)
        calc_price = nominal_price * jitter

        # Price in Lakhs and in Crores (Capped at ₹15.0 Crore = 1500 Lakhs)
        price_lakh = float(np.round(np.clip(calc_price, 3.8, 1500.0), 2))
        price_cr = float(np.round(price_lakh / 100.0, 4))

        # Features assignment aligned realistically with variant rank & segment tier
        # Base variants get foundational features, top variants get flagship luxury/ADAS
        core_features = ["Electronic Stability Program (ESP)", "Keyless Entry & Push Button Start", "Alloy Wheels", "Multi-Zone Climate Control", "Wireless Android Auto"]
        mid_features = ["Panoramic Sunroof", "360 3D Camera", "Wireless Apple CarPlay", "Matrix LED Headlamps"]
        flagship_features = ["Level 2 ADAS", "Ventilated Front & Rear Seats", "Adaptive Cruise Control", "Head-Up Display", "Burmester 4D Audio"]
        ultra_features = ["Bespoke Leather Upholstery", "Carbon Ceramic Brakes", "Air Suspension", "Massage Seats", "Soft-Close Doors", "Rear Seat Entertainment Screens"]

        chosen_features = list(core_features[:np.random.randint(2, len(core_features) + 1)])
        if t_var >= 0.25 or tier in ["premium", "luxury", "ultra_luxury", "supercar", "exotic", "hypercar"]:
            chosen_features += list(np.random.choice(mid_features, size=np.random.randint(1, len(mid_features) + 1), replace=False))
        if t_var >= 0.60 or tier in ["luxury", "ultra_luxury", "supercar", "exotic", "hypercar"]:
            chosen_features += list(np.random.choice(flagship_features, size=np.random.randint(1, len(flagship_features) + 1), replace=False))
        if tier in ["ultra_luxury", "supercar", "exotic", "hypercar"]:
            chosen_features += list(np.random.choice(ultra_features, size=np.random.randint(2, len(ultra_features) + 1), replace=False))
            if "Bespoke Leather Upholstery" not in chosen_features:
                chosen_features.append("Bespoke Leather Upholstery")
            if "Carbon Ceramic Brakes" not in chosen_features:
                chosen_features.append("Carbon Ceramic Brakes")

        features_str = ", ".join(sorted(list(set(chosen_features))))

        car_id = f"CAR{i+1:05d}"
        cars.append({
            "car_id": car_id,
            "brand": brand,
            "model": model,
            "variant": variant,
            "year": year,
            "condition": "Brand New",
            "price_lakh": price_lakh,
            "price_cr": price_cr,
            "price": price_lakh,  # Primary numeric column for backward compatibility
            "fuel_type": fuel_type,
            "transmission": transmission,
            "body_type": body_type,
            "engine_cc": engine_cc,
            "power_bhp": power_bhp,
            "mileage_kmpl": mileage_kmpl,
            "range_km": range_km,
            "seats": seats,
            "safety_rating": safety_stars,
            "features": features_str
        })

    return pd.DataFrame(cars)


def generate_new_car_user_interactions(cars_df: pd.DataFrame, num_users: int = 650, num_interactions: int = 28000, random_seed: int = 42) -> pd.DataFrame:
    """
    Simulates user browsing, inquiry, and test drive booking interactions for brand new cars.
    """
    np.random.seed(random_seed)
    user_ids = [f"U{u+1:04d}" for u in range(num_users)]

    # 6 distinct buyer personas
    personas = [
        {"name": "budget_city", "max_price_lakh": 12.0, "fuel_pref": ["Petrol", "CNG", "Electric"], "body_pref": ["Hatchback", "Sedan"]},
        {"name": "family_mid_suv", "min_price_lakh": 12.0, "max_price_lakh": 30.0, "body_pref": ["SUV", "MUV"]},
        {"name": "ev_tech_enthusiast", "min_price_lakh": 6.5, "max_price_lakh": 90.0, "fuel_pref": ["Electric"], "body_pref": ["Hatchback", "SUV", "Sedan", "Coupe"]},
        {"name": "luxury_diesel_buyer", "min_price_lakh": 70.0, "max_price_lakh": 250.0, "fuel_pref": ["Diesel"], "body_pref": ["SUV", "Sedan"]},
        {"name": "luxury_executive", "min_price_lakh": 50.0, "max_price_lakh": 250.0, "body_pref": ["Sedan", "SUV"]},
        {"name": "elite_hypercar_collector", "min_price_lakh": 350.0, "max_price_lakh": 1500.0, "body_pref": ["Coupe", "SUV", "Sedan"]}
    ]

    user_personas = {uid: np.random.choice(personas) for uid in user_ids}
    interactions = []
    car_ids = cars_df["car_id"].values

    for _ in range(num_interactions):
        user_id = np.random.choice(user_ids)
        persona = user_personas[user_id]

        if np.random.rand() < 0.75:
            cond = (cars_df["body_type"].isin(persona.get("body_pref", ["SUV", "Sedan"])))
            if "fuel_pref" in persona:
                cond &= (cars_df["fuel_type"].isin(persona["fuel_pref"]))
            if "max_price_lakh" in persona:
                cond &= (cars_df["price_lakh"] <= persona["max_price_lakh"] * 1.25)
            if "min_price_lakh" in persona:
                cond &= (cars_df["price_lakh"] >= persona["min_price_lakh"] * 0.75)

            matching_cars = cars_df[cond]["car_id"].values
            car_id = np.random.choice(matching_cars) if len(matching_cars) > 0 else np.random.choice(car_ids)
        else:
            car_id = np.random.choice(car_ids)

        car_row = cars_df[cars_df["car_id"] == car_id].iloc[0]

        # Base rating
        base_score = 3.5
        if car_row["safety_rating"] == 5:
            base_score += 0.5
        if car_row["power_bhp"] >= 300:
            base_score += 0.4

        noise = np.random.normal(0, 0.6)
        rating = int(np.clip(np.round(base_score + noise), 1, 5))

        interactions.append({
            "user_id": user_id,
            "car_id": car_id,
            "interaction_type": np.random.choice(["view_specs", "inquire_showroom", "book_test_drive"], p=[0.65, 0.23, 0.12]),
            "rating": rating,
            "timestamp": "2025-01-15"
        })

    return pd.DataFrame(interactions).drop_duplicates(subset=["user_id", "car_id"])


if __name__ == "__main__":
    print("Generating 5,200 brand new car records (₹3.5L to ₹15.0 Cr)...")
    df_new_cars = generate_brand_new_car_dataset(num_rows=5200, random_seed=42)
    cars_path = "data/processed/cars_synthesized_5k.csv"
    df_new_cars.to_csv(cars_path, index=False)
    print(f"Saved {len(df_new_cars)} brand-new cars to {cars_path}")
    print(f"Price range: ₹{df_new_cars['price_lakh'].min():.2f} Lakhs to ₹{df_new_cars['price_cr'].max():.2f} Crore")

    print("Generating brand new car user interaction logs...")
    df_inter = generate_new_car_user_interactions(df_new_cars, num_users=650, num_interactions=30000, random_seed=42)
    inter_path = "data/processed/user_interactions.csv"
    df_inter.to_csv(inter_path, index=False)
    print(f"Saved {len(df_inter)} interactions to {inter_path}")
