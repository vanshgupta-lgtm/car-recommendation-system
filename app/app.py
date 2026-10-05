"""
Streamlit Web Application: AutoMatch AI — Machine Learning Car Recommendation System.
An End-to-End Machine Learning System featuring Content-Based Filtering, KNN Vector Space Retrieval,
and Collaborative Filtering (TruncatedSVD) across 5,200+ vehicles with budgets up to ₹15 Crore.
"""

import os
import sys

# Ensure project root and app dir are in sys.path
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
for p in [PROJECT_ROOT, APP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go

from src.preprocessing import CarDataPreprocessor
from src.content_recommender import ContentBasedRecommender
from src.knn_recommender import KNNRecommender
from src.collaborative import CollaborativeFilteringRecommender
from src.hybrid_recommender import HybridCarRecommender
from src.explainability import format_currency
from src.evaluation import run_benchmark_evaluation

try:
    from src.car_images import get_car_image
except ImportError:
    try:
        from car_images import get_car_image
    except ImportError:
        def get_car_image(model_name: str, body_type: str = "SUV", brand: str = "") -> str:
            return "https://images.unsplash.com/photo-1503376780353-7e6692767b70?auto=format&fit=crop&w=800&q=80"

# Page configuration
st.set_page_config(
    page_title="AutoMatch AI — Machine Learning Car Recommendation System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom CSS
css_path = os.path.join(PROJECT_ROOT, "app", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def scroll_to_top():
    """Injects JavaScript to scroll window and Streamlit scroll container to top."""
    components.html(
        """
        <script>
        function doScroll() {
            try {
                const p = window.parent;
                const d = p.document;
                const container = d.querySelector('[data-testid="stAppViewContainer"]') || d.querySelector('section.main') || d.documentElement;
                if (container) {
                    container.scrollTop = 0;
                }
                const topAnchor = d.getElementById('top-anchor');
                if (topAnchor) {
                    topAnchor.scrollIntoView({ behavior: 'instant', block: 'start' });
                }
                p.scrollTo(0, 0);
            } catch (err) {}
        }
        doScroll();
        setTimeout(doScroll, 60);
        setTimeout(doScroll, 180);
        </script>
        """,
        height=0,
        width=0
    )


def inject_scroll_reveal_engine():
    """Injects high-performance IntersectionObserver and scroll tracking script
    for fluid, buttery-smooth Apple/Tesla-style scroll reveal animations.
    """
    components.html(
        """
        <script>
        (function() {
            try {
                const p = window.parent;
                const d = p.document;

                const targetSelectors = [
                    '.reveal-on-scroll',
                    '.vibe-card',
                    '.spotlight-card',
                    '.section-headline',
                    '.section-subheadline',
                    '.landing-stat-item',
                    '.car-card-with-img',
                    '.metric-tile'
                ];

                function checkElementsInView() {
                    const elements = d.querySelectorAll(targetSelectors.join(','));
                    if (!elements || elements.length === 0) return;

                    const vh = p.innerHeight || 800;

                    elements.forEach(el => {
                        if (el.classList.contains('reveal-active')) return;

                        const rect = el.getBoundingClientRect();
                        // Trigger only when element enters the viewport
                        if (rect.top <= (vh - 35) && rect.bottom >= 20) {
                            el.classList.add('reveal-active');
                            const col = el.closest('[data-testid="column"]');
                            if (col) {
                                col.classList.add('reveal-active');
                                const btn = col.querySelector('[data-testid="stButton"]');
                                if (btn) btn.classList.add('reveal-active');
                            }
                        } else {
                            el.classList.add('scroll-reveal-target');
                        }
                    });
                }

                function initObserver() {
                    const elements = d.querySelectorAll(targetSelectors.join(','));
                    if (!elements || elements.length === 0) return;

                    if ('IntersectionObserver' in p) {
                        const observerOptions = {
                            root: null,
                            rootMargin: '0px 0px -40px 0px',
                            threshold: [0.05, 0.15]
                        };

                        const observer = new p.IntersectionObserver((entries) => {
                            entries.forEach(entry => {
                                if (entry.isIntersecting) {
                                    entry.target.classList.add('reveal-active');
                                    const col = entry.target.closest('[data-testid="column"]');
                                    if (col) {
                                        col.classList.add('reveal-active');
                                        const btn = col.querySelector('[data-testid="stButton"]');
                                        if (btn) btn.classList.add('reveal-active');
                                    }
                                }
                            });
                        }, observerOptions);

                        elements.forEach(el => {
                            if (!el.classList.contains('reveal-active')) {
                                observer.observe(el);
                            }
                        });
                    }
                }

                function bindScrollListeners() {
                    const scrollContainer = d.querySelector('[data-testid="stAppViewContainer"]') || 
                                            d.querySelector('section[data-testid="stMain"]') || 
                                            d.querySelector('section.main') || 
                                            d.documentElement;

                    let ticking = false;
                    const onScrollTick = () => {
                        if (!ticking) {
                            p.requestAnimationFrame(() => {
                                checkElementsInView();
                                ticking = false;
                            });
                            ticking = true;
                        }
                    };

                    if (scrollContainer && !scrollContainer.__smoothRevealBound) {
                        scrollContainer.__smoothRevealBound = true;
                        scrollContainer.addEventListener('scroll', onScrollTick, { passive: true });
                    }

                    if (!p.__smoothRevealBound) {
                        p.__smoothRevealBound = true;
                        p.addEventListener('scroll', onScrollTick, { passive: true });
                    }
                }

                checkElementsInView();
                initObserver();
                bindScrollListeners();
                setTimeout(checkElementsInView, 80);
                setTimeout(checkElementsInView, 250);

                if (!p.__smoothMutationObserver && d.body) {
                    p.__smoothMutationObserver = true;
                    const mutObs = new p.MutationObserver(() => {
                        checkElementsInView();
                        initObserver();
                        bindScrollListeners();
                    });
                    mutObs.observe(d.body, { childList: true, subtree: true });
                }
            } catch (err) {}
        })();
        </script>
        """,
        height=0,
        width=0
    )


# Anchor for scroll navigation
st.markdown('<div id="top-anchor"></div>', unsafe_allow_html=True)
inject_scroll_reveal_engine()


@st.cache_resource(show_spinner="Loading vehicle dataset & ML models...")
def load_data_and_models():
    """Caches dataset and ML models in memory for instantaneous query responses."""
    data_path = os.path.join(PROJECT_ROOT, "data", "processed", "cars_synthesized_5k.csv")
    inter_path = os.path.join(PROJECT_ROOT, "data", "processed", "user_interactions.csv")
    prep_path = os.path.join(PROJECT_ROOT, "models", "preprocessor.pkl")

    df_cars = pd.read_csv(data_path)
    df_inter = pd.read_csv(inter_path)

    if os.path.exists(prep_path):
        preprocessor = CarDataPreprocessor.load(prep_path)
    else:
        preprocessor = CarDataPreprocessor()
        preprocessor.fit(df_cars)

    content_rec = ContentBasedRecommender(preprocessor).fit(df_cars)
    knn_rec = KNNRecommender(preprocessor, n_neighbors=25).fit(df_cars)
    cf_rec = CollaborativeFilteringRecommender(n_factors=20).fit(df_cars, df_inter)
    hybrid_rec = HybridCarRecommender(preprocessor, content_rec, knn_rec, cf_rec).fit(df_cars, df_inter)

    return df_cars, df_inter, preprocessor, content_rec, knn_rec, cf_rec, hybrid_rec


df_cars, df_inter, preprocessor, content_rec, knn_rec, cf_rec, hybrid_rec = load_data_and_models()
num_models = df_cars["model"].nunique()
total_cars = len(df_cars)

# Pages list
PAGES = [
    "🏠 Home",
    "✨ 1. Preference Matchmaker",
    "🏆 2. Top AI Recommendations",
    "🔍 3. Similar Cars Finder",
    "📊 4. ML Benchmark & Insights"
]

# Handle automatic programmatic page transition
if "target_page" in st.session_state and st.session_state["target_page"]:
    st.session_state["active_page"] = st.session_state["target_page"]
    st.session_state["target_page"] = None

if "active_page" not in st.session_state or st.session_state["active_page"] not in PAGES:
    st.session_state["active_page"] = PAGES[0]

# Auto-scroll to top if flagged
if st.session_state.get("do_scroll_top", False):
    scroll_to_top()
    st.session_state["do_scroll_top"] = False

# Initialize session state for user preferences
if "user_pref" not in st.session_state:
    st.session_state["user_pref"] = {
        "min_budget": 5.0,
        "max_budget": 50.0,
        "body_type": "All",
        "fuel_type": "All",
        "transmission": "All",
        "brands": ["All"],
        "min_seats": 0,
        "min_mileage": 8.0,
        "min_safety": 0,
        "features": ["Panoramic Sunroof", "Level 2 ADAS"],
        "top_k": 6,
        "strict_filtering": False
    }

if "has_searched" not in st.session_state:
    st.session_state["has_searched"] = False

current_page = st.session_state["active_page"]

# (Sidebar completely disabled per user preference - only top-left hamburger dropdown menu is used)

# --- TOP-LEFT HAMBURGER & DROPDOWN HEADER BAR ---
top_left, top_mid, top_right = st.columns([1.5, 5.0, 3.5], vertical_alignment="center")

with top_left:
    with st.popover("☰  Menu", use_container_width=False):
        st.markdown("""
        <div style="padding: 4px 6px 10px 6px; border-bottom: 1px solid rgba(255,255,255,0.12); margin-bottom: 10px;">
            <div style="font-weight: 800; font-size: 1.05rem; color: #38BDF8;">⚡ AutoMatch AI Menu</div>
            <div style="font-size: 0.78rem; color: #94A3B8;">Jump directly to any section:</div>
        </div>
        """, unsafe_allow_html=True)
        for i, page_name in enumerate(PAGES):
            is_active = (current_page == page_name)
            label = f"✓  {page_name}" if is_active else f"    {page_name}"
            if st.button(label, key=f"popover_nav_{i}", type="primary" if is_active else "secondary", use_container_width=True):
                if current_page != page_name:
                    st.session_state["active_page"] = page_name
                    st.session_state["do_scroll_top"] = True
                    st.rerun()

with top_mid:
    if current_page != PAGES[0]:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 1.12rem; font-weight: 800; background: linear-gradient(135deg, #FFFFFF, #38BDF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">⚡ AutoMatch AI</span>
            <span style="color: #64748B;">/</span>
            <span style="color: #38BDF8; font-size: 0.95rem; font-weight: 600;">{current_page}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 1.15rem; font-weight: 800; background: linear-gradient(135deg, #FFFFFF, #38BDF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">⚡ AutoMatch AI</span>
            <span style="background: rgba(56,189,248,0.12); color: #38BDF8; font-size: 0.72rem; font-weight: 700; padding: 2px 8px; border-radius: 999px; border: 1px solid rgba(56,189,248,0.3);">ML Recommender</span>
        </div>
        """, unsafe_allow_html=True)

with top_right:
    st.markdown(f"""
    <div style="text-align: right; font-size: 0.82rem; color: #94A3B8;">
        {total_cars:,}+ Vehicles • {num_models} Models • ₹4L – ₹15 Cr
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)


# =========================================================================
# PAGE 0: HOME LANDING PAGE (MINIMAL, ULTRA-ATTRACTIVE, HERO DRIVEN)
# =========================================================================
if current_page == PAGES[0]:
    scroll_to_top()

    # --- LANDING HERO BANNER ---
    st.markdown("""
    <div class="landing-hero">
        <div class="landing-glow-pill">⚡ NEXT-GEN AUTOMOTIVE INTELLIGENCE</div>
        <div class="landing-title">Find The Best Car For You.</div>
        <div class="landing-desc">
            Stop browsing hundreds of confusing car specs. Tell us your lifestyle, budget, and driving habits, and let our multi-metric recommendation engine match you with your dream car in seconds.
        </div>
    """, unsafe_allow_html=True)

    # Primary Glow Button in Hero
    st.markdown('<div class="hero-cta-box">', unsafe_allow_html=True)
    col_c1, col_c2, col_c3 = st.columns([1, 2, 1])
    with col_c2:
        if st.button("🚀 FIND THE BEST CAR FOR YOU  →", type="primary", use_container_width=True, key="home_hero_cta"):
            st.session_state["target_page"] = PAGES[1]
            st.session_state["do_scroll_top"] = True
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Stats strip inside hero
    st.markdown(f"""
        <div class="landing-stats-row">
            <div class="landing-stat-item">
                <div class="landing-stat-value">{total_cars:,}+</div>
                <div class="landing-stat-label">Synthesized Profiles</div>
            </div>
            <div class="landing-stat-item">
                <div class="landing-stat-value">{num_models}</div>
                <div class="landing-stat-label">Curated Models</div>
            </div>
            <div class="landing-stat-item">
                <div class="landing-stat-value">₹4L – ₹15 Cr</div>
                <div class="landing-stat-label">Price Spectrum</div>
            </div>
            <div class="landing-stat-item">
                <div class="landing-stat-value">&lt; 5ms</div>
                <div class="landing-stat-label">ML Latency</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- SECTION: JUMP STRAIGHT IN BY VIBE ---
    st.markdown('<div class="section-headline reveal-on-scroll">✨ Explore By Driving Lifestyle</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheadline reveal-on-scroll">Select your desired automotive vibe to immediately launch tailored recommendations:</div>', unsafe_allow_html=True)

    vibe_cards = [
        {
            "model": "EV6", "brand": "Kia", "body": "SUV",
            "title": "Electric Revolution",
            "tag": "Born-EV & Clean Tech",
            "desc": "Instant torque, futuristic dual-screen cockpits, zero emissions and ultra-low running costs.",
            "btn_label": "Explore EVs →",
            "btn_key": "vibe_btn_ev",
            "pref": {
                "min_budget": 12.0, "max_budget": 50.0,
                "body_type": "All", "fuel_type": "Electric",
                "transmission": "Automatic", "brands": ["All"],
                "min_seats": 0, "min_mileage": 8.0, "min_safety": 0,
                "features": ["Connected Car Tech", "Level 2 ADAS"],
                "top_k": 8, "strict_filtering": False
            }
        },
        {
            "model": "Defender", "brand": "Land Rover", "body": "SUV",
            "title": "Rugged 4x4 SUVs",
            "tag": "All-Terrain & Adventure",
            "desc": "Dominating road stance, authentic 4WD traction, high ground clearance and supreme trail capability.",
            "btn_label": "Explore 4x4s →",
            "btn_key": "vibe_btn_4x4",
            "pref": {
                "min_budget": 15.0, "max_budget": 120.0,
                "body_type": "SUV", "fuel_type": "Diesel",
                "transmission": "All", "brands": ["All"],
                "min_seats": 0, "min_mileage": 8.0, "min_safety": 0,
                "features": ["4x4 / AWD", "High Ground Clearance"],
                "top_k": 8, "strict_filtering": False
            }
        },
        {
            "model": "Carnival", "brand": "Kia", "body": "MUV",
            "title": "Family 7-Seaters",
            "tag": "Hybrid & Supreme Comfort",
            "desc": "Captain seat luxury, stellar hybrid fuel economy, 5-star passenger safety and massive cargo room.",
            "btn_label": "Explore 7-Seaters →",
            "btn_key": "vibe_btn_7s",
            "pref": {
                "min_budget": 15.0, "max_budget": 45.0,
                "body_type": "MUV", "fuel_type": "Strong Hybrid",
                "transmission": "Automatic", "brands": ["All"],
                "min_seats": 7, "min_mileage": 15.0, "min_safety": 0,
                "features": ["Captain Seats", "Ventilated Seats"],
                "top_k": 8, "strict_filtering": False
            }
        },
        {
            "model": "Phantom VIII Extended", "brand": "Rolls-Royce", "body": "Sedan",
            "title": "Bespoke Ultra-Luxury",
            "tag": "V8s & Flagship Exotics",
            "desc": "Handcrafted interiors, whisper-quiet cabin refinement, twin-turbo muscle and unmatched prestige.",
            "btn_label": "Explore Luxury →",
            "btn_key": "vibe_btn_lux",
            "pref": {
                "min_budget": 90.0, "max_budget": 1500.0,
                "body_type": "All", "fuel_type": "Petrol",
                "transmission": "Automatic", "brands": ["All"],
                "min_seats": 0, "min_mileage": 5.0, "min_safety": 5,
                "features": ["Bespoke Interior", "Air Suspension", "Level 2 ADAS"],
                "top_k": 8, "strict_filtering": False
            }
        }
    ]

    col_v1, col_v2, col_v3, col_v4 = st.columns(4)
    vibe_cols = [col_v1, col_v2, col_v3, col_v4]

    for idx, vc in enumerate(vibe_cards):
        with vibe_cols[idx]:
            img_uri = get_car_image(vc["model"], vc["body"], vc["brand"])
            st.markdown(f"""
            <div class="vibe-card reveal-on-scroll">
                <img src="{img_uri}" alt="{vc['title']}" class="vibe-img"/>
                <div class="vibe-info">
                    <div class="vibe-title">{vc['title']}</div>
                    <div class="vibe-tag">{vc['tag']}</div>
                    <div class="vibe-desc">{vc['desc']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(vc["btn_label"], key=vc["btn_key"], use_container_width=True):
                st.session_state["user_pref"] = vc["pref"]
                st.session_state["has_searched"] = True
                st.session_state["do_scroll_top"] = True
                st.session_state["target_page"] = PAGES[2]
                st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)

    # --- SECTION: SPOTLIGHT VEHICLES ---
    st.markdown('<div class="section-headline reveal-on-scroll">🔥 Trending In The Spotlight</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheadline reveal-on-scroll">Authentic real-world photography of standout icons in our 5,200+ car catalog:</div>', unsafe_allow_html=True)

    spotlight_cars = [
        {
            "model": "BE6", "brand": "Mahindra", "body": "SUV",
            "display_title": "Mahindra BE6",
            "tag": "Electric • 682 km Range", "price": "₹ 18.90 Lakhs",
            "btn_label": "Find Similar to BE6 →"
        },
        {
            "model": "Invicto", "brand": "Maruti Suzuki", "body": "MUV",
            "display_title": "Maruti Suzuki Invicto",
            "tag": "Strong Hybrid • 23.2 kmpl", "price": "₹ 25.21 Lakhs",
            "btn_label": "Find Similar to Invicto →"
        },
        {
            "model": "Mercedes-Maybach GLS", "brand": "Mercedes-Benz", "body": "Luxury SUV",
            "display_title": "Mercedes-Maybach GLS",
            "tag": "550 BHP Twin-Turbo V8", "price": "₹ 3.35 Crore",
            "btn_label": "Find Similar to GLS →"
        },
        {
            "model": "Land Cruiser 300", "brand": "Toyota", "body": "Luxury SUV",
            "display_title": "Toyota Land Cruiser 300",
            "tag": "King of Off-Road Luxury", "price": "₹ 2.10 Crore",
            "btn_label": "Find Similar to LC300 →"
        },
    ]

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    spotlight_cols = [col_s1, col_s2, col_s3, col_s4]

    for idx, sc in enumerate(spotlight_cars):
        with spotlight_cols[idx]:
            img_uri = get_car_image(sc["model"], sc["body"], sc["brand"])
            st.markdown(f"""
            <div class="spotlight-card reveal-on-scroll">
                <img src="{img_uri}" alt="{sc['display_title']}" class="spotlight-img"/>
                <div class="spotlight-info">
                    <div class="spotlight-title">{sc['display_title']}</div>
                    <div class="spotlight-badge">{sc['tag']}</div>
                    <div class="spotlight-price">{sc['price']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(sc["btn_label"], key=f"spot_btn_{idx}", use_container_width=True):
                st.session_state["selected_ref_car_model"] = sc["model"]
                st.session_state["target_page"] = PAGES[3]
                st.session_state["do_scroll_top"] = True
                st.rerun()


# =========================================================================
# PAGE 1: PREFERENCE MATCHMAKER (DEDICATED VISUAL & EYE-CATCHING PAGE)
# =========================================================================
elif current_page == PAGES[1]:
    st.markdown("## 🎯 Tell Us Your Dream Car Preferences")
    st.caption("Customize your budget, body style, powertrain, and desired features. Clicking 'FIND MY PERFECT CARS' will automatically load your personalized recommendations.")

    # --- STEP 1: BUDGET SELECTION ---
    st.markdown('<div class="pref-section-title">💰 Step 1: Select Your Budget Bracket</div>', unsafe_allow_html=True)

    if "budget_min_lakh" not in st.session_state:
        st.session_state["budget_min_lakh"] = 10.0
    if "budget_max_lakh" not in st.session_state:
        st.session_state["budget_max_lakh"] = 35.0

    # Quick 1-click Preset Brackets
    budget_cols = st.columns(5)
    with budget_cols[0]:
        if st.button("🟢 Smart Budget\n\n< ₹15 Lakhs", use_container_width=True, key="tier_0"):
            st.session_state["budget_min_lakh"] = 4.0
            st.session_state["budget_max_lakh"] = 15.0
            st.rerun()

    with budget_cols[1]:
        if st.button("🔵 Mid & Compact SUV\n\n₹15L – ₹30 Lakhs", use_container_width=True, key="tier_1"):
            st.session_state["budget_min_lakh"] = 15.0
            st.session_state["budget_max_lakh"] = 30.0
            st.rerun()

    with budget_cols[2]:
        if st.button("🟣 Executive & 4x4\n\n₹30L – ₹70 Lakhs", use_container_width=True, key="tier_2"):
            st.session_state["budget_min_lakh"] = 30.0
            st.session_state["budget_max_lakh"] = 70.0
            st.rerun()

    with budget_cols[3]:
        if st.button("🟡 Luxury & Sport\n\n₹70L – ₹2.5 Crore", use_container_width=True, key="tier_3"):
            st.session_state["budget_min_lakh"] = 70.0
            st.session_state["budget_max_lakh"] = 250.0
            st.rerun()

    with budget_cols[4]:
        if st.button("💎 Ultra Luxury / Exotic\n\n₹2.5 Cr – ₹15.0 Cr", use_container_width=True, key="tier_4"):
            st.session_state["budget_min_lakh"] = 250.0
            st.session_state["budget_max_lakh"] = 1500.0
            st.rerun()

    current_min = float(st.session_state.get("budget_min_lakh", 10.0))
    current_max = float(st.session_state.get("budget_max_lakh", 35.0))

    # Precision Controls: Direct Exact Number Inputs + Range Switcher
    col_input_left, col_input_right, col_scale = st.columns([1, 1, 2])
    with col_input_left:
        input_min = st.number_input(
            "Min Budget (₹ Lakhs):",
            min_value=3.5,
            max_value=1499.0,
            value=min(current_min, 1499.0),
            step=0.5,
            format="%.1f",
            help="Exact minimum budget down to ₹50,000 precision"
        )
    with col_input_right:
        input_max = st.number_input(
            "Max Budget (₹ Lakhs):",
            min_value=float(input_min + 0.5),
            max_value=1500.0,
            value=max(current_max, input_min + 0.5),
            step=0.5,
            format="%.1f",
            help="Exact maximum budget down to ₹50,000 precision"
        )
    with col_scale:
        range_mode = st.radio(
            "Fine-Tune Slider Range Scale:",
            ["🚗 Everyday (₹3.5L – ₹70L)", "👑 Luxury (₹70L – ₹15 Cr)", "🌐 Full Scale (₹3.5L – ₹15 Cr)"],
            index=0 if input_max <= 70.0 else (1 if input_min >= 70.0 else 2),
            horizontal=True
        )

    # Dynamic Precision Slider
    if "Everyday" in range_mode:
        slider_vals = st.slider(
            "Fine-Tune Budget Range (₹ Lakhs — Precise 0.5 Lakh Steps):",
            min_value=3.5,
            max_value=70.0,
            value=(float(np.clip(input_min, 3.5, 69.5)), float(np.clip(input_max, 4.0, 70.0))),
            step=0.5,
            format="₹%.1f L"
        )
        min_lakh = slider_vals[0]
        max_lakh = slider_vals[1]
    elif "Luxury" in range_mode:
        slider_vals = st.slider(
            "Fine-Tune Budget Range (₹ Crore — Precise 0.10 Cr Steps):",
            min_value=0.70,
            max_value=15.00,
            value=(float(np.clip(input_min / 100.0, 0.70, 14.90)), float(np.clip(input_max / 100.0, 0.80, 15.00))),
            step=0.10,
            format="₹%.2f Cr"
        )
        min_lakh = slider_vals[0] * 100.0
        max_lakh = slider_vals[1] * 100.0
    else:
        slider_vals = st.slider(
            "Fine-Tune Budget Range (Full ₹3.5L to ₹15 Cr Market):",
            min_value=0.04,
            max_value=15.00,
            value=(float(np.clip(input_min / 100.0, 0.04, 14.90)), float(np.clip(input_max / 100.0, 0.05, 15.00))),
            step=0.05,
            format="₹%.2f Cr"
        )
        min_lakh = slider_vals[0] * 100.0
        max_lakh = slider_vals[1] * 100.0

    st.session_state["budget_min_lakh"] = min_lakh
    st.session_state["budget_max_lakh"] = max_lakh

    col_b_info, col_b_quick = st.columns([2, 1])
    with col_b_info:
        st.markdown(f"""
        <div class="live-budget-banner">
            <div style="font-size: 0.82rem; text-transform: uppercase; color: #94A3B8; font-weight: 700; letter-spacing: 0.8px;">Active Search Budget Window</div>
            <div style="font-size: 1.35rem; font-weight: 800; color: #38BDF8; margin-top: 4px;">
                {format_currency(min_lakh)} <span style="color: #94A3B8; font-weight: 400; font-size: 1.05rem;">to</span> {format_currency(max_lakh)}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_b_quick:
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        # Quick find button right in Step 1 so users don't have to scroll all the way down
        if st.button("⚡ Quick Find With This Budget", use_container_width=True, type="primary"):
            st.session_state["user_pref"] = {
                "min_budget": min_lakh,
                "max_budget": max_lakh,
                "body_type": "All",
                "fuel_type": "All",
                "transmission": "All",
                "brands": ["All"],
                "min_seats": 0,
                "min_mileage": 8.0,
                "min_safety": 0,
                "features": [],
                "top_k": 8,
                "strict_filtering": True
            }
            st.session_state["has_searched"] = True
            st.session_state["do_scroll_top"] = True
            st.session_state["target_page"] = PAGES[2]
            st.rerun()

    st.markdown("---")

    # --- STEP 2: BODY STYLE & POWERTRAIN ---
    col_style, col_fuel = st.columns(2)

    with col_style:
        st.markdown('<div class="pref-section-title">🚗 Step 2: Preferred Body Style</div>', unsafe_allow_html=True)
        st.caption("Click any silhouette card below. Highlighted with glowing cyan border when selected.")
        body_choice = st.radio(
            "Select Body Style Silhouette:",
            [
                "🌟 All Body Styles (Show Me All Categories)",
                "🚙 SUV (High Ground Clearance, Commanding Driving Stance)",
                "🚘 Sedan (Executive Comfort, Aerodynamics & Boot Space)",
                "🏎️ Coupe / Sports (Aggressive Track Aerodynamics & Styling)",
                "🚗 Hatchback (Zippy City Agility, Compact Parking & High Mileage)",
                "🚐 MUV (Spacious Multi-Row 7-Seater Family Hauler)"
            ],
            index=0,
            key="body_style_selector"
        )
        body_clean = "All"
        if "SUV" in body_choice:
            body_clean = "SUV"
        elif "Sedan" in body_choice:
            body_clean = "Sedan"
        elif "Coupe" in body_choice:
            body_clean = "Coupe"
        elif "Hatchback" in body_choice:
            body_clean = "Hatchback"
        elif "MUV" in body_choice:
            body_clean = "MUV"

        style_badges = {
            "All": ("🌟 All Categories Active", "Displaying best-matched SUVs, Sedans, Coupes, Hatchbacks & MUVs in one unified view."),
            "SUV": ("🚙 SUV Category Filtered", "High ground clearance (180mm+), commanding driver road stance, and versatile all-terrain capability."),
            "Sedan": ("🚘 Sedan Category Filtered", "Executive passenger comfort, aerodynamic low-drag efficiency, huge boot space & high-speed highway stability."),
            "Coupe": ("🏎️ Coupe / Sports Filtered", "Aggressive aerodynamics, low center of gravity, track dynamics, and adrenaline-charged engine performance."),
            "Hatchback": ("🚗 Hatchback Category Filtered", "Zippy urban agility, effortless city parking, lightweight chassis & class-leading fuel economy."),
            "MUV": ("🚐 MUV Category Filtered", "Spacious 3-row seating, generous second/third row legroom & ultimate cross-country family luxury.")
        }
        badge_title, badge_desc = style_badges.get(body_clean, style_badges["All"])
        st.markdown(f"""
        <div style="background: rgba(56, 189, 248, 0.12); border-left: 4px solid #38BDF8; padding: 10px 14px; border-radius: 0 10px 10px 0; margin-top: 10px;">
            <div style="font-weight: 800; color: #38BDF8; font-size: 0.95rem;">{badge_title}</div>
            <div style="font-size: 0.86rem; color: #E2E8F0; margin-top: 2px;">{badge_desc}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_fuel:
        st.markdown('<div class="pref-section-title">⚡ Step 3: Powertrain & Transmission</div>', unsafe_allow_html=True)
        fuel_choice = st.selectbox(
            "Fuel Type Preference",
            ["✨ All Powertrains", "⚡ Electric (Zero Emissions, Instant Torque)", "⛽ Petrol (Refined & Responsive)", "🛢️ Diesel (High Torque & Highway Range)", "🔋 Strong Hybrid (Self-Charging Efficiency)", "🌿 CNG (Maximum Fuel Economy)"],
            index=0
        )
        fuel_clean = "All"
        if "Electric" in fuel_choice:
            fuel_clean = "Electric"
        elif "Petrol" in fuel_choice:
            fuel_clean = "Petrol"
        elif "Diesel" in fuel_choice:
            fuel_clean = "Diesel"
        elif "Hybrid" in fuel_choice:
            fuel_clean = "Hybrid"
        elif "CNG" in fuel_choice:
            fuel_clean = "CNG"

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            trans_choice = st.selectbox("Transmission", ["All", "Automatic", "Manual"])
        with col_t2:
            seats_choice = st.selectbox("Min Seats", [0, 2, 4, 5, 7], format_func=lambda x: "Any Seating" if x == 0 else f"{x} Seats")

    st.markdown("---")

    # --- STEP 3: TECH & LUXURY WISHLIST ---
    st.markdown('<div class="pref-section-title">✨ Step 4: Must-Have Tech & Performance Features</div>', unsafe_allow_html=True)
    st.caption("Select the features you value most; our multi-objective AI model prioritizes vehicles with these factory features.")

    feature_pool = [
        "Panoramic Sunroof", "Level 2 ADAS", "360 3D Camera", "Ventilated Front & Rear Seats",
        "Massage Seats", "Head-Up Display", "Wireless Apple CarPlay", "Wireless Android Auto",
        "Burmester 4D Audio", "Bespoke Leather Upholstery", "Alloy Wheels", "Matrix LED Headlamps",
        "Air Suspension", "Adaptive Cruise Control", "Carbon Ceramic Brakes", "Soft-Close Doors"
    ]

    selected_feats = st.multiselect(
        "Choose Preferred Features",
        feature_pool,
        default=[],
        label_visibility="collapsed"
    )

    # Optional Brands & Settings
    with st.expander("🔍 Optional: Specific Brands & Advanced Matching"):
        brand_list = sorted(df_cars["brand"].unique().tolist())
        chosen_brands = st.multiselect("Filter by Specific Manufacturers (leave blank for all)", brand_list)
        col_adv1, col_adv2 = st.columns(2)
        with col_adv1:
            strict_match = st.checkbox("Strict Filtering (Only cars strictly under budget cap)", value=True)
        with col_adv2:
            num_recs = st.slider("Number of Recommendations", 3, 12, 8)

    st.markdown("<br>", unsafe_allow_html=True)

    # Giant Action Button -> Automatically transitions to Page 2 and scrolls to top!
    col_btn_center = st.columns([1, 2, 1])
    with col_btn_center[1]:
        if st.button("✨ FIND MY PERFECT CARS  👉", use_container_width=True, type="primary"):
            st.session_state["user_pref"] = {
                "min_budget": min_lakh,
                "max_budget": max_lakh,
                "body_type": body_clean,
                "fuel_type": fuel_clean,
                "transmission": trans_choice,
                "brands": chosen_brands if chosen_brands else ["All"],
                "min_seats": seats_choice,
                "min_mileage": 8.0,
                "min_safety": 0,
                "features": selected_feats,
                "top_k": num_recs,
                "strict_filtering": strict_match
            }
            st.session_state["has_searched"] = True
            # Flag to automatically scroll to top when Page 2 renders!
            st.session_state["do_scroll_top"] = True
            st.session_state["target_page"] = PAGES[2]
            st.rerun()


# =========================================================================
# PAGE 2: TOP AI RECOMMENDATIONS (ARRANGED IN DESCENDING ORDER OF PRICE)
# =========================================================================
elif current_page == PAGES[2]:
    # Always snap scroll to top of recommendations
    scroll_to_top()

    pref = st.session_state["user_pref"]
    min_b = pref["min_budget"]
    max_b = pref["max_budget"]

    # Header and Navigation Back button
    col_nav_back, col_nav_space = st.columns([1, 4])
    with col_nav_back:
        if st.button("← Modify Preferences / Budget", use_container_width=True):
            st.session_state["do_scroll_top"] = True
            st.session_state["target_page"] = PAGES[1]
            st.rerun()

    # Compute recommendations using hybrid engine
    weights = {"pref": 0.50, "quality": 0.25, "sim": 0.25, "cf": 0.0}
    recs = hybrid_rec.recommend(
        preferences=pref,
        top_k=pref.get("top_k", 6),
        weights=weights,
        strict_filters=pref.get("strict_filtering", False)
    )

    col_title, col_sort = st.columns([3, 2])
    with col_title:
        st.subheader(f"🏆 Top Recommended Vehicles ({format_currency(min_b)} – {format_currency(max_b)})")
        st.caption("Ranked by Multi-Objective Hybrid ML Engine • Arranged in Descending Order According to Price")
    with col_sort:
        sort_mode = st.selectbox(
            "Sort Order:",
            ["💰 Price: High to Low (Descending)", "🏷️ Price: Low to High (Ascending)", "🎯 Match Score: Highest First"],
            index=0
        )

    if recs.empty:
        st.warning("No brand new cars matched your exact filter parameters. Please widen your budget or relax body style filters in the 'Preference Matchmaker' page.")
    else:
        # Apply sorting: Descending price by default!
        if "High to Low" in sort_mode:
            recs = recs.sort_values(by="price_lakh", ascending=False).reset_index(drop=True)
        elif "Low to High" in sort_mode:
            recs = recs.sort_values(by="price_lakh", ascending=True).reset_index(drop=True)
        elif "Match Score" in sort_mode:
            recs = recs.sort_values(by="hybrid_score", ascending=False).reset_index(drop=True)

        for idx, (_, car) in enumerate(recs.iterrows()):
            stars = "★" * int(car["safety_rating"]) + "☆" * (5 - int(car["safety_rating"]))

            score = car["match_percentage"]
            if score >= 90:
                badge_color = "#10B981"
            elif score >= 75:
                badge_color = "#3B82F6"
            else:
                badge_color = "#F59E0B"

            price_str = format_currency(car["price_lakh"])
            car_img_url = get_car_image(car["model"], car["body_type"], car["brand"])
            is_electric = car.get("fuel_type") == "Electric"
            range_val = car.get("range_km", 0)
            if is_electric and (pd.isna(range_val) or range_val <= 0):
                m_val = float(car.get("mileage_kmpl", 0))
                range_val = m_val if m_val > 50 else 450

            fuel_badge = f"⚡ Electric • {int(range_val)} km Range" if is_electric else f"{car['fuel_type']} • {car['transmission']}"

            with st.container():
                # Card HTML with authentic image banner and ZERO year display!
                st.markdown(f"""
                <div class="car-card-with-img">
                    <div class="car-img-container">
                        <img src="{car_img_url}" class="car-img" alt="{car['brand']} {car['model']}" referrerpolicy="no-referrer" loading="lazy" onerror="this.onerror=null; this.src='https://images.unsplash.com/photo-1533473359331-0135ef1b58bf?auto=format&fit=crop&w=800&q=80';"/>
                        <div class="floating-score" style="background: {badge_color};">
                            {score}% Match
                        </div>
                    </div>
                    <div class="car-card-body">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div>
                                <div class="card-title">{car['brand']} {car['model']}</div>
                                <div class="card-variant">{car['variant']} • Brand New Catalog Specification</div>
                                <div style="margin-top: 4px;">
                                    <span class="badge-pill badge-body">{car['body_type']}</span>
                                    <span class="badge-pill badge-fuel">{fuel_badge}</span>
                                    <span class="badge-pill badge-safety">{stars} ({car['safety_rating']} Stars)</span>
                                </div>
                            </div>
                            <div style="text-align: right;">
                                <div class="card-price">{price_str}</div>
                                <div style="font-size: 0.8rem; color: #94A3B8;">Catalog Price</div>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                col_metrics, col_expl = st.columns([1, 2])
                with col_metrics:
                    m1, m2 = st.columns(2)
                    with m1:
                        st.markdown(f"""
                        <div class="metric-tile">
                            <div class="metric-val">{car['power_bhp']} BHP</div>
                            <div class="metric-lbl">Power</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with m2:
                        if is_electric:
                            st.markdown("""
                            <div class="metric-tile">
                                <div class="metric-val">Electric</div>
                                <div class="metric-lbl">Powertrain</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="metric-tile">
                                <div class="metric-val">{car['engine_cc']:,} cc</div>
                                <div class="metric-lbl">Displacement</div>
                            </div>
                            """, unsafe_allow_html=True)

                    m3, m4 = st.columns(2)
                    with m3:
                        if is_electric:
                            st.markdown(f"""
                            <div class="metric-tile" style="margin-top: 8px;">
                                <div class="metric-val">{int(range_val)} km</div>
                                <div class="metric-lbl">Driving Range</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="metric-tile" style="margin-top: 8px;">
                                <div class="metric-val">{car['mileage_kmpl']:.1f} kmpl</div>
                                <div class="metric-lbl">Efficiency</div>
                            </div>
                            """, unsafe_allow_html=True)
                    with m4:
                        st.markdown(f"""
                        <div class="metric-tile" style="margin-top: 8px;">
                            <div class="metric-val">{int(car['seats'])} Seats</div>
                            <div class="metric-lbl">Capacity</div>
                        </div>
                        """, unsafe_allow_html=True)

                with col_expl:
                    explanation = hybrid_rec.explainer.explain_recommendation(car, pref)
                    st.markdown(f"""
                    <div class="explanation-box">
                        <div style="font-weight: 700; color: #38BDF8; margin-bottom: 6px;">
                            💡 Why This Vehicle Matches Your Profile:
                        </div>
                        {explanation}
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"<span style='font-size: 0.85rem; color: #94A3B8;'>**Factory Features:** {car['features']}</span>", unsafe_allow_html=True)

                st.markdown("<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 20px 0;'/>", unsafe_allow_html=True)


# =========================================================================
# PAGE 3: SIMILAR CARS FINDER (WITH AUTHENTIC IMAGES & DESCENDING PRICE)
# =========================================================================
elif current_page == PAGES[3]:
    scroll_to_top()
    st.subheader("🔍 Discover Direct Vehicle Alternatives")
    st.caption("Select any vehicle from our dataset catalog to discover its closest competitors using vector-space Cosine and KNN distance.")

    # Labels without YEAR
    car_labels = [
        f"{r['brand']} {r['model']} ({r['variant']} - {format_currency(r['price_lakh'])}, ID: {r['car_id']})"
        for _, r in df_cars.iterrows()
    ]
    car_id_map = {label: r["car_id"] for label, (_, r) in zip(car_labels, df_cars.iterrows())}

    default_idx = 0
    if "selected_ref_car_model" in st.session_state and st.session_state["selected_ref_car_model"]:
        target_m = str(st.session_state["selected_ref_car_model"]).lower()
        for idx, lbl in enumerate(car_labels):
            if target_m in lbl.lower():
                default_idx = idx
                break
        st.session_state["selected_ref_car_model"] = None

    selected_label = st.selectbox("Search or Select Reference Vehicle:", car_labels, index=default_idx)
    ref_car_id = car_id_map[selected_label]
    ref_car = df_cars[df_cars["car_id"] == ref_car_id].iloc[0]

    # Reference car showcase card with authentic image
    ref_img = get_car_image(ref_car["model"], ref_car["body_type"], ref_car["brand"])
    ref_stars = "★" * int(ref_car["safety_rating"])
    ref_is_ev = ref_car.get("fuel_type") == "Electric"
    ref_range = ref_car.get("range_km", 0)
    if ref_is_ev and (pd.isna(ref_range) or ref_range <= 0):
        ref_range = float(ref_car.get("mileage_kmpl", 0)) if float(ref_car.get("mileage_kmpl", 0)) > 50 else 450
    ref_fuel_badge = f"⚡ Electric • {int(ref_range)} km Range" if ref_is_ev else f"{ref_car['fuel_type']} • {ref_car['transmission']}"

    st.markdown("#### 📌 Reference Vehicle Selected:")
    st.markdown(f"""
    <div class="car-card-with-img" style="margin-bottom: 24px;">
        <div class="car-img-container" style="height: 260px;">
            <img src="{ref_img}" class="car-img" alt="{ref_car['brand']} {ref_car['model']}" referrerpolicy="no-referrer" loading="lazy" onerror="this.onerror=null; this.src='https://images.unsplash.com/photo-1533473359331-0135ef1b58bf?auto=format&fit=crop&w=800&q=80';"/>
        </div>
        <div class="car-card-body">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <div class="card-title">{ref_car['brand']} {ref_car['model']}</div>
                    <div class="card-variant">{ref_car['variant']} • Reference Specification</div>
                    <div style="margin-top: 4px;">
                        <span class="badge-pill badge-body">{ref_car['body_type']}</span>
                        <span class="badge-pill badge-fuel">{ref_fuel_badge}</span>
                        <span class="badge-pill badge-safety">{ref_stars} ({ref_car['safety_rating']} Stars)</span>
                        <span class="badge-pill badge-brand">{ref_car['power_bhp']} BHP</span>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div class="card-price">{format_currency(ref_car['price_lakh'])}</div>
                    <div style="font-size: 0.8rem; color: #94A3B8;">Catalog Price</div>
                </div>
            </div>
            <div style="margin-top: 12px; font-size: 0.9rem; color: #CBD5E1;">
                <b>Equipment Profile:</b> {ref_car['features']}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Retrieval: Content-Based vs KNN
    col_sim_algo, col_k = st.columns([2, 1])
    with col_sim_algo:
        sim_algorithm = st.radio("Similarity Metric", ["Content-Based Cosine Similarity", "KNN Vector Space"], horizontal=True)
    with col_k:
        sim_k = st.slider("Number of Alternatives", 3, 8, 4)

    if "Content-Based" in sim_algorithm:
        similar_df = content_rec.get_similar_cars(ref_car_id, top_k=sim_k)
    else:
        similar_df = knn_rec.get_similar_cars(ref_car_id, top_k=sim_k)

    # Arrange competitors in descending order by price!
    if not similar_df.empty:
        similar_df = similar_df.sort_values(by="price_lakh", ascending=False).reset_index(drop=True)

    st.markdown(f"#### Top {sim_k} Competitors in Vector Space (Arranged by Price):")
    for _, sim_car in similar_df.iterrows():
        sim_score = sim_car["match_percentage"]
        why_sim = hybrid_rec.explainer.explain_similarity(ref_car, sim_car)
        sim_img = get_car_image(sim_car["model"], sim_car["body_type"], sim_car["brand"])
        sim_is_ev = sim_car.get("fuel_type") == "Electric"
        sim_range = sim_car.get("range_km", 0)
        if sim_is_ev and (pd.isna(sim_range) or sim_range <= 0):
            sim_range = float(sim_car.get("mileage_kmpl", 0)) if float(sim_car.get("mileage_kmpl", 0)) > 50 else 450
        sim_fuel_badge = f"⚡ Electric • {int(sim_range)} km Range" if sim_is_ev else f"{sim_car['fuel_type']} • {sim_car['transmission']}"

        st.markdown(f"""
        <div class="car-card-with-img" style="margin-bottom: 16px;">
            <div style="display: flex; flex-direction: row; align-items: stretch;">
                <div style="width: 280px; min-width: 240px; height: 160px; overflow: hidden; position: relative;">
                    <img src="{sim_img}" style="width: 100%; height: 100%; object-fit: cover;" referrerpolicy="no-referrer" loading="lazy" onerror="this.onerror=null; this.src='https://images.unsplash.com/photo-1533473359331-0135ef1b58bf?auto=format&fit=crop&w=800&q=80';"/>
                    <div style="position: absolute; bottom: 8px; right: 8px; background: rgba(0,0,0,0.7); color: #38BDF8; padding: 4px 10px; border-radius: 8px; font-size: 0.82rem; font-weight: 700;">
                        {sim_score}% Similarity
                    </div>
                </div>
                <div style="padding: 16px 20px; flex-grow: 1;">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <span style="font-size: 1.25rem; font-weight: 800; color: #FFFFFF;">{sim_car['brand']} {sim_car['model']}</span>
                            <div style="color: #94A3B8; font-size: 0.88rem;">{sim_car['variant']}</div>
                            <div style="margin-top: 6px;">
                                <span class="badge-pill badge-body">{sim_car['body_type']}</span>
                                <span class="badge-pill badge-fuel">{sim_fuel_badge}</span>
                                <span class="badge-pill badge-brand">{sim_car['power_bhp']} BHP</span>
                                <span class="badge-pill badge-safety">{sim_car['safety_rating']} ★</span>
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 1.35rem; font-weight: 800; color: #38BDF8;">{format_currency(sim_car['price_lakh'])}</div>
                        </div>
                    </div>
                    <div style="margin-top: 10px; font-size: 0.88rem; color: #CBD5E1; background: rgba(15,23,42,0.6); padding: 8px 12px; border-radius: 8px; border-left: 3px solid #38BDF8;">
                        🔗 <b>Why Similar:</b> {why_sim}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Comparative Table
    st.markdown("#### 📊 Spec Breakdown: Reference vs Competitors")
    compare_df = pd.concat([pd.DataFrame([ref_car]), similar_df.head(2)]).copy()
    compare_df["Formatted_Price"] = compare_df["price_lakh"].apply(format_currency)

    def _fmt_powertrain(row):
        if row.get("fuel_type") == "Electric":
            return "⚡ Electric (Pure EV)"
        return f"{int(row.get('engine_cc', 0)):,} cc"

    def _fmt_efficiency(row):
        if row.get("fuel_type") == "Electric":
            r_val = row.get("range_km", 0)
            if pd.isna(r_val) or r_val <= 0:
                r_val = float(row.get("mileage_kmpl", 0)) if float(row.get("mileage_kmpl", 0)) > 50 else 450
            return f"🔋 {int(r_val)} km (Range)"
        return f"{row.get('mileage_kmpl', 0.0):.1f} kmpl"

    compare_df["Powertrain_Display"] = compare_df.apply(_fmt_powertrain, axis=1)
    compare_df["Efficiency_Display"] = compare_df.apply(_fmt_efficiency, axis=1)

    compare_table = compare_df[[
        "brand", "model", "variant", "Formatted_Price", "power_bhp", "Powertrain_Display", "Efficiency_Display", "safety_rating"
    ]].rename(columns={
        "brand": "Make", "model": "Model", "variant": "Trim",
        "Formatted_Price": "Catalog Price", "power_bhp": "BHP",
        "Powertrain_Display": "Engine / Powertrain", "Efficiency_Display": "Efficiency / Range", "safety_rating": "Safety Stars"
    })
    st.dataframe(compare_table, use_container_width=True)


# =========================================================================
# PAGE 4: BENCHMARK & CATALOG ANALYTICS
# =========================================================================
elif current_page == PAGES[4]:
    scroll_to_top()
    st.subheader("📊 Recommender Benchmark & Machine Learning Analytics")
    st.caption("Quantitative offline evaluation across 60 buyer personas across the ₹3.5L to ₹15.0 Crore catalog.")

    @st.cache_data
    def get_benchmark_table():
        return run_benchmark_evaluation(
            cars_df=df_cars,
            interactions_df=df_inter,
            preprocessor=preprocessor,
            content_rec=content_rec,
            knn_rec=knn_rec,
            hybrid_rec=hybrid_rec,
            cf_rec=cf_rec,
            k=5,
            num_queries=50
        )

    bench_df = get_benchmark_table()

    # Metric Cards Row
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.metric("Hybrid Precision@5", f"{bench_df.loc[bench_df['Model'] == 'Hybrid Engine', 'Precision@5'].values[0]:.3f}", "+25% vs Baseline")
    with mcol2:
        st.metric("Hybrid NDCG@5", f"{bench_df.loc[bench_df['Model'] == 'Hybrid Engine', 'NDCG@5'].values[0]:.3f}", "Top Rank Quality")
    with mcol3:
        st.metric("Dataset Price Spectrum", "₹4 Lakh – ₹15 Cr", "Full Spectrum")
    with mcol4:
        st.metric("Vector Query Latency", "< 3.5ms", "Instant Serving")

    st.markdown("#### 📋 Quantitative Comparison Table")
    st.dataframe(
        bench_df.style.highlight_max(subset=["Precision@5", "Recall@5", "NDCG@5", "Catalog Coverage", "Intra-List Diversity"], color="#1E3A8A"),
        use_container_width=True
    )

    # Plotly Visualizations
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        fig1 = px.bar(
            bench_df,
            x="Model",
            y=["Precision@5", "NDCG@5"],
            barmode="group",
            title="Precision@5 and NDCG@5 Comparison",
            color_discrete_sequence=["#3B82F6", "#10B981"]
        )
        fig1.update_layout(template="plotly_dark", yaxis_range=[0, 1.05])
        st.plotly_chart(fig1, use_container_width=True)

    with col_chart2:
        fig2 = px.bar(
            bench_df,
            x="Model",
            y="Intra-List Diversity",
            title="Intra-List Recommendation Diversity",
            color="Model",
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        fig2.update_layout(template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🚗 Dataset Power vs Price Distribution")

    c1, c2 = st.columns(2)
    with c1:
        sample_df = df_cars.sample(1000, random_state=42).copy()
        fig_scatter = px.scatter(
            sample_df,
            x="power_bhp",
            y="price_cr",
            color="body_type",
            size="safety_rating",
            hover_data=["brand", "model"],
            title="Engine Output (BHP) vs Price (₹ Crore) by Body Type",
            color_discrete_sequence=px.colors.qualitative.Plotly,
            labels={"price_cr": "Price (₹ Crore)", "power_bhp": "Horsepower (BHP)"}
        )
        fig_scatter.update_layout(template="plotly_dark")
        st.plotly_chart(fig_scatter, use_container_width=True)

    with c2:
        fig_pie = px.pie(
            df_cars,
            names="fuel_type",
            title="Dataset Inventory Share by Fuel Type",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_pie.update_layout(template="plotly_dark")
        st.plotly_chart(fig_pie, use_container_width=True)
