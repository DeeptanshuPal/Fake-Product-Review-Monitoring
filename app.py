import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import re
import nltk
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #2d3250);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #3d4270;
        margin: 5px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8b92b8;
        margin-top: 4px;
    }
    .verdict-fake {
        background: linear-gradient(135deg, #4a1020, #7b1535);
        border: 2px solid #e74c3c;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .verdict-genuine {
        background: linear-gradient(135deg, #0d3320, #1a5c38);
        border: 2px solid #2ecc71;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .flag-item {
        background: #1e2130;
        border-left: 3px solid #e74c3c;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.9rem;
    }
    .flag-item-ok {
        background: #1e2130;
        border-left: 3px solid #2ecc71;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.9rem;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #8b92b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 15px 0 8px 0;
    }
    .star-widget-wrap {
        background: #1e2130;
        border: 1px solid #3d4270;
        border-radius: 12px;
        padding: 14px 18px 10px 18px;
        margin: 0 0 14px 0;
    }
    .star-widget-label {
        font-size: 0.78rem;
        color: #8b92b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }
    .star-display {
        display: flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 6px;
    }
    .star-hint {
        font-size: 0.76rem;
        color: #8b92b8;
        font-style: italic;
    }
    div[data-testid="column"] button {
        border-radius: 6px !important;
        font-size: 0.78rem !important;
        padding: 2px 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open("tfidf_model.pkl",      "rb") as f: tfidf      = pickle.load(f)
    with open("classifier_model.pkl", "rb") as f: classifier = pickle.load(f)
    with open("lsa_model.pkl",        "rb") as f: lsa        = pickle.load(f)
    with open("nn_model.pkl",         "rb") as f: nn         = pickle.load(f)
    return tfidf, classifier, lsa, nn

@st.cache_data
def load_data():
    return pd.read_csv("enriched_reviews.csv", low_memory=False)

@st.cache_resource
def load_nltk():
    nltk.download("punkt",     quiet=True)
    nltk.download("wordnet",   quiet=True)
    nltk.download("stopwords", quiet=True)
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    return WordNetLemmatizer(), set(stopwords.words("english"))

tfidfModel, classifierModel, lsa_model, nn_model = load_models()
df = load_data()
lemmatizer, stop_words = load_nltk()

# ── Preprocessing ─────────────────────────────────────────────────
def preprocess_text(text):
    text  = str(text).lower()
    text  = re.sub(r"[^a-z\s]", "", text)
    text  = re.sub(r"\s+", " ", text)
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# ── Star rating display ───────────────────────────────────────────
def get_star_display(val):
    table = {
        None: ("☆☆☆☆☆", "No rating",         "#4a5175"),
        0.5:  ("½☆☆☆☆", "0.5 — Very Poor",   "#e74c3c"),
        1.0:  ("★☆☆☆☆", "1.0 — Poor",        "#e74c3c"),
        1.5:  ("★½☆☆☆", "1.5 — Poor",        "#e67e22"),
        2.0:  ("★★☆☆☆", "2.0 — Below Avg",   "#e67e22"),
        2.5:  ("★★½☆☆", "2.5 — Average",     "#f39c12"),
        3.0:  ("★★★☆☆", "3.0 — Average",     "#f39c12"),
        3.5:  ("★★★½☆", "3.5 — Good",        "#f1c40f"),
        4.0:  ("★★★★☆", "4.0 — Good",        "#2ecc71"),
        4.5:  ("★★★★½", "4.5 — Excellent",   "#2ecc71"),
        5.0:  ("★★★★★", "5.0 — Outstanding", "#27ae60"),
    }
    return table.get(val, table[None])


# ── Session state ─────────────────────────────────────────────────
if "star_rating_num" not in st.session_state:
    st.session_state.star_rating_num = None

# ── Sidebar ───────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/search.png", width=60)
st.sidebar.title("Fake Review\nDetector")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "🔍 Review Checker"],
    label_visibility="collapsed"
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Dataset:** {len(df):,} reviews")
st.sidebar.markdown(
    f"**Fake:** {(df['prediction']==1).sum():,}  |  "
    f"**Genuine:** {(df['prediction']==0).sum():,}"
)


# ════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":

    st.title("📊 Fake Review Detection — Dashboard")
    st.markdown(
        "Analysis of **{:,}** product reviews using LSA + Logistic Regression".format(len(df))
    )
    st.markdown("---")

    # ── Metrics ───────────────────────────────────────────────────
    total         = len(df)
    fake_count    = (df["prediction"] == 1).sum()
    genuine_count = (df["prediction"] == 0).sum()
    high_risk     = (df["risk_level"] == "🔴 HIGH RISK").sum()
    suspicious    = df["suspicious_ip"].sum()
    burst         = df["is_burst_reviewer"].sum()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    for col, value, label, color in zip(
        [c1, c2, c3, c4, c5, c6],
        [total, fake_count, genuine_count, high_risk, suspicious, burst],
        ["Total Reviews","Fake Reviews","Genuine Reviews",
         "High Risk","Suspicious IPs","Burst Reviewers"],
        ["#ffffff","#e74c3c","#2ecc71","#e74c3c","#f39c12","#f39c12"]
    ):
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value" style="color:{color}">{value:,}</div>'
                f'<div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1 ─────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([1.2, 1, 1])

    with col1:
        st.markdown("#### Fake vs Genuine Reviews")
        counts = df["prediction_label"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 3.5), facecolor="#0f1117")
        ax.set_facecolor("#0f1117")
        bars = ax.bar(counts.index, counts.values,
                      color=["#e74c3c","#2ecc71"], edgecolor="#ffffff", linewidth=0.5)
        ax.set_ylabel("Count", color="white")
        ax.tick_params(colors="white")
        ax.spines[["top","right","left","bottom"]].set_color("#3d4270")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 200,
                    f"{h:,}", ha="center", color="white", fontsize=9, fontweight="bold")
        ax.set_ylim(0, max(counts.values) * 1.15)
        ax.grid(axis="y", linestyle="--", alpha=0.2, color="white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### Risk Level Distribution")
        risk_counts = df["risk_level"].value_counts()
        fig, ax = plt.subplots(figsize=(4, 3.5), facecolor="#0f1117")
        ax.set_facecolor("#0f1117")
        risk_colors = {
            "🔴 HIGH RISK":   "#e74c3c",
            "🟡 MEDIUM RISK": "#f39c12",
            "🟢 LOW RISK":    "#2ecc71"
        }
        colors = [risk_colors.get(l, "#95a5a6") for l in risk_counts.index]
        ax.pie(risk_counts.values, labels=risk_counts.index, autopct="%1.1f%%",
               colors=colors, startangle=90,
               textprops={"color":"white","fontsize":8})
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col3:
        st.markdown("#### Suspicious Signal Breakdown")
        signal_data = {
            "Suspicious IP":     int(df["suspicious_ip"].sum()),
            "Burst Reviewer":    int(df["is_burst_reviewer"].sum()),
            "One-Hit Wonder":    int(df["is_one_hit_wonder"].sum()),
            "Always 5-Star":     int(df["always_5_star"].sum()),
            "High Risk Product": int(df["high_risk_product"].sum()),
            "Late Night Post":   int(df["is_late_night"].sum()),
        }
        signal_s = pd.Series(signal_data).sort_values()
        fig, ax = plt.subplots(figsize=(4, 3.5), facecolor="#0f1117")
        ax.set_facecolor("#0f1117")
        ax.barh(signal_s.index, signal_s.values,
                color="#3498db", edgecolor="#ffffff", linewidth=0.4)
        ax.tick_params(colors="white", labelsize=8)
        ax.spines[["top","right","left","bottom"]].set_color("#3d4270")
        ax.grid(axis="x", linestyle="--", alpha=0.2, color="white")
        for i, v in enumerate(signal_s.values):
            ax.text(v + 10, i, f"{v:,}", va="center", color="white", fontsize=7)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Row 2: Category ───────────────────────────────────────────
    st.markdown("#### Fake Review Rate by Product Category")
    cat_stats = df.groupby("product_category").agg(
        total=("review_id","count"),
        fake =("prediction","mean")
    ).reset_index()
    cat_stats = cat_stats[cat_stats["total"] >= 50].sort_values("fake", ascending=False)

    fig, ax = plt.subplots(figsize=(14, 4), facecolor="#0f1117")
    ax.set_facecolor("#0f1117")
    bar_colors = [
        "#e74c3c" if x > 0.6 else "#f39c12" if x > 0.4 else "#2ecc71"
        for x in cat_stats["fake"]
    ]
    ax.bar(cat_stats["product_category"], cat_stats["fake"] * 100,
           color=bar_colors, edgecolor="#ffffff", linewidth=0.4)
    ax.axhline(60, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(40, color="#f39c12", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_ylabel("Fake Review Rate (%)", color="white")
    ax.tick_params(colors="white")
    ax.set_xticklabels(cat_stats["product_category"],
                       rotation=30, ha="right", fontsize=9, color="white")
    ax.spines[["top","right","left","bottom"]].set_color("#3d4270")
    ax.grid(axis="y", linestyle="--", alpha=0.2, color="white")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Row 3: Hourly + Risk ──────────────────────────────────────
    col4, col5 = st.columns(2)

    with col4:
        st.markdown("#### Posting Pattern by Hour of Day")
        hourly_fake    = df[df["prediction"]==1].groupby("hour_of_day")["review_id"].count()
        hourly_genuine = df[df["prediction"]==0].groupby("hour_of_day")["review_id"].count()
        hours = list(range(24))
        fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="#0f1117")
        ax.set_facecolor("#0f1117")
        ax.plot(hours, [hourly_fake.get(h,0)    for h in hours],
                color="#e74c3c", marker="o", markersize=3, linewidth=2, label="Fake")
        ax.plot(hours, [hourly_genuine.get(h,0) for h in hours],
                color="#2ecc71", marker="o", markersize=3, linewidth=2, label="Genuine")
        ax.set_xlabel("Hour (0 = midnight)", color="white")
        ax.set_ylabel("Reviews", color="white")
        ax.tick_params(colors="white")
        ax.spines[["top","right","left","bottom"]].set_color("#3d4270")
        ax.legend(facecolor="#1e2130", labelcolor="white", fontsize=8)
        ax.grid(linestyle="--", alpha=0.2, color="white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col5:
        st.markdown("#### Risk Score Distribution")
        fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="#0f1117")
        ax.set_facecolor("#0f1117")
        ax.hist(df[df["prediction"]==1]["risk_score"], bins=25,
                alpha=0.7, color="#e74c3c", label="Fake",
                edgecolor="black", linewidth=0.4)
        ax.hist(df[df["prediction"]==0]["risk_score"], bins=25,
                alpha=0.7, color="#2ecc71", label="Genuine",
                edgecolor="black", linewidth=0.4)
        ax.axvline(75, color="red",    linestyle="--", linewidth=1.2)
        ax.axvline(50, color="orange", linestyle="--", linewidth=1.2)
        ax.set_xlabel("Risk Score", color="white")
        ax.set_ylabel("Count",      color="white")
        ax.tick_params(colors="white")
        ax.spines[["top","right","left","bottom"]].set_color("#3d4270")
        ax.legend(facecolor="#1e2130", labelcolor="white", fontsize=8)
        ax.grid(linestyle="--", alpha=0.2, color="white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Top 10 products ───────────────────────────────────────────
    st.markdown("#### Top 10 Most Suspicious Products")
    top_products = (
        df.groupby(["product_id","product_title"])
        .agg(
            total_reviews=("review_id",  "count"),
            fake_rate    =("prediction", "mean"),
            avg_risk     =("risk_score", "mean")
        )
        .reset_index()
        .query("total_reviews >= 5")
        .sort_values("fake_rate", ascending=False)
        .head(10)
    )
    top_products["fake_rate"] = (top_products["fake_rate"]*100).round(1).astype(str) + "%"
    top_products["avg_risk"]  = top_products["avg_risk"].round(1)
    top_products.columns      = ["Product ID","Product Title","Total Reviews",
                                  "Fake Rate","Avg Risk Score"]
    st.dataframe(top_products, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════
# PAGE 2 — REVIEW CHECKER
# ════════════════════════════════════════════════════════════════════
elif page == "🔍 Review Checker":

    st.title("🔍 Real-Time Fake Review Checker")
    st.markdown("Enter any review details below to instantly analyse it.")
    st.markdown("---")

    col_form, col_options = st.columns([2, 1])

    with col_form:

        # 1 — Headline
        review_headline_input = st.text_input(
            "Review Headline (optional)",
            placeholder='e.g. "Great product!" or "Waste of money"'
        )

        # 2 — Star rating label
        st.markdown(
            '<div class="star-widget-label">⭐ &nbsp;STAR RATING &nbsp;'
            '<span style="font-weight:300; font-style:italic; '
            'text-transform:none; letter-spacing:0">(optional)</span></div>',
            unsafe_allow_html=True
        )

        # 10 buttons: ½ 1 1½ 2 2½ 3 3½ 4 4½ 5
        half_labels = ["½","1","1½","2","2½","3","3½","4","4½","5"]
        half_values = [0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]
        btn_cols    = st.columns(10)

        for col, lbl, val in zip(btn_cols, half_labels, half_values):
            with col:
                is_sel = st.session_state.star_rating_num == val
                if st.button(
                    lbl,
                    key=f"star_{val}",
                    type="primary" if is_sel else "secondary",
                    use_container_width=True
                ):
                    st.session_state.star_rating_num = None if is_sel else val
                    st.experimental_rerun()

        # Live star display
        sv, star_label, star_color = get_star_display(st.session_state.star_rating_num)
        st.markdown(
            f'<div class="star-widget-wrap">'
            f'  <div class="star-display">'
            f'    <span style="font-size:2rem; color:{star_color}; '
            f'letter-spacing:6px; font-weight:bold">{sv}</span>'
            f'    <span style="font-size:0.95rem; font-weight:700; '
            f'color:{star_color}; margin-left:10px">{star_label}</span>'
            f'  </div>'
            f'  <div class="star-hint">Click a number to rate · click again to clear</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # 3 — Review body
        review_input = st.text_area(
            "Review Body",
            placeholder='e.g. "Amazing product! Best purchase ever! Five stars!!!"',
            height=160
        )

    with col_options:
        st.markdown("**Other Details**")
        verified = st.selectbox("Verified Purchase?", ["Y", "N"])
        st.markdown("**Reviewer & Product (optional)**")
        reviewer_id_input = st.text_input("Reviewer ID", placeholder="e.g. 20422322")
        product_id_input  = st.text_input("Product ID",  placeholder="e.g. B00MC4CED8")

    analyse_btn = st.button(
        "🔍 Analyse Review", type="primary", use_container_width=True
    )

    if analyse_btn and (review_input.strip() or review_headline_input.strip()):

        with st.spinner("Analysing..."):

            star_rating_num = st.session_state.star_rating_num

            # Combine headline + body
            full_review_text = (
                review_headline_input.strip() + " " + review_input.strip()
            ).strip()

            # ── Predict ───────────────────────────────────────────
            cleaned       = preprocess_text(full_review_text)
            X_tfidf_input = tfidfModel.transform([cleaned])
            X_lsa_input   = lsa_model.transform(X_tfidf_input)
            prediction    = classifierModel.predict(X_tfidf_input)[0]
            proba         = classifierModel.predict_proba(X_tfidf_input)[0]
            confidence    = float(proba.max())

            # ── Similarity ────────────────────────────────────────
            sim_distances, _ = nn_model.kneighbors(X_lsa_input)
            similarities     = [1 - d for d in sim_distances[0][1:]]
            avg_sim          = sum(similarities) / len(similarities) if similarities else 0

            # ── Text stats ────────────────────────────────────────
            word_count        = len(review_input.split())
            unique_word_ratio = (
                len(set(review_input.lower().split())) / max(word_count, 1)
            )
            exclamation_count = review_input.count("!")
            caps_ratio        = (
                sum(c.isupper() for c in review_input) / max(len(review_input), 1)
            )
            headline_body_same = (
                review_headline_input.strip().lower() == review_input.strip().lower()
                and review_headline_input.strip() != ""
            )
            headline_very_short = (
                len(review_headline_input.strip().split()) <= 2
                and review_headline_input.strip() != ""
            )

            # ── Star mismatch ─────────────────────────────────────
            positive_words = {"great","amazing","excellent","perfect","love",
                              "best","awesome","fantastic","wonderful"}
            negative_words = {"terrible","awful","horrible","worst","bad",
                              "broken","useless","poor","disappointing"}
            review_words   = set(review_input.lower().split())
            has_positive   = bool(review_words & positive_words)
            has_negative   = bool(review_words & negative_words)

            text_star_mismatch = False
            if star_rating_num is not None:
                if star_rating_num >= 4.5 and has_negative:
                    text_star_mismatch = True
                elif star_rating_num <= 2.0 and has_positive and not has_negative:
                    text_star_mismatch = True

            # ── Metadata lookups ──────────────────────────────────
            reviewer_info, product_info = {}, {}

            if (reviewer_id_input.strip() and
                    reviewer_id_input.strip() in df["customer_id"].astype(str).values):
                r = df[df["customer_id"].astype(str) == reviewer_id_input.strip()]
                reviewer_info = {
                    "total_reviews": len(r),
                    "avg_stars":     round(r["star_rating"].mean(), 2),
                    "is_burst":      int(r["is_burst_reviewer"].iloc[0]),
                    "is_one_hit":    int(r["is_one_hit_wonder"].iloc[0]),
                    "always_5":      int(r["always_5_star"].iloc[0]),
                }

            if (product_id_input.strip() and
                    product_id_input.strip() in df["product_id"].astype(str).values):
                p = df[df["product_id"].astype(str) == product_id_input.strip()]
                product_info = {
                    "total_reviews": len(p),
                    "fake_rate":     round(float(p["product_fake_rate"].iloc[0]), 3),
                }

            # ── Risk score ────────────────────────────────────────
            risk  = 0
            risk += confidence * 40 if prediction == 1 else (1 - confidence) * 10
            risk += avg_sim * 20
            risk += reviewer_info.get("is_burst",   0) * 10
            risk += reviewer_info.get("is_one_hit", 0) * 5
            risk += reviewer_info.get("always_5",   0) * 5
            risk += product_info.get("fake_rate",   0) * 10
            if star_rating_num is not None and star_rating_num >= 4.5 and prediction == 1:
                risk += 5
            if text_star_mismatch: risk += 5
            if headline_body_same: risk += 3
            risk_score = min(round(risk, 1), 100)
            risk_level = (
                "🔴 HIGH RISK"   if risk_score >= 75 else
                "🟡 MEDIUM RISK" if risk_score >= 50 else
                "🟢 LOW RISK"
            )

            # ── Flags ─────────────────────────────────────────────
            flags, ok_flags = [], []

            if confidence >= 0.85:
                flags.append("⚠️ Model is highly confident this is fake")
            elif confidence <= 0.55:
                flags.append("ℹ️ Model has low confidence — borderline case")
            if avg_sim >= 0.9:
                flags.append("📋 Nearly identical to many other reviews")
            elif avg_sim >= 0.75:
                flags.append("📋 Shares phrasing with many other reviews")
            if word_count < 5:
                flags.append("📝 Extremely short review — lacks detail")
            if unique_word_ratio < 0.4:
                flags.append("📝 Very repetitive language used")
            if exclamation_count >= 3:
                flags.append(f"📝 Excessive exclamation marks ({exclamation_count})")
            if caps_ratio > 0.3:
                flags.append("📝 Unusually high use of capital letters")
            if headline_body_same:
                flags.append("📝 Headline and review body are identical")
            if headline_very_short:
                flags.append(
                    f"📝 Very generic headline: \"{review_headline_input.strip()}\""
                )
            if star_rating_num is not None and star_rating_num >= 4.5 and prediction == 1:
                flags.append("⭐ 5-star rating on a review predicted fake")
            if text_star_mismatch:
                flags.append("⭐ Star rating doesn't match the sentiment of the review")
            if verified == "N" and prediction == 1:
                flags.append("🛒 Not a verified purchase")
            if reviewer_info.get("is_burst"):
                flags.append("⏱️ Reviewer posted many reviews in a very short time")
            if reviewer_info.get("is_one_hit"):
                flags.append("👤 Reviewer has only ever posted 1 review")
            if reviewer_info.get("always_5"):
                flags.append("⭐ Reviewer always gives 5-star ratings")
            if product_info.get("fake_rate", 0) > 0.7:
                flags.append(
                    f"📦 {round(product_info['fake_rate']*100)}% of this "
                    f"product's reviews are fake"
                )
            if word_count >= 20:
                ok_flags.append("✅ Review has sufficient length")
            if unique_word_ratio >= 0.7:
                ok_flags.append("✅ Review uses varied vocabulary")
            if verified == "Y":
                ok_flags.append("✅ Verified purchase")
            if star_rating_num is not None and 2.5 <= star_rating_num <= 4.0:
                ok_flags.append("✅ Balanced star rating")
            if not flags:
                ok_flags.append("✅ No strong red flags detected")

        # ── Results ───────────────────────────────────────────────
        st.markdown("---")

        if prediction == 1:
            st.markdown("""<div class="verdict-fake">
                <h2 style="color:#e74c3c; margin:0">❌ FAKE REVIEW DETECTED</h2>
                <p style="color:#ffaaaa; margin:5px 0 0 0">
                The model has classified this as a fake review</p></div>""",
                unsafe_allow_html=True)
        else:
            st.markdown("""<div class="verdict-genuine">
                <h2 style="color:#2ecc71; margin:0">✅ GENUINE REVIEW</h2>
                <p style="color:#aaffcc; margin:5px 0 0 0">
                The model has classified this as a genuine review</p></div>""",
                unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("Confidence",     f"{confidence*100:.1f}%")
        with m2: st.metric("Risk Score",     f"{risk_score} / 100")
        with m3: st.metric("Risk Level",     risk_level)
        with m4: st.metric("Avg Similarity", f"{avg_sim*100:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)
        col_flags, col_stats = st.columns(2)

        with col_flags:
            st.markdown(
                '<div class="section-header">🚩 Flags Raised</div>',
                unsafe_allow_html=True
            )
            for flag in flags:
                st.markdown(
                    f'<div class="flag-item">{flag}</div>',
                    unsafe_allow_html=True
                )
            for flag in ok_flags:
                st.markdown(
                    f'<div class="flag-item-ok">{flag}</div>',
                    unsafe_allow_html=True
                )

        with col_stats:
            st.markdown(
                '<div class="section-header">📊 Text Statistics</div>',
                unsafe_allow_html=True
            )
            stats_metrics = ["Word Count","Unique Word Ratio",
                             "Exclamation Marks","Caps Ratio"]
            stats_values  = [word_count, f"{unique_word_ratio:.2f}",
                             exclamation_count, f"{caps_ratio:.2f}"]

            if star_rating_num is not None:
                sv2, sl2, _ = get_star_display(star_rating_num)
                stats_metrics.append("Star Rating")
                stats_values.append(f"{sv2}  {sl2}")
            if review_headline_input.strip():
                stats_metrics.append("Headline")
                stats_values.append(f'"{review_headline_input.strip()}"')
                stats_metrics.append("Headline = Body?")
                stats_values.append("Yes ⚠️" if headline_body_same else "No ✅")

            st.dataframe(
                pd.DataFrame({"Metric": stats_metrics, "Value": stats_values}),
                use_container_width=True, hide_index=True
            )

            if reviewer_info:
                st.markdown(
                    '<div class="section-header">👤 Reviewer History</div>',
                    unsafe_allow_html=True
                )
                st.dataframe(pd.DataFrame({
                    "Metric": ["Total Reviews","Avg Star Rating",
                               "Burst Reviewer","Always 5-Star"],
                    "Value":  [reviewer_info["total_reviews"],
                               reviewer_info["avg_stars"],
                               "Yes ⚠️" if reviewer_info["is_burst"]  else "No ✅",
                               "Yes ⚠️" if reviewer_info["always_5"] else "No ✅"]
                }), use_container_width=True, hide_index=True)

            if product_info:
                st.markdown(
                    '<div class="section-header">📦 Product Context</div>',
                    unsafe_allow_html=True
                )
                st.dataframe(pd.DataFrame({
                    "Metric": ["Total Reviews","Fake Review Rate"],
                    "Value":  [product_info["total_reviews"],
                               f"{product_info['fake_rate']*100:.1f}%"]
                }), use_container_width=True, hide_index=True)

    elif analyse_btn:
        st.warning("Please enter at least a review headline or body to analyse.")