import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import load_fashion_mnist
from src.recommendation import (
    build_recommendation_model,
    recommend_products,
    CLASS_NAMES,
)

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="StyleAI | Fashion Intelligence",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap'
    );

    .stApp {
        background: linear-gradient(
            135deg,
            #0b1020 0%,
            #11162b 55%,
            #17142d 100%
        );
        color: #f4f5ff;
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stSidebar"] {
        background: #0a0e1b;
        border-right: 1px solid #242944;
    }

    [data-testid="stSidebar"] * {
        color: #e9eaff;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #ffffff !important;
    }

    .brand {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 28px;
        font-weight: 700;
        letter-spacing: -1px;
        margin-bottom: 4px;
    }

    .brand span {
        color: #a78bfa;
    }

    .muted {
        color: #9ba4c5;
        font-size: 14px;
    }

    .hero {
        padding: 34px;
        border: 1px solid #343052;
        border-radius: 24px;
        background:
            radial-gradient(
                circle at 85% 15%,
                rgba(139, 92, 246, 0.24),
                transparent 35%
            ),
            linear-gradient(
                125deg,
                #191c35,
                #211a3b
            );
        margin-bottom: 28px;
    }

    .eyebrow {
        color: #c4b5fd;
        text-transform: uppercase;
        letter-spacing: 3px;
        font-size: 11px;
        font-weight: 700;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(30px, 4vw, 46px);
        font-weight: 700;
        line-height: 1.15;
        margin-top: 12px;
        margin-bottom: 14px;
        color: white;
    }

    .gradient-text {
        background: linear-gradient(
            90deg,
            #c4b5fd,
            #f0abfc,
            #93c5fd
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-description {
        color: #b8bfd9;
        font-size: 15px;
        line-height: 1.8;
        max-width: 680px;
    }

    .section-label {
        font-size: 12px;
        font-weight: 700;
        color: #a5b4fc;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 8px;
    }

    .section-heading {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 25px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 5px;
    }

    .metric-card {
        background: linear-gradient(
            145deg,
            #191e35,
            #14192d
        );
        border: 1px solid #303651;
        border-radius: 18px;
        padding: 22px;
        min-height: 135px;
    }

    .metric-label {
        color: #9da7c7;
        font-size: 13px;
        margin-bottom: 12px;
    }

    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        color: #ffffff;
        font-size: 30px;
        font-weight: 700;
    }

    .metric-note {
        color: #a78bfa;
        font-size: 12px;
        margin-top: 8px;
    }

    .item-card {
        background: linear-gradient(
            145deg,
            #1a2038,
            #12182b
        );
        border: 1px solid #303650;
        border-radius: 18px;
        padding: 12px;
        margin-bottom: 12px;
    }

    .item-title {
        font-weight: 700;
        color: #f4f4ff;
        font-size: 14px;
        margin-top: 10px;
    }

    .item-subtitle {
        font-size: 12px;
        color: #9da7c7;
        margin-top: 4px;
    }

    .score-pill {
        display: inline-block;
        background: #302653;
        border: 1px solid #5b438c;
        color: #d8c8ff;
        border-radius: 20px;
        padding: 5px 10px;
        font-size: 11px;
        font-weight: 700;
        margin-top: 10px;
    }

    div.stButton > button {
        width: 100%;
        border: 1px solid #8b5cf6;
        border-radius: 12px;
        padding: 12px 16px;
        font-weight: 700;
        background: linear-gradient(
            100deg,
            #7c3aed,
            #a855f7
        );
        color: white;
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        border-color: #c4b5fd;
        background: linear-gradient(
            100deg,
            #8b5cf6,
            #c084fc
        );
        color: white;
        transform: translateY(-2px);
    }

    div[data-baseweb="select"] > div {
        background-color: #171d32;
        border-color: #363d5b;
        border-radius: 10px;
    }

    [data-testid="stImage"] img {
        border-radius: 12px;
    }

    hr {
        border-color: #2b314c;
    }

    .info-box {
        background: #171d32;
        border: 1px solid #303650;
        border-left: 4px solid #a78bfa;
        padding: 16px;
        border-radius: 12px;
        color: #bdc5df;
        font-size: 13px;
        line-height: 1.7;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# LOAD DATA AND RECOMMENDATION MODEL
# =========================================================

@st.cache_resource(show_spinner=False)
def load_application_data():
    x_train, y_train, x_test, y_test = load_fashion_mnist()

    # Use a subset for faster local recommendations.
    images = x_train[:1000]
    labels = y_train[:1000]

    _, similarity_matrix = build_recommendation_model(images)

    return images, labels, x_test, y_test, similarity_matrix


with st.spinner("✨ Initializing your AI fashion studio..."):
    images, labels, x_test, y_test, similarity_matrix = (
        load_application_data()
    )

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">Style<span>AI</span> ✦</div>
        <div class="muted">Fashion intelligence studio</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "✨ Fashion Studio",
            "📊 Analytics",
            "🧠 About the AI",
        ],
    )

    st.markdown("---")

    st.markdown("### 🎨 Your AI engine")

    st.markdown(
        """
        <div class="info-box">
        <b>Deep Learning</b><br>
        CNN image features<br><br>
        <b>Similarity Search</b><br>
        Cosine similarity<br><br>
        <b>Dataset</b><br>
        Fashion-MNIST
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.caption("STYLEAI • AI FASHION RECOMMENDER")
    st.caption("Built with Python, TensorFlow & Streamlit")

# =========================================================
# HERO HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">✦ AI-POWERED FASHION DISCOVERY</div>

        <div class="hero-title">
            Discover your next<br>
            <span class="gradient-text">favorite style.</span>
        </div>

        <div class="hero-description">
            Explore fashion through deep learning.
            Choose an item, uncover visually similar pieces,
            and discover patterns hidden inside fashion images.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# METRIC CARDS
# =========================================================

metric1, metric2, metric3, metric4 = st.columns(4)

metrics = [
    (metric1, "TRAINING IMAGES", f"{len(images):,}", "Images indexed"),
    (metric2, "FASHION CATEGORIES", "10", "Recognized classes"),
    (metric3, "TEST IMAGES", f"{len(x_test):,}", "Held-out dataset"),
    (metric4, "AI RECOMMENDATIONS", "Top 5", "Per selected item"),
]

for column, label, value, note in metrics:
    with column:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">✦ {note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")

# =========================================================
# PAGE 1: FASHION STUDIO
# =========================================================

if page == "✨ Fashion Studio":

    st.markdown(
        '<div class="section-label">YOUR PERSONAL DISCOVERY SPACE</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-heading">Find your next look</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="muted">Choose a category and an item to explore similar fashion images.</p>',
        unsafe_allow_html=True,
    )

    st.write("")

    left, right = st.columns([0.85, 1.5], gap="large")

    with left:

        st.markdown("### 🎯 Select your item")

        selected_class = st.selectbox(
            "Fashion category",
            CLASS_NAMES,
        )

        available_indices = np.where(
            labels == CLASS_NAMES.index(selected_class)
        )[0]

        if len(available_indices) == 0:
            st.warning("No items available in this category.")
            st.stop()

        selected_position = st.selectbox(
            "Choose an item",
            range(len(available_indices)),
            format_func=lambda position: (
                f"Item {position + 1} "
                f"(dataset index {available_indices[position]})"
            ),
        )

        selected_index = int(
            available_indices[selected_position]
        )

        st.markdown("#### Selected fashion item")

        st.image(
            images[selected_index],
            width=260,
            clamp=True,
        )

        st.markdown(
            f"""
            <div class="item-card">
                <div class="item-title">{selected_class}</div>
                <div class="item-subtitle">
                    Dataset item #{selected_index}
                </div>
                <div class="score-pill">AI ANALYSIS READY</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        generate = st.button(
            "✦ Discover Similar Styles",
            use_container_width=True,
        )

        if generate:
            st.session_state["selected_index"] = selected_index

    with right:

        if "selected_index" not in st.session_state:
            st.session_state["selected_index"] = selected_index

        current_index = st.session_state["selected_index"]

        st.markdown("### ✨ Recommended for you")

        st.markdown(
            '<p class="muted">Ranked by similarity in the CNN feature space.</p>',
            unsafe_allow_html=True,
        )

        recommendations = recommend_products(
            images,
            labels,
            similarity_matrix,
            image_index=current_index,
            top_n=5,
        )

        if recommendations:

            first_row = st.columns(3)

            for position, (column, item) in enumerate(
                zip(first_row, recommendations[:3]),
                start=1,
            ):

                with column:

                    st.markdown(
                        '<div class="item-card">',
                        unsafe_allow_html=True,
                    )

                    st.image(
                        item["image"],
                        use_container_width=True,
                    )

                    st.markdown(
                        f"""
                        <div class="item-title">
                            {item['class_name']}
                        </div>
                        <div class="item-subtitle">
                            Recommendation #{position}
                        </div>
                        <div class="score-pill">
                            Similarity: {item['similarity_score']:.1%}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown("</div>", unsafe_allow_html=True)

            second_row = st.columns(2)

            for position, (column, item) in enumerate(
                zip(second_row, recommendations[3:]),
                start=4,
            ):

                with column:

                    st.markdown(
                        '<div class="item-card">',
                        unsafe_allow_html=True,
                    )

                    st.image(
                        item["image"],
                        use_container_width=True,
                    )

                    st.markdown(
                        f"""
                        <div class="item-title">
                            {item['class_name']}
                        </div>
                        <div class="item-subtitle">
                            Recommendation #{position}
                        </div>
                        <div class="score-pill">
                            Similarity: {item['similarity_score']:.1%}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown("</div>", unsafe_allow_html=True)

        st.caption(
            "Similarity scores measure feature-vector similarity. "
            "They are not probabilities that you will like an item."
        )

    st.markdown("---")

    st.markdown("### 🪄 How your recommendations work")

    step1, step2, step3 = st.columns(3)

    with step1:
        st.markdown(
            """
            **01 · Choose**

            Select a fashion image from the dataset.
            """
        )

    with step2:
        st.markdown(
            """
            **02 · Understand**

            The CNN extracts visual features from the image.
            """
        )

    with step3:
        st.markdown(
            """
            **03 · Discover**

            Cosine similarity ranks visually related images.
            """
        )

# =========================================================
# PAGE 2: ANALYTICS
# =========================================================

elif page == "📊 Analytics":

    st.markdown(
        '<div class="section-label">DATA EXPLORATION</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-heading">Fashion dataset insights</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="muted">Explore the composition of the Fashion-MNIST dataset.</p>',
        unsafe_allow_html=True,
    )

    category_counts = pd.Series(
        labels
    ).value_counts().reindex(
        range(len(CLASS_NAMES)),
        fill_value=0,
    )

    distribution = pd.DataFrame(
        {
            "Category": CLASS_NAMES,
            "Images": category_counts.values,
        }
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:

        st.markdown("### Category distribution")

        fig, ax = plt.subplots(figsize=(9, 5))

        fig.patch.set_facecolor("#171b30")
        ax.set_facecolor("#171b30")

        bars = ax.barh(
            distribution["Category"],
            distribution["Images"],
        )

        for bar in bars:
            bar.set_color("#a78bfa")

        ax.tick_params(colors="#d9def5", labelsize=9)
        ax.set_xlabel(
            "Number of images",
            color="#d9def5",
        )

        ax.set_ylabel("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#414765")
        ax.spines["bottom"].set_color("#414765")

        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    with col2:

        st.markdown("### Dataset breakdown")

        st.dataframe(
            distribution,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            f"""
            <div class="info-box">
                <b>Training subset</b><br>
                {len(images):,} images used by the recommendation
                interface.<br><br>

                <b>Image dimensions</b><br>
                28 × 28 pixels, grayscale.<br><br>

                <b>Number of classes</b><br>
                {len(CLASS_NAMES)} fashion categories.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.markdown("### 🖼️ Fashion image gallery")

    chosen_gallery_category = st.selectbox(
        "Browse category",
        CLASS_NAMES,
        key="gallery_category",
    )

    gallery_indices = np.where(
        labels == CLASS_NAMES.index(chosen_gallery_category)
    )[0][:12]

    gallery_columns = st.columns(6)

    for position, image_index in enumerate(gallery_indices):

        with gallery_columns[position % 6]:

            st.image(
                images[image_index],
                use_container_width=True,
            )

            st.caption(f"Image #{image_index}")

# =========================================================
# PAGE 3: ABOUT THE AI
# =========================================================

elif page == "🧠 About the AI":

    st.markdown(
        '<div class="section-label">UNDER THE HOOD</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-heading">Inside the recommendation engine</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="muted">A simple explanation of the deep learning pipeline.</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">THE AI PIPELINE</div>
            <div class="hero-title">
                From pixels to<br>
                <span class="gradient-text">fashion similarity.</span>
            </div>
            <div class="hero-description">
                A convolutional neural network learns visual
                representations of clothing images. The system
                compares those representations to identify
                similar items.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### ⚙️ Model architecture")

    stages = [
        ("01", "Fashion-MNIST", "28 × 28 grayscale images"),
        ("02", "CNN feature extractor", "Learns visual representations"),
        ("03", "Feature vectors", "Encodes each image numerically"),
        ("04", "Cosine similarity", "Compares feature vectors"),
        ("05", "Top-5 ranking", "Returns the nearest items"),
    ]

    for number, title, description in stages:

        st.markdown(
            f"""
            <div class="metric-card" style="margin-bottom:12px;">
                <div class="metric-label">STAGE {number}</div>
                <div style="font-size:20px;font-weight:700;">
                    {title}
                </div>
                <div class="muted" style="margin-top:6px;">
                    {description}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.markdown("### 🧰 Technology stack")

    tech1, tech2, tech3, tech4 = st.columns(4)

    with tech1:
        st.markdown("**Python**")
        st.caption("Core language")

    with tech2:
        st.markdown("**TensorFlow**")
        st.caption("Deep learning")

    with tech3:
        st.markdown("**Scikit-learn**")
        st.caption("Similarity search")

    with tech4:
        st.markdown("**Streamlit**")
        st.caption("Interactive dashboard")

    st.markdown("---")

    st.markdown(
        """
        <div class="info-box">
        <b>Project scope</b><br><br>

        This prototype uses Fashion-MNIST and recommends items
        based on visual feature similarity. It does not currently
        use individual customer histories, ratings, purchase data,
        product prices, or product descriptions.<br><br>

        Its similarity scores describe visual feature similarity,
        not personal preference or recommendation accuracy.
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;padding:16px 0;color:#7f89aa;">
        <div style="font-size:17px;font-weight:700;">
            Style<span style="color:#a78bfa;">AI</span> ✦
        </div>
        <div style="font-size:12px;margin-top:6px;">
            Intelligent fashion discovery · Powered by deep learning
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)