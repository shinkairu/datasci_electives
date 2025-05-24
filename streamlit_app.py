import streamlit as st
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from app_utils import load_and_clean_data, explore_data, evaluate_model, download_results
import os
import base64
from PIL import Image

# --- Page config ---
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# --- Load background image ---
image_path = "lucy.png"
if os.path.exists(image_path):
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode()
    background_style = f"""
        <style>
        body {{
            background-image: url("data:image/png;base64,{base64_image}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
    """
    st.markdown(background_style, unsafe_allow_html=True)

# --- Custom CSS styling with embedded background ---
st.markdown(f"""
    <style>
    .main {{
        background: rgba(255, 255, 255, 0.85);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1);
        animation: fadeIn 1.2s ease-in-out;
    }}
    h1, h2, h3 {{
        color: #6a1b9a;
        margin-bottom: 0.5rem;
    }}
    .stButton>button {{
        background: #7b1fa2;
        color: white;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        border: none;
        transition: transform 0.2s ease;
    }}
    .stButton>button:hover {{
        background: #4a0072;
        transform: scale(1.02);
    }}
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    </style>
""", unsafe_allow_html=True)

# Tabs for Dashboard and Result
tabs = st.tabs(["📊 DASHBOARD", "📈 RESULTS"])

# Load and preview data
df = load_and_clean_data()

with tabs[0]:
    st.markdown("""
    <div class="main">
    """, unsafe_allow_html=True)

    st.markdown("### 💳 Credit Card Fraud Detection Dashboard")
    st.write("An interactive ML-powered dashboard for fraud detection.")

    if st.toggle("Show Dataset Preview"):
        st.markdown("### 🔍 Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

    with st.expander("📊 Run Exploratory Data Analysis (EDA)"):
        explore_data(df)

    st.markdown("### 🤖 Train Fraud Detection Model")
    target = "Class"
    X = df.drop(columns=[target])
    y = df[target]

    left, center, right = st.columns([1, 2, 1])
    with center:
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test set size (%)", 10, 50, 30)
        with col2:
            random_state = st.number_input("Random seed", value=42)

        if st.button("🚀 Train Model"):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size / 100, random_state=random_state, stratify=y
            )
            model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
            model.fit(X_train, y_train)

            st.session_state.model = model
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.trained = True
            st.success("✅ Model trained successfully! Check 'Results' tab for output.")

    st.markdown("""</div>""", unsafe_allow_html=True)

with tabs[1]:
    if st.session_state.get("trained", False):
        st.markdown("""
        <div class="main">
        """, unsafe_allow_html=True)

        st.markdown("### 📈 Evaluation Metrics and Visualizations")
        model = st.session_state.model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test

        report_df = evaluate_model(model, X_test, y_test)

        st.markdown("### 📋 Classification Report")
        st.dataframe(report_df, use_container_width=True)

        # Download button
        csv = download_results(report_df)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="fraud_model_report.csv">📥 Download Results as CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

        st.markdown("""
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please train the model first in the Dashboard tab.")
