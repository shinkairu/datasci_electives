import streamlit as st
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import sys
import os
sys.path.append(os.path.dirname(__file__))
from app_utils import load_and_clean_data, explore_data, evaluate_model, download_results

import base64

# --- Page config ---
st.set_page_config(page_title="🌸 Lucy's Fraud Detection Dashboard", layout="wide")

# --- Background Image Functions ---
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(image_file):
    bin_str = get_base64_of_bin_file(image_file)
    page_bg_img = f"""
    <style>
    body {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: #f3f0ff;
        font-family: 'Segoe UI', sans-serif;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# --- Set background if image exists ---
image_path = os.path.join(os.path.dirname(__file__), "lucy.png")
if os.path.exists(image_path):
    set_png_as_page_bg(image_path)
else:
    st.warning("⚠️ Background image not found.")

# --- Custom CSS for Font & Style ---
st.markdown("""
    <style>
    .main {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 18px;
        padding: 2rem;
        margin: 2rem 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(10px);
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3 {
        color: #ffffff;
        font-family: 'Segoe UI Semibold', sans-serif;
        font-size: 1.8rem;
    }
    .stMarkdown p {
        font-size: 1.1rem;
        color: #f0f0f0;
    }
    .stButton>button {
        background-color: #ab47bc;
        color: white;
        font-weight: bold;
        font-size: 1rem;
        padding: 0.6rem 1.2rem;
        border-radius: 12px;
        border: none;
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #7b1fa2;
        transform: scale(1.05);
    }
    .stDataFrame {
        background-color: #ffffff;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Tabs for Dashboard and Result ---
tabs = st.tabs(["📊 Dashboard", "📈 Results"])

# --- Load and preview data ---
df = load_and_clean_data()

with tabs[0]:
    st.markdown('<div class="main">', unsafe_allow_html=True)

    st.markdown("## 💳 Credit Card Fraud Detection")
    st.markdown("Welcome to Lucy’s AI dashboard for detecting fraud. 🧠✨")

    if st.toggle("🔍 Show Dataset Preview"):
        st.dataframe(df.head(), use_container_width=True)

    with st.expander("📊 Run EDA"):
        explore_data(df)

    st.markdown("### 🤖 Train Model")
    target = "Class"
    X = df.drop(columns=[target])
    y = df[target]

    left, center, right = st.columns([1, 2, 1])
    with center:
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test Size (%)", 10, 50, 30)
        with col2:
            random_state = st.number_input("Random Seed", value=42)

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
            st.success("✅ Model trained successfully!")

    st.markdown("</div>", unsafe_allow_html=True)

# --- Results Tab ---
with tabs[1]:
    if st.session_state.get("trained", False):
        st.markdown('<div class="main">', unsafe_allow_html=True)

        st.markdown("## 📈 Evaluation & Report")
        model = st.session_state.model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test

        report_df = evaluate_model(model, X_test, y_test)
        st.markdown("### 📋 Classification Report")
        st.dataframe(report_df, use_container_width=True)

        csv = download_results(report_df)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="fraud_model_report.csv">📥 Download Results as CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please train the model in the Dashboard tab.")

