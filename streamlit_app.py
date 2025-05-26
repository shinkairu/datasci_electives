import streamlit as st
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import sys
import os

sys.path.append(os.path.dirname(__file__))
from app_utils import load_and_clean_data, explore_data, evaluate_model, download_results

import base64
from PIL import Image

# Add project directory to path
sys.path.append(os.path.dirname(__file__))
from app_utils import load_and_clean_data, explore_data, evaluate_model, download_results

# --- Page Config ---
st.set_page_config(page_title="Credit Card Fraud Detector 💳", layout="wide")

# --- Background image setup ---
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

def set_background(image_path):
    bin_str = get_base64_of_bin_file(image_path)
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }}
    .main-container {{
        background: rgba(255, 255, 255, 0.8);
        border-radius: 16px;
        padding: 3rem;
        margin-top: 2rem;
        box-shadow: 0 0 25px rgba(0,0,0,0.15);
        animation: fadeIn 1.2s ease-in-out;
    }}
    h1, h2, h3 {{
        color: #4A0072;
        text-shadow: 1px 1px #ffffff;
        font-size: 2.2em;
    }}
    .stButton>button {{
        background-color: #7b1fa2;
        color: white;
        border-radius: 10px;
        padding: 10px 24px;
        font-size: 20px;
    }}
    
    html, body, p, div {{
        color: #2e003e !important;  /* Dark purple text */
        font-size: 18px !important;  /* Larger base font */
        font-family: 'Segoe UI', sans-serif;
    }}
    
    .stButton>button {{
        background-color: #7b1fa2;
        color: white;
        border-radius: 10px;
        padding: 10px 24px;
        font-size: 20px;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }}

    .stButton>button:hover {{
        background-color: #4a0072;
    }}
    

    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    </style>
    """
    
    st.markdown(css, unsafe_allow_html=True)

# Set image as background
background_path = os.path.join(os.path.dirname(__file__), "lucy.png")
if os.path.exists(background_path):
    set_background(background_path)
else:
    st.warning("⚠️ Background image 'lucy.png' not found.")

# Load and preview data
df = load_and_clean_data()

# Tabs for Dashboard and Result
tabs = st.tabs(["📊 DASHBOARD", "📈 RESULTS"])

with tabs[0]:
    st.markdown("""
    <div class="main">
    """, unsafe_allow_html=True)

    st.title("Lucy's Fraud Detection AI")
    st.subheader("🔍 Powered by XGBoost | Smart Detection for Smarter Security")

    if st.toggle("📁 Preview Dataset Head"):
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

        if st.button("🚀 Train XGBoost Model"):
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
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        st.subheader("📈 Model Evaluation")

        model = st.session_state.model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test

        report_df = evaluate_model(model, X_test, y_test)

        st.markdown("### 📋 Classification Report")
        st.dataframe(report_df, use_container_width=True)

        csv = download_results(report_df)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="lucy_fraud_report.csv">📥 Download CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Train the model first in the Dashboard tab.")
