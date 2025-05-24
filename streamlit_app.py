import streamlit as st
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from app_utils import load_and_clean_data, explore_data, evaluate_model, download_results
import base64

# Page config
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# Custom CSS styling for a purple-themed interface with motion and background image
st.markdown("""
    <style>
    body {
        background-color: #1e002f;
        background-image: url('data:image/png;base64,{base64_image}');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .main {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        padding: 2rem;
        animation: fadeIn 1s ease-in-out;
    }
    h1, h2, h3, h4 {
        color: #ba68c8;
    }
    .stButton>button {
        background-color: #7e57c2;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #512da8;
    }
    .stSlider > div[data-baseweb="slider"] > div {
        background: linear-gradient(90deg, #8e24aa, #ce93d8);
    }
    .stDataFrame thead tr th {
        background-color: #f3e5f5;
        color: #4a148c;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    </style>
""".replace("{base64_image}", base64.b64encode(open("/mnt/data/3a108fb6-884d-4af6-b569-a3da2c10aad9.png", "rb").read()).decode()), unsafe_allow_html=True)

# Tabs for Dashboard and Result
tabs = st.tabs(["📊 Dashboard", "📈 Results"])

# Load and preview data
df = load_and_clean_data()

with tabs[0]:
    st.title("💳 Credit Card Fraud Detection Dashboard")
    st.write("A visual and interactive way to explore fraud detection using machine learning.")

    if st.checkbox("Show Dataset Preview"):
        st.subheader("🔍 Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

    with st.expander("📊 Run Exploratory Data Analysis (EDA)"):
        explore_data(df)

    st.subheader("🤖 Train Fraud Detection Model")
    target = "Class"
    X = df.drop(columns=[target])
    y = df[target]

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

with tabs[1]:
    if st.session_state.get("trained", False):
        model = st.session_state.model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test

        st.subheader("📈 Evaluation Metrics and Visualizations")
        report_df = evaluate_model(model, X_test, y_test)

        st.write("### 📋 Classification Report")
        st.dataframe(report_df.style.background_gradient(cmap='Purples'), use_container_width=True)

        # Download button
        csv = download_results(report_df)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="fraud_model_report.csv">📥 Download Results as CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please train the model first in the Dashboard tab.")
