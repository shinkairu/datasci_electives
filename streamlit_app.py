import streamlit as st
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from app_utils import load_and_clean_data, explore_data, evaluate_model

# Page config
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# Custom CSS styling for a purple-themed interface
st.markdown("""
    <style>
    body {
        background-color: #f8f0fa;
    }
    .main {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 2rem;
    }
    h1, h2, h3, h4 {
        color: #6a1b9a;
    }
    .stButton>button {
        background-color: #7e57c2;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .stSlider > div[data-baseweb="slider"] > div {
        background: linear-gradient(90deg, #8e24aa, #ce93d8);
    }
    .stDataFrame thead tr th {
        background-color: #f3e5f5;
        color: #4a148c;
    }
    </style>
""", unsafe_allow_html=True)

# App Title
st.title("💳 Credit Card Fraud Detection Dashboard")
st.write("A visual and interactive way to explore fraud detection using machine learning.")

# Load and preview data
df = load_and_clean_data()
st.subheader("🔍 Dataset Preview")
with st.expander("Show first 5 rows of data"):
    st.dataframe(df.head(), use_container_width=True)

# EDA section
with st.expander("📊 Run Exploratory Data Analysis (EDA)"):
    run_eda = st.checkbox("Run EDA", value=False)
    if run_eda:
        explore_data(df)

# Modeling Section
st.subheader("🤖 Train Fraud Detection Model")
target = "Class"
X = df.drop(columns=[target])
y = df[target]

col1, col2 = st.columns(2)
with col1:
    test_size = st.slider("Test set size (%)", 10, 50, 30)
with col2:
    random_state = st.number_input("Random seed", value=42)

run_model = st.button("Train Model")

if run_model:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size / 100, random_state=random_state, stratify=y
    )

    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    st.success("✅ Model trained successfully!")

    # Evaluation Section
    st.subheader("📈 Evaluation Metrics and Visualizations")
    report_df = evaluate_model(model, X_test, y_test)
    st.write("### 📋 Classification Report")
    st.dataframe(report_df.style.background_gradient(cmap='Purples'), use_container_width=True)
