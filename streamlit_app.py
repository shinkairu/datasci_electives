import streamlit as st
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import sys
import os
import base64
from PIL import Image

sys.path.append(os.path.dirname(__file__))    # custom utilitiy function for better pathing
from app_utils import load_and_clean_data, explore_data, evaluate_model, download_results

sys.path.append(os.path.dirname(__file__))
from app_utils import load_and_clean_data, explore_data, evaluate_model, plot_confusion_matrix, plot_roc_curve, show_feature_importance, download_results

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
    .main-tab-container {{
        background: rgba(255, 255, 255, 0.85);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 0 25px rgba(0,0,0,0.15);
        animation: fadeIn 1.2s ease-in-out;
    }}
    .info-box {{
        background: rgba(255, 255, 255, 0.95);
        border-left: 6px solid #7b1fa2;
        padding: 1.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }}
    h1, h2, h3 {{
        color: #4A0072;
        text-shadow: 1px 1px #ffffff;
        font-size: 2.5em;
    }}
    html, body, p, div {{
        color: #2e003e !important;
        font-size: 18px !important;
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
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# set image as background
bg_path = os.path.join(os.path.dirname(__file__), "lucy.png")
if os.path.exists(bg_path):
    set_background(bg_path)

# load and preview cleaned data
df = load_and_clean_data()

# tabs
tabs = st.tabs(["🏠 HOME", "🔬 PROTOTYPE", "📬 CONTACT"])

with tabs[0]:
    st.markdown("""
<div class='info-box'>
    <h3>✅ Test Box</h3>
    <p>This is a test to see if the CSS box works.</p>
</div>
""", unsafe_allow_html=True)
    
    st.markdown("<div class='main-tab-container'>", unsafe_allow_html=True)
    st.title("Credit Card Fraud Detection Tool")
    
    # Intro Description
    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
    st.markdown("""
    ### 🚀 This tool leverages **XGBoost** to detect fraudulent credit card transactions.
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Dataset Info
    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
    st.markdown("""
    #### 📂 Dataset Info:
    - This dataset involves real-world anonymized credit card data.
    - It has 31 features: principal components, amount, class.
    - Note: The dataset is highly imbalanced: only ~0.17% fraud cases.
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Purpose
    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
    st.markdown("""
    #### 💡 Purpose:
    This prototype helps analysts:
    - Explore data visually  
    - Train a fraud detection model  
    - View model evaluation metrics  
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True) 

# home tab
with tabs[1]:
    st.markdown("<div class='main-tab-container'>", unsafe_allow_html=True)
    
    st.subheader("📊 Dataset & EDA")
    if st.toggle("📁 Preview Dataset Head"):
        st.dataframe(df.head(), use_container_width=True)

    with st.expander("📊 Run Exploratory Data Analysis (EDA)"):
        with st.spinner("Running EDA..."):
        explore_data(df)

    st.subheader("⚙️ Train Fraud Detection Model")
    target = "Class"
    X = df.drop(columns=[target])
    y = df[target]

    test_size = st.slider("Test set size (%)", 10, 50, 30)
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
        st.success("✅ Model trained successfully! Scroll down to view results.")

    # added the 'results' as a subsection
    if st.session_state.get("trained", False):
        st.subheader("📈 Model Evaluation & Results")

        if st.toggle("📋 Show Evaluation Report & Graphs"):
            model = st.session_state.model
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test

            report_df = evaluate_model(model, X_test, y_test)
            st.markdown("### 📋 Classification Report")
            st.dataframe(report_df, use_container_width=True)

            # Add visualizations here
            st.markdown("### 📊 Confusion Matrix")
            st.pyplot(plot_confusion_matrix(model, X_test, y_test))

            st.markdown("### 📈 ROC Curve")
            st.pyplot(plot_roc_curve(model, X_test, y_test))
            
            st.markdown("### 🔍 Feature Importance")
            show_feature_importance(model)

            csv = download_results(report_df)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="fraud_report.csv">📥 Download CSV</a>'
            st.markdown(href, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# contact info tab
with tabs[2]:
    st.markdown("<div class='main-tab-container'>", unsafe_allow_html=True)
    st.title("📬 Contact")
    
    st.markdown("**Developer**: Shinkairu", unsafe_allow_html=True)
    st.markdown("**GitHub**: [github.com/shinkairu](https://github.com/shinkairu)", unsafe_allow_html=True)
    st.markdown("**Email**: your_email@example.com", unsafe_allow_html=True)
    st.markdown("> This is a sample Streamlit prototype developed for educational and demonstration purposes.", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
