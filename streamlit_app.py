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
from app_utils import load_and_clean_data, explore_data, evaluate_model, show_metric_bar_chart, plot_confusion_matrix, plot_roc_curve, show_feature_importance, download_results

# page configuration
st.set_page_config(page_title="Credit Card Fraud Detector 💳", layout="wide")

# background image setup
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

    .css-1bzp7po {{
        justify-content: center !important;
    }}

    button[data-baseweb="tab"] {{
        background-color: rgba(255, 255, 255, 0.65);
        border: 2px solid #7b1fa2;
        color: #4A0072;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 12px;
        margin: 5px;
        transition: all 0.3s ease;
    }}

    button[data-baseweb="tab"]:hover {{
        background-color: #e1bee7;
        border-color: #4a0072;
        color: #2e003e;
    }}

    button[data-baseweb="tab"][aria-selected="true"] {{
        background-color: #7b1fa2;
        color: white;
        border-color: #4a0072;
    }}

    .main-tab-container {{
        background: rgba(255, 255, 255, 0.65);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 0 25px rgba(0,0,0,0.15);
        animation: fadeIn 1.2s ease-in-out;
    }}

    .info-box {{
        background: rgba(255, 255, 255, 0.65);
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
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
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
    st.title("Credit Card Fraud Detection Tool 💳")

    # Description
    st.markdown("""
    <div class='info-box'>
        <h3>🚀 Description</h3>
        <p>This tool leverages <strong>XGBoost</strong> to detect fraudulent credit card transactions using a real-world dataset.</p>
    </div>
    """, unsafe_allow_html=True)

    # Dataset Info
    st.markdown("""
    <div class='info-box'>
        <h4>📂 Dataset Info</h4>
        <ul>
            <li>This dataset involves real-world anonymized credit card data.</li>
            <li>It includes 31 features: principal components, amount, and class.</li>
            <li>Note: The dataset is highly imbalanced: only ~0.17% fraud cases.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Purpose
    st.markdown("""
    <div class='info-box'>
        <h4>💡 Purpose</h4>
        <p>This prototype helps analysts:</p>
        <ul>
            <li>Explore data visually</li>
            <li>Train a fraud detection model</li>
            <li>View model evaluation metrics</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# home tab
with tabs[1]:
    st.title("📊 Dataset & EDA")
    if st.toggle("📁 Preview Dataset Head"):
        st.dataframe(df.head(), use_container_width=True)

    with st.expander("📊 Run Exploratory Data Analysis (EDA)"):
        with st.spinner("Running EDA..."):
            explore_data(df)
            
    st.markdown("<div class='main-tab-container'>", unsafe_allow_html=True)
    
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

        if st.expander("📋 Show Evaluation Report & Graphs"):
            model = st.session_state.model
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test

            report_df = evaluate_model(model, X_test, y_test)
            st.markdown("### 📋 Classification Report")
            st.dataframe(report_df, use_container_width=True)

            st.write("### 📈 Evaluation Metrics")
            st.pyplot(show_metric_bar_chart(metrics))

            st.markdown("### 📊 Confusion Matrix")
            st.pyplot(plot_confusion_matrix(model, X_test, y_test))

            st.markdown("### 📈 ROC Curve")
            st.pyplot(plot_roc_curve(model, X_test, y_test))
            
            st.markdown("### 🔍 Feature Importance")
            st.pyplot(show_feature_importance(model))

            csv = download_results(report_df)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="fraud_report.csv">📥 Download CSV</a>'
            st.markdown(href, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# contact info tab
with tabs[2]:
    st.markdown("""
    <div class='main-tab-container'>
        <h1>📬 Contact</h1>
        <div class='info-box'>
            <p><strong>Developed by:</strong>: Alexine, Lyca, Fredric and Kit ^^ </p>
            <p><strong>GitHub</strong>: <a href='https://github.com/shinkairu' target='_blank'>github.com/shinkairu</a></p>
            <p><strong>Email</strong>: bestgroupever@gmail.com</p>
            <blockquote>This is only a sample Streamlit prototype developed for our CpE Elective (Data Science) Project.Thank you!</blockquote>
        </div>
    </div>
    """, unsafe_allow_html=True)
