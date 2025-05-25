import streamlit as st
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd
import base64
import os
import sys

# --- Page Config ---
st.set_page_config(page_title="Lucy’s AI Fraud Detector 💳", layout="wide")

# --- Set background ---
background_path = os.path.join(os.path.dirname(__file__), "lucy.png")
if os.path.exists(background_path):
    set_background(background_path)
else:
    st.warning("⚠️ Background image 'lucy.png' not found.")

# --- Sidebar Navigation ---
st.sidebar.title("🌐 Navigation")
section = st.sidebar.radio("Go to", ["🏠 Home", "🧪 Prototype", "📬 Contact Info"])

if section == "🏠 Home":
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.title("📊 Welcome to Lucy’s AI Fraud Detector")
    st.markdown("""
    This tool leverages **XGBoost** to detect fraudulent credit card transactions.

    #### 📂 Dataset Info:
    - Sourced from a real-world dataset of anonymized credit card transactions.
    - Includes 31 features: anonymized principal components, time, amount, and class.
    - Highly imbalanced: only ~0.17% are fraud cases.

    #### 💡 Purpose:
    This prototype aims to help analysts:
    - Explore data visually
    - Train a fraud detection model
    - View model evaluation and performance charts
    """)
    st.markdown('</div>', unsafe_allow_html=True)

elif section == "🧪 Prototype":
    sub_tab = st.sidebar.radio("Prototype", ["💳 Prototype Dashboard", "📈 Result Tab"])

    # --- Load Data ---
    df = load_and_clean_data()

    if sub_tab == "💳 Prototype Dashboard":
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        st.title("Lucy's Fraud Detection AI")
        st.subheader("🔍 Powered by XGBoost | Smart Detection for Smarter Security")

        if st.toggle("📁 Show Raw Dataset"):
            st.dataframe(df.head(), use_container_width=True)

        with st.expander("📊 Run Data Exploration"):
            explore_data(df)

        st.markdown("### 🤖 Train Model")
        target = "Class"
        X = df.drop(columns=[target])
        y = df[target]

        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test Set Size (%)", 10, 50, 30)
        with col2:
            random_state = st.number_input("Random Seed", min_value=0, value=42)

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
            st.success("✅ Model trained! Check the Results tab.")

        st.markdown('</div>', unsafe_allow_html=True)

    elif sub_tab == "📈 Result Tab":
        if st.session_state.get("trained", False):
            st.markdown('<div class="main">', unsafe_allow_html=True)

            st.markdown("## 📈 Evaluation & Report")
            model = st.session_state.model
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            report_df = evaluate_model(model, X_test, y_test)
            st.markdown("### 📋 Classification Report")
            st.dataframe(report_df, use_container_width=True)

            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
            import matplotlib.pyplot as plt

            # --- Confusion Matrix & ROC Curve ---
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🔍 Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
                fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
                disp.plot(ax=ax_cm, cmap="Purples", colorbar=False)
                st.pyplot(fig_cm)

            with col2:
                st.markdown("#### 🧠 ROC Curve")
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                fig_roc, ax_roc = plt.subplots(figsize=(4, 4))
                ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
                ax_roc.plot([0, 1], [0, 1], color='navy', linestyle='--')
                ax_roc.set_xlabel("False Positive Rate")
                ax_roc.set_ylabel("True Positive Rate")
                ax_roc.set_title("ROC Curve")
                ax_roc.legend(loc="lower right")
                st.pyplot(fig_roc)

            col3, col4 = st.columns(2)
            with col3:
                st.markdown("#### 🔄 Precision-Recall Curve")
                precision, recall, _ = precision_recall_curve(y_test, y_prob)
                fig_pr, ax_pr = plt.subplots(figsize=(4, 4))
                ax_pr.plot(recall, precision, color="blue", lw=2)
                ax_pr.set_xlabel("Recall")
                ax_pr.set_ylabel("Precision")
                ax_pr.set_title("Precision-Recall")
                st.pyplot(fig_pr)

            # --- Download Button ---
            csv = download_results(report_df)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="fraud_model_report.csv">📥 Download Results as CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please train the model first in the Dashboard tab.")

elif section == "📬 Contact Info":
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.title("📬 Contact")
    st.markdown("""
    **Developer**: Shinkairu  
    **GitHub**: [github.com/shinkairu](https://github.com/shinkairu)  
    **Email**: your_email@example.com  

    > This is a sample Streamlit prototype developed for educational and demonstration purposes.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
