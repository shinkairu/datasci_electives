import streamlit as st
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from app_utils import load_and_clean_data, explore_data, evaluate_model

# Streamlit app layout
st.title("Credit Card Fraud Detection Dashboard")

# Load and preview data
df = load_and_clean_data()
st.write("### Dataset Preview")
st.dataframe(df.head())

# EDA section
if st.checkbox("Run Exploratory Data Analysis (EDA)"):
    explore_data(df)

# Modeling
st.write("## Train Fraud Detection Model")
target = "Class"
X = df.drop(columns=[target])
y = df[target]

test_size = st.slider("Test set size (%)", 10, 50, 30)
random_state = st.number_input("Random seed", value=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size / 100, random_state=random_state, stratify=y
)

model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

st.success("Model trained successfully!")

# Evaluation
st.write("## Evaluation Metrics and Visualizations")
report_df = evaluate_model(model, X_test, y_test)
st.dataframe(report_df)
