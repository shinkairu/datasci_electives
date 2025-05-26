# === app_utils.py ===

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from xgboost import plot_importance
import kagglehub


def load_and_clean_data():
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    df = pd.read_csv(f"{path}/creditcard.csv")
    df.drop_duplicates(inplace=True)
    df['Amount'] = StandardScaler().fit_transform(df[['Amount']])
    df.drop(columns='Time', inplace=True)
    return df


def explore_data(df):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 5))
    sns.countplot(x='Class', data=df, palette=["#4a148c", "#db76db"])
    plt.title("Class Distribution")
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.histplot(df, x="Amount", hue="Class", bins=50, kde=True,
                 element="step", stat="density", common_norm=False,
                 palette=["#4a148c", "#db76db"])
    plt.title("Transaction Amount by Class")
    plt.show()

    corr = df.corr()
    top_corr = corr['Class'].abs().sort_values(ascending=False)[1:11]
    selected = top_corr.index.tolist() + ['Class']
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[selected].corr(), annot=True, cmap="Purples", fmt=".2f")
    plt.title("Top Feature Correlations")
    plt.show()

    plt.figure(figsize=(6, 5))
    sns.boxplot(x='Class', y='Amount', data=df, palette=["#4a148c", "#db76db"])
    plt.yscale("log")
    plt.title("Boxplot of Amount by Class")
    plt.show()

    top3 = top_corr.index[:3].tolist()
    sns.pairplot(df.sample(2000), vars=top3, hue="Class",
                 palette=["#4a148c", "#db76db"], corner=True)
    plt.suptitle("Top 3 Features by Class", y=1.02)
    plt.show()


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
        'Score': [acc, prec, rec, f1, auc]
    })

    return metrics_df


def show_feature_importance(model):
    plt.figure(figsize=(10, 6))
    plot_importance(model, importance_type='gain', max_num_features=10,
                    title='Top 10 Important Features', color='#8e44ad')
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model, X_test, y_test):
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    return fig


def plot_roc_curve(model, X_test, y_test):
    fig, ax = plt.subplots()
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    return fig


def download_results(report_df):
    return report_df.to_csv(index=False)
