# libraries and other utility stuffs
# === app_utils.py ===
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, RocCurveDisplay,
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

    plt.figure(figsize=(8, 5))
    sns.barplot(x='Metric', y='Score', data=metrics_df, palette="Purples_r")
    plt.ylim(0, 1)
    plt.title("Model Evaluation Metrics")
    plt.tight_layout()
    plt.show()

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=["Legit", "Fraud"],
                yticklabels=["Legit", "Fraud"])
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    plot_importance(model, importance_type='gain', max_num_features=10,
                    title='Top 10 Important Features', color='#8e44ad')
    plt.tight_layout()
    plt.show()

     return metrics_df

def download_results(report_df):
    return report_df.to_csv(index=False)
