import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from imblearn.over_sampling import SMOTE

# ---------- LOAD DATA ----------
df = pd.read_csv("data/optimal_10k_dataset.csv")
df.columns = df.columns.str.replace('‚Äì','-').str.strip()

target = "Fraudulent Transaction (Yes/No)"

# ---------- FEATURE ENGINEERING ----------
df["Amount_to_AccountAge"] = df["Transaction Amount ($)"] / (df["Account Age (Days)"] + 1e-6)

df["Night_Transaction"] = (df["Transaction Hour of Day"] > 0.8).astype(int)

df["High_Amount"] = (df["Transaction Amount ($)"] >
                     df["Transaction Amount ($)"].quantile(0.9)).astype(int)

df["Young_Account"] = (df["Account Age (Days)"] <
                       df["Account Age (Days)"].quantile(0.1)).astype(int)

df["Device_Risk"] = (
    df["Device Used – Mobile"] * 1 +
    df["Device Used – Tablet"] * 0.8 +
    df["Device Used – Desktop"] * 0.5
)

df["Payment_Risk"] = (
    df["Payment Method – PayPal"] * 1 +
    df["Payment Method – Bank Transfer"] * 0.9 +
    df["Payment Method – Credit Card"] * 0.6 +
    df["Payment Method – Debit Card"] * 0.5
)

# ---------- SPLIT ----------
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Train:", X_train.shape)
print("Test:", X_test.shape)

# ---------- SCALING ----------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------- SMOTE ----------
smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print(pd.Series(y_train).value_counts())

# ---------- BASE MODELS ----------
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

xgb = XGBClassifier(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

lgb = LGBMClassifier(
    n_estimators=600,
    learning_rate=0.05,
    random_state=42
)

# ---------- STACKING MODEL ----------
stack = StackingClassifier(
    estimators=[
        ("rf", rf),
        ("xgb", xgb),
        ("lgb", lgb)
    ],
    final_estimator=LogisticRegression(),
    n_jobs=-1
)

stack.fit(X_train, y_train)

# ---------- PREDICTION PROBABILITIES ----------
y_prob = stack.predict_proba(X_test)[:,1]

# ---------- THRESHOLD OPTIMIZATION ----------
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print("\nBest Threshold:", best_threshold)

y_pred = (y_prob > best_threshold).astype(int)

# ---------- METRICS ----------
print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

roc = roc_auc_score(y_test, y_prob)
print("\nROC AUC:", roc)

# ---------- ROC CURVE ----------
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label="ROC AUC = %.3f" % roc)
plt.plot([0,1],[0,1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# ---------- PRECISION RECALL CURVE ----------
plt.figure()

plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")

plt.show()

# ---------- FEATURE IMPORTANCE ----------
# using XGBoost importance for interpretability
xgb.fit(X_train, y_train)

importances = pd.Series(xgb.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print("\nTop Fraud Features")
print(importances.head(10))

importances.head(10).plot(kind="bar")

plt.title("Top Fraud Predictors")
plt.show()

# --- CURRENT STATS SUMMARY ---
print("="*50)
print("MODEL PERFORMANCE SUMMARY")
print("="*50)
print(f"ROC AUC:   {roc:.4f}")
print(f"Threshold: {best_threshold:.4f}")
print()
print(classification_report(y_test, y_pred, 
      target_names=["Legit", "Fraud"]))
print(confusion_matrix(y_test, y_pred))