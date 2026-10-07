import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ---------- LOAD DATA ----------
df = pd.read_csv("data/optimal_10k_dataset.csv")
df.columns = df.columns.str.replace('‚Äì','-').str.strip()

target = "Fraudulent Transaction (Yes/No)"

# ---------- FEATURE ENGINEERING ----------
df["Amount_to_AccountAge"] = df["Transaction Amount ($)"] / (df["Account Age (Days)"] + 1e-6)

df["Night_Transaction"] = (df["Transaction Hour of Day"] > 0.8).astype(int)

df["High_Amount"] = (df["Transaction Amount ($)"] > df["Transaction Amount ($)"].quantile(0.9)).astype(int)

df["Young_Account"] = (df["Account Age (Days)"] < df["Account Age (Days)"].quantile(0.1)).astype(int)

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

# ---------- MODEL ----------
model = XGBClassifier(
    n_estimators=800,
    max_depth=7,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

# ---------- PREDICTIONS ----------
y_prob = model.predict_proba(X_test)[:,1]

threshold = 0.4
y_pred = (y_prob > threshold).astype(int)

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