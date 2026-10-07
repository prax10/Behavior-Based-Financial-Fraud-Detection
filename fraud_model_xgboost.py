import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

from xgboost import XGBClassifier

# ---------- LOAD DATA ----------
df = pd.read_csv("data/optimal_10k_dataset.csv")

df.columns = df.columns.str.replace('‚Äì','-').str.strip()

target = "Fraudulent Transaction (Yes/No)"

X = df.drop(columns=[target])
y = df[target]

# ---------- TRAIN TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Train:", X_train.shape)
print("Test:", X_test.shape)

# ---------- MODEL ----------
model = XGBClassifier(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

# ---------- PREDICTIONS ----------
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:,1]

# ---------- METRICS ----------
acc = accuracy_score(y_test, y_pred)

print("\nAccuracy:", acc)

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ---------- ROC CURVE ----------
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()

plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc)
plt.plot([0,1], [0,1])

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()
plt.show()

print("\nROC AUC:", roc_auc)

# ---------- FEATURE IMPORTANCE ----------
importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print("\nTop Features")
print(importances.head(10))

importances.head(10).plot(kind="bar")

plt.title("Top Fraud Predictors")
plt.show()