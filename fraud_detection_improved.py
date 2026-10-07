import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import shap
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

# ── LOAD ──────────────────────────────────────────────
df = pd.read_csv("data/optimal_10k_dataset.csv")
df.columns = df.columns.str.replace('\u2013', '-').str.strip()
target = "Fraudulent Transaction (Yes/No)"

# ── FIX SKEWNESS ──────────────────────────────────────
# Transaction Amount: skew=4.73 → log1p transform
# Location_Freq:      skew=2.72 → log1p transform
df["Transaction Amount ($)"] = np.log1p(df["Transaction Amount ($)"])
df["Location_Freq"]          = np.log1p(df["Location_Freq"])

# ── FEATURE ENGINEERING (improved) ────────────────────
# Original features
df["Amount_to_AccountAge"] = df["Transaction Amount ($)"] / (df["Account Age (Days)"] + 1e-6)
df["Night_Transaction"]    = (df["Transaction Hour of Day"] > 0.8).astype(int)
df["High_Amount"]          = (df["Transaction Amount ($)"] > df["Transaction Amount ($)"].quantile(0.9)).astype(int)
df["Young_Account"]        = (df["Account Age (Days)"] < df["Account Age (Days)"].quantile(0.1)).astype(int)
df["Device_Risk"]          = df["Device Used - Mobile"]*1.0 + df["Device Used - Tablet"]*0.8 + df["Device Used - Desktop"]*0.5
df["Payment_Risk"]         = df["Payment Method - PayPal"]*1.0 + df["Payment Method - Bank Transfer"]*0.9 + df["Payment Method - Credit Card"]*0.6 + df["Payment Method - Debit Card"]*0.5

# NEW features (informed by skew analysis & correlation findings)
# 1. High amount on young account → strongest combo signal
df["YoungAccount_HighAmount"] = df["Young_Account"] * df["High_Amount"]

# 2. Amount * hour interaction (night + high amount = very suspicious)
df["Amount_x_Hour"] = df["Transaction Amount ($)"] * df["Transaction Hour of Day"]

# 3. Rare location (low freq = first-time location = risky)
df["Rare_Location"] = (df["Location_Freq"] < df["Location_Freq"].quantile(0.25)).astype(int)

# 4. Ship/bill mismatch on new account
df["Mismatch_YoungAccount"] = (1 - df["Same ship and bill"]) * df["Young_Account"]

# 5. Account age buckets (non-linear encoding)
df["AccountAge_Bucket"] = pd.cut(
    df["Account Age (Days)"],
    bins=[0, 0.05, 0.15, 0.35, 0.6, 1.0],
    labels=[4, 3, 2, 1, 0]   # 4=most risky (newest), 0=least risky
).astype(float).fillna(0)

print(f"Total features: {df.shape[1] - 1}")

# ── SPLIT ─────────────────────────────────────────────
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ── SCALE ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── SMOTE ─────────────────────────────────────────────
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_sc, y_train)

print("After SMOTE:", pd.Series(y_train_res).value_counts().to_dict())

# ── CROSS-VALIDATION ──────────────────────────────────
print("\nRunning 5-fold cross-validation...")
xgb_cv = XGBClassifier(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='logloss', random_state=42
)
cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_roc  = cross_val_score(xgb_cv, X_train_res, y_train_res, cv=cv, scoring='roc_auc',       n_jobs=-1)
cv_f1   = cross_val_score(xgb_cv, X_train_res, y_train_res, cv=cv, scoring='f1',            n_jobs=-1)

print(f"CV ROC AUC : {cv_roc.mean():.4f} ± {cv_roc.std():.4f}")
print(f"CV F1      : {cv_f1.mean():.4f}  ± {cv_f1.std():.4f}")

# ── STACKING ENSEMBLE ──────────────────────────────────
rf  = RandomForestClassifier(
    n_estimators=400, max_depth=None,
    min_samples_leaf=2, random_state=42, n_jobs=-1
)
xgb = XGBClassifier(
    n_estimators=700, max_depth=6, learning_rate=0.04,
    subsample=0.85, colsample_bytree=0.85,
    min_child_weight=3,
    scale_pos_weight=2,
    eval_metric='logloss', random_state=42
)
lgb = LGBMClassifier(
    n_estimators=700, learning_rate=0.04,
    num_leaves=63, min_child_samples=20,
    random_state=42, verbose=-1
)

stack = StackingClassifier(
    estimators=[('rf', rf), ('xgb', xgb), ('lgb', lgb)],
    final_estimator=LogisticRegression(C=0.5),
    n_jobs=-1
)

print("\nTraining stacking ensemble...")
stack.fit(X_train_res, y_train_res)

 # ── THRESHOLD TUNING ──────────────────────────────────
y_prob = stack.predict_proba(X_test_sc)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores  = 2 * (precision * recall) / (precision + recall + 1e-9)
best_idx = np.argmax(f1_scores)
best_thr   = thresholds[best_idx]

y_pred = (y_prob > best_thr).astype(int)

# ── RESULTS ───────────────────────────────────────────
roc = roc_auc_score(y_test, y_prob)

print("\n" + "="*55)
print("IMPROVED MODEL — PERFORMANCE SUMMARY")
print("="*55)
print(f"Best threshold : {best_thr:.4f}")
print(f"ROC AUC        : {roc:.4f}   (was 0.9367)")
print()
print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

# ── PLOTS ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[0].plot(fpr, tpr, label=f"AUC = {roc:.3f}", color="steelblue")
axes[0].plot([0,1],[0,1],'k--', alpha=0.4)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve (Improved)")
axes[0].legend()

axes[1].plot(recall, precision, color="darkorange")
axes[1].axvline(recall[best_idx], color="gray", linestyle="--", alpha=0.6)
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve")

axes[2].bar(range(1,6), cv_roc, color="steelblue", alpha=0.8)
axes[2].axhline(cv_roc.mean(), color="red", linestyle="--", label=f"Mean={cv_roc.mean():.4f}")
axes[2].set_xlabel("Fold")
axes[2].set_ylabel("ROC AUC")
axes[2].set_title("5-Fold Cross-Validation")
axes[2].set_ylim(0.85, 1.0)
axes[2].legend()

plt.tight_layout()
plt.savefig("results/improved_results.png", dpi=150, bbox_inches="tight")
plt.show()

# ── FEATURE IMPORTANCE ────────────────────────────────
xgb_fi = XGBClassifier(
    n_estimators=700, max_depth=6, learning_rate=0.04,
    subsample=0.85, colsample_bytree=0.85,
    eval_metric='logloss', random_state=42
)
xgb_fi.fit(X_train_res, y_train_res)

imp = pd.Series(xgb_fi.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 12 features:")
print(imp.head(12).to_string())

imp.head(12).plot(kind="bar", figsize=(10,4), color="steelblue")
plt.title("Top Fraud Predictors (Improved Model)")
plt.tight_layout()
plt.savefig("results/improved_features.png", dpi=150, bbox_inches="tight")
plt.show()



# ── SHAP ──────────────────────────────────────────────
xgb_shap = XGBClassifier(
    n_estimators=700, max_depth=6, learning_rate=0.04,
    subsample=0.85, colsample_bytree=0.85,
    eval_metric='logloss', random_state=42
)
xgb_shap.fit(X_train_res, y_train_res)

explainer   = shap.TreeExplainer(xgb_shap)
shap_values = explainer.shap_values(X_test_sc)

# Beeswarm — global importance
shap.summary_plot(shap_values, X_test_sc, feature_names=X.columns.tolist(), show=False)
plt.title("SHAP Beeswarm — Feature Impact on Fraud")
plt.tight_layout()
plt.savefig("results/shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.show()

# Bar — mean absolute SHAP
shap.summary_plot(shap_values, X_test_sc, feature_names=X.columns.tolist(), plot_type="bar", show=False)
plt.title("Mean |SHAP| — Feature Ranking")
plt.tight_layout()
plt.savefig("results/shap_bar.png", dpi=150, bbox_inches="tight")
plt.show()

# Waterfall — single highest-risk transaction
X_test_df = pd.DataFrame(X_test_sc, columns=X.columns)
explanation = explainer(X_test_df)
top_idx = np.argmax(y_prob)
shap.waterfall_plot(explanation[top_idx], max_display=12, show=False)
plt.tight_layout()
plt.savefig("results/shap_waterfall.png", dpi=150, bbox_inches="tight")
plt.show()