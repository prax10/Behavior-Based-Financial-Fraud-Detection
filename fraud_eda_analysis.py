import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------- LOAD DATA ----------------
path = "data/optimal_10k_dataset.csv"
df = pd.read_csv(path)

df.columns = df.columns.str.replace('‚Äì','-').str.strip()

target = "Fraudulent Transaction (Yes/No)"

print("Dataset shape:", df.shape)
print("\nFraud distribution:")
print(df[target].value_counts())

# ---------------- HEATMAP ----------------
plt.figure(figsize=(6,8))

sns.heatmap(
    df.corr()[[target]].sort_values(by=target, ascending=False),
    annot=True,
    cmap="coolwarm"
)

plt.title("Feature Correlation with Fraud")
plt.show()

# ---------------- CLASS DISTRIBUTION ----------------
plt.figure()

sns.countplot(x=target, data=df)

plt.title("Class Distribution (Fraud vs Non-Fraud)")
plt.xlabel("Fraud (1) vs Non-Fraud (0)")
plt.ylabel("Count")

plt.show()

# ---------------- TRANSACTION AMOUNT VS FRAUD ----------------
plt.figure()

sns.boxplot(
    x=target,
    y="Transaction Amount ($)",
    data=df
)

plt.title("Transaction Amount vs Fraud")
plt.show()

# ---------------- FRAUD VS DEVICE ----------------
device_cols = [c for c in df.columns if "Device Used" in c]

device_df = df.groupby(target)[device_cols].sum()

device_df.T.plot(kind="bar")

plt.title("Fraud vs Device Type")
plt.ylabel("Count")
plt.show()

# ---------------- FRAUD VS PAYMENT METHOD ----------------
payment_cols = [c for c in df.columns if "Payment Method" in c]

payment_df = df.groupby(target)[payment_cols].sum()

payment_df.T.plot(kind="bar")

plt.title("Fraud vs Payment Method")
plt.ylabel("Count")
plt.show()

# ---------------- FRAUD VS PRODUCT CATEGORY ----------------
product_cols = [c for c in df.columns if "Product Category" in c]

product_df = df.groupby(target)[product_cols].sum()

product_df.T.plot(kind="bar")

plt.title("Fraud vs Product Category")
plt.ylabel("Count")
plt.show()

# ---------------- NUMERIC DISTRIBUTIONS ----------------
numeric_cols = df.select_dtypes(include="number").columns

for col in numeric_cols:
    plt.figure()
    sns.histplot(df[col], bins=40)
    plt.title(f"Distribution: {col}")
    plt.show()

print("\nEDA complete.")