import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ---------- LOAD DATA ----------
path = "data/Processed_data_FD.csv"
df = pd.read_csv(path)

print("Shape:", df.shape)
print("\nColumns:\n", df.columns)

# ---------- BASIC CLEANING ----------
df.columns = df.columns.str.replace('‚Äì','-')
df.columns = df.columns.str.strip()

print("\nAfter cleaning column names:")
print(df.columns)

# ---------- DATA TYPES ----------
print("\nData types:")
print(df.dtypes)

# ---------- MISSING VALUES ----------
print("\nMissing values per column:")
print(df.isnull().sum())

# ---------- DUPLICATES ----------
print("\nDuplicate rows:", df.duplicated().sum())

# ---------- BASIC STATS ----------
print("\nSummary statistics:")
print(df.describe())

# ---------- CHECK NORMALIZATION RANGE ----------
print("\nMin values:")
print(df.min())

print("\nMax values:")
print(df.max())

# ---------- TARGET DISTRIBUTION ----------
target = "Fraudulent Transaction (Yes/No)"

print("\nFraud distribution:")
print(df[target].value_counts())

print("\nFraud ratio:")
print(df[target].value_counts(normalize=True))

# ---------- CLASS IMBALANCE METRIC ----------
fraud_counts = df[target].value_counts()
imbalance_ratio = fraud_counts.max() / fraud_counts.min()
print("\nImbalance ratio:", imbalance_ratio)

# ---------- FEATURE CORRELATION ----------
corr = df.corr()

plt.figure(figsize=(12,10))
plt.imshow(corr, cmap="coolwarm")
plt.colorbar()
plt.title("Correlation Matrix")
plt.show()

# ---------- FEATURE CORRELATION WITH FRAUD ----------
fraud_corr = corr[target].sort_values(ascending=False)
print("\nTop correlations with fraud:")
print(fraud_corr)

# ---------- CHECK ONE HOT GROUPS ----------
product_cols = [c for c in df.columns if "Product Category" in c]
payment_cols = [c for c in df.columns if "Payment Method" in c]
device_cols = [c for c in df.columns if "Device Used" in c]

print("\nProduct category distribution:")
print(df[product_cols].sum())

print("\nPayment method distribution:")
print(df[payment_cols].sum())

print("\nDevice distribution:")
print(df[device_cols].sum())

# ---------- VERIFY ONE HOT VALIDITY ----------
print("\nRows violating product one-hot encoding:")
print((df[product_cols].sum(axis=1) != 1).sum())

print("\nRows violating payment one-hot encoding:")
print((df[payment_cols].sum(axis=1) != 1).sum())

print("\nRows violating device one-hot encoding:")
print((df[device_cols].sum(axis=1) != 1).sum())

# ---------- DISTRIBUTION CHECK ----------
numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    plt.figure()
    df[col].hist(bins=50)
    plt.title(col)
    plt.show()

# ---------- STRATIFIED SAMPLE (OPTIMAL 10K) ----------
sample_10k, _ = train_test_split(
    df,
    train_size=10000,
    stratify=df[target],
    random_state=42
)

print("\nSample shape:", sample_10k.shape)

print("\nSample fraud distribution:")
print(sample_10k[target].value_counts())

# ---------- VERIFY SAMPLE REPRESENTATION ----------
print("\nSample product distribution:")
print(sample_10k[product_cols].sum())

print("\nSample payment distribution:")
print(sample_10k[payment_cols].sum())

print("\nSample device distribution:")
print(sample_10k[device_cols].sum())

# ---------- SAVE DATASET ----------
output_path = "data/sample_10k_dataset.csv"
sample_10k.to_csv(output_path, index=False)

print("\nSaved optimized dataset to:", output_path)