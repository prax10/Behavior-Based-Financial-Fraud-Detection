import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

full_path = "data/Processed_data_FD.csv"
sample_path = "data/optimal_10k_dataset.csv"

df_full = pd.read_csv(full_path)
df_sample = pd.read_csv(sample_path)

df_full.columns = df_full.columns.str.replace('‚Äì','-').str.strip()
df_sample.columns = df_sample.columns.str.replace('‚Äì','-').str.strip()

target = "Fraudulent Transaction (Yes/No)"

print("FULL DATASET SHAPE:", df_full.shape)
print("SAMPLE DATASET SHAPE:", df_sample.shape)

print("\nFULL FRAUD DISTRIBUTION")
print(df_full[target].value_counts(normalize=True))

print("\nSAMPLE FRAUD DISTRIBUTION")
print(df_sample[target].value_counts(normalize=True))

numeric_cols = df_full.select_dtypes(include=np.number).columns

print("\nNUMERIC FEATURE COMPARISON (mean difference)")
for col in numeric_cols:
    full_mean = df_full[col].mean()
    sample_mean = df_sample[col].mean()
    diff = abs(full_mean - sample_mean)
    print(col, "diff:", diff)

product_cols = [c for c in df_full.columns if "Product Category" in c]
payment_cols = [c for c in df_full.columns if "Payment Method" in c]
device_cols = [c for c in df_full.columns if "Device Used" in c]

print("\nPRODUCT CATEGORY DISTRIBUTION")
print("FULL")
print(df_full[product_cols].sum())
print("SAMPLE")
print(df_sample[product_cols].sum())

print("\nPAYMENT METHOD DISTRIBUTION")
print("FULL")
print(df_full[payment_cols].sum())
print("SAMPLE")
print(df_sample[payment_cols].sum())

print("\nDEVICE DISTRIBUTION")
print("FULL")
print(df_full[device_cols].sum())
print("SAMPLE")
print(df_sample[device_cols].sum())

print("\nONE-HOT VALIDATION")

print("Product errors:", (df_sample[product_cols].sum(axis=1) != 1).sum())
print("Payment errors:", (df_sample[payment_cols].sum(axis=1) != 1).sum())
print("Device errors:", (df_sample[device_cols].sum(axis=1) != 1).sum())

for col in numeric_cols:
    plt.figure()
    df_full[col].hist(bins=50, alpha=0.5, label="full")
    df_sample[col].hist(bins=50, alpha=0.5, label="sample")
    plt.title(col)
    plt.legend()
    plt.show()