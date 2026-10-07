import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

# ---------- LOAD DATA ----------
path = "data/Processed_data_FD.csv"
df = pd.read_csv(path)

df.columns = df.columns.str.replace('‚Äì','-')
df.columns = df.columns.str.strip()

target = "Fraudulent Transaction (Yes/No)"

# ---------- SPLIT FRAUD / NON FRAUD ----------
fraud = df[df[target] == 1]
nonfraud = df[df[target] == 0]

print("Fraud rows:", len(fraud))
print("Non-fraud rows:", len(nonfraud))

# ---------- TARGET SAMPLE SIZE ----------
fraud_target = 3000
nonfraud_target = 7000

# ---------- FRAUD SAMPLING (random since already rare) ----------
fraud_sample = fraud.sample(n=fraud_target, random_state=42)

# ---------- NON FRAUD REPRESENTATIVE SAMPLING ----------
features = nonfraud.drop(columns=[target])

scaler = StandardScaler()
X = scaler.fit_transform(features)

kmeans = MiniBatchKMeans(
    n_clusters=nonfraud_target,
    batch_size=10000,
    random_state=42
)

kmeans.fit(X)

centers = kmeans.cluster_centers_

# Find closest real transaction to each cluster center
indices = []

for center in centers:
    distances = np.linalg.norm(X - center, axis=1)
    idx = np.argmin(distances)
    indices.append(idx)

nonfraud_sample = nonfraud.iloc[indices]

# ---------- COMBINE ----------
optimal_df = pd.concat([fraud_sample, nonfraud_sample])

# shuffle
optimal_df = optimal_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nFinal dataset size:", optimal_df.shape)

print("\nClass distribution:")
print(optimal_df[target].value_counts())

# ---------- SAVE ----------
output_path = "data/optimal_10k_dataset.csv"
optimal_df.to_csv(output_path, index=False)

print("\nSaved:", output_path)