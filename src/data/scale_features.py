from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

raw = Path("data/raw/elliptic/elliptic_txs_features.csv")
out = Path("data/processed")
mapping = pd.read_csv(out / "node_mapping.csv")
scaler = StandardScaler()
chunk_size = 10000

start = 0
for chunk in pd.read_csv(raw, header=None, chunksize=chunk_size):
    end = start + len(chunk)
    assert np.array_equal(chunk.iloc[:, 0].to_numpy(),
                          mapping["txId"].iloc[start:end].to_numpy())
    train_rows = chunk.iloc[:, 1].to_numpy() <= 32
    if train_rows.any():
        scaler.partial_fit(chunk.iloc[train_rows, 2:].to_numpy(dtype=np.float64))
    start = end

assert start == len(mapping)
joblib.dump(scaler, out / "train_scaler.joblib")

scaled = np.lib.format.open_memmap(
    out / "scaled_features.npy",
    mode="w+",
    dtype=np.float32,
    shape=(len(mapping), 165),
)

start = 0
for chunk in pd.read_csv(raw, header=None, chunksize=chunk_size):
    end = start + len(chunk)
    scaled[start:end] = scaler.transform(
        chunk.iloc[:, 2:].to_numpy(dtype=np.float64)
    ).astype(np.float32)
    start = end

scaled.flush()
print("Scaler fit on nodes:", scaler.n_samples_seen_)
print("Scaled features shape:", scaled.shape)
print("Saved training scaler and scaled features")
