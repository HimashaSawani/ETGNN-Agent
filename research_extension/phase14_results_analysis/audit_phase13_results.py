import pandas as pd
import numpy as np
import torch
from pathlib import Path

VAL_CSV = Path("research_extension/phase13_advanced_investigation/PHASE13_VAL_SELECTION.csv")
TEST_CSV = Path("research_extension/phase13_advanced_investigation/PHASE13_TEST_METRICS.csv")
CKPT_DIR = Path("research_extension/phase13_advanced_investigation/checkpoints")

print("=== 1. VALIDATION SELECTION RECORDS ===")
df_val = pd.read_csv(VAL_CSV)
print(f"Total validation rows: {len(df_val)}")
print(df_val)

print("\n=== 2. TEST EVALUATION RECORDS ===")
df_test = pd.read_csv(TEST_CSV)
print(f"Total test rows: {len(df_test)}")
print(df_test[["config", "seed", "tau_star", "test_ap", "test_roc", "test_f1", "test_prec", "test_rec"]])

print("\n=== 3. GROUPBY SUMMARY ACROSS SEEDS ===")
grp = df_test.groupby("config").agg(
    count=("test_ap", "count"),
    test_ap_mean=("test_ap", "mean"),
    test_ap_std=("test_ap", "std"),
    test_roc_mean=("test_roc", "mean"),
    test_roc_std=("test_roc", "std"),
    test_f1_mean=("test_f1", "mean"),
    test_f1_std=("test_f1", "std"),
    ap_s0_mean=("ap_s0_k0", "mean"),
    ap_s1_mean=("ap_s1_k12", "mean"),
    ap_s2_mean=("ap_s2_k3plus", "mean"),
)
print(grp.to_string())

print("\n=== 4. CHECKPOINT AUDIT ===")
ckpts = list(CKPT_DIR.glob("*.pt"))
print(f"Total checkpoints found: {len(ckpts)}")
for c in sorted(ckpts):
    d = torch.load(c, map_location="cpu", weights_only=False)
    print(f"{c.name:32s} | Size: {c.stat().st_size:,} bytes | Keys: {list(d.keys())} | Epoch: {d.get('epoch')} | Val AP: {d.get('val_ap'):.4f} | Loss: {d.get('loss_type')}")

