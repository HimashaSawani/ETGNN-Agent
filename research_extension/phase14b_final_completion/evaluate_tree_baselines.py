import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

import json
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

from src.models.time_encoding import encode_time

# Load test data and masks
nodes = pd.read_csv("data/processed/nodes_with_masks.csv")
features = np.load("data/processed/scaled_features.npy", mmap_mode="r")
time = encode_time(torch.tensor(nodes["time_step"].to_numpy())).numpy()

val_mask = nodes["val_mask"].to_numpy(dtype=bool)
test_mask = nodes["test_mask"].to_numpy(dtype=bool)

y_val = nodes.loc[val_mask, "label"].to_numpy()
y_test = nodes.loc[test_mask, "label"].to_numpy()

print(f"Validation nodes: {val_mask.sum():,} | Test nodes: {test_mask.sum():,}")
print(f"Test label distribution -> 0: {(y_test==0).sum():,}, 1: {(y_test==1).sum():,}")

# Degree stratification
test_graph = torch.load("data/processed/test_graph.pt", map_location="cpu", weights_only=True)
test_indices = torch.where(test_graph['test_mask'])[0]
dst_nodes = test_graph['edge_index'][1]
deg_map = torch.bincount(dst_nodes, minlength=test_graph['x'].size(0))
test_degrees = deg_map[test_indices].numpy()

s0_idx = np.where(test_degrees == 0)[0]
s1_idx = np.where((test_degrees >= 1) & (test_degrees <= 2))[0]
s2_idx = np.where(test_degrees >= 3)[0]

results = []

def evaluate_model(name, val_probs, test_probs):
    val_ap = average_precision_score(y_val, val_probs)
    # Tune threshold tau* on validation
    best_tau = 0.5
    best_f1 = -1.0
    for tau in np.linspace(0.01, 0.99, 99):
        f1 = f1_score(y_val, (val_probs >= tau).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = tau

    test_ap = average_precision_score(y_test, test_probs)
    test_roc = roc_auc_score(y_test, test_probs)
    test_pred = (test_probs >= best_tau).astype(int)
    test_f1 = f1_score(y_test, test_pred, zero_division=0)
    test_prec = precision_score(y_test, test_pred, zero_division=0)
    test_rec = recall_score(y_test, test_pred, zero_division=0)

    ap_s0 = average_precision_score(y_test[s0_idx], test_probs[s0_idx])
    ap_s1 = average_precision_score(y_test[s1_idx], test_probs[s1_idx])
    ap_s2 = average_precision_score(y_test[s2_idx], test_probs[s2_idx])

    res = {
        "model": name,
        "tau_star": best_tau,
        "val_ap": val_ap,
        "val_f1": best_f1,
        "test_ap": test_ap,
        "test_roc": test_roc,
        "test_f1": test_f1,
        "test_prec": test_prec,
        "test_rec": test_rec,
        "ap_s0": ap_s0,
        "ap_s1": ap_s1,
        "ap_s2": ap_s2
    }
    results.append(res)
    print(f"\n--- {name} ---")
    print(f"Val AP: {val_ap:.4f} | Optimal Tau*: {best_tau:.2f} (Val F1: {best_f1:.4f})")
    print(f"Test AP: {test_ap:.4f} | ROC-AUC: {test_roc:.4f} | F1: {test_f1:.4f} | Prec: {test_prec:.4f} | Rec: {test_rec:.4f}")
    print(f"Degree Strata AP -> S0 (k=0): {ap_s0:.4f} | S1 (k=1..2): {ap_s1:.4f} | S2 (k>=3): {ap_s2:.4f}")

# 1. Random Forest (165 features)
rf_path = Path("results/checkpoints/random_forest.joblib")
if rf_path.exists():
    rf = joblib.load(rf_path)
    val_probs = rf.predict_proba(features[val_mask])[:, 1]
    test_probs = rf.predict_proba(features[test_mask])[:, 1]
    evaluate_model("RandomForest_165feat", val_probs, test_probs)

# 2. Random Forest with Time (181 features)
rf_time_path = Path("results/checkpoints/rf_with_time.joblib")
if rf_time_path.exists():
    rf_t = joblib.load(rf_time_path)
    x_val = np.concatenate((features[val_mask], time[val_mask]), axis=1)
    x_test = np.concatenate((features[test_mask], time[test_mask]), axis=1)
    val_probs = rf_t.predict_proba(x_val)[:, 1]
    test_probs = rf_t.predict_proba(x_test)[:, 1]
    evaluate_model("RandomForest_181feat_with_time", val_probs, test_probs)

# 3. Random Forest Local (93 features)
rf_loc_path = Path("results/checkpoints/rf_local.joblib")
if rf_loc_path.exists():
    rf_l = joblib.load(rf_loc_path)
    val_probs = rf_l.predict_proba(features[val_mask, :93])[:, 1]
    test_probs = rf_l.predict_proba(features[test_mask, :93])[:, 1]
    evaluate_model("RandomForest_93local", val_probs, test_probs)

# 4. XGBoost (165 features)
xgb_path = Path("results/checkpoints/xgboost.json")
if xgb_path.exists():
    xgb = XGBClassifier()
    xgb.load_model(str(xgb_path))
    val_probs = xgb.predict_proba(features[val_mask])[:, 1]
    test_probs = xgb.predict_proba(features[test_mask])[:, 1]
    evaluate_model("XGBoost_165feat", val_probs, test_probs)

df_out = pd.DataFrame(results)
df_out.to_csv("research_extension/phase14b_final_completion/TREE_BASELINES_TEST_EVALUATION.csv", index=False)
print("\n[SUCCESS] Evaluated all trained tree baselines on chronological test split.")

