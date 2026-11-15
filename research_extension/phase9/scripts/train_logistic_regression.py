import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, f1_score, precision_score, recall_score

def evaluate_binary(y_true, y_prob, threshold=0.5):
    p, r, _ = precision_recall_curve(y_true, y_prob)
    auprc = float(auc(r, p))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    y_pred = (y_prob >= threshold).astype(int)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {
        "auprc": auprc,
        "roc_auc": roc_auc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "threshold": threshold
    }

def run_logistic_regression():
    processed = Path("data/processed")
    nodes = pd.read_csv(processed / "nodes_with_masks.csv")
    features = np.load(processed / "scaled_features.npy", mmap_mode="r")

    train_mask = nodes["train_mask"].to_numpy(dtype=bool)
    val_mask = nodes["val_mask"].to_numpy(dtype=bool)
    test_mask = nodes["test_mask"].to_numpy(dtype=bool)

    y_train = nodes.loc[train_mask, "label"].to_numpy()
    y_val = nodes.loc[val_mask, "label"].to_numpy()
    y_test = nodes.loc[test_mask, "label"].to_numpy()

    configs = [
        {"name": "Full (165f)", "dim": 165, "save_name": "logistic_regression_full"},
        {"name": "Local (93f)", "dim": 93, "save_name": "logistic_regression_local"}
    ]

    C_values = [0.001, 0.01, 0.1, 1.0, 10.0]
    class_weights = [None, "balanced"]
    results_summary = []

    out_ckpt_dir = Path("research_extension/phase9/checkpoints")
    out_ckpt_dir.mkdir(parents=True, exist_ok=True)

    test_probabilities = {}

    for cfg in configs:
        dim = cfg["dim"]
        name = cfg["name"]
        save_name = cfg["save_name"]
        print(f"\n==========================================")
        print(f"Training Logistic Regression: {name}")
        print(f"==========================================")

        X_tr = features[train_mask, :dim]
        X_va = features[val_mask, :dim]
        X_te = features[test_mask, :dim]

        best_val_auprc = -1.0
        best_model = None
        best_params = {}

        for cw in class_weights:
            for c in C_values:
                clf = LogisticRegression(
                    C=c,
                    class_weight=cw,
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=42,
                    n_jobs=-1
                )
                clf.fit(X_tr, y_train)
                val_prob = clf.predict_proba(X_va)[:, 1]
                val_metrics = evaluate_binary(y_val, val_prob)

                print(f"  Params: C={c}, class_weight={cw} -> Val AUPRC: {val_metrics['auprc']:.4f}, Val F1: {val_metrics['f1']:.4f}")

                if val_metrics["auprc"] > best_val_auprc:
                    best_val_auprc = val_metrics["auprc"]
                    best_model = clf
                    best_params = {"C": c, "class_weight": cw, "val_metrics": val_metrics}

        print(f"\n[Selected Best Configuration for {name}]: C={best_params['C']}, class_weight={best_params['class_weight']} (Val AUPRC: {best_val_auprc:.4f})")

        # Save model
        model_path = out_ckpt_dir / f"{save_name}.joblib"
        joblib.dump(best_model, model_path)
        print(f"Saved model checkpoint: {model_path}")

        # Evaluate on Test
        test_prob = best_model.predict_proba(X_te)[:, 1]
        test_metrics = evaluate_binary(y_test, test_prob)
        test_probabilities[save_name] = test_prob

        print(f"[Test Evaluation for {name}]:")
        for k, v in test_metrics.items():
            if isinstance(v, float):
                print(f"  {k:<12}: {v:.4f}")
            else:
                print(f"  {k:<12}: {v}")

        results_summary.append({
            "model_condition": name,
            "feature_dim": dim,
            "best_C": best_params["C"],
            "best_class_weight": str(best_params["class_weight"]),
            "val_auprc": best_params["val_metrics"]["auprc"],
            "val_f1": best_params["val_metrics"]["f1"],
            "val_roc_auc": best_params["val_metrics"]["roc_auc"],
            "test_auprc": test_metrics["auprc"],
            "test_f1": test_metrics["f1"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_roc_auc": test_metrics["roc_auc"],
            "test_tp": test_metrics["tp"],
            "test_fp": test_metrics["fp"],
            "test_tn": test_metrics["tn"],
            "test_fn": test_metrics["fn"],
        })

    df_results = pd.DataFrame(results_summary)
    summary_path = Path("research_extension/phase9/LOGISTIC_REGRESSION_RESULTS.csv")
    df_results.to_csv(summary_path, index=False)
    print(f"\nSaved Logistic Regression summary: {summary_path}")

    # Save test probabilities
    np.savez_compressed(
        "research_extension/phase9/logistic_regression_test_probabilities.npz",
        y_test=y_test,
        **test_probabilities
    )
    print("Saved test probabilities to research_extension/phase9/logistic_regression_test_probabilities.npz")

if __name__ == "__main__":
    run_logistic_regression()

