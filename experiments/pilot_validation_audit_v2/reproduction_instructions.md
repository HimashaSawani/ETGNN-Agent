# Reproduction Instructions: ETGNN Validation Feasibility Pilot Evidence Audit

This document provides exact, deterministic commands to reproduce the pilot verification, candidate edge reconciliation, and statistical recomputation from scratch.

---

## 1. Environment & Prerequisites
- Python 3.10+ (or Python 3.13 virtual environment)
- PyTorch (CPU or CUDA)
- PyTorch Geometric (`torch_geometric`)
- Required libraries: `pandas`, `numpy`, `matplotlib`, `scikit-learn`

---

## 2. Command-Line Reproduction

### Step 1: Run the Standalone Evidence Audit
To verify all 700 perturbation trials, selected edge bounds, candidate edge reconciliations, FP seed breakdowns, and Jaccard stability:

```powershell
# From workspace root: d:\ETGNN-Fraud-Detection
.\.venv\Scripts\python.exe src/evaluation/audit_pilot_evidence.py
```
This command runs in approximately 1 second and outputs `experiments/pilot_validation_audit_v2/verification_output.txt`.

### Step 2: Run Full Numerical Model Verification
To numerically validate baseline test metrics, explanation deletion controls, and multi-seed ablation averages:

```powershell
.\.venv\Scripts\python.exe src/evaluation/verify_results_and_figures.py
```

### Step 3: Optional Re-run of the Live Validation Pilot (Hosted Colab or CPU)
To re-run the 20-node perturbation pilot from scratch without exhausting local 4 GB RAM:
1. Open Google Colab with `ETGNN_Validation_Audit_Colab.ipynb`.
2. Ensure `ETGNN_Colab_Bundle.zip` is available.
3. In Cell 8, set `FORCE_RERUN = True`.
4. Run all cells sequentially. Results are atomically recorded and support seamless resumption if interrupted.
