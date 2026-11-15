# Google Colab Pro Execution Guide
## Workflow, Hardware Fallback, and Artifact Provenance for GPU-Intensive Experiments

**Project:** ETGNN-Agent  
**Document:** `docs/phase5/COLAB_PRO_EXECUTION_GUIDE.md`  
**Notebook Directory:** [`notebooks/phase5/`](file:///D:/ETGNN-Fraud-Detection/notebooks/phase5/)  

---

### 1. Colab Notebook Suite

Three dedicated research notebooks are provided for optional GPU acceleration:
1. [`phase5_drift_analysis.ipynb`](file:///D:/ETGNN-Fraud-Detection/notebooks/phase5/phase5_drift_analysis.ipynb): High-throughput feature distribution and topology drift analysis across all 49 snapshots.
2. [`phase5_candidate_training.ipynb`](file:///D:/ETGNN-Fraud-Detection/notebooks/phase5/phase5_candidate_training.ipynb): Controlled candidate model training with multi-seed logging and artifact isolation.
3. [`phase5_candidate_evaluation.ipynb`](file:///D:/ETGNN-Fraud-Detection/notebooks/phase5/phase5_candidate_evaluation.ipynb): Candidate vs baseline multi-metric comparison and promotion gate execution.

---

### 2. Execution Protocol & Prerequisites

#### A. Hardware Detection and Fallback
Each notebook dynamically detects hardware acceleration:
```python
import torch
if torch.cuda.is_available():
    print('Using GPU:', torch.cuda.get_device_name(0))
else:
    print('Running on CPU fallback.')
```
The codebase automatically falls back to CPU execution if no GPU is assigned.

#### B. Drive Mounting & Data Path Configuration
```python
from pathlib import Path
# Uncomment if executing inside Colab Pro with Google Drive
# from google.colab import drive
# drive.mount('/content/drive')
# DATA_DIR = Path('/content/drive/MyDrive/ETGNN-Fraud-Detection/data/processed')
```

#### C. SHA-256 Checksum Verification
Before running any analysis, notebooks verify dataset file hashes against [baseline_manifest.json](file:///D:/ETGNN-Fraud-Detection/artifacts/baseline_manifest.json):
```python
import hashlib
assert hashlib.sha256(open('data/processed/nodes_with_masks.csv', 'rb').read()).hexdigest() == 'ee6832a5aa57f2032cf32f9f66350a83001ed4d1b2792000624edae9c6d1643d'
```

---

### 3. Safety Constraints

- **No Secrets in Notebooks:** API keys and credentials are never hardcoded in notebook cells.
- **No Overwriting Production Checkpoints:** Candidate models write strictly to `artifacts/phase5_candidates/`.
- **No Unapproved Training Runs:** GPU training must only be executed when drift justification is formally established.

