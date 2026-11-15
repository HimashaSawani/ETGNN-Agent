import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from src.models.etgnn import ETGNNClassifier

DATA_DIR = torch.load("data/processed/test_graph.pt", map_location="cpu", weights_only=True)
val_data = torch.load("data/processed/validation_graph.pt", map_location="cpu", weights_only=True)

m0 = ETGNNClassifier()
ckpt = torch.load("results/checkpoints/etgnn.pt", map_location="cpu", weights_only=False)
m0.load_state_dict(ckpt.get("state_dict", ckpt))
m0.eval()

# Let's check evaluation on test graph directly with raw logits vs softmax
test_mask = DATA_DIR['test_mask']
test_true = DATA_DIR['y'][test_mask].numpy()

with torch.no_grad():
    test_logits = m0(DATA_DIR['x'], DATA_DIR['edge_index'], DATA_DIR['time_step'])
    probs_col1 = F.softmax(test_logits[test_mask], dim=1)[:, 1].numpy()
    logits_col1 = test_logits[test_mask][:, 1].numpy()
    diff_logits = (test_logits[test_mask][:, 1] - test_logits[test_mask][:, 0]).numpy()

print("AP from softmax col 1:", average_precision_score(test_true, probs_col1))
print("AP from logit col 1:  ", average_precision_score(test_true, logits_col1))
print("AP from logit diff:   ", average_precision_score(test_true, diff_logits))

# Now let's check: What was loaded in Colab?
# Did Colab have results/checkpoints/etgnn.pt or was it something else?
# Let's check what was inside ETGNN_Phase13_Input_Bundle.zip:
import zipfile
with zipfile.ZipFile("ETGNN_Phase13_Input_Bundle.zip") as z:
    for name in z.namelist():
        if "etgnn.pt" in name:
            print("Zip entry:", name, z.getinfo(name).file_size)
