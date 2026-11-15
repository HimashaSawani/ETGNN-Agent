import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from src.models.etgnn import ETGNNClassifier

m0 = ETGNNClassifier()
ckpt = torch.load("results/checkpoints/etgnn.pt", map_location="cpu")
m0.load_state_dict(ckpt)
m0.eval()

test_data = torch.load("data/processed/test_graph.pt", map_location="cpu", weights_only=True)
test_mask = test_data['test_mask']
test_true = test_data['y'][test_mask].numpy()

with torch.no_grad():
    logits = m0(test_data['x'], test_data['edge_index'], test_data['time_step'])
    probs = F.softmax(logits[test_mask], dim=1)[:, 1].numpy()

ap = average_precision_score(test_true, probs)
roc = roc_auc_score(test_true, probs)
print(f"M0 Test AUPRC: {ap:.6f}")
print(f"M0 Test ROC-AUC: {roc:.6f}")
