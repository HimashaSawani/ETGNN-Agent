import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from src.evaluation.metrics import evaluate_binary
from src.models.etgnn import ETGNNClassifier

torch.manual_seed(42)
torch.set_num_threads(2)

train = torch.load("data/processed/train_graph.pt",
                   map_location="cpu", weights_only=True)
val = torch.load("data/processed/validation_graph.pt",
                 map_location="cpu", weights_only=True)

model = ETGNNClassifier(input_dim=93)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

train_labels = train["y"][train["train_mask"]]
weight = torch.tensor([1.0, float((train_labels == 0).sum() /
                                  (train_labels == 1).sum())])

best_auprc = -1.0
bad_epochs = 0
Path("results/checkpoints").mkdir(parents=True, exist_ok=True)

for epoch in range(1, 51):
    model.train()
    optimizer.zero_grad()
    logits = model(train["x"][:, :93], train["edge_index"], train["time_step"])
    loss = F.cross_entropy(logits[train["train_mask"]],
                           train_labels, weight=weight)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(val["x"][:, :93], val["edge_index"], val["time_step"])
        probability = torch.softmax(val_logits[val["val_mask"]], dim=1)[:, 1].numpy()
        y_val = val["y"][val["val_mask"]].numpy()
        auprc = average_precision_score(y_val, probability)

    if auprc > best_auprc + 0.0001:
        best_auprc = auprc
        best_epoch = epoch
        bad_epochs = 0
        torch.save(model.state_dict(), "results/checkpoints/etgnn_local.pt")
    else:
        bad_epochs += 1

    if epoch == 1 or epoch % 5 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.4f}, val_AUPRC={auprc:.4f}", flush=True)
    if bad_epochs >= 8:
        break

model.load_state_dict(torch.load("results/checkpoints/etgnn_local.pt",
                                 map_location="cpu", weights_only=True))
model.eval()
with torch.no_grad():
    logits = model(val["x"][:, :93], val["edge_index"], val["time_step"])
    probability = torch.softmax(logits[val["val_mask"]], dim=1)[:, 1].numpy()

scores = evaluate_binary(val["y"][val["val_mask"]].numpy(), probability)
Path("results/tables/etgnn_local_validation.json").write_text(
    json.dumps({"best_epoch": best_epoch, **scores}, indent=2),
    encoding="utf-8",
)
print("Local ETGNN best epoch:", best_epoch)
print(json.dumps(scores, indent=2))
