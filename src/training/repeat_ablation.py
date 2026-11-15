import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from src.models.etgnn import ETGNNClassifier
from src.models.matched_gat import MatchedGATClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["etgnn", "matched_gat"], required=True)
parser.add_argument("--seed", type=int, required=True)
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.set_num_threads(2)
train = torch.load("data/processed/train_graph.pt",
                   map_location="cpu", weights_only=True)
val = torch.load("data/processed/validation_graph.pt",
                 map_location="cpu", weights_only=True)

model_class = ETGNNClassifier if args.model == "etgnn" else MatchedGATClassifier
model = model_class()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
labels = train["y"][train["train_mask"]]
weight = torch.tensor([1.0, float((labels == 0).sum() / (labels == 1).sum())])
y_val = val["y"][val["val_mask"]].numpy()

best_auprc = -1.0
best_epoch = 0
bad_epochs = 0

for epoch in range(1, 51):
    model.train()
    optimizer.zero_grad()
    logits = model(train["x"], train["edge_index"], train["time_step"])
    loss = F.cross_entropy(logits[train["train_mask"]], labels, weight=weight)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(val["x"], val["edge_index"], val["time_step"])
        probability = torch.softmax(
            logits[val["val_mask"]], dim=1
        )[:, 1].numpy()
    auprc = average_precision_score(y_val, probability)

    if auprc > best_auprc + 0.0001:
        best_auprc = auprc
        best_epoch = epoch
        bad_epochs = 0
    else:
        bad_epochs += 1
    if bad_epochs >= 8:
        break

output = Path("results/tables/ablation")
output.mkdir(parents=True, exist_ok=True)
result = {
    "model": args.model,
    "seed": args.seed,
    "best_epoch": best_epoch,
    "validation_auprc": best_auprc,
}
(output / f"{args.model}_seed_{args.seed}.json").write_text(
    json.dumps(result, indent=2), encoding="utf-8"
)
print(result, flush=True)
