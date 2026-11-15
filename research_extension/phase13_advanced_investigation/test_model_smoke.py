"""
Unit and smoke tests for Phase 13 model architectures:
- ETGNNClassifier
- ResidualETGNNClassifier
- ReliabilityAwareETGNNClassifier
- DeepParameterMatchedMLP
- FocalLoss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.models.etgnn import ETGNNClassifier
from src.models.reliability_etgnn import (
    ResidualETGNNClassifier,
    ReliabilityAwareETGNNClassifier,
    DeepParameterMatchedMLP
)

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with class-imbalance alpha and focusing gamma.
    """
    def __init__(self, alpha=0.75, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        # alpha weight for class 1, (1-alpha) for class 0
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        focal_loss = alpha_t * (1.0 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss

def run_smoke_tests():
    print("=== STARTING PHASE 13 MODEL SMOKE TESTS ===")
    device = torch.device("cpu")
    num_nodes = 50
    input_dim = 165
    time_dim = 16
    
    # Synthetic batch: 50 nodes, 60 edges, random timesteps
    x = torch.randn(num_nodes, input_dim)
    src = torch.randint(0, num_nodes, (60,))
    dst = torch.randint(0, num_nodes, (60,))
    edge_index = torch.stack([src, dst], dim=0)
    time_step = torch.randint(1, 50, (num_nodes,))
    targets = torch.randint(0, 2, (num_nodes,))

    # 1. Test ETGNNClassifier (Frozen Baseline M0)
    print("\n--- Test 1: ETGNNClassifier (M0) ---")
    m0 = ETGNNClassifier(input_dim=input_dim, time_dim=time_dim)
    out0 = m0(x, edge_index, time_step)
    assert out0.shape == (num_nodes, 2), f"M0 output shape mismatch: {out0.shape}"
    loss0 = F.cross_entropy(out0, targets)
    loss0.backward()
    param_count0 = sum(p.numel() for p in m0.parameters() if p.requires_grad)
    print(f"  [PASS] Output shape: {out0.shape}, Loss: {loss0.item():.4f}, Trainable params: {param_count0}")

    # 2. Test DeepParameterMatchedMLP (M2)
    print("\n--- Test 2: DeepParameterMatchedMLP (M2) ---")
    m2 = DeepParameterMatchedMLP(input_dim=input_dim, time_dim=time_dim)
    out2 = m2(x, edge_index, time_step)
    assert out2.shape == (num_nodes, 2), f"M2 output shape mismatch: {out2.shape}"
    loss2 = F.cross_entropy(out2, targets)
    loss2.backward()
    param_count2 = sum(p.numel() for p in m2.parameters() if p.requires_grad)
    print(f"  [PASS] Output shape: {out2.shape}, Loss: {loss2.item():.4f}, Trainable params: {param_count2}")

    # 3. Test ResidualETGNNClassifier (M1)
    print("\n--- Test 3: ResidualETGNNClassifier (M1) ---")
    m1 = ResidualETGNNClassifier(input_dim=input_dim, time_dim=time_dim)
    out1 = m1(x, edge_index, time_step)
    assert out1.shape == (num_nodes, 2), f"M1 output shape mismatch: {out1.shape}"
    loss1 = F.cross_entropy(out1, targets)
    loss1.backward()
    param_count1 = sum(p.numel() for p in m1.parameters() if p.requires_grad)
    print(f"  [PASS] Output shape: {out1.shape}, Loss: {loss1.item():.4f}, Trainable params: {param_count1}")

    # 4. Test ReliabilityAwareETGNNClassifier (M3)
    print("\n--- Test 4: ReliabilityAwareETGNNClassifier (M3) ---")
    m3 = ReliabilityAwareETGNNClassifier(input_dim=input_dim, time_dim=time_dim)
    out3, gate_dict = m3(x, edge_index, time_step, return_gates=True)
    assert out3.shape == (num_nodes, 2), f"M3 output shape mismatch: {out3.shape}"
    assert "gates1" in gate_dict and "gates2" in gate_dict, "Gate dictionary incomplete"
    print(f"  Gates1 shape: {gate_dict['gates1'].shape}, min: {gate_dict['gates1'].min():.4f}, max: {gate_dict['gates1'].max():.4f}")
    
    # Check gradient flow to gate_mlp
    loss3 = F.cross_entropy(out3, targets)
    loss3.backward()
    gate_grad_norm = sum(p.grad.norm().item() for p in m3.conv1.gate_mlp.parameters() if p.grad is not None)
    assert gate_grad_norm > 0, "Gate MLP received ZERO gradients!"
    param_count3 = sum(p.numel() for p in m3.parameters() if p.requires_grad)
    print(f"  [PASS] Output shape: {out3.shape}, Gate grad norm: {gate_grad_norm:.6f}, Trainable params: {param_count3}")

    # 5. Test Disconnected Nodes (E=0) for M3
    print("\n--- Test 5: Disconnected Node Behavior (Empty Edges E=0) ---")
    empty_edges = torch.empty((2, 0), dtype=torch.long)
    out_empty, gates_empty = m3(x, empty_edges, time_step, return_gates=True)
    assert out_empty.shape == (num_nodes, 2), f"Empty edges output shape mismatch: {out_empty.shape}"
    assert not torch.isnan(out_empty).any(), "NaN detected in empty edge forward pass!"
    print(f"  [PASS] Successfully handled E=0 empty graph without NaN. Output shape: {out_empty.shape}")

    # 6. Test Focal Loss
    print("\n--- Test 6: Focal Loss vs Cross Entropy ---")
    focal_fn = FocalLoss(alpha=0.75, gamma=2.0)
    fl_val = focal_fn(out3, targets)
    ce_val = F.cross_entropy(out3, targets)
    assert not torch.isnan(fl_val), "Focal Loss output is NaN"
    print(f"  [PASS] Cross-Entropy: {ce_val.item():.4f}, Focal Loss: {fl_val.item():.4f}")

    print("\n=== ALL 6 QUALITY & SAFETY SMOKE TESTS PASSED CLEANLY ===")
    return True

if __name__ == "__main__":
    success = run_smoke_tests()
    if success:
        sys.exit(0)
    else:
        sys.exit(1)

