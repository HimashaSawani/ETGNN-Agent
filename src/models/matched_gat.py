import torch
import torch.nn.functional as F

from src.models.etgnn import ETGNNClassifier


class MatchedGATClassifier(ETGNNClassifier):
    def forward(self, x, edge_index, time_step):
        zero_time = torch.zeros(
            (x.shape[0], self.time_dim), dtype=x.dtype, device=x.device
        )
        combined = torch.cat((x, zero_time), dim=1)
        h = self.conv1(combined, edge_index).relu()
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(h, edge_index)
