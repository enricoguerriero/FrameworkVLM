import torch
import torch.nn as nn

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, hidden_dim)
        attn_weights = self.attn(x).squeeze(-1)  # (batch, seq_len)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(~mask, float('-inf'))
        attn_scores = torch.softmax(attn_weights, dim=1)  # (batch, seq_len)
        pooled = (x * attn_scores.unsqueeze(-1)).sum(1)   # (batch, hidden_dim)
        return pooled.float()