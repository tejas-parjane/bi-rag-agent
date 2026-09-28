"""Tiny, reproducible PyTorch metric-learning experiment for the portfolio project.

It learns a projection that makes matching query/document feature pairs closer than
non-matching pairs. Replace the synthetic pairs with labelled internal retrieval data
when available. This is deliberately separate from the production embedding model.
"""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)
FEATURE_DIM, EMBEDDING_DIM = 32, 16


class RetrieverProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(FEATURE_DIM, 24), nn.ReLU(), nn.Linear(24, EMBEDDING_DIM))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(x), dim=-1)


def make_pairs(samples: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.randn(samples, FEATURE_DIM)
    return base + 0.08 * torch.randn_like(base), base + 0.08 * torch.randn_like(base)


def train(epochs: int = 120) -> RetrieverProjection:
    queries, documents = make_pairs()
    model = RetrieverProjection()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    labels = torch.arange(len(queries))
    for epoch in range(epochs):
        logits = model(queries) @ model(documents).T / 0.07
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if epoch % 30 == 0:
            print(f"epoch={epoch:03d} contrastive_loss={loss.item():.4f}")
    return model


if __name__ == "__main__":
    model = train()
    output = Path("artifacts/retriever_projection.pt")
    output.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), output)
    print(f"saved={output}")
