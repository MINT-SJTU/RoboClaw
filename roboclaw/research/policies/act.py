from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ModuleNotFoundError:
    torch = None
    F = None
    nn = None


def _numeric_items(values: dict[str, Any], *, nested: tuple[str, ...] = ()) -> list[tuple[str, float]]:
    source = values
    for key in nested:
        if isinstance(source.get(key), dict):
            source = source[key]
            break
    return sorted(
        (str(key), float(value))
        for key, value in source.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )


def _vector(values: dict[str, Any], dim: int, *, nested: tuple[str, ...] = ()) -> tuple[list[str], list[float]]:
    items = _numeric_items(values, nested=nested)
    keys = [key for key, _ in items][:dim]
    data = [value for _, value in items][:dim]
    keys.extend(f"joint_{i}" for i in range(len(keys), dim))
    data.extend(0.0 for _ in range(len(data), dim))
    return keys, data


@dataclass(frozen=True)
class ACTConfig:
    state_dim: int = 6
    action_dim: int = 6
    chunk_size: int = 10
    hidden_dim: int = 128
    n_heads: int = 4
    n_encoder_layers: int = 2
    n_decoder_layers: int = 1
    dropout: float = 0.1
    use_vae: bool = False
    latent_dim: int = 16
    lr: float = 1e-4


class ACTPolicy(nn.Module if nn is not None else object):
    def __init__(self, config: ACTConfig):
        if nn is None:
            raise ModuleNotFoundError("torch is required for roboclaw.research.policies.act")
        super().__init__()
        self.config = config
        self.state_encoder = nn.Linear(config.state_dim, config.hidden_dim)
        self.action_encoder = nn.Linear(config.action_dim, config.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_encoder_layers)
        layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=config.n_decoder_layers)
        self.action_head = nn.Linear(config.hidden_dim, config.action_dim)
        self.query_embed = nn.Embedding(config.chunk_size, config.hidden_dim)

    def forward(self, state, actions=None):
        memory = self.encoder(self.state_encoder(state).unsqueeze(1))
        query = self.query_embed.weight.unsqueeze(0).expand(state.shape[0], -1, -1)
        if actions is not None:
            query = query + self.action_encoder(actions)
        predicted = self.action_head(self.decoder(query, memory))
        losses = {}
        if actions is not None:
            losses["l1_loss"] = F.l1_loss(predicted, actions)
        return predicted, losses

    def predict(self, state: dict[str, Any]) -> dict[str, float]:
        if torch is None:
            raise ModuleNotFoundError("torch is required for roboclaw.research.policies.act")
        keys, values = _vector(state, self.config.state_dim, nested=("joint_positions",))
        device = next(self.parameters()).device
        tensor = torch.tensor([values], dtype=torch.float32, device=device)
        was_training = self.training
        self.eval()
        with torch.no_grad():
            predicted, _ = self.forward(tensor)
        if was_training:
            self.train()
        action = predicted[0, 0].detach().cpu().tolist()
        return {keys[i]: float(action[i]) for i in range(self.config.action_dim)}
