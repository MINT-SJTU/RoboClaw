from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _rows(root: Path) -> list[dict]:
    jsonl_path = root / "data" / "episodes.jsonl"
    if jsonl_path.exists():
        return [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    parquet_path = root / "data" / "episodes.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"No dataset found under {root}")
    import pyarrow.parquet as pq

    return pq.read_table(parquet_path).to_pylist()


def _numeric(values: dict, *keys: str) -> list[float]:
    source = values
    for key in keys:
        if isinstance(source.get(key), dict):
            source = source[key]
            break
    items = sorted(
        (str(key), float(value))
        for key, value in source.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    return [value for _, value in items]


def _fit(values: list[float], dim: int) -> list[float]:
    return (values[:dim] + [0.0] * dim)[:dim]


def _tensors(root: Path, config):
    import torch

    episodes = defaultdict(list)
    for row in _rows(root):
        episodes[int(row.get("episode_index", 0))].append(row)
    states, actions = [], []
    for rows in episodes.values():
        rows.sort(key=lambda row: int(row.get("frame_index", 0)))
        frames = []
        for row in rows:
            state = json.loads(row["state"]) if isinstance(row.get("state"), str) else dict(row.get("state") or {})
            action = json.loads(row["action"]) if isinstance(row.get("action"), str) else dict(row.get("action") or {})
            state_vec = _fit(_numeric(state, "joint_positions"), config.state_dim)
            raw_action = _numeric(action, "joint_positions", "targets", "positions", "values")
            action_vec = _fit(raw_action or state_vec, config.action_dim)
            frames.append((state_vec, action_vec))
        for index, (state_vec, _) in enumerate(frames):
            chunk = [frames[min(index + step, len(frames) - 1)][1] for step in range(config.chunk_size)]
            states.append(state_vec)
            actions.append(chunk)
    if not states:
        raise ValueError("Dataset does not contain numeric state/action frames.")
    return torch.tensor(states, dtype=torch.float32), torch.tensor(actions, dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--algorithm", default="act")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    if args.algorithm != "act":
        raise SystemExit(f"Unsupported algorithm: {args.algorithm}")

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from roboclaw.research.policies.act import ACTConfig, ACTPolicy

    config = ACTConfig(lr=args.lr)
    state, actions = _tensors(Path(args.dataset), config)
    loader = DataLoader(TensorDataset(state, actions), batch_size=args.batch_size, shuffle=True)
    model = ACTPolicy(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        last_loss = 0.0
        for batch_state, batch_actions in loader:
            optimizer.zero_grad()
            _, losses = model(batch_state, batch_actions)
            loss = losses["l1_loss"]
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach())
        print(f"epoch {epoch + 1}/{args.epochs} loss={last_loss:.4f}", flush=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "policy.pt")


if __name__ == "__main__":
    main()
