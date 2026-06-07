"""GRPO advantage and logprob utilities — experimental.

Ported from dexbotic.exp.rl.rl_trainer. Not validated on live hardware.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import torch
import torch.nn.functional as F

try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

    _FLASH_ATTN_AVAILABLE = True
except ImportError:  # pragma: no cover - optional acceleration path
    _FLASH_ATTN_AVAILABLE = False


def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # [experimental]
    if _FLASH_ATTN_AVAILABLE:
        b = logits.shape[:-1]
        d = logits.shape[-1]
        out = -cross_entropy_loss(logits.reshape(-1, d), labels.reshape(-1))[0]
        return out.view(*b)
    logp = F.log_softmax(logits, dim=-1)
    return torch.gather(logp, -1, labels.unsqueeze(-1)).squeeze(-1)


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    # [experimental]
    pd = F.softmax(logits, dim=-1)
    return torch.logsumexp(logits, dim=-1) - (pd * logits).sum(dim=-1)


def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    # [experimental] Ported from Dexbotic SimpleVLA-RL GRPO implementation.
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)
    id2score: dict[int, list[torch.Tensor]] = defaultdict(list)
    id2mean: dict[int, torch.Tensor] = {}
    id2std: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[int(index[i].item())].append(scores[i])
        for idx, group in id2score.items():
            t = torch.stack(group)
            id2mean[idx] = t.mean() if len(group) > 1 else torch.tensor(0.0, device=scores.device)
            id2std[idx] = t.std() if len(group) > 1 else torch.tensor(1.0, device=scores.device)
        for i in range(bsz):
            key = int(index[i].item())
            scores[i] = (scores[i] - id2mean[key]) / (id2std[key] + epsilon)
        scores = scores.unsqueeze(-1).expand(-1, response_length) * eos_mask
    return scores, scores


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    axis: Optional[int] = None,
) -> torch.Tensor:
    # [experimental]
    mask_sum = mask.sum(axis=axis)
    masked_sum = (values * mask).sum(axis=axis)
    result = masked_sum / torch.clamp(mask_sum, min=1e-8)
    return torch.where(mask_sum > 0, result, masked_sum * 0.0)


def kl_penalty(
    logprob: torch.Tensor,
    ref_logprob: torch.Tensor,
    mode: str = "kl",
) -> torch.Tensor:
    # [experimental]
    if mode == "kl":
        return logprob - ref_logprob
    if mode == "abs":
        return (logprob - ref_logprob).abs()
    if mode == "mse":
        return 0.5 * (logprob - ref_logprob).square()
    raise NotImplementedError(f"kl_penalty mode {mode!r} not supported")
