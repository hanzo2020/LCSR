from __future__ import annotations

import torch


LCSR_CONSENSUS_CHOICES = ("mean", "geo", "benefit_risk")
LCSR_CONSENSUS_EPS = 1e-6


def map_cosine_to_unit_interval(score: torch.Tensor) -> torch.Tensor:
    return (score + 1.0) / 2.0


def compute_consensus_score(
    s_id: torch.Tensor,
    s_low: torch.Tensor,
    s_mid: torch.Tensor,
    s_high: torch.Tensor,
    consensus: str = "mean",
    eps: float = LCSR_CONSENSUS_EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    if consensus not in LCSR_CONSENSUS_CHOICES:
        raise ValueError(f"Unsupported consensus: {consensus}")

    s_id_u = map_cosine_to_unit_interval(s_id)
    s_low_u = map_cosine_to_unit_interval(s_low)
    s_mid_u = map_cosine_to_unit_interval(s_mid)
    s_high_u = map_cosine_to_unit_interval(s_high)

    stacked = torch.stack([s_id_u, s_low_u, s_mid_u, s_high_u], dim=0)
    if consensus == "mean":
        c = stacked.mean(dim=0)
        score = s_id_u * c
    else:
        c = (s_id_u * s_low_u * s_mid_u * s_high_u + eps).pow(0.25)
        score = s_id_u * c
        if consensus == "benefit_risk":
            risk = stacked.std(dim=0, unbiased=False)
            score = score / (eps + risk)
    return score, c
