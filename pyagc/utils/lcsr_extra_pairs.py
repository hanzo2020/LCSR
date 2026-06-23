from __future__ import annotations

from dataclasses import dataclass

import torch

from pyagc.utils.lcsr_nonedge_diag import _build_adjacency_csr
from pyagc.utils.lcsr_pair_source_comparison import (
    _topk_nonedge_from_reps,
    prepare_pair_source_representations,
)


@dataclass
class LCSRExtraPairPackage:
    top_indices: torch.Tensor
    pair_weights: torch.Tensor | None
    confidence_scores: torch.Tensor | None
    candidate_source: str
    weight_source: str
    rho: float
    filter_k: int
    topk: int


def _compute_pair_scores(
    rep: torch.Tensor,
    top_indices: torch.Tensor,
    row_block_size: int = 4096,
) -> torch.Tensor:
    num_nodes, topk = top_indices.shape
    scores = torch.full((num_nodes, topk), float("nan"), dtype=torch.float32)

    for row_start in range(0, num_nodes, row_block_size):
        row_end = min(row_start + row_block_size, num_nodes)
        partner = top_indices[row_start:row_end]
        valid = partner >= 0
        if not valid.any():
            continue

        block_rows = rep[row_start:row_end]
        block_scores = torch.full((row_end - row_start, topk), float("nan"), dtype=torch.float32)

        valid_partner = partner.clamp_min(0)
        partner_rep = rep[valid_partner]
        pair_score = (block_rows.unsqueeze(1) * partner_rep).sum(dim=-1)
        block_scores[valid] = pair_score[valid]
        scores[row_start:row_end] = block_scores

    return scores


def _minmax_normalize_selected(scores: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    valid = torch.isfinite(scores)
    if not valid.any():
        return torch.zeros_like(scores, dtype=torch.float32)

    values = scores[valid]
    score_min = values.min()
    score_max = values.max()
    normalized = torch.zeros_like(scores, dtype=torch.float32)
    normalized[valid] = (values - score_min) / (score_max - score_min + eps)
    return normalized


def build_lcsr_extra_pair_package(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    candidate_source: str,
    weight_source: str,
    topk: int,
    filter_k: int,
    rho: float,
    work_device: torch.device,
    row_block_size: int = 256,
    col_block_size: int = 2048,
) -> LCSRExtraPairPackage:
    if candidate_source not in {"raw", "lowpass"}:
        raise ValueError(f"Unsupported candidate_source: {candidate_source}")
    if weight_source not in {"none", "fcrs_mu", "lcsr_mu", "fcrs_lcb", "lcsr_lcb"}:
        raise ValueError(f"Unsupported weight_source: {weight_source}")

    reps = prepare_pair_source_representations(
        x=x,
        edge_index=edge_index,
        embeddings=x,
        filter_k=filter_k,
        work_device=work_device,
    )
    adj_csr = _build_adjacency_csr(edge_index=edge_index, num_nodes=x.size(0))

    candidate_reps = {
        "raw": [reps["raw"]],
        "lowpass": [reps["low"]],
    }[candidate_source]

    _, top_indices = _topk_nonedge_from_reps(
        reps=candidate_reps,
        adj_csr=adj_csr,
        max_k=topk,
        work_device=work_device,
        row_block_size=row_block_size,
        col_block_size=col_block_size,
    )

    pair_weights = None
    confidence_scores = None
    if weight_source != "none":
        s_raw = _compute_pair_scores(reps["raw"], top_indices)
        s_low = _compute_pair_scores(reps["low"], top_indices)
        s_mid = _compute_pair_scores(reps["mid"], top_indices)
        s_high = _compute_pair_scores(reps["high"], top_indices)
        all_scores = torch.stack([s_raw, s_low, s_mid, s_high], dim=0)
        valid = torch.isfinite(all_scores)
        valid_count = valid.sum(dim=0)
        safe_scores = torch.where(valid, all_scores, torch.zeros_like(all_scores))
        mu = safe_scores.sum(dim=0) / valid_count.clamp_min(1)
        centered = torch.where(valid, all_scores - mu.unsqueeze(0), torch.zeros_like(all_scores))
        std = torch.sqrt((centered.pow(2).sum(dim=0) / valid_count.clamp_min(1)).clamp_min(0.0))
        mu = mu.masked_fill(valid_count == 0, float("nan"))
        std = std.masked_fill(valid_count == 0, float("nan"))

        if weight_source in {"fcrs_mu", "lcsr_mu"}:
            confidence_scores = mu
        else:
            confidence_scores = mu - float(rho) * std
        pair_weights = _minmax_normalize_selected(confidence_scores)

    return LCSRExtraPairPackage(
        top_indices=top_indices,
        pair_weights=pair_weights,
        confidence_scores=confidence_scores,
        candidate_source=candidate_source,
        weight_source=weight_source,
        rho=float(rho),
        filter_k=int(filter_k),
        topk=int(topk),
    )


build_fcrs_extra_pair_package = build_lcsr_extra_pair_package
