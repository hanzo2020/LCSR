from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F

from pyagc.utils.lcsr_consensus import compute_consensus_score, map_cosine_to_unit_interval
from pyagc.utils.lcsr_nonedge_diag import _build_adjacency_csr
from pyagc.utils.lcsr_nonedge_diag import _sample_random_nonedge
from pyagc.utils.lcsr_pair_source_comparison import (
    evaluate_nonedge_source,
    prepare_pair_source_representations,
    select_nonedge_topk_by_source,
)


@dataclass
class LCSRPairPackage:
    mode_name: str
    rectify_variant: str
    add_score_name: str
    reliability_mode: str
    use_local_calibration: bool
    use_mutual: bool
    add_pair_index: torch.Tensor
    drop_pair_index: torch.Tensor
    add_pair_scores: torch.Tensor
    drop_pair_scores: torch.Tensor
    candidate_pool_size: int
    margin: float
    rho: float
    kmax: int
    release: bool
    budget_match: bool
    raw_swap_count: int
    raw_unique_add_count: int
    raw_unique_drop_count: int
    matched_unique_add_count: int
    matched_unique_drop_count: int
    add_ratio: float
    drop_ratio: float
    dynamic_k_mean: float
    dynamic_k_std: float
    dynamic_k_max: int
    active_node_ratio: float
    mean_p_add: float
    mean_p_drop: float
    support_gain: float
    a_plus_same_ratio: float | None
    a_minus_cross_ratio: float | None
    sfpa_same_ratio: float | None
    random_nonedge_same_ratio: float | None
    observed_edge_same_ratio: float | None
    observed_edge_cross_ratio: float | None
    mean_static_add_score: float | None = None
    mean_embedding_add_score: float | None = None
    support_source: str = "freq"
    source_gap_raw: float | None = None
    source_gap_freq: float | None = None
    source_gap_mu: float | None = None
    source_gap_raw_mul_mu: float | None = None
    source_weight_raw: float | None = None
    source_weight_freq: float | None = None
    source_weight_mu: float | None = None
    source_weight_raw_mul_mu: float | None = None
    risk_budget_scale: float | None = None
    refine_mode: str | None = None
    refine_alpha: float | None = None
    refine_emb_floor: float | None = None
    changed_add_count_vs_static: int | None = None
    overlap_ratio_vs_static: float | None = None


@dataclass
class LCSRSwapDiagnosticPackage:
    mode_name: str
    rectify_variant: str
    add_score_name: str
    reliability_mode: str
    use_local_calibration: bool
    use_mutual: bool
    candidate_pool_size: int
    margin: float
    rho: float
    kmax: int
    budget_match: bool
    num_nodes: int
    num_edges: int
    raw_swap_count: int
    matched_swap_count: int
    raw_swap_records: list[dict[str, float | int | bool]]
    matched_swap_records: list[dict[str, float | int | bool]]
    observed_edge_same_ratio: float | None
    observed_edge_cross_ratio: float | None
    random_nonedge_same_ratio: float | None
    sfpa_reference_pairs: list[dict[str, float | int | bool]]


def _canonical_pair(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def _compute_pair_scores_for_pairs(
    reps: dict[str, torch.Tensor],
    left: np.ndarray,
    right: np.ndarray,
    reliability_mode: str,
    support_source: str = "freq",
    source_weights: dict[str, float] | None = None,
    batch_size: int = 131072,
) -> torch.Tensor:
    scores = torch.empty(len(left), dtype=torch.float32)
    if len(left) == 0:
        return scores
    for start in range(0, len(left), batch_size):
        end = min(start + batch_size, len(left))
        l_idx = torch.from_numpy(left[start:end]).long()
        r_idx = torch.from_numpy(right[start:end]).long()
        s_id = (reps["raw"][l_idx] * reps["raw"][r_idx]).sum(dim=-1)
        if reliability_mode == "identity_only":
            score = map_cosine_to_unit_interval(s_id)
        else:
            s_low = (reps["low"][l_idx] * reps["low"][r_idx]).sum(dim=-1)
            if reliability_mode == "low_only":
                score = map_cosine_to_unit_interval(s_low)
            else:
                s_mid = (reps["mid"][l_idx] * reps["mid"][r_idx]).sum(dim=-1)
                s_high = (reps["high"][l_idx] * reps["high"][r_idx]).sum(dim=-1)
                raw_u = map_cosine_to_unit_interval(s_id)
                low_u = map_cosine_to_unit_interval(s_low)
                mid_u = map_cosine_to_unit_interval(s_mid)
                high_u = map_cosine_to_unit_interval(s_high)
                mu = torch.stack([raw_u, low_u, mid_u, high_u], dim=0).mean(dim=0)
                freq = raw_u * mu
                raw_mul_mu = raw_u * mu
                if support_source == "raw":
                    score = raw_u
                elif support_source == "mu":
                    score = mu
                elif support_source in {"freq", "raw_mul_mu"}:
                    score = freq if support_source == "freq" else raw_mul_mu
                elif support_source == "source_adaptive":
                    weights = source_weights or {
                        "raw": 0.25,
                        "freq": 0.25,
                        "mu": 0.25,
                        "raw_mul_mu": 0.25,
                    }
                    score = (
                        float(weights.get("raw", 0.0)) * raw_u
                        + float(weights.get("freq", 0.0)) * freq
                        + float(weights.get("mu", 0.0)) * mu
                        + float(weights.get("raw_mul_mu", 0.0)) * raw_mul_mu
                    )
                else:
                    score, _ = compute_consensus_score(s_id, s_low, s_mid, s_high, consensus="mean")
        scores[start:end] = score.cpu()
    return scores


def _resolve_support_source(reliability_mode: str, support_source: str | None) -> str:
    if support_source is not None:
        return support_source
    return {
        "full": "freq",
        "identity_only": "raw",
        "low_only": "mu",
    }.get(reliability_mode, "freq")


def _softmax_weights_from_gaps(gaps: dict[str, float]) -> dict[str, float]:
    keys = ["raw", "freq", "mu", "raw_mul_mu"]
    values = np.asarray([float(gaps.get(key, 0.0)) for key in keys], dtype=np.float64)
    values = values - values.max()
    exp = np.exp(values)
    denom = float(exp.sum()) if float(exp.sum()) > 0 else 1.0
    return {key: float(val / denom) for key, val in zip(keys, exp)}


def _estimate_source_adaptive_metadata(
    reps: dict[str, torch.Tensor],
    adj_csr,
    seed: int,
    sample_per_node: int = 1,
) -> tuple[dict[str, float], dict[str, float], float]:
    coo = adj_csr.tocoo()
    mask = coo.row < coo.col
    obs_left = coo.row[mask].astype(np.int64, copy=False)
    obs_right = coo.col[mask].astype(np.int64, copy=False)
    obs_scores = {}
    for source in ["raw", "freq", "mu", "raw_mul_mu"]:
        score = _compute_pair_scores_for_pairs(
            reps,
            obs_left,
            obs_right,
            reliability_mode="full",
            support_source=source,
        )
        obs_scores[source] = float(score.mean().item()) if score.numel() > 0 else 0.0

    rng = np.random.default_rng(seed + 7919)
    num_nodes = adj_csr.shape[0]
    degree = np.diff(adj_csr.indptr)
    rand_left = []
    rand_right = []
    for node_id in range(num_nodes):
        take = min(sample_per_node, max(int(num_nodes - 1 - degree[node_id]), 0))
        if take <= 0:
            continue
        neighbors = adj_csr.indices[adj_csr.indptr[node_id]:adj_csr.indptr[node_id + 1]]
        sampled = _sample_random_nonedge(
            rng=rng,
            node_id=node_id,
            candidate_count=take,
            num_nodes=num_nodes,
            neighbors=neighbors,
        )
        for cand in sampled.tolist():
            pair = _canonical_pair(int(node_id), int(cand))
            rand_left.append(pair[0])
            rand_right.append(pair[1])
    if rand_left:
        rand_left_arr = np.asarray(rand_left, dtype=np.int64)
        rand_right_arr = np.asarray(rand_right, dtype=np.int64)
    else:
        rand_left_arr = np.empty((0,), dtype=np.int64)
        rand_right_arr = np.empty((0,), dtype=np.int64)
    rand_scores = {}
    for source in ["raw", "freq", "mu", "raw_mul_mu"]:
        score = _compute_pair_scores_for_pairs(
            reps,
            rand_left_arr,
            rand_right_arr,
            reliability_mode="full",
            support_source=source,
        )
        rand_scores[source] = float(score.mean().item()) if score.numel() > 0 else 0.0
    gaps = {source: float(obs_scores[source] - rand_scores[source]) for source in obs_scores}
    weights = _softmax_weights_from_gaps(gaps)
    risk_budget_scale = max(weights.values()) if weights else 1.0
    return gaps, weights, float(risk_budget_scale)


def _compute_observed_support_scores(adj_csr, reps: dict[str, torch.Tensor], reliability_mode: str):
    coo = adj_csr.tocoo()
    mask = coo.row < coo.col
    left = coo.row[mask].astype(np.int64, copy=False)
    right = coo.col[mask].astype(np.int64, copy=False)
    scores = _compute_pair_scores_for_pairs(reps, left, right, reliability_mode=reliability_mode)
    return left, right, scores


def _build_sorted_observed_score_lists(
    num_nodes: int,
    left: np.ndarray,
    right: np.ndarray,
    scores: torch.Tensor,
):
    buckets: list[list[float]] = [[] for _ in range(num_nodes)]
    for u, v, score in zip(left.tolist(), right.tolist(), scores.tolist()):
        buckets[u].append(score)
        buckets[v].append(score)
    sorted_scores = []
    for items in buckets:
        if items:
            sorted_scores.append(np.sort(np.asarray(items, dtype=np.float32)))
        else:
            sorted_scores.append(np.empty((0,), dtype=np.float32))
    return sorted_scores


def _build_sorted_candidate_score_lists(pool_scores: torch.Tensor) -> list[np.ndarray]:
    sorted_scores = []
    for node_id in range(pool_scores.size(0)):
        values = pool_scores[node_id].detach().cpu().numpy()
        valid = values[np.isfinite(values)]
        if valid.size > 0:
            sorted_scores.append(np.sort(valid.astype(np.float32, copy=False)))
        else:
            sorted_scores.append(np.empty((0,), dtype=np.float32))
    return sorted_scores


def _local_percentile(sorted_scores: list[np.ndarray], node_id: int, score: float) -> float:
    node_scores = sorted_scores[node_id]
    rank = np.searchsorted(node_scores, score, side="right")
    return float((1 + rank) / (node_scores.size + 1))


def _global_percentile(sorted_scores: list[np.ndarray], node_id: int, score: float) -> float:
    node_scores = sorted_scores[node_id]
    rank = np.searchsorted(node_scores, score, side="right")
    return float((1 + rank) / (node_scores.size + 1))


def _pair_same_ratio(y: torch.Tensor | None, pair_to_value: dict[tuple[int, int], float]) -> float | None:
    if y is None or not pair_to_value:
        return None
    y_cpu = y.detach().cpu()
    valid_nodes = torch.ones(y_cpu.size(0), dtype=torch.bool)
    if torch.is_floating_point(y_cpu):
        valid_nodes = ~torch.isnan(y_cpu)
    left = []
    right = []
    for u, v in pair_to_value.keys():
        if valid_nodes[u] and valid_nodes[v]:
            left.append(u)
            right.append(v)
    if not left:
        return None
    left_t = torch.tensor(left, dtype=torch.long)
    right_t = torch.tensor(right, dtype=torch.long)
    return float((y_cpu[left_t] == y_cpu[right_t]).float().mean().item())


def _build_pair_index_from_pairs(pair_to_value: dict[tuple[int, int], float]) -> torch.Tensor:
    if not pair_to_value:
        return torch.empty((2, 0), dtype=torch.long)
    pairs = sorted(pair_to_value.keys())
    left = torch.tensor([u for u, _ in pairs], dtype=torch.long)
    right = torch.tensor([v for _, v in pairs], dtype=torch.long)
    return torch.stack([left, right], dim=0)


def _build_pair_index_and_scores_from_pairs(
    pair_to_value: dict[tuple[int, int], float]
) -> tuple[torch.Tensor, torch.Tensor]:
    if not pair_to_value:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float32)
    pairs = sorted(pair_to_value.keys())
    left = torch.tensor([u for u, _ in pairs], dtype=torch.long)
    right = torch.tensor([v for _, v in pairs], dtype=torch.long)
    scores = torch.tensor([float(pair_to_value[pair]) for pair in pairs], dtype=torch.float32)
    return torch.stack([left, right], dim=0), scores


def _prepare_observed_candidates(
    num_nodes: int,
    obs_left: np.ndarray,
    obs_right: np.ndarray,
    obs_scores: torch.Tensor,
    sorted_obs_scores: list[np.ndarray],
    use_local_calibration: bool,
    use_mutual: bool,
) -> list[list[dict[str, float | int]]]:
    observed_by_node: list[list[dict[str, float | int]]] = [[] for _ in range(num_nodes)]
    for u, v, score in zip(obs_left.tolist(), obs_right.tolist(), obs_scores.tolist()):
        if use_local_calibration:
            p_u_to_v = _local_percentile(sorted_obs_scores, u, score)
            p_v_to_u = _local_percentile(sorted_obs_scores, v, score)
        else:
            p_u_to_v = float(score)
            p_v_to_u = float(score)
        p_drop = max(p_u_to_v, p_v_to_u) if use_mutual else 0.5 * (p_u_to_v + p_v_to_u)
        observed_by_node[u].append(
            {"nbr": v, "score": float(score), "p_local": p_u_to_v, "p_peer": p_v_to_u, "p_drop": p_drop}
        )
        observed_by_node[v].append(
            {"nbr": u, "score": float(score), "p_local": p_v_to_u, "p_peer": p_u_to_v, "p_drop": p_drop}
        )
    return observed_by_node


def _prepare_nonedge_candidates(
    num_nodes: int,
    pool_scores: torch.Tensor,
    pool_indices: torch.Tensor,
    sorted_obs_scores: list[np.ndarray],
    use_local_calibration: bool,
    use_mutual: bool,
) -> list[list[dict[str, float | int]]]:
    candidate_by_node: list[list[dict[str, float | int]]] = [[] for _ in range(num_nodes)]
    for node_id in range(num_nodes):
        node_items: list[dict[str, float | int]] = []
        for cand, score in zip(pool_indices[node_id].tolist(), pool_scores[node_id].tolist()):
            if cand < 0 or not np.isfinite(score):
                continue
            if use_local_calibration:
                p_i_to_j = _local_percentile(sorted_obs_scores, node_id, float(score))
                p_j_to_i = _local_percentile(sorted_obs_scores, cand, float(score))
            else:
                p_i_to_j = float(score)
                p_j_to_i = float(score)
            p_add = min(p_i_to_j, p_j_to_i) if use_mutual else 0.5 * (p_i_to_j + p_j_to_i)
            node_items.append(
                {"nbr": cand, "score": float(score), "p_local": p_i_to_j, "p_peer": p_j_to_i, "p_add": p_add}
            )
        candidate_by_node[node_id] = node_items
    return candidate_by_node


def _prepare_observed_candidates_degree_shrink(
    num_nodes: int,
    obs_left: np.ndarray,
    obs_right: np.ndarray,
    obs_scores: torch.Tensor,
    sorted_obs_scores: list[np.ndarray],
    sorted_pool_scores: list[np.ndarray],
    alpha_by_node: np.ndarray,
) -> list[list[dict[str, float | int]]]:
    observed_by_node: list[list[dict[str, float | int]]] = [[] for _ in range(num_nodes)]
    for u, v, score in zip(obs_left.tolist(), obs_right.tolist(), obs_scores.tolist()):
        p_local_u = _local_percentile(sorted_obs_scores, u, float(score))
        p_local_v = _local_percentile(sorted_obs_scores, v, float(score))
        p_global_u = _global_percentile(sorted_pool_scores, u, float(score))
        p_global_v = _global_percentile(sorted_pool_scores, v, float(score))
        p_tilde_u = float(alpha_by_node[u] * p_local_u + (1.0 - alpha_by_node[u]) * p_global_u)
        p_tilde_v = float(alpha_by_node[v] * p_local_v + (1.0 - alpha_by_node[v]) * p_global_v)
        p_drop = 0.5 * (p_tilde_u + p_tilde_v)
        observed_by_node[u].append(
            {
                "nbr": v,
                "score": float(score),
                "p_local": p_local_u,
                "p_global": p_global_u,
                "p_tilde": p_tilde_u,
                "p_peer_tilde": p_tilde_v,
                "p_drop": p_drop,
            }
        )
        observed_by_node[v].append(
            {
                "nbr": u,
                "score": float(score),
                "p_local": p_local_v,
                "p_global": p_global_v,
                "p_tilde": p_tilde_v,
                "p_peer_tilde": p_tilde_u,
                "p_drop": p_drop,
            }
        )
    return observed_by_node


def _prepare_nonedge_candidates_degree_shrink(
    num_nodes: int,
    pool_scores: torch.Tensor,
    pool_indices: torch.Tensor,
    sorted_obs_scores: list[np.ndarray],
    sorted_pool_scores: list[np.ndarray],
    alpha_by_node: np.ndarray,
    add_score_name: str,
) -> list[list[dict[str, float | int]]]:
    candidate_by_node: list[list[dict[str, float | int]]] = [[] for _ in range(num_nodes)]
    for node_id in range(num_nodes):
        node_items: list[dict[str, float | int]] = []
        for cand, score in zip(pool_indices[node_id].tolist(), pool_scores[node_id].tolist()):
            if cand < 0 or not np.isfinite(score):
                continue
            p_local_i = _local_percentile(sorted_obs_scores, node_id, float(score))
            p_local_j = _local_percentile(sorted_obs_scores, int(cand), float(score))
            p_global_i = _global_percentile(sorted_pool_scores, node_id, float(score))
            p_global_j = _global_percentile(sorted_pool_scores, int(cand), float(score))
            p_tilde_i = float(alpha_by_node[node_id] * p_local_i + (1.0 - alpha_by_node[node_id]) * p_global_i)
            p_tilde_j = float(alpha_by_node[int(cand)] * p_local_j + (1.0 - alpha_by_node[int(cand)]) * p_global_j)
            p_global = 0.5 * (p_global_i + p_global_j)
            p_shrink = 0.5 * (p_tilde_i + p_tilde_j)
            p_add = p_global if add_score_name == "global" else p_shrink
            node_items.append(
                {
                    "nbr": cand,
                    "score": float(score),
                    "p_local": p_local_i,
                    "p_global": p_global_i,
                    "p_tilde": p_tilde_i,
                    "p_peer_tilde": p_tilde_j,
                    "p_add": p_add,
                }
            )
        candidate_by_node[node_id] = node_items
    return candidate_by_node


def _collect_swaps_v2(
    observed_by_node: list[list[dict[str, float | int]]],
    candidate_by_node: list[list[dict[str, float | int]]],
    degrees: np.ndarray,
    margin: float,
    rho: float,
    kmax: int,
):
    swap_records: list[dict[str, float | int | tuple[int, int]]] = []
    dynamic_k = np.zeros(len(observed_by_node), dtype=np.int64)

    for node_id in range(len(observed_by_node)):
        observed_items = observed_by_node[node_id]
        candidate_items = candidate_by_node[node_id]
        if not observed_items or not candidate_items:
            continue

        observed_items = sorted(observed_items, key=lambda item: float(item["p_drop"]))
        candidate_items = sorted(candidate_items, key=lambda item: float(item["p_add"]), reverse=True)
        node_budget = min(int(kmax), int(ceil(float(rho) * max(int(degrees[node_id]), 0))))
        if node_budget <= 0:
            continue

        max_swaps = min(len(observed_items), len(candidate_items), node_budget)
        for t in range(max_swaps):
            drop_item = observed_items[t]
            add_item = candidate_items[t]
            p_drop = float(drop_item["p_drop"])
            p_add = float(add_item["p_add"])
            gain = p_add - p_drop
            if gain <= margin:
                break
            drop_nbr = int(drop_item["nbr"])
            add_nbr = int(add_item["nbr"])
            swap_records.append(
                {
                    "anchor": node_id,
                    "drop_nbr": drop_nbr,
                    "add_nbr": add_nbr,
                    "add_pair": _canonical_pair(node_id, add_nbr),
                    "drop_pair": _canonical_pair(node_id, drop_nbr),
                    "p_add": p_add,
                    "p_drop": p_drop,
                    "gain": gain,
                    "r_add": float(add_item["score"]),
                    "r_drop": float(drop_item["score"]),
                    "r_gain": float(add_item["score"]) - float(drop_item["score"]),
                    "degree": int(degrees[node_id]),
                }
            )
            dynamic_k[node_id] += 1
    return swap_records, dynamic_k


def _collect_decoupled_global_add_swaps(
    observed_by_node: list[list[dict[str, float | int]]],
    candidate_by_node_gate: list[list[dict[str, float | int]]],
    candidate_by_node_global: list[list[dict[str, float | int]]],
    degrees: np.ndarray,
    margin: float,
    rho: float,
    kmax: int,
):
    swap_records: list[dict[str, float | int | tuple[int, int]]] = []
    dynamic_k = np.zeros(len(observed_by_node), dtype=np.int64)

    for node_id in range(len(observed_by_node)):
        observed_items = observed_by_node[node_id]
        gate_items = candidate_by_node_gate[node_id]
        candidate_items = candidate_by_node_global[node_id]
        if not observed_items or not candidate_items or not gate_items:
            continue

        observed_items = sorted(observed_items, key=lambda item: float(item["p_drop"]))
        gate_items = sorted(gate_items, key=lambda item: float(item["p_add"]), reverse=True)
        candidate_items = sorted(candidate_items, key=lambda item: float(item["score"]), reverse=True)
        node_budget = min(int(kmax), int(ceil(float(rho) * max(int(degrees[node_id]), 0))))
        if node_budget <= 0:
            continue

        max_swaps = min(len(observed_items), len(candidate_items), len(gate_items), node_budget)
        accepted_drop_count = 0
        for t in range(max_swaps):
            drop_item = observed_items[t]
            add_item_for_gate = gate_items[t]
            if float(add_item_for_gate["p_add"]) - float(drop_item["p_drop"]) <= margin:
                break
            accepted_drop_count += 1

        for t in range(accepted_drop_count):
            drop_item = observed_items[t]
            add_item = candidate_items[t]
            drop_nbr = int(drop_item["nbr"])
            add_nbr = int(add_item["nbr"])
            p_drop = float(drop_item["p_drop"])
            p_add = float(add_item["score"])
            swap_records.append(
                {
                    "anchor": node_id,
                    "drop_nbr": drop_nbr,
                    "add_nbr": add_nbr,
                    "add_pair": _canonical_pair(node_id, add_nbr),
                    "drop_pair": _canonical_pair(node_id, drop_nbr),
                    "p_add": p_add,
                    "p_drop": p_drop,
                    "gain": p_add,
                    "r_add": float(add_item["score"]),
                    "r_drop": float(drop_item["score"]),
                    "r_gain": float(add_item["score"]) - float(drop_item["score"]),
                    "degree": int(degrees[node_id]),
                }
            )
            dynamic_k[node_id] += 1
    return swap_records, dynamic_k


def _dedup_pair_scores(
    swap_records: list[dict[str, float | int | tuple[int, int]]],
    pair_key: str,
    value_key: str,
) -> dict[tuple[int, int], float]:
    pair_to_value: dict[tuple[int, int], float] = {}
    for record in swap_records:
        pair = record[pair_key]
        assert isinstance(pair, tuple)
        value = float(record[value_key])
        prev = pair_to_value.get(pair)
        if prev is None or value > prev:
            pair_to_value[pair] = value
    return pair_to_value


def _match_swap_records(
    swap_records: list[dict[str, float | int | tuple[int, int]]],
    enable_budget_match: bool,
):
    if not enable_budget_match:
        return list(swap_records)

    ordered = sorted(swap_records, key=lambda item: float(item["gain"]), reverse=True)
    selected_records = []
    seen_add = set()
    seen_drop = set()
    for record in ordered:
        add_pair = record["add_pair"]
        drop_pair = record["drop_pair"]
        assert isinstance(add_pair, tuple)
        assert isinstance(drop_pair, tuple)
        if add_pair in seen_add or drop_pair in seen_drop:
            continue
        seen_add.add(add_pair)
        seen_drop.add(drop_pair)
        selected_records.append(record)
    return selected_records


def _match_swaps(
    swap_records: list[dict[str, float | int | tuple[int, int]]],
    enable_budget_match: bool,
):
    selected_records = _match_swap_records(
        swap_records=swap_records,
        enable_budget_match=enable_budget_match,
    )
    selected_add = _dedup_pair_scores(selected_records, "add_pair", "p_add")
    selected_drop = _dedup_pair_scores(selected_records, "drop_pair", "p_drop")
    return selected_add, selected_drop


def _build_edge_same_ratio(y: torch.Tensor | None, obs_same: float | None) -> tuple[float | None, float | None]:
    if y is None or obs_same is None:
        return None, None
    return obs_same, 1.0 - obs_same


def _build_reference_nonedge_pairs(
    pool_scores: torch.Tensor,
    pool_indices: torch.Tensor,
):
    pair_to_score: dict[tuple[int, int], float] = {}
    for node_id in range(pool_indices.size(0)):
        for cand, score in zip(pool_indices[node_id].tolist(), pool_scores[node_id].tolist()):
            if cand < 0 or not np.isfinite(score):
                continue
            pair = _canonical_pair(node_id, int(cand))
            prev = pair_to_score.get(pair)
            value = float(score)
            if prev is None or value > prev:
                pair_to_score[pair] = value
    ranked_pairs = sorted(pair_to_score.items(), key=lambda item: item[1], reverse=True)
    return [
        {"u": int(pair[0]), "v": int(pair[1]), "score": float(score)}
        for pair, score in ranked_pairs
    ]


def _build_embedding_candidate_score_maps(
    embeddings: torch.Tensor,
    pool_indices: torch.Tensor,
) -> tuple[list[dict[int, float]], list[np.ndarray]]:
    z = F.normalize(embeddings.detach().cpu().float(), p=2, dim=-1)
    score_maps: list[dict[int, float]] = []
    sorted_scores: list[np.ndarray] = []
    for node_id in range(pool_indices.size(0)):
        cand = pool_indices[node_id]
        valid = cand >= 0
        if not bool(valid.any()):
            score_maps.append({})
            sorted_scores.append(np.empty((0,), dtype=np.float32))
            continue
        cand_ids = cand[valid].long()
        node_vec = z[node_id].unsqueeze(0)
        emb_cos = (node_vec * z[cand_ids]).sum(dim=-1)
        emb_score = map_cosine_to_unit_interval(emb_cos).cpu().numpy().astype(np.float32, copy=False)
        mapping = {int(c): float(s) for c, s in zip(cand_ids.tolist(), emb_score.tolist())}
        score_maps.append(mapping)
        sorted_scores.append(np.sort(emb_score) if emb_score.size > 0 else np.empty((0,), dtype=np.float32))
    return score_maps, sorted_scores


def _embedding_percentile(sorted_scores: list[np.ndarray], node_id: int, score: float) -> float:
    node_scores = sorted_scores[node_id]
    rank = np.searchsorted(node_scores, score, side="right")
    return float((1 + rank) / (node_scores.size + 1))


def _pair_set_from_index(pair_index: torch.Tensor) -> set[tuple[int, int]]:
    if pair_index is None or pair_index.numel() == 0:
        return set()
    return {
        _canonical_pair(int(u), int(v))
        for u, v in zip(pair_index[0].tolist(), pair_index[1].tolist())
    }


def _annotate_swap_records(
    swap_records: list[dict[str, float | int | tuple[int, int]]],
    y: torch.Tensor | None,
):
    y_cpu = None if y is None else y.detach().cpu()
    annotated = []
    for record in swap_records:
        add_pair = record["add_pair"]
        drop_pair = record["drop_pair"]
        assert isinstance(add_pair, tuple)
        assert isinstance(drop_pair, tuple)
        add_same = None
        drop_cross = None
        if y_cpu is not None:
            add_same = bool(y_cpu[add_pair[0]] == y_cpu[add_pair[1]])
            drop_cross = bool(y_cpu[drop_pair[0]] != y_cpu[drop_pair[1]])
        annotated.append(
            {
                "anchor": int(record["anchor"]),
                "degree": int(record["degree"]),
                "drop_u": int(drop_pair[0]),
                "drop_v": int(drop_pair[1]),
                "drop_nbr": int(record["drop_nbr"]),
                "add_u": int(add_pair[0]),
                "add_v": int(add_pair[1]),
                "add_nbr": int(record["add_nbr"]),
                "p_add": float(record["p_add"]),
                "p_drop": float(record["p_drop"]),
                "gain": float(record["gain"]),
                "r_add": float(record["r_add"]),
                "r_drop": float(record["r_drop"]),
                "r_gain": float(record["r_gain"]),
                "a_plus_same_label": add_same,
                "a_minus_cross_label": drop_cross,
            }
        )
    return annotated


def build_lcsr_pair_package(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor | None,
    filter_k: int,
    candidate_pool_size: int,
    seed: int,
    work_device: torch.device,
    margin: float = 0.0,
    rho: float = 1.0,
    kmax: int = 32,
    release: bool = False,
    budget_match: bool = False,
    reliability_mode: str = "full",
    support_source: str | None = None,
    risk_budget: bool = False,
    use_local_calibration: bool = True,
    use_mutual: bool = True,
    row_block_size: int = 256,
    col_block_size: int = 2048,
    mode_name: str = "lcsr",
    rectify_variant: str = "v2",
    add_score_name: str = "local",
) -> LCSRPairPackage:
    support_source = _resolve_support_source(reliability_mode, support_source)
    reps = prepare_pair_source_representations(
        x=x,
        edge_index=edge_index,
        embeddings=x,
        filter_k=filter_k,
        work_device=work_device,
    )
    adj_csr = _build_adjacency_csr(edge_index=edge_index, num_nodes=x.size(0))
    num_nodes = x.size(0)
    num_edges = int(adj_csr.nnz // 2)
    degrees = np.diff(adj_csr.indptr).astype(np.int64, copy=False)

    source_gaps = {"raw": None, "freq": None, "mu": None, "raw_mul_mu": None}
    source_weights = {"raw": None, "freq": None, "mu": None, "raw_mul_mu": None}
    risk_budget_scale = None
    adaptive_weights = None
    gaps, adaptive_weights_all, adaptive_scale = _estimate_source_adaptive_metadata(
        reps=reps,
        adj_csr=adj_csr,
        seed=seed,
    )
    source_gaps.update(gaps)
    if support_source == "source_adaptive":
        adaptive_weights = adaptive_weights_all
        source_weights.update(adaptive_weights_all)
        risk_budget_scale = adaptive_scale
    coo = adj_csr.tocoo()
    mask = coo.row < coo.col
    obs_left = coo.row[mask].astype(np.int64, copy=False)
    obs_right = coo.col[mask].astype(np.int64, copy=False)
    obs_scores = _compute_pair_scores_for_pairs(
        reps,
        obs_left,
        obs_right,
        reliability_mode=reliability_mode,
        support_source=support_source,
        source_weights=adaptive_weights,
    )
    sorted_obs_scores = _build_sorted_observed_score_lists(num_nodes, obs_left, obs_right, obs_scores)

    source_name = {
        "raw": "raw",
        "freq": "semantic_frequency",
        "mu": "lcsr_mu",
        "raw_mul_mu": "raw_mul_lcsr",
        "source_adaptive": "semantic_frequency",
    }[support_source]
    pool_scores, pool_indices = select_nonedge_topk_by_source(
        source_name=source_name,
        x=x,
        edge_index=edge_index,
        embeddings=None,
        filter_k=filter_k,
        topk=candidate_pool_size,
        seed=seed,
        consensus="mean",
        work_device=work_device,
        row_block_size=row_block_size,
        col_block_size=col_block_size,
    )
    if support_source == "source_adaptive":
        # Rebuild pool scores using adaptive weighted source combination on the retrieved candidate pool.
        adaptive_pool_scores = torch.full_like(pool_scores, float("-inf"))
        for node_id in range(num_nodes):
            valid = pool_indices[node_id] >= 0
            if not bool(valid.any()):
                continue
            left = np.full(int(valid.sum().item()), node_id, dtype=np.int64)
            right = pool_indices[node_id][valid].detach().cpu().numpy().astype(np.int64, copy=False)
            adaptive_vals = _compute_pair_scores_for_pairs(
                reps,
                left,
                right,
                reliability_mode="full",
                support_source="source_adaptive",
                source_weights=adaptive_weights,
            )
            adaptive_pool_scores[node_id, valid] = adaptive_vals
        pool_scores = adaptive_pool_scores
    observed_by_node = _prepare_observed_candidates(
        num_nodes=num_nodes,
        obs_left=obs_left,
        obs_right=obs_right,
        obs_scores=obs_scores,
        sorted_obs_scores=sorted_obs_scores,
        use_local_calibration=use_local_calibration,
        use_mutual=use_mutual,
    )
    candidate_by_node = _prepare_nonedge_candidates(
        num_nodes=num_nodes,
        pool_scores=pool_scores,
        pool_indices=pool_indices,
        sorted_obs_scores=sorted_obs_scores,
        use_local_calibration=use_local_calibration,
        use_mutual=use_mutual,
    )

    if rectify_variant == "v2":
        effective_rho = float(rho)
        effective_kmax = int(kmax)
        if support_source == "source_adaptive" and risk_budget and risk_budget_scale is not None:
            effective_rho = float(rho) * float(risk_budget_scale)
            effective_kmax = max(1, int(np.floor(float(kmax) * float(risk_budget_scale))))
        swap_records, dynamic_k = _collect_swaps_v2(
            observed_by_node=observed_by_node,
            candidate_by_node=candidate_by_node,
            degrees=degrees,
            margin=margin,
            rho=effective_rho,
            kmax=effective_kmax,
        )
    elif rectify_variant == "decoupled_global_add":
        candidate_by_node_global = _prepare_nonedge_candidates(
            num_nodes=num_nodes,
            pool_scores=pool_scores,
            pool_indices=pool_indices,
            sorted_obs_scores=sorted_obs_scores,
            use_local_calibration=False,
            use_mutual=False,
        )
        swap_records, dynamic_k = _collect_decoupled_global_add_swaps(
            observed_by_node=observed_by_node,
            candidate_by_node_gate=candidate_by_node,
            candidate_by_node_global=candidate_by_node_global,
            degrees=degrees,
            margin=margin,
            rho=rho,
            kmax=kmax,
        )
    elif rectify_variant == "degree_shrink":
        sorted_pool_scores = _build_sorted_candidate_score_lists(pool_scores)
        median_degree = float(np.median(degrees)) if degrees.size > 0 else 0.0
        alpha_by_node = degrees.astype(np.float64) / np.maximum(degrees.astype(np.float64) + median_degree, 1e-12)
        observed_by_node_shrink = _prepare_observed_candidates_degree_shrink(
            num_nodes=num_nodes,
            obs_left=obs_left,
            obs_right=obs_right,
            obs_scores=obs_scores,
            sorted_obs_scores=sorted_obs_scores,
            sorted_pool_scores=sorted_pool_scores,
            alpha_by_node=alpha_by_node,
        )
        candidate_by_node_shrink = _prepare_nonedge_candidates_degree_shrink(
            num_nodes=num_nodes,
            pool_scores=pool_scores,
            pool_indices=pool_indices,
            sorted_obs_scores=sorted_obs_scores,
            sorted_pool_scores=sorted_pool_scores,
            alpha_by_node=alpha_by_node,
            add_score_name=add_score_name,
        )
        effective_rho = float(rho)
        effective_kmax = int(kmax)
        if support_source == "source_adaptive" and risk_budget and risk_budget_scale is not None:
            effective_rho = float(rho) * float(risk_budget_scale)
            effective_kmax = max(1, int(np.floor(float(kmax) * float(risk_budget_scale))))
        swap_records, dynamic_k = _collect_swaps_v2(
            observed_by_node=observed_by_node_shrink,
            candidate_by_node=candidate_by_node_shrink,
            degrees=degrees,
            margin=margin,
            rho=effective_rho,
            kmax=effective_kmax,
        )
    else:
        raise ValueError(f"Unknown rectify_variant: {rectify_variant}")
    raw_add_pairs = _dedup_pair_scores(swap_records, "add_pair", "p_add")
    raw_drop_pairs = _dedup_pair_scores(swap_records, "drop_pair", "p_drop")
    matched_add_pairs, matched_drop_pairs = _match_swaps(
        swap_records=swap_records,
        enable_budget_match=budget_match,
    )

    active_nodes = int((dynamic_k > 0).sum())
    sfpa_diag = None
    if y is not None:
        sfpa_diag = evaluate_nonedge_source(
            source_name="semantic_frequency",
            x=x,
            edge_index=edge_index,
            y=y,
            embeddings=None,
            filter_k=filter_k,
            topks=[1],
            seed=seed,
            consensus="mean",
            work_device=work_device,
            row_block_size=row_block_size,
            col_block_size=col_block_size,
        )[1]

    observed_edge_same, observed_edge_cross = _build_edge_same_ratio(
        y=y,
        obs_same=None if sfpa_diag is None else float(sfpa_diag["edge_same"]),
    )
    mean_p_add = float(np.mean(list(matched_add_pairs.values()))) if matched_add_pairs else 0.0
    mean_p_drop = float(np.mean(list(matched_drop_pairs.values()))) if matched_drop_pairs else 0.0
    add_pair_index, add_pair_scores = _build_pair_index_and_scores_from_pairs(matched_add_pairs)
    drop_pair_index, drop_pair_scores = _build_pair_index_and_scores_from_pairs(matched_drop_pairs)

    return LCSRPairPackage(
        mode_name=mode_name,
        rectify_variant=rectify_variant,
        add_score_name=add_score_name,
        reliability_mode=reliability_mode,
        use_local_calibration=bool(use_local_calibration),
        use_mutual=bool(use_mutual),
        add_pair_index=add_pair_index,
        drop_pair_index=drop_pair_index,
        add_pair_scores=add_pair_scores,
        drop_pair_scores=drop_pair_scores,
        candidate_pool_size=int(candidate_pool_size),
        margin=float(margin),
        rho=float(rho),
        kmax=int(kmax),
        release=bool(release),
        budget_match=bool(budget_match),
        raw_swap_count=len(swap_records),
        raw_unique_add_count=len(raw_add_pairs),
        raw_unique_drop_count=len(raw_drop_pairs),
        matched_unique_add_count=len(matched_add_pairs),
        matched_unique_drop_count=len(matched_drop_pairs),
        add_ratio=(len(matched_add_pairs) / num_edges) if num_edges > 0 else 0.0,
        drop_ratio=(len(matched_drop_pairs) / num_edges) if num_edges > 0 else 0.0,
        dynamic_k_mean=float(dynamic_k.mean()) if dynamic_k.size > 0 else 0.0,
        dynamic_k_std=float(dynamic_k.std()) if dynamic_k.size > 0 else 0.0,
        dynamic_k_max=int(dynamic_k.max()) if dynamic_k.size > 0 else 0,
        active_node_ratio=(active_nodes / num_nodes) if num_nodes > 0 else 0.0,
        mean_p_add=mean_p_add,
        mean_p_drop=mean_p_drop,
        support_gain=mean_p_add - mean_p_drop,
        a_plus_same_ratio=_pair_same_ratio(y, matched_add_pairs),
        a_minus_cross_ratio=None
        if _pair_same_ratio(y, matched_drop_pairs) is None
        else 1.0 - float(_pair_same_ratio(y, matched_drop_pairs)),
        sfpa_same_ratio=None if sfpa_diag is None else float(sfpa_diag["same"]),
        random_nonedge_same_ratio=None if sfpa_diag is None else float(sfpa_diag["random"]),
        observed_edge_same_ratio=observed_edge_same,
        observed_edge_cross_ratio=observed_edge_cross,
        mean_static_add_score=mean_p_add,
        mean_embedding_add_score=None,
        support_source=support_source,
        source_gap_raw=source_gaps["raw"],
        source_gap_freq=source_gaps["freq"],
        source_gap_mu=source_gaps["mu"],
        source_gap_raw_mul_mu=source_gaps["raw_mul_mu"],
        source_weight_raw=source_weights["raw"],
        source_weight_freq=source_weights["freq"],
        source_weight_mu=source_weights["mu"],
        source_weight_raw_mul_mu=source_weights["raw_mul_mu"],
        risk_budget_scale=risk_budget_scale,
        refine_mode=None,
        refine_alpha=None,
        refine_emb_floor=None,
        changed_add_count_vs_static=0,
        overlap_ratio_vs_static=1.0 if matched_add_pairs else 0.0,
    )


def build_lcsr_pair_package_with_embedding_refinement(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor | None,
    embeddings: torch.Tensor,
    filter_k: int,
    candidate_pool_size: int,
    seed: int,
    work_device: torch.device,
    margin: float = 0.0,
    rho: float = 1.0,
    kmax: int = 32,
    budget_match: bool = False,
    reliability_mode: str = "full",
    row_block_size: int = 256,
    col_block_size: int = 2048,
    mode_name: str = "lcsr_refined",
    rectify_variant: str = "degree_shrink",
    add_score_name: str = "global",
    refine_mode: str = "blend",
    refine_alpha: float = 0.2,
    refine_emb_floor: float = 0.7,
) -> LCSRPairPackage:
    static_package = build_lcsr_pair_package(
        x=x,
        edge_index=edge_index,
        y=y,
        filter_k=filter_k,
        candidate_pool_size=candidate_pool_size,
        seed=seed,
        work_device=work_device,
        margin=margin,
        rho=rho,
        kmax=kmax,
        release=False,
        budget_match=budget_match,
        reliability_mode=reliability_mode,
        use_local_calibration=True,
        use_mutual=True,
        row_block_size=row_block_size,
        col_block_size=col_block_size,
        mode_name=mode_name,
        rectify_variant=rectify_variant,
        add_score_name=add_score_name,
    )
    if rectify_variant != "degree_shrink" or add_score_name != "global":
        raise ValueError("Embedding refinement currently supports only degree_shrink + global add.")

    reps = prepare_pair_source_representations(
        x=x,
        edge_index=edge_index,
        embeddings=x,
        filter_k=filter_k,
        work_device=work_device,
    )
    adj_csr = _build_adjacency_csr(edge_index=edge_index, num_nodes=x.size(0))
    num_nodes = x.size(0)
    degrees = np.diff(adj_csr.indptr).astype(np.int64, copy=False)
    obs_left, obs_right, obs_scores = _compute_observed_support_scores(
        adj_csr,
        reps,
        reliability_mode=reliability_mode,
    )
    sorted_obs_scores = _build_sorted_observed_score_lists(num_nodes, obs_left, obs_right, obs_scores)
    source_name = {
        "full": "semantic_frequency",
        "identity_only": "raw",
        "low_only": "lowpass",
    }[reliability_mode]
    pool_scores, pool_indices = select_nonedge_topk_by_source(
        source_name=source_name,
        x=x,
        edge_index=edge_index,
        embeddings=None,
        filter_k=filter_k,
        topk=candidate_pool_size,
        seed=seed,
        consensus="mean",
        work_device=work_device,
        row_block_size=row_block_size,
        col_block_size=col_block_size,
    )
    sorted_pool_scores = _build_sorted_candidate_score_lists(pool_scores)
    median_degree = float(np.median(degrees)) if degrees.size > 0 else 0.0
    alpha_by_node = degrees.astype(np.float64) / np.maximum(degrees.astype(np.float64) + median_degree, 1e-12)
    observed_by_node_shrink = _prepare_observed_candidates_degree_shrink(
        num_nodes=num_nodes,
        obs_left=obs_left,
        obs_right=obs_right,
        obs_scores=obs_scores,
        sorted_obs_scores=sorted_obs_scores,
        sorted_pool_scores=sorted_pool_scores,
        alpha_by_node=alpha_by_node,
    )
    candidate_by_node_static = _prepare_nonedge_candidates_degree_shrink(
        num_nodes=num_nodes,
        pool_scores=pool_scores,
        pool_indices=pool_indices,
        sorted_obs_scores=sorted_obs_scores,
        sorted_pool_scores=sorted_pool_scores,
        alpha_by_node=alpha_by_node,
        add_score_name=add_score_name,
    )
    emb_score_maps, emb_sorted_scores = _build_embedding_candidate_score_maps(embeddings=embeddings, pool_indices=pool_indices)
    eps = 1e-6

    static_pair_to_score = {
        _canonical_pair(int(u), int(v)): float(s)
        for u, v, s in zip(
            static_package.add_pair_index[0].tolist(),
            static_package.add_pair_index[1].tolist(),
            static_package.add_pair_scores.tolist(),
        )
    }

    if refine_mode == "filter":
        filtered_pairs: dict[tuple[int, int], float] = {}
        emb_pair_scores: dict[tuple[int, int], float] = {}
        for pair, static_score in static_pair_to_score.items():
            u, v = pair
            s_uv = emb_score_maps[u].get(v)
            s_vu = emb_score_maps[v].get(u)
            if s_uv is None and s_vu is None:
                continue
            emb_raw = float(max(s for s in [s_uv, s_vu] if s is not None))
            emb_p_u = _embedding_percentile(emb_sorted_scores, u, emb_raw) if s_uv is not None else 0.0
            emb_p_v = _embedding_percentile(emb_sorted_scores, v, emb_raw) if s_vu is not None else 0.0
            emb_score = 0.5 * (emb_p_u + emb_p_v)
            emb_pair_scores[pair] = emb_score
            if emb_score >= float(refine_emb_floor):
                filtered_pairs[pair] = static_score
        matched_add_pairs = filtered_pairs
        matched_drop_pairs = {
            _canonical_pair(int(u), int(v)): float(s)
            for u, v, s in zip(
                static_package.drop_pair_index[0].tolist(),
                static_package.drop_pair_index[1].tolist(),
                static_package.drop_pair_scores.tolist(),
            )
        }
        add_pair_index, add_pair_scores = _build_pair_index_and_scores_from_pairs(matched_add_pairs)
        drop_pair_index, drop_pair_scores = _build_pair_index_and_scores_from_pairs(matched_drop_pairs)
        mean_static_add_score = float(np.mean(list(matched_add_pairs.values()))) if matched_add_pairs else 0.0
        mean_emb_add_score = float(np.mean([emb_pair_scores[p] for p in matched_add_pairs])) if matched_add_pairs else 0.0
        overlap_ratio = (len(matched_add_pairs) / max(len(static_pair_to_score), 1)) if static_pair_to_score else 0.0
        changed_add_count = max(len(static_pair_to_score) - len(matched_add_pairs), 0)
        return LCSRPairPackage(
            mode_name=mode_name,
            rectify_variant=rectify_variant,
            add_score_name=add_score_name,
            reliability_mode=reliability_mode,
            use_local_calibration=True,
            use_mutual=True,
            add_pair_index=add_pair_index,
            drop_pair_index=drop_pair_index,
            add_pair_scores=add_pair_scores,
            drop_pair_scores=drop_pair_scores,
            candidate_pool_size=int(candidate_pool_size),
            margin=float(margin),
            rho=float(rho),
            kmax=int(kmax),
            release=False,
            budget_match=bool(budget_match and len(matched_add_pairs) == len(matched_drop_pairs)),
            raw_swap_count=static_package.raw_swap_count,
            raw_unique_add_count=static_package.raw_unique_add_count,
            raw_unique_drop_count=static_package.raw_unique_drop_count,
            matched_unique_add_count=len(matched_add_pairs),
            matched_unique_drop_count=len(matched_drop_pairs),
            add_ratio=(len(matched_add_pairs) / max(int(adj_csr.nnz // 2), 1)),
            drop_ratio=static_package.drop_ratio,
            dynamic_k_mean=static_package.dynamic_k_mean,
            dynamic_k_std=static_package.dynamic_k_std,
            dynamic_k_max=static_package.dynamic_k_max,
            active_node_ratio=static_package.active_node_ratio,
            mean_p_add=mean_static_add_score,
            mean_p_drop=static_package.mean_p_drop,
            support_gain=mean_static_add_score - static_package.mean_p_drop,
            a_plus_same_ratio=_pair_same_ratio(y, matched_add_pairs),
            a_minus_cross_ratio=static_package.a_minus_cross_ratio,
            sfpa_same_ratio=static_package.sfpa_same_ratio,
            random_nonedge_same_ratio=static_package.random_nonedge_same_ratio,
            observed_edge_same_ratio=static_package.observed_edge_same_ratio,
            observed_edge_cross_ratio=static_package.observed_edge_cross_ratio,
            mean_static_add_score=mean_static_add_score,
            mean_embedding_add_score=mean_emb_add_score,
            refine_mode=refine_mode,
            refine_alpha=None,
            refine_emb_floor=float(refine_emb_floor),
            changed_add_count_vs_static=changed_add_count,
            overlap_ratio_vs_static=overlap_ratio,
        )

    candidate_by_node_refined: list[list[dict[str, float | int]]] = [[] for _ in range(num_nodes)]
    for node_id in range(num_nodes):
        items = []
        for item in candidate_by_node_static[node_id]:
            nbr = int(item["nbr"])
            static_score = float(item["p_add"])
            emb_raw = emb_score_maps[node_id].get(nbr)
            if emb_raw is None:
                continue
            emb_score = _embedding_percentile(emb_sorted_scores, node_id, emb_raw)
            if refine_mode == "blend":
                refined = (1.0 - float(refine_alpha)) * static_score + float(refine_alpha) * emb_score
            elif refine_mode == "geom":
                refined = float(np.sqrt(max(static_score, eps) * max(emb_score, eps)))
            else:
                raise ValueError(f"Unknown refine_mode: {refine_mode}")
            new_item = dict(item)
            new_item["p_add"] = float(refined)
            new_item["static_add"] = static_score
            new_item["emb_add"] = float(emb_score)
            items.append(new_item)
        candidate_by_node_refined[node_id] = items

    swap_records, dynamic_k = _collect_swaps_v2(
        observed_by_node=observed_by_node_shrink,
        candidate_by_node=candidate_by_node_refined,
        degrees=degrees,
        margin=margin,
        rho=rho,
        kmax=kmax,
    )
    matched_add_pairs, matched_drop_pairs = _match_swaps(
        swap_records=swap_records,
        enable_budget_match=budget_match,
    )
    add_pair_index, add_pair_scores = _build_pair_index_and_scores_from_pairs(matched_add_pairs)
    drop_pair_index, drop_pair_scores = _build_pair_index_and_scores_from_pairs(matched_drop_pairs)
    refined_pair_set = set(matched_add_pairs.keys())
    static_pair_set = set(static_pair_to_score.keys())
    overlap = len(refined_pair_set & static_pair_set)
    changed_count = len(refined_pair_set.symmetric_difference(static_pair_set))

    mean_static_add_score = None
    mean_emb_add_score = None
    if refined_pair_set:
        static_vals = []
        emb_vals = []
        for node_id in range(num_nodes):
            for item in candidate_by_node_refined[node_id]:
                pair = _canonical_pair(node_id, int(item["nbr"]))
                if pair in refined_pair_set:
                    static_vals.append(float(item["static_add"]))
                    emb_vals.append(float(item["emb_add"]))
        if static_vals:
            mean_static_add_score = float(np.mean(static_vals))
            mean_emb_add_score = float(np.mean(emb_vals))

    return LCSRPairPackage(
        mode_name=mode_name,
        rectify_variant=rectify_variant,
        add_score_name=add_score_name,
        reliability_mode=reliability_mode,
        use_local_calibration=True,
        use_mutual=True,
        add_pair_index=add_pair_index,
        drop_pair_index=drop_pair_index,
        add_pair_scores=add_pair_scores,
        drop_pair_scores=drop_pair_scores,
        candidate_pool_size=int(candidate_pool_size),
        margin=float(margin),
        rho=float(rho),
        kmax=int(kmax),
        release=False,
        budget_match=bool(budget_match),
        raw_swap_count=len(swap_records),
        raw_unique_add_count=len(_dedup_pair_scores(swap_records, "add_pair", "p_add")),
        raw_unique_drop_count=len(_dedup_pair_scores(swap_records, "drop_pair", "p_drop")),
        matched_unique_add_count=len(matched_add_pairs),
        matched_unique_drop_count=len(matched_drop_pairs),
        add_ratio=(len(matched_add_pairs) / max(int(adj_csr.nnz // 2), 1)),
        drop_ratio=(len(matched_drop_pairs) / max(int(adj_csr.nnz // 2), 1)),
        dynamic_k_mean=float(dynamic_k.mean()) if dynamic_k.size > 0 else 0.0,
        dynamic_k_std=float(dynamic_k.std()) if dynamic_k.size > 0 else 0.0,
        dynamic_k_max=int(dynamic_k.max()) if dynamic_k.size > 0 else 0,
        active_node_ratio=(int((dynamic_k > 0).sum()) / num_nodes) if num_nodes > 0 else 0.0,
        mean_p_add=float(np.mean(list(matched_add_pairs.values()))) if matched_add_pairs else 0.0,
        mean_p_drop=float(np.mean(list(matched_drop_pairs.values()))) if matched_drop_pairs else 0.0,
        support_gain=(
            (float(np.mean(list(matched_add_pairs.values()))) if matched_add_pairs else 0.0)
            - (float(np.mean(list(matched_drop_pairs.values()))) if matched_drop_pairs else 0.0)
        ),
        a_plus_same_ratio=_pair_same_ratio(y, matched_add_pairs),
        a_minus_cross_ratio=None
        if _pair_same_ratio(y, matched_drop_pairs) is None
        else 1.0 - float(_pair_same_ratio(y, matched_drop_pairs)),
        sfpa_same_ratio=static_package.sfpa_same_ratio,
        random_nonedge_same_ratio=static_package.random_nonedge_same_ratio,
        observed_edge_same_ratio=static_package.observed_edge_same_ratio,
        observed_edge_cross_ratio=static_package.observed_edge_cross_ratio,
        mean_static_add_score=mean_static_add_score,
        mean_embedding_add_score=mean_emb_add_score,
        refine_mode=refine_mode,
        refine_alpha=float(refine_alpha) if refine_mode == "blend" else None,
        refine_emb_floor=float(refine_emb_floor) if refine_mode == "filter" else None,
        changed_add_count_vs_static=changed_count,
        overlap_ratio_vs_static=(overlap / max(len(static_pair_set), 1)) if static_pair_set else 0.0,
    )


def build_lcsr_swap_diagnostic_package(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor | None,
    filter_k: int,
    candidate_pool_size: int,
    seed: int,
    work_device: torch.device,
    margin: float = 0.0,
    rho: float = 1.0,
    kmax: int = 32,
    budget_match: bool = False,
    reliability_mode: str = "full",
    use_local_calibration: bool = True,
    use_mutual: bool = True,
    row_block_size: int = 256,
    col_block_size: int = 2048,
    mode_name: str = "lcsr",
    rectify_variant: str = "v2",
    add_score_name: str = "local",
) -> LCSRSwapDiagnosticPackage:
    reps = prepare_pair_source_representations(
        x=x,
        edge_index=edge_index,
        embeddings=x,
        filter_k=filter_k,
        work_device=work_device,
    )
    adj_csr = _build_adjacency_csr(edge_index=edge_index, num_nodes=x.size(0))
    num_nodes = x.size(0)
    num_edges = int(adj_csr.nnz // 2)
    degrees = np.diff(adj_csr.indptr).astype(np.int64, copy=False)

    obs_left, obs_right, obs_scores = _compute_observed_support_scores(
        adj_csr,
        reps,
        reliability_mode=reliability_mode,
    )
    sorted_obs_scores = _build_sorted_observed_score_lists(num_nodes, obs_left, obs_right, obs_scores)
    observed_by_node = _prepare_observed_candidates(
        num_nodes=num_nodes,
        obs_left=obs_left,
        obs_right=obs_right,
        obs_scores=obs_scores,
        sorted_obs_scores=sorted_obs_scores,
        use_local_calibration=use_local_calibration,
        use_mutual=use_mutual,
    )

    source_name = {
        "full": "semantic_frequency",
        "identity_only": "raw",
        "low_only": "lowpass",
    }[reliability_mode]
    pool_scores, pool_indices = select_nonedge_topk_by_source(
        source_name=source_name,
        x=x,
        edge_index=edge_index,
        embeddings=None,
        filter_k=filter_k,
        topk=candidate_pool_size,
        seed=seed,
        consensus="mean",
        work_device=work_device,
        row_block_size=row_block_size,
        col_block_size=col_block_size,
    )
    candidate_by_node = _prepare_nonedge_candidates(
        num_nodes=num_nodes,
        pool_scores=pool_scores,
        pool_indices=pool_indices,
        sorted_obs_scores=sorted_obs_scores,
        use_local_calibration=use_local_calibration,
        use_mutual=use_mutual,
    )

    swap_records, _ = _collect_swaps_v2(
        observed_by_node=observed_by_node,
        candidate_by_node=candidate_by_node,
        degrees=degrees,
        margin=margin,
        rho=rho,
        kmax=kmax,
    )
    matched_swap_records = _match_swap_records(
        swap_records=swap_records,
        enable_budget_match=budget_match,
    )

    sfpa_diag = None
    if y is not None:
        sfpa_diag = evaluate_nonedge_source(
            source_name="semantic_frequency",
            x=x,
            edge_index=edge_index,
            y=y,
            embeddings=None,
            filter_k=filter_k,
            topks=[1],
            seed=seed,
            consensus="mean",
            work_device=work_device,
            row_block_size=row_block_size,
            col_block_size=col_block_size,
        )[1]
    observed_edge_same, observed_edge_cross = _build_edge_same_ratio(
        y=y,
        obs_same=None if sfpa_diag is None else float(sfpa_diag["edge_same"]),
    )
    return LCSRSwapDiagnosticPackage(
        mode_name=mode_name,
        rectify_variant=rectify_variant,
        add_score_name=add_score_name,
        reliability_mode=reliability_mode,
        use_local_calibration=bool(use_local_calibration),
        use_mutual=bool(use_mutual),
        candidate_pool_size=int(candidate_pool_size),
        margin=float(margin),
        rho=float(rho),
        kmax=int(kmax),
        budget_match=bool(budget_match),
        num_nodes=int(num_nodes),
        num_edges=int(num_edges),
        raw_swap_count=len(swap_records),
        matched_swap_count=len(matched_swap_records),
        raw_swap_records=_annotate_swap_records(swap_records, y),
        matched_swap_records=_annotate_swap_records(matched_swap_records, y),
        observed_edge_same_ratio=observed_edge_same,
        observed_edge_cross_ratio=observed_edge_cross,
        random_nonedge_same_ratio=None if sfpa_diag is None else float(sfpa_diag["random"]),
        sfpa_reference_pairs=_build_reference_nonedge_pairs(pool_scores, pool_indices),
    )
