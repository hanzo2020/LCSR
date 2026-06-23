from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch_geometric.utils import add_remaining_self_loops, to_undirected

from pyagc.utils.lcsr_consensus import map_cosine_to_unit_interval


def _build_normalized_adj(edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype, device: torch.device):
    edge_index = add_remaining_self_loops(edge_index, num_nodes=num_nodes)[0]
    edge_index = to_undirected(edge_index, num_nodes=num_nodes).to(device)

    values = torch.ones(edge_index.size(1), dtype=dtype, device=device)
    row, col = edge_index
    deg = torch.zeros(num_nodes, dtype=dtype, device=device)
    deg.scatter_add_(0, row, values)
    deg_inv_sqrt = deg.clamp_min(1e-12).pow(-0.5)
    norm_values = deg_inv_sqrt[row] * values * deg_inv_sqrt[col]
    return torch.sparse_coo_tensor(
        edge_index,
        norm_values,
        (num_nodes, num_nodes),
        dtype=dtype,
        device=device,
    ).coalesce()


def _propagate_k(x: torch.Tensor, norm_adj: torch.Tensor, num_hops: int):
    x1 = torch.sparse.mm(norm_adj, x)
    xk = x1
    for _ in range(1, max(num_hops, 1)):
        xk = torch.sparse.mm(norm_adj, xk)
    return x1, xk


def _build_adjacency_csr(edge_index: torch.Tensor, num_nodes: int) -> sp.csr_matrix:
    edge_cpu = edge_index.detach().cpu().numpy()
    values = np.ones(edge_cpu.shape[1], dtype=np.bool_)
    adj = sp.coo_matrix((values, (edge_cpu[0], edge_cpu[1])), shape=(num_nodes, num_nodes))
    adj = (adj + adj.T).astype(np.bool_).tocsr()
    adj.setdiag(False)
    adj.eliminate_zeros()
    return adj


def _prepare_representations(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    filter_k: int,
    work_device: torch.device,
):
    x_work = x.to(work_device)
    edge_index_work = edge_index.to(work_device)
    norm_adj = _build_normalized_adj(
        edge_index=edge_index_work,
        num_nodes=x.size(0),
        dtype=x_work.dtype,
        device=work_device,
    )
    x1, xk = _propagate_k(x_work, norm_adj, num_hops=filter_k)
    reps = {
        "id": F.normalize(x_work, p=2, dim=-1).cpu(),
        "low": F.normalize(xk, p=2, dim=-1).cpu(),
        "mid": F.normalize(x1 - xk, p=2, dim=-1).cpu(),
        "high": F.normalize(x_work - x1, p=2, dim=-1).cpu(),
    }
    return reps


def _pair_scores_from_source(
    reps: dict[str, torch.Tensor],
    anchor: int,
    candidate_ids: np.ndarray,
    support_source: str,
) -> torch.Tensor:
    anchor_index = torch.full((int(candidate_ids.size),), int(anchor), dtype=torch.long)
    cand_index = torch.from_numpy(candidate_ids.astype(np.int64, copy=False))
    raw = map_cosine_to_unit_interval((reps["id"][anchor_index] * reps["id"][cand_index]).sum(dim=-1))
    if support_source == "raw":
        return raw
    low = map_cosine_to_unit_interval((reps["low"][anchor_index] * reps["low"][cand_index]).sum(dim=-1))
    mid = map_cosine_to_unit_interval((reps["mid"][anchor_index] * reps["mid"][cand_index]).sum(dim=-1))
    high = map_cosine_to_unit_interval((reps["high"][anchor_index] * reps["high"][cand_index]).sum(dim=-1))
    mu = (raw + low + mid + high) * 0.25
    if support_source == "mu":
        return mu
    if support_source in {"freq", "raw_mul_mu", "source_adaptive"}:
        return raw * mu
    return raw * mu


def _build_cache_path(
    cache_dir: str | Path,
    dataset_name: str,
    support_source: str,
    filter_k: int,
    bank_size: int,
) -> Path:
    safe_dataset = dataset_name.replace("/", "_").replace("\\", "_")
    safe_source = support_source.replace("/", "_").replace("\\", "_")
    return Path(cache_dir) / f"{safe_dataset}_src-{safe_source}_k-{filter_k}_bank-{bank_size}.pt"


def build_or_load_lcsr_candidate_bank(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    dataset_name: str,
    support_source: str,
    filter_k: int,
    bank_size: int,
    cache_dir: str | Path,
    work_device: torch.device,
    logger=None,
):
    cache_path = _build_cache_path(
        cache_dir=cache_dir,
        dataset_name=dataset_name,
        support_source=support_source,
        filter_k=filter_k,
        bank_size=bank_size,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        bank = payload["bank"].cpu().long()
        meta = dict(payload.get("meta", {}))
        meta.update(
            {
                "cache_hit": True,
                "cache_path": str(cache_path),
                "cache_size_bytes": int(cache_path.stat().st_size),
                "bank_shape": [int(bank.size(0)), int(bank.size(1))],
            }
        )
        if logger is not None:
            logger.info(
                f"LCSR candidate bank cache hit: path={cache_path} "
                f"shape={tuple(bank.shape)} size_bytes={meta['cache_size_bytes']}"
            )
        return bank, meta

    start = perf_counter()
    adj_csr = _build_adjacency_csr(edge_index=edge_index, num_nodes=int(x.size(0)))
    reps = _prepare_representations(
        x=x,
        edge_index=edge_index,
        filter_k=filter_k,
        work_device=work_device,
    )
    num_nodes = int(x.size(0))
    bank = torch.full((num_nodes, int(bank_size)), -1, dtype=torch.long)
    degrees = np.diff(adj_csr.indptr).astype(np.int64, copy=False)
    shortlist_size = max(int(bank_size) * 4, 128)
    frontier_unique_counts: list[int] = []
    selected_counts: list[int] = []

    for node_id in range(num_nodes):
        nbrs = adj_csr.indices[adj_csr.indptr[node_id]:adj_csr.indptr[node_id + 1]]
        if nbrs.size == 0:
            frontier_unique_counts.append(0)
            selected_counts.append(0)
            continue
        frontier_parts = [
            adj_csr.indices[adj_csr.indptr[int(nbr)]:adj_csr.indptr[int(nbr) + 1]]
            for nbr in nbrs
        ]
        if not frontier_parts:
            frontier_unique_counts.append(0)
            selected_counts.append(0)
            continue
        frontier = np.concatenate(frontier_parts)
        if frontier.size == 0:
            frontier_unique_counts.append(0)
            selected_counts.append(0)
            continue
        uniq, counts = np.unique(frontier.astype(np.int64, copy=False), return_counts=True)
        valid = uniq != node_id
        if valid.any():
            neighbor_mask = np.asarray(adj_csr[node_id, uniq].toarray()).reshape(-1).astype(bool, copy=False)
            valid &= ~neighbor_mask
        uniq = uniq[valid]
        counts = counts[valid]
        frontier_unique_counts.append(int(uniq.size))
        if uniq.size == 0:
            selected_counts.append(0)
            continue
        if uniq.size > shortlist_size:
            top_idx = np.argpartition(counts, -shortlist_size)[-shortlist_size:]
            uniq = uniq[top_idx]
            counts = counts[top_idx]
        score = _pair_scores_from_source(
            reps=reps,
            anchor=node_id,
            candidate_ids=uniq,
            support_source=support_source,
        )
        if uniq.size > int(bank_size):
            top = torch.topk(score, k=int(bank_size))
            chosen = uniq[top.indices.cpu().numpy()]
        else:
            order = torch.argsort(score, descending=True)
            chosen = uniq[order.cpu().numpy()]
        take = min(int(bank_size), int(chosen.size))
        if take > 0:
            bank[node_id, :take] = torch.from_numpy(chosen[:take].astype(np.int64, copy=False))
        selected_counts.append(int(take))

    build_s = perf_counter() - start
    meta = {
        "cache_hit": False,
        "cache_path": str(cache_path),
        "bank_shape": [int(bank.size(0)), int(bank.size(1))],
        "bank_size": int(bank_size),
        "dataset": dataset_name,
        "support_source": support_source,
        "filter_k": int(filter_k),
        "num_nodes": int(num_nodes),
        "mean_degree": float(degrees.mean()) if degrees.size > 0 else 0.0,
        "candidate_bank_build_s": float(build_s),
        "frontier_unique_mean": float(np.mean(frontier_unique_counts)) if frontier_unique_counts else 0.0,
        "frontier_unique_max": int(np.max(frontier_unique_counts)) if frontier_unique_counts else 0,
        "selected_count_mean": float(np.mean(selected_counts)) if selected_counts else 0.0,
        "selected_count_max": int(np.max(selected_counts)) if selected_counts else 0,
    }
    torch.save({"bank": bank, "meta": meta}, cache_path)
    meta["cache_size_bytes"] = int(cache_path.stat().st_size)
    if logger is not None:
        logger.info(
            f"LCSR candidate bank built: path={cache_path} "
            f"shape={tuple(bank.shape)} build_s={build_s:.2f} "
            f"frontier_mean={meta['frontier_unique_mean']:.2f} "
            f"selected_mean={meta['selected_count_mean']:.2f}"
        )
    return bank, meta
