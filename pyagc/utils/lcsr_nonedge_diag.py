from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch_geometric.utils import add_remaining_self_loops, to_undirected


@dataclass
class NonEdgeDiagResult:
    dataset: str
    score: str
    filter_k: int
    topks: list[int]
    results: dict[int, dict[str, Any]]
    baseline: dict[str, Any]


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
        "raw": F.normalize(x_work, p=2, dim=-1).cpu(),
        "low": F.normalize(xk, p=2, dim=-1).cpu(),
        "mid": F.normalize(x1 - xk, p=2, dim=-1).cpu(),
        "high": F.normalize(x_work - x1, p=2, dim=-1).cpu(),
    }
    return reps


def _build_adjacency_csr(edge_index: torch.Tensor, num_nodes: int) -> sp.csr_matrix:
    edge_cpu = edge_index.detach().cpu().numpy()
    values = np.ones(edge_cpu.shape[1], dtype=np.bool_)
    adj = sp.coo_matrix((values, (edge_cpu[0], edge_cpu[1])), shape=(num_nodes, num_nodes))
    adj = (adj + adj.T).astype(np.bool_).tocsr()
    adj.setdiag(False)
    adj.eliminate_zeros()
    return adj


def _sample_random_nonedge(
    rng: np.random.Generator,
    node_id: int,
    candidate_count: int,
    num_nodes: int,
    neighbors: np.ndarray,
) -> np.ndarray:
    if candidate_count <= 0:
        return np.empty((0,), dtype=np.int64)

    sampled = np.empty((candidate_count,), dtype=np.int64)
    filled = 0
    while filled < candidate_count:
        remaining = candidate_count - filled
        proposal = rng.integers(0, num_nodes - 1, size=remaining * 2)
        proposal = np.where(proposal >= node_id, proposal + 1, proposal)
        is_neighbor = np.isin(proposal, neighbors, assume_unique=False)
        valid = proposal[~is_neighbor]
        take = min(valid.size, remaining)
        if take > 0:
            sampled[filled:filled + take] = valid[:take]
            filled += take
    return sampled


def _topk_nonedge_scores(
    reps: dict[str, torch.Tensor],
    adj_csr: sp.csr_matrix,
    max_k: int,
    work_device: torch.device,
    row_block_size: int,
    col_block_size: int,
):
    num_nodes = next(iter(reps.values())).size(0)
    top_values = torch.full((num_nodes, max_k), float("-inf"), dtype=torch.float32)
    top_indices = torch.full((num_nodes, max_k), -1, dtype=torch.long)

    rep_names = ("raw", "low", "mid", "high")

    for row_start in range(0, num_nodes, row_block_size):
        row_end = min(row_start + row_block_size, num_nodes)
        row_size = row_end - row_start

        block_values = torch.full((row_size, max_k), float("-inf"), device=work_device)
        block_indices = torch.full((row_size, max_k), -1, device=work_device, dtype=torch.long)

        row_chunks = {
            name: reps[name][row_start:row_end].to(work_device)
            for name in rep_names
        }

        for col_start in range(0, num_nodes, col_block_size):
            col_end = min(col_start + col_block_size, num_nodes)
            col_size = col_end - col_start

            scores = None
            for name in rep_names:
                row_chunk = row_chunks[name]
                col_chunk = reps[name][col_start:col_end].to(work_device)
                partial = row_chunk @ col_chunk.T
                scores = partial if scores is None else scores + partial
            scores = scores / float(len(rep_names))

            invalid = torch.from_numpy(
                adj_csr[row_start:row_end, col_start:col_end].toarray()
            ).to(work_device)
            scores = scores.masked_fill(invalid, float("-inf"))

            if row_start < col_end and col_start < row_end:
                diag_start = max(row_start, col_start)
                diag_end = min(row_end, col_end)
                diag_len = max(0, diag_end - diag_start)
                if diag_len > 0:
                    row_ids = torch.arange(diag_start - row_start, diag_start - row_start + diag_len, device=work_device)
                    col_ids = torch.arange(diag_start - col_start, diag_start - col_start + diag_len, device=work_device)
                    scores[row_ids, col_ids] = float("-inf")

            cand_values, cand_local_idx = torch.topk(scores, k=min(max_k, col_size), dim=1)
            cand_indices = cand_local_idx + col_start

            merged_values = torch.cat([block_values, cand_values], dim=1)
            merged_indices = torch.cat([block_indices, cand_indices], dim=1)
            best_values, best_pos = torch.topk(merged_values, k=max_k, dim=1)
            best_indices = torch.gather(merged_indices, 1, best_pos)
            best_indices = best_indices.masked_fill(~torch.isfinite(best_values), -1)

            block_values = best_values
            block_indices = best_indices

        top_values[row_start:row_end] = block_values.cpu()
        top_indices[row_start:row_end] = block_indices.cpu()

    return top_values, top_indices


def run_lcsr_nonedge_diagnostic(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    dataset_name: str,
    topks: list[int],
    filter_k: int,
    logger=None,
    seed: int = 0,
    work_device: torch.device | None = None,
    row_block_size: int = 256,
    col_block_size: int = 2048,
):
    if not topks:
        raise ValueError("topks must not be empty")

    topks = sorted({int(k) for k in topks if int(k) > 0})
    max_k = max(topks)
    num_nodes = x.size(0)

    if work_device is None:
        work_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if logger is not None:
        logger.info(
            f"[FCRS non-edge diag] start dataset={dataset_name} "
            f"filter_k={filter_k} topks={topks} device={work_device}"
        )

    reps = _prepare_representations(
        x=x,
        edge_index=edge_index,
        filter_k=filter_k,
        work_device=work_device,
    )
    adj_csr = _build_adjacency_csr(edge_index=edge_index, num_nodes=num_nodes)
    top_values, top_indices = _topk_nonedge_scores(
        reps=reps,
        adj_csr=adj_csr,
        max_k=max_k,
        work_device=work_device,
        row_block_size=row_block_size,
        col_block_size=col_block_size,
    )

    y_cpu = y.detach().cpu()
    valid_nodes = torch.ones(num_nodes, dtype=torch.bool)
    if torch.is_floating_point(y_cpu):
        valid_nodes = ~torch.isnan(y_cpu)

    edge_src = edge_index[0].detach().cpu()
    edge_dst = edge_index[1].detach().cpu()
    edge_valid = valid_nodes[edge_src] & valid_nodes[edge_dst]
    edge_same = (y_cpu[edge_src][edge_valid] == y_cpu[edge_dst][edge_valid]).float().mean().item()

    rng = np.random.default_rng(seed)
    degree = np.diff(adj_csr.indptr)

    results = {}
    for k in topks:
        selected_idx = top_indices[:, :k]
        selected_val = top_values[:, :k]
        selected_mask = (selected_idx >= 0) & torch.isfinite(selected_val)

        anchor_ids = torch.arange(num_nodes).unsqueeze(1).expand(-1, k)
        flat_anchor = anchor_ids[selected_mask]
        flat_partner = selected_idx[selected_mask]
        flat_score = selected_val[selected_mask]

        coverage = int((selected_mask.sum(dim=1) > 0).sum().item())
        selected = int(flat_partner.numel())

        pair_valid = valid_nodes[flat_anchor] & valid_nodes[flat_partner]
        same = (y_cpu[flat_anchor][pair_valid] == y_cpu[flat_partner][pair_valid]).float()
        same_ratio = same.mean().item() if same.numel() > 0 else float("nan")

        random_same_parts = []
        selected_per_node = selected_mask.sum(dim=1).cpu().numpy()
        for node_id, take in enumerate(selected_per_node.tolist()):
            if take <= 0 or not valid_nodes[node_id]:
                continue
            candidate_count = int(num_nodes - 1 - degree[node_id])
            if candidate_count <= 0:
                continue
            neighbors = adj_csr.indices[adj_csr.indptr[node_id]:adj_csr.indptr[node_id + 1]]
            sampled = _sample_random_nonedge(
                rng=rng,
                node_id=node_id,
                candidate_count=take,
                num_nodes=num_nodes,
                neighbors=neighbors,
            )
            partner = torch.from_numpy(sampled)
            valid = valid_nodes[partner]
            if valid.any():
                same_part = (y_cpu[partner][valid] == y_cpu[node_id]).float()
                random_same_parts.append(same_part)
        if random_same_parts:
            random_same = torch.cat(random_same_parts).mean().item()
        else:
            random_same = float("nan")

        ranked_valid = pair_valid & torch.isfinite(flat_score)
        valid_scores = flat_score[ranked_valid]
        valid_same = (y_cpu[flat_anchor][ranked_valid] == y_cpu[flat_partner][ranked_valid]).float()
        if valid_scores.numel() > 0:
            q = max(1, int(np.ceil(valid_scores.numel() * 0.2)))
            order = torch.argsort(valid_scores)
            bottom_same = valid_same[order[:q]].mean().item()
            top_same = valid_same[order[-q:]].mean().item()
            gap = top_same - bottom_same
        else:
            top_same = float("nan")
            bottom_same = float("nan")
            gap = float("nan")

        item = {
            "selected": selected,
            "coverage": coverage,
            "same": same_ratio,
            "random": random_same,
            "edge_same": edge_same,
            "top20": top_same,
            "bottom20": bottom_same,
            "gap": gap,
        }
        results[k] = item

        if logger is not None:
            logger.info(
                "[FCRS non-edge diag] "
                f"dataset={dataset_name} score=mu k={k} "
                f"selected={selected} coverage={coverage} "
                f"same={same_ratio:.4f} random={random_same:.4f} edge_same={edge_same:.4f} "
                f"top20={top_same:.4f} bottom20={bottom_same:.4f} gap={gap:.4f}"
            )

    baseline = {
        "num_nodes": num_nodes,
        "num_edges": int(edge_index.size(1)),
        "valid_label_nodes": int(valid_nodes.sum().item()),
    }
    return NonEdgeDiagResult(
        dataset=dataset_name,
        score="mu",
        filter_k=filter_k,
        topks=topks,
        results=results,
        baseline=baseline,
    )


run_fcrs_nonedge_diagnostic = run_lcsr_nonedge_diagnostic
