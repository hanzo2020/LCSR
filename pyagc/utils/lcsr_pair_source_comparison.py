from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from pyagc.utils.lcsr_consensus import compute_consensus_score
from pyagc.utils.lcsr_nonedge_diag import (
    _build_adjacency_csr,
    _prepare_representations,
    _sample_random_nonedge,
)


@dataclass
class PairSourceComparisonResult:
    dataset: str
    filter_k: int
    topks: list[int]
    sources: dict[str, dict[str, Any]]
    baseline: dict[str, Any]


def prepare_pair_source_representations(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    embeddings: torch.Tensor,
    filter_k: int,
    work_device: torch.device,
):
    reps = _prepare_representations(
        x=x,
        edge_index=edge_index,
        filter_k=filter_k,
        work_device=work_device,
    )
    reps["final_embedding"] = F.normalize(embeddings.detach().to(work_device), p=2, dim=-1).cpu()
    return reps


def _topk_nonedge_from_reps(
    reps: list[torch.Tensor],
    adj_csr,
    max_k: int,
    work_device: torch.device,
    row_block_size: int,
    col_block_size: int,
):
    num_nodes = reps[0].size(0)
    top_values = torch.full((num_nodes, max_k), float("-inf"), dtype=torch.float32)
    top_indices = torch.full((num_nodes, max_k), -1, dtype=torch.long)

    for row_start in range(0, num_nodes, row_block_size):
        row_end = min(row_start + row_block_size, num_nodes)
        row_size = row_end - row_start

        block_values = torch.full((row_size, max_k), float("-inf"), device=work_device)
        block_indices = torch.full((row_size, max_k), -1, dtype=torch.long, device=work_device)
        row_chunks = [rep[row_start:row_end].to(work_device) for rep in reps]

        for col_start in range(0, num_nodes, col_block_size):
            col_end = min(col_start + col_block_size, num_nodes)
            col_reps = [rep[col_start:col_end].to(work_device) for rep in reps]

            scores = None
            for row_rep, col_rep in zip(row_chunks, col_reps):
                part = row_rep @ col_rep.T
                scores = part if scores is None else scores + part
            scores = scores / float(len(reps))

            invalid = torch.from_numpy(
                adj_csr[row_start:row_end, col_start:col_end].toarray()
            ).to(work_device)
            scores = scores.masked_fill(invalid, float("-inf"))

            if row_start < col_end and col_start < row_end:
                diag_start = max(row_start, col_start)
                diag_end = min(row_end, col_end)
                diag_len = max(0, diag_end - diag_start)
                if diag_len > 0:
                    row_ids = torch.arange(
                        diag_start - row_start,
                        diag_start - row_start + diag_len,
                        device=work_device,
                    )
                    col_ids = torch.arange(
                        diag_start - col_start,
                        diag_start - col_start + diag_len,
                        device=work_device,
                    )
                    scores[row_ids, col_ids] = float("-inf")

            cand_values, cand_local_idx = torch.topk(scores, k=min(max_k, scores.size(1)), dim=1)
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


def _topk_nonedge_from_score_fn(
    reps: dict[str, torch.Tensor],
    adj_csr,
    max_k: int,
    work_device: torch.device,
    row_block_size: int,
    col_block_size: int,
    score_fn,
):
    num_nodes = reps["raw"].size(0)
    top_values = torch.full((num_nodes, max_k), float("-inf"), dtype=torch.float32)
    top_indices = torch.full((num_nodes, max_k), -1, dtype=torch.long)

    for row_start in range(0, num_nodes, row_block_size):
        row_end = min(row_start + row_block_size, num_nodes)
        row_size = row_end - row_start

        block_values = torch.full((row_size, max_k), float("-inf"), device=work_device)
        block_indices = torch.full((row_size, max_k), -1, dtype=torch.long, device=work_device)
        row_reps = {
            name: rep[row_start:row_end].to(work_device)
            for name, rep in reps.items()
        }

        for col_start in range(0, num_nodes, col_block_size):
            col_end = min(col_start + col_block_size, num_nodes)
            col_reps = {
                name: rep[col_start:col_end].to(work_device)
                for name, rep in reps.items()
            }

            scores = score_fn(row_reps, col_reps)

            invalid = torch.from_numpy(
                adj_csr[row_start:row_end, col_start:col_end].toarray()
            ).to(work_device)
            scores = scores.masked_fill(invalid, float("-inf"))

            if row_start < col_end and col_start < row_end:
                diag_start = max(row_start, col_start)
                diag_end = min(row_end, col_end)
                diag_len = max(0, diag_end - diag_start)
                if diag_len > 0:
                    row_ids = torch.arange(
                        diag_start - row_start,
                        diag_start - row_start + diag_len,
                        device=work_device,
                    )
                    col_ids = torch.arange(
                        diag_start - col_start,
                        diag_start - col_start + diag_len,
                        device=work_device,
                    )
                    scores[row_ids, col_ids] = float("-inf")

            cand_values, cand_local_idx = torch.topk(scores, k=min(max_k, scores.size(1)), dim=1)
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


def _topk_random_nonedge(
    adj_csr,
    num_nodes: int,
    max_k: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    degree = np.diff(adj_csr.indptr)
    top_indices = torch.full((num_nodes, max_k), -1, dtype=torch.long)
    top_values = torch.full((num_nodes, max_k), float("-inf"), dtype=torch.float32)

    for node_id in range(num_nodes):
        candidate_count = int(num_nodes - 1 - degree[node_id])
        if candidate_count <= 0:
            continue
        take = min(max_k, candidate_count)
        neighbors = adj_csr.indices[adj_csr.indptr[node_id]:adj_csr.indptr[node_id + 1]]
        sampled = _sample_random_nonedge(
            rng=rng,
            node_id=node_id,
            candidate_count=take,
            num_nodes=num_nodes,
            neighbors=neighbors,
        )
        scores = rng.random(take, dtype=np.float32)
        order = np.argsort(scores)
        sampled = sampled[order]
        scores = scores[order]
        top_indices[node_id, :take] = torch.from_numpy(sampled)
        top_values[node_id, :take] = torch.from_numpy(scores)

    return top_values, top_indices


def select_nonedge_topk_by_source(
    source_name: str,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    embeddings: torch.Tensor | None,
    filter_k: int,
    topk: int,
    seed: int,
    consensus: str = "mean",
    work_device: torch.device | None = None,
    row_block_size: int = 256,
    col_block_size: int = 2048,
):
    if work_device is None:
        work_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    adj_csr = _build_adjacency_csr(edge_index=edge_index, num_nodes=x.size(0))

    if source_name == "random_nonedge":
        return _topk_random_nonedge(
            adj_csr=adj_csr,
            num_nodes=x.size(0),
            max_k=topk,
            seed=seed,
        )

    reps = prepare_pair_source_representations(
        x=x,
        edge_index=edge_index,
        embeddings=x if embeddings is None else embeddings,
        filter_k=filter_k,
        work_device=work_device,
    )

    source_map = {
        "raw": [reps["raw"]],
        "raw_cosine": [reps["raw"]],
        "semantic_identity": [reps["raw"]],
        "lowpass": [reps["low"]],
        "lowpass_cosine": [reps["low"]],
        "fcrs_mu": [reps["raw"], reps["low"], reps["mid"], reps["high"]],
        "lcsr_mu": [reps["raw"], reps["low"], reps["mid"], reps["high"]],
        "final_embedding": [reps["final_embedding"]],
        "final_embedding_cosine": [reps["final_embedding"]],
    }
    if source_name in {"raw_mul_fcrs", "raw_mul_lcsr", "semantic_frequency"}:
        return _topk_nonedge_from_score_fn(
            reps={
                "raw": reps["raw"],
                "low": reps["low"],
                "mid": reps["mid"],
                "high": reps["high"],
            },
            adj_csr=adj_csr,
            max_k=topk,
            work_device=work_device,
            row_block_size=row_block_size,
            col_block_size=col_block_size,
            score_fn=lambda row_reps, col_reps: compute_consensus_score(
                row_reps["raw"] @ col_reps["raw"].T,
                row_reps["low"] @ col_reps["low"].T,
                row_reps["mid"] @ col_reps["mid"].T,
                row_reps["high"] @ col_reps["high"].T,
                consensus=consensus,
            )[0],
        )
    if source_name not in source_map:
        raise ValueError(f"Unknown source_name: {source_name}")

    return _topk_nonedge_from_reps(
        reps=source_map[source_name],
        adj_csr=adj_csr,
        max_k=topk,
        work_device=work_device,
        row_block_size=row_block_size,
        col_block_size=col_block_size,
    )


def evaluate_nonedge_source(
    source_name: str,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    embeddings: torch.Tensor | None,
    filter_k: int,
    topks: list[int],
    seed: int,
    consensus: str = "mean",
    work_device: torch.device | None = None,
    row_block_size: int = 256,
    col_block_size: int = 2048,
):
    topks = sorted({int(k) for k in topks if int(k) > 0})
    if not topks:
        raise ValueError("topks must not be empty")

    if work_device is None:
        work_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    adj_csr = _build_adjacency_csr(edge_index=edge_index, num_nodes=x.size(0))
    top_values, top_indices = select_nonedge_topk_by_source(
        source_name=source_name,
        x=x,
        edge_index=edge_index,
        embeddings=embeddings,
        filter_k=filter_k,
        topk=max(topks),
        seed=seed,
        consensus=consensus,
        work_device=work_device,
        row_block_size=row_block_size,
        col_block_size=col_block_size,
    )
    return _evaluate_selection(
        top_values=top_values,
        top_indices=top_indices,
        y=y,
        edge_index=edge_index,
        adj_csr=adj_csr,
        topks=topks,
        seed=seed + 17,
    )


def _evaluate_selection(
    top_values: torch.Tensor,
    top_indices: torch.Tensor,
    y: torch.Tensor,
    edge_index: torch.Tensor,
    adj_csr,
    topks: list[int],
    seed: int,
):
    num_nodes = y.size(0)
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
        mean_score = float(flat_score.mean().item()) if flat_score.numel() > 0 else float("nan")

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
                random_same_parts.append((y_cpu[partner][valid] == y_cpu[node_id]).float())
        random_same = torch.cat(random_same_parts).mean().item() if random_same_parts else float("nan")

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
            bottom_same = float("nan")
            top_same = float("nan")
            gap = float("nan")

        results[k] = {
            "selected": selected,
            "coverage": coverage,
            "mean_score": mean_score,
            "same": same_ratio,
            "random": random_same,
            "edge_same": edge_same,
            "top20": top_same,
            "bottom20": bottom_same,
            "gap": gap,
        }

    return results


def run_pair_source_comparison(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    embeddings: torch.Tensor,
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

    if work_device is None:
        work_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    topks = sorted({int(k) for k in topks if int(k) > 0})
    max_k = max(topks)

    reps = prepare_pair_source_representations(
        x=x,
        edge_index=edge_index,
        embeddings=embeddings,
        filter_k=filter_k,
        work_device=work_device,
    )
    adj_csr = _build_adjacency_csr(edge_index=edge_index, num_nodes=x.size(0))

    source_defs = {
        "random_nonedge": None,
        "raw_cosine": [reps["raw"]],
        "lowpass_cosine": [reps["low"]],
        "lcsr_mu": [reps["raw"], reps["low"], reps["mid"], reps["high"]],
        "final_embedding_cosine": [reps["final_embedding"]],
    }

    source_results = {}
    for offset, (source_name, source_reps) in enumerate(source_defs.items()):
        if logger is not None:
            logger.info(
                f"[FCRS pair source] start dataset={dataset_name} source={source_name} "
                f"filter_k={filter_k} topks={topks} device={work_device}"
            )

        if source_reps is None:
            top_values, top_indices = _topk_random_nonedge(
                adj_csr=adj_csr,
                num_nodes=x.size(0),
                max_k=max_k,
                seed=seed + offset * 9973,
            )
        else:
            top_values, top_indices = _topk_nonedge_from_reps(
                reps=source_reps,
                adj_csr=adj_csr,
                max_k=max_k,
                work_device=work_device,
                row_block_size=row_block_size,
                col_block_size=col_block_size,
            )

        metrics = _evaluate_selection(
            top_values=top_values,
            top_indices=top_indices,
            y=y,
            edge_index=edge_index,
            adj_csr=adj_csr,
            topks=topks,
            seed=seed + offset * 9973 + 17,
        )
        source_results[source_name] = metrics

        if logger is not None:
            for k in topks:
                item = metrics[k]
                logger.info(
                    "[FCRS pair source] "
                    f"dataset={dataset_name} source={source_name} k={k} "
                    f"selected={item['selected']} coverage={item['coverage']} "
                    f"same={item['same']:.4f} random={item['random']:.4f} edge_same={item['edge_same']:.4f} "
                    f"top20={item['top20']:.4f} bottom20={item['bottom20']:.4f} gap={item['gap']:.4f}"
                )

    baseline = {
        "num_nodes": int(x.size(0)),
        "num_edges": int(edge_index.size(1)),
    }
    return PairSourceComparisonResult(
        dataset=dataset_name,
        filter_k=filter_k,
        topks=topks,
        sources=source_results,
        baseline=baseline,
    )
