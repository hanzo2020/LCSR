import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from time import perf_counter
from torch_geometric.data import Data
from torch.nn import Module
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import add_remaining_self_loops, to_undirected

from pyagc.models.base import LossOutput
from pyagc.models.ns4gc import NS4GC
from pyagc.utils.lcsr_consensus import compute_consensus_score, map_cosine_to_unit_interval
from pyagc.utils.lcsr import _match_swap_records


class FCRS_AGC(NS4GC):
    r"""Behaviorally identical shell of NS4GC for FCRS diagnostics.

    The current FCRS_AGC benchmark intentionally preserves the exact NS4GC
    training path. Any FCRS-specific analysis should be implemented as
    post-hoc diagnostics outside of the training loss / forward pass.
    """

    def __init__(
        self,
        encoder: Module,
        transform1: BaseTransform,
        transform2: BaseTransform,
        s: float = 0.6,
        tau: float = 0.1,
        lam: float = 1.0,
        gam: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            encoder=encoder,
            transform1=transform1,
            transform2=transform2,
            s=s,
            tau=tau,
            lam=lam,
            gam=gam,
        )
        self.extra_config = kwargs
        self.fcrs_extra_loss = kwargs.get("fcrs_extra_loss", False)
        self.fcrs_extra_source = kwargs.get("fcrs_extra_source", "lcsr_mu")
        self.fcrs_extra_mode = kwargs.get("fcrs_extra_mode", "plain")
        self.fcrs_consensus = kwargs.get("fcrs_consensus", "mean")
        self.fcrs_candidate_source = kwargs.get("fcrs_candidate_source", "raw")
        self.fcrs_weight_source = kwargs.get("fcrs_weight_source", "none")
        self.fcrs_lcb_rho = float(kwargs.get("fcrs_lcb_rho", 1.0))
        self.fcrs_extra_k = int(kwargs.get("fcrs_extra_k", 1))
        self.fcrs_extra_lambda = float(kwargs.get("fcrs_extra_lambda", 0.0))
        self.fcrs_extra_warmup = int(kwargs.get("fcrs_extra_warmup", 0))
        self.fcrs_extra_ramp_epochs = int(kwargs.get("fcrs_extra_ramp_epochs", 0))
        self.fcrs_positive_loss = kwargs.get("fcrs_positive_loss", "linear")
        self.fcrs_positive_margin = float(kwargs.get("fcrs_positive_margin", 0.8))
        self.fcrs_positive_temperature = float(kwargs.get("fcrs_positive_temperature", 0.05))
        self.fcrs_positive_quantile = float(kwargs.get("fcrs_positive_quantile", 0.2))
        self.fcrs_saturation_tau = float(kwargs.get("fcrs_saturation_tau", 0.80))
        self.fcrs_saturation_temp = float(kwargs.get("fcrs_saturation_temp", 0.03))
        self.fcrs_plus_dropout = float(kwargs.get("fcrs_plus_dropout", 0.0))
        self.fcrs_filter_k = int(kwargs.get("fcrs_filter_k", 1))
        self.fcrs_batch_local_admission = bool(kwargs.get("fcrs_batch_local_admission", False))
        self.fcrs_risk_budget = bool(kwargs.get("fcrs_risk_budget", False))
        self.fcrs_extra_top_indices = None
        self.fcrs_extra_pair_index = None
        self.fcrs_positive_pair_index = None
        self.fcrs_release_pair_index = None
        self.fcrs_release_pair_keys = None
        self.fcrs_release_pair_weights = None
        self.fcrs_release_hard = True
        self.fcrs_extra_pair_weights = None
        self.fcrs_positive_pair_keys = None
        self.fcrs_positive_pair_weights = None
        self.fcrs_num_nodes = None
        self.fcrs_batch_row_block = int(kwargs.get(
            "fcrs_batch_row_block",
            512 if self.fcrs_batch_local_admission else 256,
        ))
        self.fcrs_batch_col_block = int(kwargs.get(
            "fcrs_batch_col_block",
            8192 if self.fcrs_batch_local_admission else 2048,
        ))
        self.fcrs_complete_mode = bool(kwargs.get("fcrs_complete_mode", False))
        self.fcrs_message_passing_edges_unchanged = True
        self.fcrs_a_minus_not_used_as_negative = True
        self.fcrs_positive_release_applied = False
        self.fcrs_last_complete_stats = None
        self.fcrs_runtime_profile = bool(kwargs.get("fcrs_runtime_profile", False))
        self.fcrs_runtime_profile_batches = int(kwargs.get("fcrs_runtime_profile_batches", 10))
        self.fcrs_runtime_profile_only = bool(kwargs.get("fcrs_runtime_profile_only", False))
        self.fcrs_runtime_profile_rows = []
        self.fcrs_runtime_profile_meta = {}
        self._fcrs_runtime_pending_data_transfer_s = None
        self._fcrs_runtime_active = None
        self._fcrs_runtime_backward_start = None
        self._fcrs_runtime_finalize_ctx = None
        self.fcrs_admission_audit = bool(kwargs.get("fcrs_admission_audit", False))
        self.fcrs_admission_audit_batches = int(kwargs.get("fcrs_admission_audit_batches", 20))
        self.fcrs_admission_audit_only = bool(kwargs.get("fcrs_admission_audit_only", False))
        self.fcrs_admission_audit_rows = []
        self.fcrs_batch_local_semantics = kwargs.get("fcrs_batch_local_semantics", "legacy_topk")
        self.fcrs_lcsr_candidate_pool_size = int(kwargs.get("fcrs_lcsr_candidate_pool_size", self.fcrs_extra_k))
        self.fcrs_lcsr_margin = float(kwargs.get("fcrs_lcsr_margin", 0.0))
        self.fcrs_lcsr_rho = float(kwargs.get("fcrs_lcsr_rho", 1.0))
        self.fcrs_lcsr_kmax = int(kwargs.get("fcrs_lcsr_kmax", self.fcrs_extra_k))
        self.fcrs_lcsr_budget_match = bool(kwargs.get("fcrs_lcsr_budget_match", False))
        self.fcrs_lcsr_disable_local_calibration = bool(kwargs.get("fcrs_lcsr_disable_local_calibration", False))
        self.fcrs_lcsr_disable_mutual = bool(kwargs.get("fcrs_lcsr_disable_mutual", False))
        self.fcrs_global_degree = None
        self.fcrs_candidate_bank = None
        self.fcrs_candidate_bank_meta = {}

    def _runtime_sync(self, device: torch.device | None = None):
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                return
        if isinstance(device, torch.device) and device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)

    def _runtime_now(self, device: torch.device | None = None) -> float:
        self._runtime_sync(device)
        return perf_counter()

    def _runtime_profile_enabled_for_batch(self, current_epoch: int | None) -> bool:
        return (
            self.fcrs_runtime_profile
            and self.fcrs_batch_local_admission
            and self.fcrs_extra_loss
            and current_epoch is not None
            and current_epoch > self.fcrs_extra_warmup
            and len(self.fcrs_runtime_profile_rows) < self.fcrs_runtime_profile_batches
        )

    def _runtime_profile_add(self, key: str, seconds: float):
        if self._fcrs_runtime_active is None:
            return
        self._fcrs_runtime_active[key] = self._fcrs_runtime_active.get(key, 0.0) + float(seconds)

    def capture_runtime_profile_data_transfer(self, data_wait_s: float):
        self._fcrs_runtime_pending_data_transfer_s = float(data_wait_s)

    def begin_runtime_profile_backward(self):
        if self._fcrs_runtime_active is None:
            self._fcrs_runtime_backward_start = None
            return
        device = self._fcrs_runtime_active.get("device")
        self._fcrs_runtime_backward_start = self._runtime_now(device)

    def end_runtime_profile_backward(self):
        if self._fcrs_runtime_active is None or self._fcrs_runtime_backward_start is None:
            return
        device = self._fcrs_runtime_active.get("device")
        elapsed = self._runtime_now(device) - self._fcrs_runtime_backward_start
        self._runtime_profile_add("backward_optimizer_s", elapsed)
        self._fcrs_runtime_backward_start = None
        if self._fcrs_runtime_finalize_ctx is not None:
            batch, batch_local_stats, pair_index = self._fcrs_runtime_finalize_ctx
            self._runtime_profile_finish_batch(batch=batch, batch_local_stats=batch_local_stats, pair_index=pair_index)
            self._fcrs_runtime_finalize_ctx = None

    def runtime_profile_should_stop(self) -> bool:
        return self.should_stop_training_early()

    def should_stop_training_early(self) -> bool:
        runtime_done = self.fcrs_runtime_profile_only and len(self.fcrs_runtime_profile_rows) >= self.fcrs_runtime_profile_batches
        audit_done = self.fcrs_admission_audit_only and len(self.fcrs_admission_audit_rows) >= self.fcrs_admission_audit_batches
        return bool(runtime_done or audit_done)

    def _runtime_profile_start_batch(self, current_epoch: int | None, device: torch.device):
        if not self._runtime_profile_enabled_for_batch(current_epoch):
            self._fcrs_runtime_active = None
            self._fcrs_runtime_pending_data_transfer_s = None
            self._fcrs_runtime_finalize_ctx = None
            return False
        self._fcrs_runtime_active = {
            "epoch": int(current_epoch),
            "device": device,
            "dataloader_data_transfer_s": float(self._fcrs_runtime_pending_data_transfer_s or 0.0),
        }
        self._fcrs_runtime_pending_data_transfer_s = None
        return True

    def _runtime_profile_finish_batch(self, batch: Data, batch_local_stats: dict[str, float], pair_index: Tensor | None):
        if self._fcrs_runtime_active is None:
            return
        extra_pairs = float(batch_local_stats.get("extra_pairs", 0.0))
        if extra_pairs <= 0:
            self._fcrs_runtime_active = None
            return
        row = dict(self._fcrs_runtime_active)
        row.pop("device", None)
        row["batch_size"] = int(getattr(batch, "batch_size", 0))
        row["num_nodes"] = int(batch.x.size(0))
        row["num_edges"] = int(batch.edge_index.size(1))
        row["extra_pairs"] = extra_pairs
        row["pair_count"] = int(pair_index.size(1)) if pair_index is not None else 0
        total = 0.0
        for key, value in row.items():
            if key.endswith("_s"):
                total += float(value)
        row["profiled_total_s"] = total
        self.fcrs_runtime_profile_rows.append(row)
        self._fcrs_runtime_active = None

    def get_runtime_profile_summary(self):
        rows = self.fcrs_runtime_profile_rows
        if not rows:
            return None
        module_keys = [
            "dataloader_data_transfer_s",
            "ns4gc_forward_s",
            "ns4gc_base_loss_s",
            "lcsr_frequency_descriptor_s",
            "lcsr_candidate_generation_scoring_s",
            "lcsr_topk_admission_pair_assembly_s",
            "lcsr_pair_loss_s",
            "diagnostics_logging_s",
            "backward_optimizer_s",
        ]
        means = {}
        total = 0.0
        for key in module_keys:
            vals = [float(row.get(key, 0.0)) for row in rows]
            mean_val = float(sum(vals) / max(len(vals), 1))
            means[key] = mean_val
            total += mean_val
        ranked = sorted(
            (
                {
                    "module": key,
                    "mean_s": means[key],
                    "share": (means[key] / total) if total > 0 else 0.0,
                }
                for key in module_keys
            ),
            key=lambda item: item["mean_s"],
            reverse=True,
        )
        meta = {
            "profiled_batches": len(rows),
            "batch_size_mean": float(sum(row.get("batch_size", 0) for row in rows) / len(rows)),
            "num_nodes_mean": float(sum(row.get("num_nodes", 0) for row in rows) / len(rows)),
            "num_edges_mean": float(sum(row.get("num_edges", 0) for row in rows) / len(rows)),
            "extra_pairs_mean": float(sum(row.get("extra_pairs", 0.0) for row in rows) / len(rows)),
            "pair_count_mean": float(sum(row.get("pair_count", 0) for row in rows) / len(rows)),
            "profiled_total_s_mean": float(sum(row.get("profiled_total_s", 0.0) for row in rows) / len(rows)),
        }
        meta.update(self.fcrs_runtime_profile_meta)
        return {
            "rows": rows,
            "modules": means,
            "ranked": ranked,
            "meta": meta,
        }

    def _admission_audit_enabled_for_batch(self, current_epoch: int | None) -> bool:
        return (
            self.fcrs_admission_audit
            and self.fcrs_batch_local_admission
            and self.fcrs_extra_loss
            and current_epoch is not None
            and current_epoch > self.fcrs_extra_warmup
            and len(self.fcrs_admission_audit_rows) < self.fcrs_admission_audit_batches
        )

    def set_global_degree(self, degree: Tensor | np.ndarray | None):
        if degree is None:
            self.fcrs_global_degree = None
            return
        if isinstance(degree, np.ndarray):
            self.fcrs_global_degree = torch.from_numpy(degree.astype(np.int64, copy=False))
            return
        self.fcrs_global_degree = degree.detach().cpu().long()

    def set_batch_local_candidate_bank(self, candidate_bank: Tensor | None, meta: dict | None = None):
        self.fcrs_candidate_bank = None if candidate_bank is None else candidate_bank.detach().cpu().long()
        self.fcrs_candidate_bank_meta = {} if meta is None else dict(meta)

    def get_admission_audit_summary(self):
        if not self.fcrs_admission_audit_rows:
            return None
        max_count = 0
        for row in self.fcrs_admission_audit_rows:
            if row["admitted_count_per_anchor"]:
                max_count = max(max_count, max(row["admitted_count_per_anchor"]))
        k = max(max_count, int(self.fcrs_extra_k), int(self.fcrs_lcsr_kmax), 1)
        total_anchors = 0
        hist = {str(i): 0 for i in range(k + 1)}
        all_admitted_scores = []
        rows = self.fcrs_admission_audit_rows
        for row in rows:
            total_anchors += len(row["admitted_count_per_anchor"])
            for key, value in row["admitted_fraction_by_count"].items():
                hist[str(key)] = hist.get(str(key), 0) + int(round(float(value) * len(row["admitted_count_per_anchor"])))
            all_admitted_scores.extend(row["admitted_scores"])
        fraction_by_count = {
            key: (value / total_anchors if total_anchors > 0 else 0.0)
            for key, value in sorted(hist.items(), key=lambda item: int(item[0]))
        }
        quantiles = {}
        if all_admitted_scores:
            scores_t = torch.tensor(all_admitted_scores, dtype=torch.float32)
            for q in [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]:
                quantiles[f"q{int(q*100):02d}"] = float(torch.quantile(scores_t, q).item())
        return {
            "config": {
                "batch_local_semantics": self.fcrs_batch_local_semantics,
                "fcrs_extra_k": int(self.fcrs_extra_k),
                "fcrs_extra_warmup": int(self.fcrs_extra_warmup),
                "fcrs_extra_source": self.fcrs_extra_source,
                "fcrs_risk_budget": bool(self.fcrs_risk_budget),
                "lcsr_candidate_pool_size": int(self.fcrs_lcsr_candidate_pool_size),
                "lcsr_margin": float(self.fcrs_lcsr_margin),
                "lcsr_rho": float(self.fcrs_lcsr_rho),
                "lcsr_kmax": int(self.fcrs_lcsr_kmax),
                "lcsr_budget_match": bool(self.fcrs_lcsr_budget_match),
            },
            "effective_batches": len(rows),
            "batches": rows,
            "aggregate": {
                "anchors_total": total_anchors,
                "admitted_fraction_by_count": fraction_by_count,
                "admitted_score_quantiles": quantiles,
            },
        }

    def _resolve_support_source(self) -> str:
        source = self.fcrs_extra_source
        if source == "semantic_frequency":
            return "freq"
        if source == "raw":
            return "raw"
        if source in {"fcrs_mu", "lcsr_mu"}:
            return "mu"
        if source in {"raw_mul_fcrs", "raw_mul_lcsr"}:
            return "raw_mul_mu"
        if source == "source_adaptive":
            return "source_adaptive"
        return "freq"

    def _batch_source_components(self, row_reps: dict[str, Tensor], col_reps: dict[str, Tensor]):
        s_id = row_reps["id"] @ col_reps["id"].T
        s_low = row_reps["low"] @ col_reps["low"].T
        s_mid = row_reps["mid"] @ col_reps["mid"].T
        s_high = row_reps["high"] @ col_reps["high"].T
        raw_u = map_cosine_to_unit_interval(s_id)
        low_u = map_cosine_to_unit_interval(s_low)
        mid_u = map_cosine_to_unit_interval(s_mid)
        high_u = map_cosine_to_unit_interval(s_high)
        mu = (raw_u + low_u + mid_u + high_u) * 0.25
        freq = raw_u * mu
        return {
            "raw": raw_u,
            "mu": mu,
            "freq": freq,
            "raw_mul_mu": raw_u * mu,
        }

    def _batch_adaptive_metadata(self, batch_size: int, num_nodes: int, edge_index: Tensor, comps: dict[str, Tensor]):
        src = edge_index[0]
        dst = edge_index[1]
        obs_mask = (src < batch_size) & (dst < batch_size) & (src != dst)
        gaps = {"raw": 0.0, "freq": 0.0, "mu": 0.0, "raw_mul_mu": 0.0}
        weights = {"raw": 0.25, "freq": 0.25, "mu": 0.25, "raw_mul_mu": 0.25}
        if not bool(obs_mask.any()):
            return gaps, weights, 0.25
        obs_src = src[obs_mask]
        obs_dst = dst[obs_mask]
        neighbors = [set() for _ in range(batch_size)]
        for u, v in zip(obs_src.tolist(), obs_dst.tolist()):
            if 0 <= u < batch_size:
                neighbors[u].add(v)
        rng = np.random.default_rng(batch_size + num_nodes + self.fcrs_filter_k)
        rand_src: list[int] = []
        rand_dst: list[int] = []
        for node_id in range(batch_size):
            for _ in range(16):
                cand = int(rng.integers(0, num_nodes))
                if cand == node_id or cand in neighbors[node_id]:
                    continue
                rand_src.append(node_id)
                rand_dst.append(cand)
                break
        if not rand_src:
            return gaps, weights, 0.25
        rand_src_t = torch.tensor(rand_src, dtype=torch.long, device=edge_index.device)
        rand_dst_t = torch.tensor(rand_dst, dtype=torch.long, device=edge_index.device)
        keys = ["raw", "freq", "mu", "raw_mul_mu"]
        values = []
        for key in keys:
            obs_mean = float(comps[key][obs_src, obs_dst].mean().item())
            rand_mean = float(comps[key][rand_src_t, rand_dst_t].mean().item())
            gaps[key] = obs_mean - rand_mean
            values.append(gaps[key])
        arr = np.asarray(values, dtype=np.float64)
        arr = arr - arr.max()
        exp = np.exp(arr)
        exp = exp / max(exp.sum(), 1e-12)
        weights = {key: float(val) for key, val in zip(keys, exp)}
        return gaps, weights, float(exp.max())

    def _effective_extra_lambda(self, current_epoch: int | None) -> float:
        base = float(self.fcrs_extra_lambda)
        if base <= 0:
            return 0.0
        if current_epoch is None:
            return base
        if current_epoch <= self.fcrs_extra_warmup:
            return 0.0
        ramp = max(int(self.fcrs_extra_ramp_epochs), 0)
        if ramp <= 0:
            return base
        progress = min(max(current_epoch - self.fcrs_extra_warmup, 0), ramp)
        return base * (float(progress) / float(ramp))

    def _apply_plus_dropout(
        self,
        pair_index: Tensor | None,
        pair_weight: Tensor | None,
        device: torch.device,
        current_epoch: int | None,
    ):
        if pair_index is None or pair_index.numel() == 0:
            return pair_index, pair_weight
        drop_prob = float(min(max(self.fcrs_plus_dropout, 0.0), 1.0))
        if drop_prob <= 0.0 or current_epoch is None:
            return pair_index, pair_weight
        keep_mask = torch.rand(pair_index.size(1), device=device) >= drop_prob
        if not bool(keep_mask.any()):
            keep_mask[torch.randint(pair_index.size(1), (1,), device=device)] = True
        pair_index = pair_index[:, keep_mask]
        if pair_weight is not None:
            pair_weight = pair_weight[keep_mask]
        return pair_index, pair_weight

    def set_extra_pairs(self, top_indices: Tensor | None, pair_weights: Tensor | None = None):
        self.fcrs_extra_pair_index = None
        if top_indices is None:
            self.fcrs_extra_top_indices = None
            self.fcrs_extra_pair_weights = None
            self.fcrs_num_nodes = None
            return
        self.fcrs_extra_top_indices = top_indices.detach().cpu().long()
        self.fcrs_extra_pair_weights = None if pair_weights is None else pair_weights.detach().cpu().float()
        self.fcrs_num_nodes = int(top_indices.size(0))

    def set_extra_pair_index(self, pair_index: Tensor | None, num_nodes: int, pair_weights: Tensor | None = None):
        self.fcrs_extra_top_indices = None
        if pair_index is None:
            self.fcrs_extra_pair_index = None
            self.fcrs_extra_pair_weights = None
            self.fcrs_num_nodes = None
            return
        self.fcrs_extra_pair_index = pair_index.detach().cpu().long()
        self.fcrs_extra_pair_weights = None if pair_weights is None else pair_weights.detach().cpu().float()
        self.fcrs_num_nodes = int(num_nodes)

    def set_positive_pair_index(self, pair_index: Tensor | None, num_nodes: int, pair_weights: Tensor | None = None):
        if pair_index is None or pair_index.numel() == 0:
            self.fcrs_positive_pair_index = None
            self.fcrs_positive_pair_keys = None
            self.fcrs_positive_pair_weights = None
            return
        pair_index = pair_index.detach().cpu().long()
        self.fcrs_positive_pair_index = pair_index
        left = torch.minimum(pair_index[0], pair_index[1])
        right = torch.maximum(pair_index[0], pair_index[1])
        self.fcrs_positive_pair_keys = left * int(num_nodes) + right
        self.fcrs_positive_pair_weights = None if pair_weights is None else pair_weights.detach().cpu().float()
        self.fcrs_num_nodes = int(num_nodes)

    def set_release_pair_index(
        self,
        pair_index: Tensor | None,
        num_nodes: int,
        pair_weights: Tensor | None = None,
        hard_release: bool = True,
    ):
        if pair_index is None or pair_index.numel() == 0:
            self.fcrs_release_pair_index = None
            self.fcrs_release_pair_keys = None
            self.fcrs_release_pair_weights = None
            self.fcrs_release_hard = True
            return
        pair_index = pair_index.detach().cpu().long()
        left = torch.minimum(pair_index[0], pair_index[1])
        right = torch.maximum(pair_index[0], pair_index[1])
        keys = left * int(num_nodes) + right
        order = torch.argsort(keys)
        self.fcrs_release_pair_index = torch.stack([left[order], right[order]], dim=0)
        self.fcrs_release_pair_keys = keys[order]
        self.fcrs_release_pair_weights = None if pair_weights is None else pair_weights.detach().cpu().float()[order]
        self.fcrs_release_hard = bool(hard_release)
        self.fcrs_num_nodes = int(num_nodes)

    def _filter_release_edges(self, edge_index: Tensor, num_nodes: int) -> Tensor:
        if (
            self.fcrs_release_pair_keys is None
            or edge_index.numel() == 0
            or not self.fcrs_release_hard
        ):
            return edge_index
        src = edge_index[0].detach().cpu().long()
        dst = edge_index[1].detach().cpu().long()
        left = torch.minimum(src, dst)
        right = torch.maximum(src, dst)
        edge_keys = left * int(num_nodes) + right
        keep_mask = ~torch.isin(edge_keys, self.fcrs_release_pair_keys)
        if bool(keep_mask.all()):
            return edge_index
        filtered = edge_index[:, keep_mask.to(edge_index.device)]
        return filtered

    def _lookup_release_edge_weights(self, edge_index: Tensor, num_nodes: int) -> Tensor | None:
        if self.fcrs_release_pair_keys is None or self.fcrs_release_pair_weights is None or edge_index.numel() == 0:
            return None
        src = edge_index[0].detach().cpu().long()
        dst = edge_index[1].detach().cpu().long()
        left = torch.minimum(src, dst)
        right = torch.maximum(src, dst)
        edge_keys = left * int(num_nodes) + right
        keys = self.fcrs_release_pair_keys
        pos = torch.searchsorted(keys, edge_keys)
        weights = torch.ones(edge_keys.size(0), dtype=torch.float32)
        valid = pos < keys.numel()
        if valid.any():
            matched_pos = pos[valid]
            matched_keys = keys[matched_pos]
            exact = matched_keys == edge_keys[valid]
            if exact.any():
                valid_indices = torch.nonzero(valid, as_tuple=False).view(-1)[exact]
                weights[valid_indices] = self.fcrs_release_pair_weights[matched_pos[exact]]
        return weights.to(edge_index.device)

    def _expand_pair_index_bidirectional(self, pair_index: Tensor | None) -> Tensor | None:
        if pair_index is None or pair_index.numel() == 0:
            return None
        rev = torch.stack([pair_index[1], pair_index[0]], dim=0)
        return torch.cat([pair_index, rev], dim=1)

    def _expand_pair_weight_bidirectional(self, pair_weight: Tensor | None) -> Tensor | None:
        if pair_weight is None or pair_weight.numel() == 0:
            return None
        return torch.cat([pair_weight, pair_weight], dim=0)

    def _map_global_pair_index_to_batch(
        self,
        pair_index: Tensor | None,
        pair_weight: Tensor | None,
        batch: Data,
    ):
        if pair_index is None or not hasattr(batch, "n_id"):
            return None, None
        seed_global = batch.n_id[:batch.batch_size].detach().cpu()
        subgraph_global = batch.n_id.detach().cpu()
        inverse = torch.full((self.fcrs_num_nodes,), -1, dtype=torch.long)
        inverse[subgraph_global] = torch.arange(subgraph_global.size(0), dtype=torch.long)
        seed_inverse = torch.full((self.fcrs_num_nodes,), -1, dtype=torch.long)
        seed_inverse[seed_global] = torch.arange(batch.batch_size, dtype=torch.long)

        left_global = pair_index[0]
        right_global = pair_index[1]
        left_local = seed_inverse[left_global]
        right_local = inverse[right_global]
        mask = (left_local >= 0) & (right_local >= 0)
        if not mask.any():
            return None, None
        batch_pair_index = torch.stack([left_local[mask], right_local[mask]], dim=0).to(batch.x.device)
        batch_pair_weight = None
        if pair_weight is not None:
            batch_pair_weight = pair_weight[mask].to(batch.x.device)
        return batch_pair_index, batch_pair_weight

    def _rectify_full_edge_index_for_loss(self, edge_index: Tensor) -> Tensor:
        return self._filter_release_edges(edge_index=edge_index, num_nodes=self.fcrs_num_nodes)

    def _rectify_batch_edge_index_for_loss(self, batch: Data, batch_edge_index: Tensor) -> Tensor:
        if self.fcrs_release_pair_keys is None or batch_edge_index.numel() == 0 or not hasattr(batch, "n_id"):
            return batch_edge_index
        global_nodes = batch.n_id.detach().cpu().long()
        src_global = global_nodes[batch_edge_index[0].detach().cpu().long()]
        dst_global = global_nodes[batch_edge_index[1].detach().cpu().long()]
        global_edge_index = torch.stack([src_global, dst_global], dim=0)
        filtered_global = self._filter_release_edges(global_edge_index, num_nodes=self.fcrs_num_nodes)
        if filtered_global.size(1) == global_edge_index.size(1):
            return batch_edge_index
        keep_mask = ~torch.isin(
            torch.minimum(src_global, dst_global) * int(self.fcrs_num_nodes)
            + torch.maximum(src_global, dst_global),
            self.fcrs_release_pair_keys,
        )
        return batch_edge_index[:, keep_mask.to(batch_edge_index.device)]

    def _lookup_batch_release_edge_weights(self, batch: Data, batch_edge_index: Tensor) -> Tensor | None:
        if (
            self.fcrs_release_pair_keys is None
            or self.fcrs_release_pair_weights is None
            or batch_edge_index.numel() == 0
            or not hasattr(batch, "n_id")
        ):
            return None
        global_nodes = batch.n_id.detach().cpu().long()
        src_global = global_nodes[batch_edge_index[0].detach().cpu().long()]
        dst_global = global_nodes[batch_edge_index[1].detach().cpu().long()]
        global_edge_index = torch.stack([src_global, dst_global], dim=0)
        return self._lookup_release_edge_weights(global_edge_index.to(batch_edge_index.device), self.fcrs_num_nodes)

    def _build_full_spa_exclusion_edge_index(self, edge_index: Tensor) -> Tensor:
        exclusion = edge_index
        if self.fcrs_complete_mode and self.fcrs_positive_pair_index is not None and self.fcrs_positive_pair_index.numel() > 0:
            add_pairs = self._expand_pair_index_bidirectional(self.fcrs_positive_pair_index)
            exclusion = torch.cat([exclusion, add_pairs.to(edge_index.device)], dim=1)
        return exclusion

    def _build_batch_spa_exclusion_edge_index(self, batch: Data, batch_edge_index: Tensor) -> Tensor:
        exclusion = batch_edge_index
        if not self.fcrs_complete_mode:
            return exclusion
        add_pair_index, _ = self._map_global_pair_index_to_batch(self.fcrs_positive_pair_index, None, batch)
        if add_pair_index is None:
            return exclusion
        seed_mask = add_pair_index[1] < batch.batch_size
        if not seed_mask.any():
            return exclusion
        add_pairs = add_pair_index[:, seed_mask]
        add_pairs = self._expand_pair_index_bidirectional(add_pairs)
        return torch.cat([exclusion, add_pairs], dim=1)

    def _build_full_extra_pair_index(self):
        if self.fcrs_extra_pair_index is not None:
            pair_index = self.fcrs_extra_pair_index
            pair_weight = self.fcrs_extra_pair_weights
            if pair_index.numel() == 0:
                return None, None
            return pair_index, pair_weight
        if self.fcrs_extra_top_indices is None:
            return None, None
        partner = self.fcrs_extra_top_indices[:, :self.fcrs_extra_k]
        mask = partner >= 0
        if not mask.any():
            return None, None
        anchor = torch.arange(partner.size(0)).unsqueeze(1).expand_as(partner)
        pair_index = torch.stack([anchor[mask], partner[mask]], dim=0)
        pair_weight = None
        if self.fcrs_extra_pair_weights is not None:
            pair_weight = self.fcrs_extra_pair_weights[:, :self.fcrs_extra_k][mask]
        return pair_index, pair_weight

    def _build_batch_extra_pair_index(self, batch: Data):
        if self.fcrs_extra_pair_index is not None and hasattr(batch, "n_id"):
            return self._map_global_pair_index_to_batch(
                self.fcrs_extra_pair_index,
                self.fcrs_extra_pair_weights,
                batch,
            )
        if self.fcrs_extra_top_indices is None or not hasattr(batch, "n_id"):
            return None, None
        seed_global = batch.n_id[:batch.batch_size].detach().cpu()
        inverse = torch.full((self.fcrs_num_nodes,), -1, dtype=torch.long)
        inverse[seed_global] = torch.arange(batch.batch_size, dtype=torch.long)

        partner_global = self.fcrs_extra_top_indices[seed_global][:, :self.fcrs_extra_k]
        partner_local = inverse[partner_global.clamp_min(0)]
        mask = (partner_global >= 0) & (partner_local >= 0)
        if not mask.any():
            return None, None

        anchor_local = torch.arange(batch.batch_size, dtype=torch.long).unsqueeze(1).expand_as(partner_global)
        pair_index = torch.stack([anchor_local[mask], partner_local[mask]], dim=0).to(batch.x.device)
        pair_weight = None
        if self.fcrs_extra_pair_weights is not None:
            pair_weight = self.fcrs_extra_pair_weights[seed_global][:, :self.fcrs_extra_k][mask].to(batch.x.device)
        return pair_index, pair_weight

    def _build_batch_positive_pair_index(self, batch: Data):
        return self._map_global_pair_index_to_batch(
            self._expand_pair_index_bidirectional(self.fcrs_positive_pair_index),
            self._expand_pair_weight_bidirectional(self.fcrs_positive_pair_weights),
            batch,
        )

    def _build_normalized_adj(self, edge_index: Tensor, num_nodes: int, dtype: torch.dtype, device: torch.device):
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

    def _propagate_k(self, x: Tensor, norm_adj: Tensor, num_hops: int):
        x1 = torch.sparse.mm(norm_adj, x)
        xk = x1
        for _ in range(1, max(num_hops, 1)):
            xk = torch.sparse.mm(norm_adj, xk)
        return x1, xk

    @torch.no_grad()
    def _get_admission_input_features(self, batch_x):
        if isinstance(batch_x, Tensor):
            return batch_x
        if hasattr(self.encoder, "encode_tabular"):
            return self.encoder.encode_tabular(batch_x)
        raise TypeError("Batch-local FCRS admission expects a Tensor input or an encoder with encode_tabular().")

    @torch.no_grad()
    def _build_batch_local_semantic_frequency_pairs_legacy(self, batch: Data, batch_input: Tensor, reps: dict[str, Tensor]):
        profile_active = self._fcrs_runtime_active is not None
        profile_device = batch.x.device if profile_active else None
        batch_size = int(getattr(batch, "batch_size", 0))
        num_nodes = int(batch_input.size(0))
        topk = max(int(self.fcrs_extra_k), 1)
        support_source = self._resolve_support_source()
        adaptive_gaps = {"raw": 0.0, "freq": 0.0, "mu": 0.0, "raw_mul_mu": 0.0}
        adaptive_weights = {"raw": 0.25, "freq": 0.25, "mu": 0.25, "raw_mul_mu": 0.25}
        risk_scale = 1.0
        if support_source == "source_adaptive":
            full_components = self._batch_source_components(reps, reps)
            adaptive_gaps, adaptive_weights, risk_scale = self._batch_adaptive_metadata(
                batch_size=batch_size,
                num_nodes=num_nodes,
                edge_index=batch.edge_index,
                comps=full_components,
            )

        row_block = min(self.fcrs_batch_row_block, batch_size)
        col_block = min(self.fcrs_batch_col_block, num_nodes)
        top_values = torch.full((batch_size, topk), float("-inf"), dtype=batch_input.dtype, device=batch_input.device)
        top_indices = torch.full((batch_size, topk), -1, dtype=torch.long, device=batch_input.device)
        audit_active = self._admission_audit_enabled_for_batch(getattr(self, "_current_epoch_for_audit", None))
        candidate_count_per_anchor = (
            torch.zeros(batch_size, dtype=torch.long, device=batch_input.device)
            if audit_active else None
        )

        edge_src = batch.edge_index[0]
        edge_dst = batch.edge_index[1]
        target_edge_mask = edge_src < batch_size
        target_edge_src = edge_src[target_edge_mask]
        target_edge_dst = edge_dst[target_edge_mask]
        score_elapsed = 0.0
        select_elapsed = 0.0
        dense_score_elements = 0

        for row_start in range(0, batch_size, row_block):
            row_end = min(row_start + row_block, batch_size)
            row_size = row_end - row_start

            block_values = torch.full((row_size, topk), float("-inf"), dtype=batch_input.dtype, device=batch_input.device)
            block_indices = torch.full((row_size, topk), -1, dtype=torch.long, device=batch_input.device)
            row_reps = {name: rep[row_start:row_end] for name, rep in reps.items()}

            row_edge_mask = (target_edge_src >= row_start) & (target_edge_src < row_end)
            row_edge_src = target_edge_src[row_edge_mask] - row_start
            row_edge_dst = target_edge_dst[row_edge_mask]

            for col_start in range(0, num_nodes, col_block):
                col_end = min(col_start + col_block, num_nodes)
                col_reps = {name: rep[col_start:col_end] for name, rep in reps.items()}
                t_score = self._runtime_now(profile_device) if profile_active else None
                comps = self._batch_source_components(row_reps, col_reps)
                if support_source == "raw":
                    scores = comps["raw"]
                elif support_source == "mu":
                    scores = comps["mu"]
                elif support_source == "raw_mul_mu":
                    scores = comps["raw_mul_mu"]
                else:
                    scores = comps["freq"]
                if support_source == "source_adaptive":
                    scores = (
                        adaptive_weights["raw"] * comps["raw"]
                        + adaptive_weights["freq"] * comps["freq"]
                        + adaptive_weights["mu"] * comps["mu"]
                        + adaptive_weights["raw_mul_mu"] * comps["raw_mul_mu"]
                    )
                if profile_active and t_score is not None:
                    score_elapsed += self._runtime_now(profile_device) - t_score
                    dense_score_elements += int(scores.size(0) * scores.size(1))

                t_select = self._runtime_now(profile_device) if profile_active else None
                edge_block_mask = (row_edge_dst >= col_start) & (row_edge_dst < col_end)
                if edge_block_mask.any():
                    scores[row_edge_src[edge_block_mask], row_edge_dst[edge_block_mask] - col_start] = float("-inf")

                if row_start < col_end and col_start < row_end:
                    diag_start = max(row_start, col_start)
                    diag_end = min(row_end, col_end)
                    diag_len = max(0, diag_end - diag_start)
                    if diag_len > 0:
                        row_ids = torch.arange(diag_start - row_start, diag_start - row_start + diag_len, device=batch_input.device)
                        col_ids = torch.arange(diag_start - col_start, diag_start - col_start + diag_len, device=batch_input.device)
                        scores[row_ids, col_ids] = float("-inf")
                if audit_active:
                    candidate_count_per_anchor[row_start:row_end] += torch.isfinite(scores).sum(dim=1).to(torch.long)

                cand_values, cand_local_idx = torch.topk(scores, k=min(topk, scores.size(1)), dim=1)
                cand_indices = cand_local_idx + col_start

                merged_values = torch.cat([block_values, cand_values], dim=1)
                merged_indices = torch.cat([block_indices, cand_indices], dim=1)
                best_values, best_pos = torch.topk(merged_values, k=topk, dim=1)
                best_indices = torch.gather(merged_indices, 1, best_pos)
                best_indices = best_indices.masked_fill(~torch.isfinite(best_values), -1)
                block_values = best_values
                block_indices = best_indices
                if profile_active and t_select is not None:
                    select_elapsed += self._runtime_now(profile_device) - t_select

            top_values[row_start:row_end] = block_values
            top_indices[row_start:row_end] = block_indices

        if profile_active:
            self._runtime_profile_add("lcsr_candidate_generation_scoring_s", score_elapsed)
            self._runtime_profile_add("lcsr_topk_admission_pair_assembly_s", select_elapsed)
            self.fcrs_runtime_profile_meta["candidate_score_tensor_shape"] = [int(batch_size), int(num_nodes)]
            self.fcrs_runtime_profile_meta["candidate_pool_semantics"] = "legacy_full_sampled_subgraph_scan"
            self.fcrs_runtime_profile_meta["dense_score_elements_per_batch"] = dense_score_elements
            self.fcrs_runtime_profile_meta["row_block"] = int(row_block)
            self.fcrs_runtime_profile_meta["col_block"] = int(col_block)

        t_diag = self._runtime_now(profile_device) if profile_active else None
        mask = (top_indices >= 0) & torch.isfinite(top_values)
        if not mask.any():
            return None, None, self._empty_batch_local_stats()
        threshold_value = None
        threshold_type = "none"
        pass_count_per_anchor = None
        if support_source == "source_adaptive" and self.fcrs_risk_budget:
            keep_fraction = max(min(risk_scale, 1.0), 0.25)
            finite_scores = top_values[mask]
            keep_count = max(1, int(np.ceil(float(finite_scores.numel()) * keep_fraction)))
            keep_values = torch.topk(finite_scores, k=min(keep_count, finite_scores.numel())).values
            threshold = keep_values.min()
            threshold_value = float(threshold.item())
            threshold_type = "global_topk_score_floor"
            mask = mask & (top_values >= threshold)
            if not bool(mask.any()):
                return None, None, self._empty_batch_local_stats()
            if audit_active:
                pass_count_per_anchor = (top_values >= threshold).sum(dim=1).to(torch.long)
        elif audit_active:
            pass_count_per_anchor = candidate_count_per_anchor.clone()

        anchor = torch.arange(batch_size, device=batch_input.device).unsqueeze(1).expand_as(top_indices)
        pair_index = torch.stack([anchor[mask], top_indices[mask]], dim=0)
        mean_score = float(top_values[mask].mean().item())
        if audit_active:
            admitted_count_per_anchor = mask.sum(dim=1).to(torch.long)
            admitted_scores = top_values[mask].detach().cpu().float()
            fraction_by_count = {}
            for admitted in range(topk + 1):
                fraction_by_count[str(admitted)] = float((admitted_count_per_anchor == admitted).float().mean().item())
            score_quantiles = {}
            if admitted_scores.numel() > 0:
                for q in [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]:
                    score_quantiles[f"q{int(q*100):02d}"] = float(torch.quantile(admitted_scores, q).item())
            self.fcrs_admission_audit_rows.append({
                "batch_index": len(self.fcrs_admission_audit_rows) + 1,
                "epoch": int(getattr(self, "_current_epoch_for_audit", -1)),
                "batch_size": int(batch_size),
                "num_nodes": int(num_nodes),
                "candidate_count_per_anchor": candidate_count_per_anchor.detach().cpu().tolist(),
                "pass_count_per_anchor": pass_count_per_anchor.detach().cpu().tolist() if pass_count_per_anchor is not None else None,
                "admitted_count_per_anchor": admitted_count_per_anchor.detach().cpu().tolist(),
                "admitted_fraction_by_count": fraction_by_count,
                "admitted_score_quantiles": score_quantiles,
                "threshold_type": threshold_type,
                "threshold_value": threshold_value,
                "admitted_scores": admitted_scores.tolist(),
            })
        if profile_active and t_diag is not None:
            self._runtime_profile_add("diagnostics_logging_s", self._runtime_now(profile_device) - t_diag)
        return pair_index, None, {
            "extra_pairs": float(mask.sum().item()),
            "admit_score": mean_score,
            "gap_raw": float(adaptive_gaps["raw"]),
            "gap_freq": float(adaptive_gaps["freq"]),
            "gap_mu": float(adaptive_gaps["mu"]),
            "gap_raw_mul_mu": float(adaptive_gaps["raw_mul_mu"]),
            "weight_raw": float(adaptive_weights["raw"]),
            "weight_freq": float(adaptive_weights["freq"]),
            "weight_mu": float(adaptive_weights["mu"]),
            "weight_raw_mul_mu": float(adaptive_weights["raw_mul_mu"]),
            "risk_budget_scale": float(risk_scale),
        }

    @staticmethod
    def _batch_local_percentile(score_lists: list[np.ndarray], node_id: int, score: float) -> float:
        values = score_lists[node_id]
        rank = np.searchsorted(values, score, side="right")
        return float((1 + rank) / (values.size + 1))

    @staticmethod
    def _batch_local_canonical_pair(u: int, v: int) -> tuple[int, int]:
        return (u, v) if u < v else (v, u)

    @torch.no_grad()
    def _empty_batch_local_stats(self):
        return {
            "extra_pairs": 0.0,
            "admit_score": 0.0,
            "gap_raw": 0.0,
            "gap_freq": 0.0,
            "gap_mu": 0.0,
            "gap_raw_mul_mu": 0.0,
            "weight_raw": 0.0,
            "weight_freq": 0.0,
            "weight_mu": 0.0,
            "weight_raw_mul_mu": 0.0,
            "risk_budget_scale": 1.0,
        }

    @torch.no_grad()
    def _build_batch_local_semantic_frequency_pairs_aligned(self, batch: Data, batch_input: Tensor, reps: dict[str, Tensor]):
        profile_active = self._fcrs_runtime_active is not None
        profile_device = batch.x.device if profile_active else None
        batch_size = int(getattr(batch, "batch_size", 0))
        num_nodes = int(batch_input.size(0))
        if batch_size <= 0 or num_nodes <= 1:
            return None, None, self._empty_batch_local_stats()

        candidate_pool_size = max(int(self.fcrs_lcsr_candidate_pool_size), 1)
        rho = float(self.fcrs_lcsr_rho)
        kmax = max(int(self.fcrs_lcsr_kmax), 1)
        margin = float(self.fcrs_lcsr_margin)
        use_local_calibration = not self.fcrs_lcsr_disable_local_calibration
        use_mutual = not self.fcrs_lcsr_disable_mutual
        support_source = self._resolve_support_source()

        t_prep = self._runtime_now(profile_device) if profile_active else None
        edge_src = batch.edge_index[0].detach().cpu().numpy().astype(np.int64, copy=False)
        edge_dst = batch.edge_index[1].detach().cpu().numpy().astype(np.int64, copy=False)
        valid_mask = edge_src != edge_dst
        edge_src = edge_src[valid_mask]
        edge_dst = edge_dst[valid_mask]
        order = np.argsort(edge_src, kind="stable")
        edge_src = edge_src[order]
        edge_dst = edge_dst[order]
        local_degree = np.bincount(edge_src, minlength=num_nodes).astype(np.int64, copy=False)
        indptr = np.zeros(num_nodes + 1, dtype=np.int64)
        indptr[1:] = np.cumsum(local_degree)
        if profile_active and t_prep is not None:
            self._runtime_profile_add("lcsr_candidate_generation_scoring_s", 0.0)

        t_obs = self._runtime_now(profile_device) if profile_active else None
        local_edge_src = torch.from_numpy(edge_src).to(batch_input.device, dtype=torch.long)
        local_edge_dst = torch.from_numpy(edge_dst).to(batch_input.device, dtype=torch.long)
        obs_scores_t = self._batch_pair_scores_by_source(
            support_source=support_source,
            left_index=local_edge_src,
            right_index=local_edge_dst,
            reps=reps,
        )
        obs_scores_np = obs_scores_t.detach().cpu().numpy().astype(np.float32, copy=False)
        obs_score_lists: list[list[float]] = [[] for _ in range(num_nodes)]
        for src, score in zip(edge_src.tolist(), obs_scores_np.tolist()):
            obs_score_lists[src].append(float(score))
        sorted_obs_scores = [
            np.sort(np.asarray(items, dtype=np.float32)) if items else np.empty((0,), dtype=np.float32)
            for items in obs_score_lists
        ]
        if profile_active and t_obs is not None:
            self._runtime_profile_add("lcsr_frequency_descriptor_s", self._runtime_now(profile_device) - t_obs)

        t_select = self._runtime_now(profile_device) if profile_active else None
        anchor_candidate_ids = torch.full(
            (batch_size, candidate_pool_size),
            -1,
            dtype=torch.long,
            device=batch_input.device,
        )
        observed_by_node: list[list[dict[str, float | int]]] = [[] for _ in range(batch_size)]
        candidate_count_per_anchor = np.zeros(batch_size, dtype=np.int64)
        pass_count_per_anchor = np.zeros(batch_size, dtype=np.int64)
        dynamic_budget_per_anchor = np.zeros(batch_size, dtype=np.int64)
        if self.fcrs_global_degree is not None and hasattr(batch, "n_id"):
            global_degree = self.fcrs_global_degree[batch.n_id[:batch_size].detach().cpu().long()].numpy().astype(np.int64, copy=False)
        else:
            global_degree = local_degree[:batch_size]

        for anchor in range(batch_size):
            obs_neighbors = edge_dst[indptr[anchor]:indptr[anchor + 1]]
            if obs_neighbors.size == 0:
                continue
            obs_neighbor_set = set(int(v) for v in obs_neighbors.tolist())
            dynamic_budget_per_anchor[anchor] = min(kmax, int(np.ceil(rho * max(int(global_degree[anchor]), 0))))

            frontier = []
            for nbr in obs_neighbors.tolist():
                frontier.extend(edge_dst[indptr[int(nbr)]:indptr[int(nbr) + 1]].tolist())
            if not frontier:
                continue
            frontier_arr = np.unique(np.asarray(frontier, dtype=np.int64))
            frontier_arr = frontier_arr[(frontier_arr != anchor)]
            if frontier_arr.size == 0:
                continue
            keep_mask = np.array([cand not in obs_neighbor_set for cand in frontier_arr.tolist()], dtype=bool)
            frontier_arr = frontier_arr[keep_mask]
            if frontier_arr.size == 0:
                continue

            frontier_ids = torch.from_numpy(frontier_arr).to(batch_input.device, dtype=torch.long)
            anchor_ids = torch.full((frontier_ids.numel(),), anchor, dtype=torch.long, device=batch_input.device)
            coarse_scores = map_cosine_to_unit_interval(
                (reps["id"][anchor_ids] * reps["id"][frontier_ids]).sum(dim=-1)
            )
            if frontier_ids.numel() > candidate_pool_size:
                top_pos = torch.topk(coarse_scores, k=candidate_pool_size).indices
                frontier_ids = frontier_ids[top_pos]
            anchor_candidate_ids[anchor, :frontier_ids.numel()] = frontier_ids
            candidate_count_per_anchor[anchor] = int(frontier_ids.numel())

            obs_local_scores = []
            row_slice_scores = obs_scores_np[indptr[anchor]:indptr[anchor + 1]]
            for nbr, obs_score in zip(obs_neighbors.tolist(), row_slice_scores.tolist()):
                obs_score = float(obs_score)
                if use_local_calibration:
                    p_anchor = self._batch_local_percentile(sorted_obs_scores, anchor, obs_score)
                    p_peer = self._batch_local_percentile(sorted_obs_scores, int(nbr), obs_score)
                else:
                    p_anchor = obs_score
                    p_peer = obs_score
                p_drop = max(p_anchor, p_peer) if use_mutual else 0.5 * (p_anchor + p_peer)
                obs_local_scores.append(
                    {"nbr": int(nbr), "score": obs_score, "p_drop": p_drop}
                )
            observed_by_node[anchor] = obs_local_scores
        if profile_active and t_select is not None:
            self._runtime_profile_add("lcsr_candidate_generation_scoring_s", self._runtime_now(profile_device) - t_select)

        t_score = self._runtime_now(profile_device) if profile_active else None
        valid_candidate_mask = anchor_candidate_ids >= 0
        candidate_scores = torch.full(
            (batch_size, candidate_pool_size),
            float("-inf"),
            dtype=batch_input.dtype,
            device=batch_input.device,
        )
        if bool(valid_candidate_mask.any()):
            flat_anchor = torch.arange(batch_size, device=batch_input.device).unsqueeze(1).expand_as(anchor_candidate_ids)[valid_candidate_mask]
            flat_cand = anchor_candidate_ids[valid_candidate_mask]
            scored = self._batch_pair_scores_by_source(
                support_source=support_source,
                left_index=flat_anchor,
                right_index=flat_cand,
                reps=reps,
            )
            candidate_scores[valid_candidate_mask] = scored
        if profile_active and t_score is not None:
            self._runtime_profile_add("lcsr_candidate_generation_scoring_s", self._runtime_now(profile_device) - t_score)

        t_gate = self._runtime_now(profile_device) if profile_active else None
        swap_records: list[dict[str, float | int | tuple[int, int]]] = []
        admitted_count_per_anchor = np.zeros(batch_size, dtype=np.int64)
        admitted_scores = []
        admitted_support_scores = []
        for anchor in range(batch_size):
            valid = valid_candidate_mask[anchor]
            if not bool(valid.any()):
                continue
            node_candidates = anchor_candidate_ids[anchor][valid].detach().cpu().tolist()
            node_scores = candidate_scores[anchor][valid].detach().cpu().tolist()
            candidate_items = []
            for cand, score in zip(node_candidates, node_scores):
                score = float(score)
                if not np.isfinite(score):
                    continue
                if use_local_calibration:
                    p_anchor = self._batch_local_percentile(sorted_obs_scores, anchor, score)
                    p_peer = self._batch_local_percentile(sorted_obs_scores, int(cand), score)
                else:
                    p_anchor = score
                    p_peer = score
                p_add = min(p_anchor, p_peer) if use_mutual else 0.5 * (p_anchor + p_peer)
                candidate_items.append({"nbr": int(cand), "score": score, "p_add": p_add})
            pass_count_per_anchor[anchor] = len(candidate_items)
            if not observed_by_node[anchor] or not candidate_items:
                continue
            node_budget = min(kmax, int(np.ceil(rho * max(int(global_degree[anchor]), 0))))
            dynamic_budget_per_anchor[anchor] = node_budget
            if node_budget <= 0:
                continue
            observed_items = sorted(observed_by_node[anchor], key=lambda item: float(item["p_drop"]))
            ranked_candidates = sorted(candidate_items, key=lambda item: float(item["p_add"]), reverse=True)
            max_swaps = min(len(observed_items), len(ranked_candidates), node_budget)
            for offset in range(max_swaps):
                drop_item = observed_items[offset]
                add_item = ranked_candidates[offset]
                gain = float(add_item["p_add"]) - float(drop_item["p_drop"])
                if gain <= margin:
                    break
                drop_nbr = int(drop_item["nbr"])
                add_nbr = int(add_item["nbr"])
                swap_records.append(
                    {
                        "anchor": int(anchor),
                        "drop_nbr": drop_nbr,
                        "add_nbr": add_nbr,
                        "add_pair": self._batch_local_canonical_pair(int(anchor), add_nbr),
                        "drop_pair": self._batch_local_canonical_pair(int(anchor), drop_nbr),
                        "p_add": float(add_item["p_add"]),
                        "p_drop": float(drop_item["p_drop"]),
                        "gain": gain,
                        "r_add": float(add_item["score"]),
                        "r_drop": float(drop_item["score"]),
                        "r_gain": float(add_item["score"]) - float(drop_item["score"]),
                        "degree": int(global_degree[anchor]),
                    }
                )
        selected_records = _match_swap_records(
            swap_records=swap_records,
            enable_budget_match=self.fcrs_lcsr_budget_match,
        )
        if profile_active and t_gate is not None:
            self._runtime_profile_add("lcsr_topk_admission_pair_assembly_s", self._runtime_now(profile_device) - t_gate)

        if not selected_records:
            if self._admission_audit_enabled_for_batch(getattr(self, "_current_epoch_for_audit", None)):
                fraction_by_count = {str(i): float((admitted_count_per_anchor == i).mean()) for i in range(kmax + 1)}
                self.fcrs_admission_audit_rows.append({
                    "batch_index": len(self.fcrs_admission_audit_rows) + 1,
                    "epoch": int(getattr(self, "_current_epoch_for_audit", -1)),
                    "batch_size": int(batch_size),
                    "num_nodes": int(num_nodes),
                    "candidate_count_per_anchor": candidate_count_per_anchor.tolist(),
                    "pass_count_per_anchor": pass_count_per_anchor.tolist(),
                    "admitted_count_per_anchor": admitted_count_per_anchor.tolist(),
                    "admitted_fraction_by_count": fraction_by_count,
                    "admitted_score_quantiles": {},
                    "threshold_type": "pairwise_margin_gate",
                    "threshold_value": margin,
                    "admitted_scores": [],
                })
            if profile_active:
                self.fcrs_runtime_profile_meta["candidate_score_tensor_shape"] = [int(batch_size), int(candidate_pool_size)]
                self.fcrs_runtime_profile_meta["candidate_pool_semantics"] = "aligned_bounded_pool"
            return None, None, self._empty_batch_local_stats()

        pair_left = []
        pair_right = []
        for record in selected_records:
            anchor = int(record["anchor"])
            add_nbr = int(record["add_nbr"])
            pair_left.append(anchor)
            pair_right.append(add_nbr)
            admitted_count_per_anchor[anchor] += 1
            admitted_scores.append(float(record["r_add"]))
            admitted_support_scores.append(float(record["r_add"]))
        pair_index = torch.tensor([pair_left, pair_right], dtype=torch.long, device=batch_input.device)
        mean_score = float(np.mean(admitted_support_scores)) if admitted_support_scores else 0.0

        if profile_active:
            self.fcrs_runtime_profile_meta["candidate_score_tensor_shape"] = [int(batch_size), int(candidate_pool_size)]
            self.fcrs_runtime_profile_meta["candidate_pool_semantics"] = "aligned_bounded_pool"
            self.fcrs_runtime_profile_meta["bounded_candidate_scores_per_batch"] = int(valid_candidate_mask.sum().item())

        if self._admission_audit_enabled_for_batch(getattr(self, "_current_epoch_for_audit", None)):
            fraction_by_count = {str(i): float((admitted_count_per_anchor == i).mean()) for i in range(kmax + 1)}
            score_quantiles = {}
            if admitted_scores:
                admitted_scores_t = torch.tensor(admitted_scores, dtype=torch.float32)
                for q in [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]:
                    score_quantiles[f"q{int(q*100):02d}"] = float(torch.quantile(admitted_scores_t, q).item())
            self.fcrs_admission_audit_rows.append({
                "batch_index": len(self.fcrs_admission_audit_rows) + 1,
                "epoch": int(getattr(self, "_current_epoch_for_audit", -1)),
                "batch_size": int(batch_size),
                "num_nodes": int(num_nodes),
                "candidate_count_per_anchor": candidate_count_per_anchor.tolist(),
                "pass_count_per_anchor": pass_count_per_anchor.tolist(),
                "admitted_count_per_anchor": admitted_count_per_anchor.tolist(),
                "admitted_fraction_by_count": fraction_by_count,
                "admitted_score_quantiles": score_quantiles,
                "threshold_type": "pairwise_margin_gate",
                "threshold_value": margin,
                "admitted_scores": admitted_scores,
            })

        return pair_index, None, {
            "extra_pairs": float(len(selected_records)),
            "admit_score": mean_score,
            "gap_raw": 0.0,
            "gap_freq": 0.0,
            "gap_mu": 0.0,
            "gap_raw_mul_mu": 0.0,
            "weight_raw": 0.0,
            "weight_freq": 0.0,
            "weight_mu": 0.0,
            "weight_raw_mul_mu": 0.0,
            "risk_budget_scale": 1.0,
        }

    @torch.no_grad()
    def _build_batch_local_semantic_frequency_pairs_candidate_bank_v2(self, batch: Data, batch_input: Tensor, reps: dict[str, Tensor]):
        profile_active = self._fcrs_runtime_active is not None
        profile_device = batch.x.device if profile_active else None
        batch_size = int(getattr(batch, "batch_size", 0))
        num_nodes = int(batch_input.size(0))
        if batch_size <= 0 or num_nodes <= 1 or not hasattr(batch, "n_id"):
            return None, None, self._empty_batch_local_stats()
        if self.fcrs_candidate_bank is None:
            return None, None, self._empty_batch_local_stats()

        support_source = self._resolve_support_source()
        candidate_pool_size = max(int(self.fcrs_lcsr_candidate_pool_size), 1)
        rho = float(self.fcrs_lcsr_rho)
        kmax = max(int(self.fcrs_lcsr_kmax), 1)
        margin = float(self.fcrs_lcsr_margin)
        use_local_calibration = not self.fcrs_lcsr_disable_local_calibration
        use_mutual = not self.fcrs_lcsr_disable_mutual
        device = batch_input.device

        t_lookup = self._runtime_now(profile_device) if profile_active else None
        global_nodes_cpu = batch.n_id.detach().cpu().long()
        seed_global_cpu = global_nodes_cpu[:batch_size]
        bank_global_cpu = self.fcrs_candidate_bank.index_select(0, seed_global_cpu)
        bank_global = bank_global_cpu.to(device=device, dtype=torch.long)
        global_to_local = torch.full((int(self.fcrs_candidate_bank.size(0)),), -1, dtype=torch.long, device=device)
        global_nodes = global_nodes_cpu.to(device=device, dtype=torch.long)
        global_to_local[global_nodes] = torch.arange(global_nodes.numel(), device=device, dtype=torch.long)
        candidate_local = torch.full_like(bank_global, -1)
        nonneg_mask = bank_global >= 0
        if bool(nonneg_mask.any()):
            candidate_local[nonneg_mask] = global_to_local[bank_global[nonneg_mask]]
        if profile_active and t_lookup is not None:
            self._runtime_profile_add("lcsr_candidate_bank_lookup_s", self._runtime_now(profile_device) - t_lookup)

        t_filter = self._runtime_now(profile_device) if profile_active else None
        anchor_local = torch.arange(batch_size, device=device, dtype=torch.long).unsqueeze(1).expand_as(candidate_local)
        valid_candidate_mask = (candidate_local >= 0) & (candidate_local != anchor_local)
        candidate_count_per_anchor = valid_candidate_mask.sum(dim=1).to(torch.long)
        if profile_active and t_filter is not None:
            self._runtime_profile_add("lcsr_candidate_filtering_s", self._runtime_now(profile_device) - t_filter)

        t_score = self._runtime_now(profile_device) if profile_active else None
        bank_width = int(candidate_local.size(1))
        candidate_scores = torch.full(
            (batch_size, bank_width),
            float("-inf"),
            dtype=batch_input.dtype,
            device=device,
        )
        if bool(valid_candidate_mask.any()):
            flat_anchor = anchor_local[valid_candidate_mask]
            flat_cand = candidate_local[valid_candidate_mask]
            scored = self._batch_pair_scores_by_source(
                support_source=support_source,
                left_index=flat_anchor,
                right_index=flat_cand,
                reps=reps,
            )
            candidate_scores[valid_candidate_mask] = scored
        shortlist_k = min(candidate_pool_size, bank_width)
        top_candidate_scores, top_candidate_pos = torch.topk(candidate_scores, k=shortlist_k, dim=1)
        top_candidate_local = torch.gather(candidate_local, 1, top_candidate_pos)
        top_candidate_global = torch.gather(bank_global, 1, top_candidate_pos)
        top_candidate_valid = torch.isfinite(top_candidate_scores) & (top_candidate_local >= 0)
        if profile_active and t_score is not None:
            self._runtime_profile_add("lcsr_candidate_generation_scoring_s", self._runtime_now(profile_device) - t_score)

        t_drop = self._runtime_now(profile_device) if profile_active else None
        edge_src = batch.edge_index[0]
        edge_dst = batch.edge_index[1]
        nonself_mask = edge_src != edge_dst
        edge_src = edge_src[nonself_mask]
        edge_dst = edge_dst[nonself_mask]
        if edge_src.numel() == 0:
            return None, None, self._empty_batch_local_stats()
        edge_scores = self._batch_pair_scores_by_source(
            support_source=support_source,
            left_index=edge_src,
            right_index=edge_dst,
            reps=reps,
        )
        local_degree = torch.bincount(edge_src, minlength=num_nodes).to(torch.long)
        max_local_degree = int(local_degree.max().item()) if local_degree.numel() > 0 else 0
        if max_local_degree <= 0:
            return None, None, self._empty_batch_local_stats()
        edge_offsets = torch.cat(
            [
                torch.zeros(1, device=device, dtype=torch.long),
                torch.cumsum(local_degree, dim=0)[:-1],
            ],
            dim=0,
        )
        edge_sort_key = edge_src.to(torch.float32) * 2.0 + edge_scores.to(torch.float32)
        edge_order = torch.argsort(edge_sort_key)
        sorted_edge_src = edge_src[edge_order]
        sorted_edge_dst = edge_dst[edge_order]
        sorted_edge_scores = edge_scores[edge_order]
        sorted_edge_rank = torch.arange(sorted_edge_src.numel(), device=device, dtype=torch.long) - edge_offsets[sorted_edge_src]
        obs_sorted_all = torch.full(
            (num_nodes, max_local_degree),
            float("inf"),
            dtype=batch_input.dtype,
            device=device,
        )
        obs_sorted_all[sorted_edge_src, sorted_edge_rank] = sorted_edge_scores
        if profile_active and t_drop is not None:
            self._runtime_profile_add("lcsr_frequency_descriptor_s", self._runtime_now(profile_device) - t_drop)

        t_gate = self._runtime_now(profile_device) if profile_active else None
        seed_obs_mask = edge_src < batch_size
        seed_obs_src = edge_src[seed_obs_mask]
        seed_obs_dst = edge_dst[seed_obs_mask]
        seed_obs_scores = edge_scores[seed_obs_mask]
        seed_obs_counts = torch.bincount(seed_obs_src, minlength=batch_size).to(torch.long)
        if seed_obs_scores.numel() == 0:
            return None, None, self._empty_batch_local_stats()
        seed_offsets = torch.cat(
            [
                torch.zeros(1, device=device, dtype=torch.long),
                torch.cumsum(seed_obs_counts, dim=0)[:-1],
            ],
            dim=0,
        )
        if use_local_calibration:
            seed_anchor_rank = torch.searchsorted(
                obs_sorted_all[seed_obs_src],
                seed_obs_scores.unsqueeze(-1),
                right=True,
            ).squeeze(-1)
            seed_peer_rank = torch.searchsorted(
                obs_sorted_all[seed_obs_dst],
                seed_obs_scores.unsqueeze(-1),
                right=True,
            ).squeeze(-1)
            seed_anchor_p = (1.0 + seed_anchor_rank.to(batch_input.dtype)) / (local_degree[seed_obs_src].to(batch_input.dtype) + 1.0)
            seed_peer_p = (1.0 + seed_peer_rank.to(batch_input.dtype)) / (local_degree[seed_obs_dst].to(batch_input.dtype) + 1.0)
        else:
            seed_anchor_p = seed_obs_scores
            seed_peer_p = seed_obs_scores
        seed_p_drop = torch.maximum(seed_anchor_p, seed_peer_p) if use_mutual else 0.5 * (seed_anchor_p + seed_peer_p)
        seed_sort_key = seed_obs_src.to(torch.float32) * 2.0 + seed_p_drop.to(torch.float32)
        seed_order = torch.argsort(seed_sort_key)
        seed_sorted_src = seed_obs_src[seed_order]
        seed_sorted_dst = seed_obs_dst[seed_order]
        seed_sorted_p_drop = seed_p_drop[seed_order]
        seed_sorted_r_drop = seed_obs_scores[seed_order]
        seed_sorted_rank = torch.arange(seed_sorted_src.numel(), device=device, dtype=torch.long) - seed_offsets[seed_sorted_src]
        max_seed_degree = int(seed_obs_counts.max().item()) if seed_obs_counts.numel() > 0 else 0
        if max_seed_degree <= 0:
            return None, None, self._empty_batch_local_stats()
        drop_padded = torch.full((batch_size, max_seed_degree), float("inf"), dtype=batch_input.dtype, device=device)
        drop_r_padded = torch.full((batch_size, max_seed_degree), float("-inf"), dtype=batch_input.dtype, device=device)
        drop_local_padded = torch.full((batch_size, max_seed_degree), -1, dtype=torch.long, device=device)
        drop_padded[seed_sorted_src, seed_sorted_rank] = seed_sorted_p_drop
        drop_r_padded[seed_sorted_src, seed_sorted_rank] = seed_sorted_r_drop
        drop_local_padded[seed_sorted_src, seed_sorted_rank] = seed_sorted_dst

        if use_local_calibration:
            anchor_rank = torch.empty_like(top_candidate_local, dtype=torch.long)
            peer_rank = torch.empty_like(top_candidate_local, dtype=torch.long)
            anchor_boundaries = obs_sorted_all[:batch_size].contiguous()
            for col in range(shortlist_k):
                value_col = top_candidate_scores[:, col:col + 1]
                anchor_rank[:, col] = torch.searchsorted(
                    anchor_boundaries,
                    value_col,
                    right=True,
                ).squeeze(-1)
                peer_boundaries = obs_sorted_all[top_candidate_local[:, col].clamp_min(0)].contiguous()
                peer_rank[:, col] = torch.searchsorted(
                    peer_boundaries,
                    value_col,
                    right=True,
                ).squeeze(-1)
            candidate_anchor_p = (1.0 + anchor_rank.to(batch_input.dtype)) / (local_degree[:batch_size].to(batch_input.dtype).unsqueeze(1) + 1.0)
            candidate_peer_p = (1.0 + peer_rank.to(batch_input.dtype)) / (local_degree[top_candidate_local.clamp_min(0)].to(batch_input.dtype) + 1.0)
        else:
            candidate_anchor_p = top_candidate_scores
            candidate_peer_p = top_candidate_scores
        p_add = torch.minimum(candidate_anchor_p, candidate_peer_p) if use_mutual else 0.5 * (candidate_anchor_p + candidate_peer_p)
        p_add = p_add.masked_fill(~top_candidate_valid, float("-inf"))

        add_p_sorted, add_order = torch.topk(p_add, k=shortlist_k, dim=1)
        add_score_sorted = torch.gather(top_candidate_scores, 1, add_order)
        add_local_sorted = torch.gather(top_candidate_local, 1, add_order)
        add_global_sorted = torch.gather(top_candidate_global, 1, add_order)
        add_valid_sorted = torch.isfinite(add_p_sorted) & (add_local_sorted >= 0)

        if self.fcrs_global_degree is not None:
            global_degree = self.fcrs_global_degree[seed_global_cpu].to(device=device, dtype=torch.long)
        else:
            global_degree = local_degree[:batch_size]
        node_budget = torch.ceil(global_degree.to(torch.float32) * rho).to(torch.long).clamp_min(0).clamp_max(kmax)

        rank_width = min(shortlist_k, max_seed_degree, kmax)
        if rank_width <= 0:
            return None, None, self._empty_batch_local_stats()
        add_p_rank = add_p_sorted[:, :rank_width]
        add_score_rank = add_score_sorted[:, :rank_width]
        add_local_rank = add_local_sorted[:, :rank_width]
        add_global_rank = add_global_sorted[:, :rank_width]
        add_valid_rank = add_valid_sorted[:, :rank_width]
        drop_p_rank = drop_padded[:, :rank_width]
        drop_r_rank = drop_r_padded[:, :rank_width]
        drop_local_rank = drop_local_padded[:, :rank_width]
        rank_ids = torch.arange(rank_width, device=device, dtype=torch.long).unsqueeze(0)
        base_valid = add_valid_rank & (drop_local_rank >= 0)
        margin_pass = base_valid & ((add_p_rank - drop_p_rank) > margin)
        post_margin_count_per_anchor = margin_pass.sum(dim=1).to(torch.long)
        budget_valid = margin_pass & (rank_ids < node_budget.unsqueeze(1))

        flat_anchor_rank = torch.arange(batch_size, device=device, dtype=torch.long).unsqueeze(1).expand_as(add_local_rank)
        flat_anchor = flat_anchor_rank[budget_valid]
        flat_add_local = add_local_rank[budget_valid]
        flat_add_global = add_global_rank[budget_valid]
        flat_drop_local = drop_local_rank[budget_valid]
        flat_gain = (add_p_rank - drop_p_rank)[budget_valid]
        flat_r_add = add_score_rank[budget_valid]
        flat_r_drop = drop_r_rank[budget_valid]

        if flat_anchor.numel() == 0:
            admitted_count_per_anchor = torch.zeros(batch_size, dtype=torch.long, device=device)
            if self._admission_audit_enabled_for_batch(getattr(self, "_current_epoch_for_audit", None)):
                fraction_by_count = {str(i): float((admitted_count_per_anchor == i).float().mean().item()) for i in range(kmax + 1)}
                self.fcrs_admission_audit_rows.append({
                    "batch_index": len(self.fcrs_admission_audit_rows) + 1,
                    "epoch": int(getattr(self, "_current_epoch_for_audit", -1)),
                    "batch_size": int(batch_size),
                    "num_nodes": int(num_nodes),
                    "candidate_count_per_anchor": candidate_count_per_anchor.detach().cpu().tolist(),
                    "pass_count_per_anchor": post_margin_count_per_anchor.detach().cpu().tolist(),
                    "admitted_count_per_anchor": admitted_count_per_anchor.detach().cpu().tolist(),
                    "admitted_fraction_by_count": fraction_by_count,
                    "admitted_score_quantiles": {},
                    "threshold_type": "pairwise_margin_gate",
                    "threshold_value": margin,
                    "admitted_scores": [],
                })
            if profile_active and t_gate is not None:
                self._runtime_profile_add("lcsr_margin_gate_s", self._runtime_now(profile_device) - t_gate)
                self.fcrs_runtime_profile_meta["candidate_score_tensor_shape"] = [int(batch_size), int(bank_width)]
                self.fcrs_runtime_profile_meta["candidate_pool_semantics"] = "candidate_bank_v2"
                self.fcrs_runtime_profile_meta["bounded_candidate_scores_per_batch"] = int(valid_candidate_mask.sum().item())
                self.fcrs_runtime_profile_meta["candidate_pool_size_effective"] = int(shortlist_k)
            return None, None, self._empty_batch_local_stats()

        sort_order = torch.argsort(flat_gain, descending=True)
        flat_anchor = flat_anchor[sort_order]
        flat_add_local = flat_add_local[sort_order]
        flat_add_global = flat_add_global[sort_order]
        flat_drop_local = flat_drop_local[sort_order]
        flat_gain = flat_gain[sort_order]
        flat_r_add = flat_r_add[sort_order]
        flat_r_drop = flat_r_drop[sort_order]

        if self.fcrs_lcsr_budget_match:
            add_u = torch.minimum(seed_global_cpu.to(device=device, dtype=torch.long)[flat_anchor], flat_add_global)
            add_v = torch.maximum(seed_global_cpu.to(device=device, dtype=torch.long)[flat_anchor], flat_add_global)
            drop_global = global_nodes[flat_drop_local]
            drop_u = torch.minimum(seed_global_cpu.to(device=device, dtype=torch.long)[flat_anchor], drop_global)
            drop_v = torch.maximum(seed_global_cpu.to(device=device, dtype=torch.long)[flat_anchor], drop_global)
            total_nodes = int(self.fcrs_candidate_bank.size(0))
            add_code = add_u * total_nodes + add_v
            drop_code = drop_u * total_nodes + drop_v
            seen_add: set[int] = set()
            seen_drop: set[int] = set()
            keep_index = []
            for idx, (a_code, d_code) in enumerate(zip(add_code.detach().cpu().tolist(), drop_code.detach().cpu().tolist())):
                if a_code in seen_add or d_code in seen_drop:
                    continue
                seen_add.add(a_code)
                seen_drop.add(d_code)
                keep_index.append(idx)
            if keep_index:
                keep_index_t = torch.tensor(keep_index, device=device, dtype=torch.long)
                flat_anchor = flat_anchor[keep_index_t]
                flat_add_local = flat_add_local[keep_index_t]
                flat_gain = flat_gain[keep_index_t]
                flat_r_add = flat_r_add[keep_index_t]
                flat_r_drop = flat_r_drop[keep_index_t]
            else:
                flat_anchor = flat_anchor[:0]
                flat_add_local = flat_add_local[:0]
                flat_gain = flat_gain[:0]
                flat_r_add = flat_r_add[:0]
                flat_r_drop = flat_r_drop[:0]

        admitted_count_per_anchor = torch.bincount(flat_anchor, minlength=batch_size).to(torch.long)
        admitted_total = int(flat_anchor.numel())
        if profile_active and t_gate is not None:
            self._runtime_profile_add("lcsr_margin_gate_s", self._runtime_now(profile_device) - t_gate)

        t_pair = self._runtime_now(profile_device) if profile_active else None
        pair_index = torch.stack([flat_anchor, flat_add_local], dim=0)
        mean_score = float(flat_r_add.mean().item()) if flat_r_add.numel() > 0 else 0.0
        if profile_active and t_pair is not None:
            self._runtime_profile_add("lcsr_topk_admission_pair_assembly_s", self._runtime_now(profile_device) - t_pair)
            self.fcrs_runtime_profile_meta["candidate_score_tensor_shape"] = [int(batch_size), int(bank_width)]
            self.fcrs_runtime_profile_meta["candidate_pool_semantics"] = "candidate_bank_v2"
            self.fcrs_runtime_profile_meta["bounded_candidate_scores_per_batch"] = int(valid_candidate_mask.sum().item())
            self.fcrs_runtime_profile_meta["candidate_pool_size_effective"] = int(shortlist_k)
            for key, value in self.fcrs_candidate_bank_meta.items():
                self.fcrs_runtime_profile_meta[f"candidate_bank_{key}"] = value

        if self._admission_audit_enabled_for_batch(getattr(self, "_current_epoch_for_audit", None)):
            fraction_by_count = {str(i): float((admitted_count_per_anchor == i).float().mean().item()) for i in range(kmax + 1)}
            score_quantiles = {}
            if flat_r_add.numel() > 0:
                admitted_scores_t = flat_r_add.detach().float().cpu()
                for q in [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]:
                    score_quantiles[f"q{int(q*100):02d}"] = float(torch.quantile(admitted_scores_t, q).item())
            self.fcrs_admission_audit_rows.append({
                "batch_index": len(self.fcrs_admission_audit_rows) + 1,
                "epoch": int(getattr(self, "_current_epoch_for_audit", -1)),
                "batch_size": int(batch_size),
                "num_nodes": int(num_nodes),
                "candidate_count_per_anchor": candidate_count_per_anchor.detach().cpu().tolist(),
                "pass_count_per_anchor": post_margin_count_per_anchor.detach().cpu().tolist(),
                "admitted_count_per_anchor": admitted_count_per_anchor.detach().cpu().tolist(),
                "admitted_fraction_by_count": fraction_by_count,
                "admitted_score_quantiles": score_quantiles,
                "threshold_type": "pairwise_margin_gate",
                "threshold_value": margin,
                "admitted_scores": flat_r_add.detach().cpu().tolist(),
            })

        return pair_index, None, {
            "extra_pairs": float(admitted_total),
            "admit_score": mean_score,
            "gap_raw": 0.0,
            "gap_freq": 0.0,
            "gap_mu": 0.0,
            "gap_raw_mul_mu": 0.0,
            "weight_raw": 0.0,
            "weight_freq": 0.0,
            "weight_mu": 0.0,
            "weight_raw_mul_mu": 0.0,
            "risk_budget_scale": 1.0,
        }

    @torch.no_grad()
    def _batch_pair_scores_by_source(self, support_source: str, left_index: Tensor, right_index: Tensor, reps: dict[str, Tensor]) -> Tensor:
        raw = map_cosine_to_unit_interval((reps["id"][left_index] * reps["id"][right_index]).sum(dim=-1))
        if support_source == "raw":
            return raw
        low = map_cosine_to_unit_interval((reps["low"][left_index] * reps["low"][right_index]).sum(dim=-1))
        mid = map_cosine_to_unit_interval((reps["mid"][left_index] * reps["mid"][right_index]).sum(dim=-1))
        high = map_cosine_to_unit_interval((reps["high"][left_index] * reps["high"][right_index]).sum(dim=-1))
        mu = (raw + low + mid + high) * 0.25
        if support_source == "mu":
            return mu
        if support_source in {"freq", "raw_mul_mu", "source_adaptive"}:
            return raw * mu
        return raw * mu

    @torch.no_grad()
    def _build_batch_local_semantic_frequency_pairs(self, batch: Data):
        if (
            not self.fcrs_extra_loss
            or self.fcrs_extra_mode != "plain"
            or self.fcrs_extra_source not in {
                "semantic_frequency",
                "raw_mul_fcrs",
                "raw_mul_lcsr",
                "raw",
                "fcrs_mu",
                "lcsr_mu",
                "source_adaptive",
            }
            or not self.fcrs_batch_local_admission
        ):
            return None, None, self._empty_batch_local_stats()

        batch_input = self._get_admission_input_features(batch.x)
        batch_size = int(getattr(batch, "batch_size", 0))
        num_nodes = int(batch_input.size(0))
        if batch_size <= 0 or num_nodes <= 1:
            return None, None, self._empty_batch_local_stats()

        profile_active = self._fcrs_runtime_active is not None
        profile_device = batch.x.device if profile_active else None
        t_desc = self._runtime_now(profile_device) if profile_active else None
        norm_adj = self._build_normalized_adj(
            edge_index=batch.edge_index,
            num_nodes=num_nodes,
            dtype=batch_input.dtype,
            device=batch_input.device,
        )
        x1, xk = self._propagate_k(batch_input, norm_adj, num_hops=self.fcrs_filter_k)
        reps = {
            "id": F.normalize(batch_input, p=2, dim=-1),
            "low": F.normalize(xk, p=2, dim=-1),
            "mid": F.normalize(x1 - xk, p=2, dim=-1),
            "high": F.normalize(batch_input - x1, p=2, dim=-1),
        }
        if profile_active and t_desc is not None:
            self._runtime_profile_add("lcsr_frequency_descriptor_s", self._runtime_now(profile_device) - t_desc)
        if self.fcrs_batch_local_semantics == "legacy_topk":
            return self._build_batch_local_semantic_frequency_pairs_legacy(batch, batch_input, reps)
        if self.fcrs_batch_local_semantics == "candidate_bank_v2":
            return self._build_batch_local_semantic_frequency_pairs_candidate_bank_v2(batch, batch_input, reps)
        return self._build_batch_local_semantic_frequency_pairs_aligned(batch, batch_input, reps)

    def _reduce_weighted(self, values: Tensor, weights: Tensor | None) -> Tensor:
        if values.numel() == 0:
            return torch.zeros((), device=values.device, dtype=values.dtype)
        if weights is None:
            return values.mean()
        weights = weights.detach().to(values.device, dtype=values.dtype)
        return (weights * values).sum() / weights.sum().clamp_min(1e-12)

    def _compute_extra_positive_loss(
        self,
        extra_selected: Tensor,
        extra_pair_weight: Tensor | None = None,
    ):
        device = extra_selected.device
        if extra_selected.numel() == 0:
            zero = torch.zeros((), device=device, dtype=extra_selected.dtype)
            return zero, zero, zero, zero, zero

        extra_sim = self._reduce_weighted(extra_selected, extra_pair_weight)
        weights = None if extra_pair_weight is None else extra_pair_weight.detach().to(device=device, dtype=extra_selected.dtype)
        margin_value = torch.full((), float("nan"), device=device, dtype=extra_selected.dtype)
        saturation_gamma = torch.full((), float("nan"), device=device, dtype=extra_selected.dtype)

        linear_terms = 1.0 - extra_selected
        linear_loss = self._reduce_weighted(linear_terms, weights)

        if self.fcrs_positive_loss == "linear":
            # `1 - sim` keeps the same gradient as `-sim` while making the
            # minimized objective a non-negative positive loss.
            loss_extra = linear_loss
            active_pair_ratio = torch.ones((), device=device, dtype=extra_selected.dtype)
        elif self.fcrs_positive_loss == "hinge":
            margin = float(self.fcrs_positive_margin)
            margin_value = torch.full((), margin, device=device, dtype=extra_selected.dtype)
            violation = margin - extra_selected
            loss_terms = F.relu(violation)
            active_mask = (violation > 0).to(extra_selected.dtype)
            loss_extra = self._reduce_weighted(loss_terms, weights)
            active_pair_ratio = self._reduce_weighted(active_mask, weights)
        elif self.fcrs_positive_loss == "softplus_hinge":
            margin = float(self.fcrs_positive_margin)
            margin_value = torch.full((), margin, device=device, dtype=extra_selected.dtype)
            temp = max(float(self.fcrs_positive_temperature), 1e-6)
            scaled = (margin - extra_selected) / temp
            loss_terms = temp * F.softplus(scaled)
            active_mask = ((margin - extra_selected) > 0).to(extra_selected.dtype)
            loss_extra = self._reduce_weighted(loss_terms, weights)
            active_pair_ratio = self._reduce_weighted(active_mask, weights)
        elif self.fcrs_positive_loss == "quantile_hinge":
            q = float(min(max(self.fcrs_positive_quantile, 0.0), 1.0))
            margin_value = torch.quantile(extra_selected.detach(), q=q)
            violation = margin_value - extra_selected
            loss_terms = F.relu(violation)
            active_mask = (violation > 0).to(extra_selected.dtype)
            loss_extra = self._reduce_weighted(loss_terms, weights)
            active_pair_ratio = self._reduce_weighted(active_mask, weights)
        elif self.fcrs_positive_loss == "saturation_gate":
            q = float(min(max(self.fcrs_positive_quantile, 0.0), 1.0))
            margin_value = torch.quantile(extra_selected.detach(), q=q)
            violation = margin_value - extra_selected
            quantile_terms = F.relu(violation)
            quantile_active = (violation > 0).to(extra_selected.dtype)
            quantile_loss = self._reduce_weighted(quantile_terms, weights)
            quantile_active_ratio = self._reduce_weighted(quantile_active, weights)
            tau_sat = float(self.fcrs_saturation_tau)
            temp_sat = max(float(self.fcrs_saturation_temp), 1e-6)
            saturation_gamma = torch.sigmoid((extra_sim.detach() - tau_sat) / temp_sat)
            loss_extra = (1.0 - saturation_gamma) * linear_loss + saturation_gamma * quantile_loss
            active_pair_ratio = (1.0 - saturation_gamma) + saturation_gamma * quantile_active_ratio
        else:
            raise ValueError(f"Unknown fcrs_positive_loss: {self.fcrs_positive_loss}")
        return loss_extra, extra_sim, active_pair_ratio, margin_value, saturation_gamma

    def _compute_loss(
        self,
        z1: Tensor,
        z2: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None = None,
        extra_pair_index: Tensor | None = None,
        extra_pair_weight: Tensor | None = None,
        current_epoch: int | None = None,
        extra_pair_right: Tensor | None = None,
        positive_pair_index: Tensor | None = None,
        positive_pair_weight: Tensor | None = None,
        positive_pair_right: Tensor | None = None,
        spa_exclusion_edge_index: Tensor | None = None,
    ):
        device = z1.device
        profile_active = self._fcrs_runtime_active is not None
        z1 = F.normalize(z1, p=2, dim=-1)
        z2 = F.normalize(z2, p=2, dim=-1)

        t_base = self._runtime_now(device) if profile_active else None
        S = z1 @ z2.T
        num_nodes = z1.size(0)
        loss_ali = -torch.diag(S).mean()

        mask = torch.ones((num_nodes, num_nodes), device=device, dtype=torch.bool)
        mask.fill_diagonal_(False)
        exclusion_edge_index = edge_index if spa_exclusion_edge_index is None else spa_exclusion_edge_index
        excl_src, excl_dst = exclusion_edge_index
        mask[excl_src, excl_dst] = False

        src, dst = edge_index

        positive_scores = []
        positive_weights = []
        if src.numel() > 0:
            positive_scores.append(S[src, dst])
            positive_weights.append(
                edge_weight if edge_weight is not None else torch.ones(src.numel(), device=device, dtype=S.dtype)
            )
        if positive_pair_index is not None and positive_pair_index.numel() > 0:
            if positive_pair_right is None:
                positive_scores.append(S[positive_pair_index[0], positive_pair_index[1]])
            else:
                positive_right = F.normalize(positive_pair_right, p=2, dim=-1)
                positive_scores.append(
                    (z1[positive_pair_index[0]] * positive_right[positive_pair_index[1]]).sum(dim=-1)
                )
            positive_weights.append(
                positive_pair_weight
                if positive_pair_weight is not None
                else torch.ones(positive_pair_index.size(1), device=device, dtype=S.dtype)
            )

        if positive_scores:
            positive_selected = torch.cat(positive_scores, dim=0)
            positive_weight = torch.cat(positive_weights, dim=0).to(positive_selected.device)
            loss_nei = -(positive_selected * positive_weight).sum() / positive_weight.sum().clamp_min(1e-12)
        else:
            loss_nei = torch.zeros((), device=device)

        S_spa = torch.masked_select(S, mask)
        S_spa = torch.sigmoid((S_spa - self.s) / self.tau)
        loss_spa = S_spa.mean()
        loss_ns4gc = loss_ali + self.lam * loss_nei + self.gam * loss_spa

        loss_extra = torch.zeros((), device=device)
        extra_sim = torch.zeros((), device=device)
        extra_ratio = torch.zeros((), device=device)
        active_pair_ratio = torch.zeros((), device=device)
        positive_margin_value = torch.full((), float("nan"), device=device)
        saturation_gamma = torch.full((), float("nan"), device=device)
        effective_extra_lambda = self._effective_extra_lambda(current_epoch)
        extra_active = (
            self.fcrs_extra_loss
            and effective_extra_lambda > 0
            and extra_pair_index is not None
            and (current_epoch is None or current_epoch > self.fcrs_extra_warmup)
        )
        if extra_pair_index is not None and extra_pair_index.numel() > 0:
            if profile_active and t_base is not None:
                base_elapsed = self._runtime_now(device) - t_base
                self._runtime_profile_add("ns4gc_base_loss_s", base_elapsed)
                t_base = None
            t_pair = self._runtime_now(device) if profile_active else None
            if extra_pair_right is None:
                extra_selected = S[extra_pair_index[0], extra_pair_index[1]]
            else:
                extra_right = F.normalize(extra_pair_right, p=2, dim=-1)
                extra_selected = (z1[extra_pair_index[0]] * extra_right[extra_pair_index[1]]).sum(dim=-1)
            loss_extra_raw, extra_sim, active_pair_ratio, positive_margin_value, saturation_gamma = self._compute_extra_positive_loss(
                extra_selected,
                extra_pair_weight=extra_pair_weight,
            )
            if profile_active and t_pair is not None:
                self._runtime_profile_add("lcsr_pair_loss_s", self._runtime_now(device) - t_pair)
        if extra_active and extra_pair_index.numel() > 0:
            loss_extra = loss_extra_raw
            extra_ratio = effective_extra_lambda * loss_extra / loss_ns4gc.detach().abs().clamp_min(1e-12)

        loss = loss_ns4gc + effective_extra_lambda * loss_extra
        if profile_active and t_base is not None:
            base_elapsed = self._runtime_now(device) - t_base
            self._runtime_profile_add("ns4gc_base_loss_s", base_elapsed)
        return (
            loss,
            loss_ali,
            loss_nei,
            loss_spa,
            loss_ns4gc,
            loss_extra,
            extra_sim,
            extra_ratio,
            effective_extra_lambda,
            active_pair_ratio,
            positive_margin_value,
            saturation_gamma,
        )

    def loss(self, x: Tensor, edge_index: Tensor, current_epoch: int | None = None, **kwargs) -> LossOutput:
        z1, z2 = self(x, edge_index, **kwargs)
        extra_pair_index, extra_pair_weight = self._build_full_extra_pair_index()
        neighbor_edge_index = self._rectify_full_edge_index_for_loss(edge_index)
        neighbor_edge_weight = self._lookup_release_edge_weights(edge_index, self.fcrs_num_nodes)
        spa_exclusion_edge_index = self._build_full_spa_exclusion_edge_index(edge_index)
        positive_pair_index = self._expand_pair_index_bidirectional(self.fcrs_positive_pair_index)
        positive_pair_weight = self._expand_pair_weight_bidirectional(self.fcrs_positive_pair_weights)
        if extra_pair_index is not None:
            extra_pair_index = extra_pair_index.to(edge_index.device)
        if extra_pair_weight is not None:
            extra_pair_weight = extra_pair_weight.to(edge_index.device)
        if positive_pair_index is not None:
            positive_pair_index = positive_pair_index.to(edge_index.device)
        if positive_pair_weight is not None:
            positive_pair_weight = positive_pair_weight.to(edge_index.device)
        extra_pair_index, extra_pair_weight = self._apply_plus_dropout(
            extra_pair_index,
            extra_pair_weight,
            device=edge_index.device,
            current_epoch=current_epoch,
        )
        (
            loss,
            loss_ali,
            loss_nei,
            loss_spa,
            loss_ns4gc,
            loss_extra,
            extra_sim,
            extra_ratio,
            effective_extra_lambda,
            active_pair_ratio,
            positive_margin_value,
            saturation_gamma,
        ) = self._compute_loss(
            z1,
            z2,
            neighbor_edge_index,
            edge_weight=neighbor_edge_weight,
            extra_pair_index=extra_pair_index,
            extra_pair_weight=extra_pair_weight,
            current_epoch=current_epoch,
            positive_pair_index=positive_pair_index,
            positive_pair_weight=positive_pair_weight,
            spa_exclusion_edge_index=spa_exclusion_edge_index,
        )
        components = {
            'ali': loss_ali.detach(),
            'nei': loss_nei.detach(),
            'spa': loss_spa.detach(),
            'ns4gc': loss_ns4gc.detach(),
        }
        if self.fcrs_extra_loss:
            components['extra'] = loss_extra.detach()
            components['extra_sim'] = extra_sim.detach()
            components['extra_ratio'] = extra_ratio.detach()
            components['effective_extra_lambda'] = float(effective_extra_lambda)
            components['active_pair_ratio'] = active_pair_ratio.detach()
            components['positive_margin_value'] = positive_margin_value.detach()
            components['saturation_gamma'] = saturation_gamma.detach()
        return LossOutput(total=loss, components=components)

    def loss_batch(self, batch: Data, current_epoch: int | None = None) -> LossOutput:
        self._current_epoch_for_audit = current_epoch
        self._runtime_profile_start_batch(current_epoch=current_epoch, device=batch.x.device)
        t_forward = self._runtime_now(batch.x.device) if self._fcrs_runtime_active is not None else None
        z1_full, z2_full = self(batch.x, batch.edge_index)
        if self._fcrs_runtime_active is not None and t_forward is not None:
            self._runtime_profile_add("ns4gc_forward_s", self._runtime_now(batch.x.device) - t_forward)
        z1 = z1_full[:batch.batch_size]
        z2 = z2_full[:batch.batch_size]

        batch_mask = (batch.edge_index[0] < batch.batch_size) & (batch.edge_index[1] < batch.batch_size)
        batch_edge_index_observed = batch.edge_index[:, batch_mask]
        spa_exclusion_edge_index = self._build_batch_spa_exclusion_edge_index(batch, batch_edge_index_observed)
        batch_edge_index = self._rectify_batch_edge_index_for_loss(batch=batch, batch_edge_index=batch_edge_index_observed)
        batch_edge_weight = self._lookup_batch_release_edge_weights(batch, batch_edge_index)
        extra_pair_index, extra_pair_weight = self._build_batch_extra_pair_index(batch)
        positive_pair_index, positive_pair_weight = self._build_batch_positive_pair_index(batch)
        batch_local_stats = {
            "extra_pairs": 0.0,
            "admit_score": 0.0,
            "gap_raw": 0.0,
            "gap_freq": 0.0,
            "gap_mu": 0.0,
            "gap_raw_mul_mu": 0.0,
            "weight_raw": 0.0,
            "weight_freq": 0.0,
            "weight_mu": 0.0,
            "weight_raw_mul_mu": 0.0,
            "risk_budget_scale": 1.0,
        }
        extra_pair_right = None
        positive_pair_right = None
        if extra_pair_index is not None and self.fcrs_extra_pair_index is not None:
            extra_pair_right = z2_full
        if positive_pair_index is not None and self.fcrs_positive_pair_index is not None:
            positive_pair_right = z2_full
        if extra_pair_index is None and self.fcrs_batch_local_admission:
            extra_pair_index, extra_pair_weight, batch_local_stats = self._build_batch_local_semantic_frequency_pairs(batch)
            if extra_pair_index is not None:
                extra_pair_right = z2_full
        extra_pair_index, extra_pair_weight = self._apply_plus_dropout(
            extra_pair_index,
            extra_pair_weight,
            device=batch.x.device,
            current_epoch=current_epoch,
        )
        (
            loss,
            loss_ali,
            loss_nei,
            loss_spa,
            loss_ns4gc,
            loss_extra,
            extra_sim,
            extra_ratio,
            effective_extra_lambda,
            active_pair_ratio,
            positive_margin_value,
            saturation_gamma,
        ) = self._compute_loss(
            z1,
            z2,
            batch_edge_index,
            edge_weight=batch_edge_weight,
            extra_pair_index=extra_pair_index,
            extra_pair_weight=extra_pair_weight,
            current_epoch=current_epoch,
            extra_pair_right=extra_pair_right,
            positive_pair_index=positive_pair_index,
            positive_pair_weight=positive_pair_weight,
            positive_pair_right=positive_pair_right,
            spa_exclusion_edge_index=spa_exclusion_edge_index,
        )
        t_diag = self._runtime_now(batch.x.device) if self._fcrs_runtime_active is not None else None
        components = {
            'ali': loss_ali.detach(),
            'nei': loss_nei.detach(),
            'spa': loss_spa.detach(),
            'ns4gc': loss_ns4gc.detach(),
        }
        if self.fcrs_extra_loss:
            components['extra'] = loss_extra.detach()
            components['extra_sim'] = extra_sim.detach()
            components['extra_ratio'] = extra_ratio.detach()
            components['effective_extra_lambda'] = float(effective_extra_lambda)
            components['active_pair_ratio'] = active_pair_ratio.detach()
            components['positive_margin_value'] = positive_margin_value.detach()
            components['saturation_gamma'] = saturation_gamma.detach()
            components['extra_pairs'] = batch_local_stats['extra_pairs']
            components['admit_score'] = batch_local_stats['admit_score']
            components['gap_raw'] = batch_local_stats['gap_raw']
            components['gap_freq'] = batch_local_stats['gap_freq']
            components['gap_mu'] = batch_local_stats['gap_mu']
            components['gap_raw_mul_mu'] = batch_local_stats['gap_raw_mul_mu']
            components['weight_raw'] = batch_local_stats['weight_raw']
            components['weight_freq'] = batch_local_stats['weight_freq']
            components['weight_mu'] = batch_local_stats['weight_mu']
            components['weight_raw_mul_mu'] = batch_local_stats['weight_raw_mul_mu']
            components['risk_budget_scale'] = batch_local_stats['risk_budget_scale']
        if self._fcrs_runtime_active is not None and t_diag is not None:
            self._runtime_profile_add("diagnostics_logging_s", self._runtime_now(batch.x.device) - t_diag)
        if self._fcrs_runtime_active is not None:
            self._fcrs_runtime_finalize_ctx = (batch, batch_local_stats, extra_pair_index)
        return LossOutput(total=loss, components=components)
