import argparse
import csv
import json
import os
import time
from statistics import mean
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from pyagc.clusters import KMeansClusterHead
from pyagc.data import get_dataset
from pyagc.encoders import create_tuned_gnn
from pyagc.metrics import label_metrics, structure_metrics
from pyagc.models import LCSR
from pyagc.utils.lcsr_consensus import LCSR_CONSENSUS_CHOICES
from pyagc.utils.lcsr_extra_pairs import build_lcsr_extra_pair_package
from pyagc.utils.lcsr import (
    build_lcsr_pair_package,
    build_lcsr_pair_package_with_embedding_refinement,
)
from pyagc.utils.lcsr_candidate_bank import build_or_load_lcsr_candidate_bank
from pyagc.utils.lcsr_pair_source_comparison import (
    evaluate_nonedge_source,
    run_pair_source_comparison,
    select_nonedge_topk_by_source,
)
from pyagc.transforms import GSSLTransform
from pyagc.utils import CheckpointManager, get_training_config, get_logger, set_seed
from pyagc.utils.lcsr_nonedge_diag import run_lcsr_nonedge_diagnostic


def train_full_batch(model, data, optimizer, conf, logger, device, ckpt_manager=None,
                     resume_from_ckpt=False, epoch_refine_callback=None):
    """Full-batch training for small graphs."""
    logger.info("=" * 60)
    logger.info("Training Mode: Full-batch")
    logger.info("=" * 60)

    data = data.to(device)
    epochs = conf.get('epochs', 200)
    patience = conf.get('patience', 50)
    save_every = conf.get('save_every', None)  # None means no periodic saves

    # Resume from checkpoint if requested
    start_epoch = 1
    best_loss = float('inf')
    patience_counter = 0

    if resume_from_ckpt and ckpt_manager is not None:
        checkpoint = ckpt_manager.load_checkpoint(
            model, optimizer, load_best=False, device=device
        )
        if checkpoint is not None:
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            patience_counter = checkpoint.get('patience_counter', 0)
            logger.info(f"Resuming training from epoch {start_epoch}")

    epoch_times = []
    epoch_stats = []
    start_time = time.time()

    epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        loss = model.train_full(data, optimizer, epoch, verbose=epoch == 1 or (epoch % 10 == 0),
                                current_epoch=epoch)
        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)
        if getattr(model, 'last_epoch_loss_stats', None) is not None:
            epoch_stats.append(model.last_epoch_loss_stats)
            components = model.last_epoch_loss_stats.get('components', {})
            effective_extra_lambda = float(components.get('effective_extra_lambda', 0.0))
            positive_loss = float(components.get('positive_loss', components.get('extra', 0.0)))
            extra_pair_count = float(components.get('extra_pair_count', components.get('extra_pairs', 0.0)))
            active_flag = bool(effective_extra_lambda > 0 and extra_pair_count > 0)
            logger.info(
                "LCSR_EPOCH_AUDIT "
                f"epoch={epoch} "
                f"effective_extra_lambda={effective_extra_lambda:.6f} "
                f"positive_loss={positive_loss:.6f} "
                f"extra_pair_count={extra_pair_count:.2f} "
                f"active_flag={int(active_flag)}"
            )

        # Determine if this is the best model
        is_best = loss < best_loss
        if is_best:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch_refine_callback is not None:
            epoch_refine_callback(model=model, epoch=epoch, logger=logger)

        # Save checkpoint based on configuration
        if ckpt_manager is not None:
            # Always save if it's the best or last epoch
            should_save = is_best or (epoch == epochs)

            # Save periodically if save_every is specified
            if save_every is not None and save_every > 0:
                should_save = should_save or (epoch % save_every == 0)

            if should_save:
                ckpt_manager.save_checkpoint(
                    model, optimizer, epoch, loss, is_best=is_best,
                    additional_info={'patience_counter': patience_counter}
                )

        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    total_time = time.time() - start_time
    avg_epoch_time = np.mean(epoch_times) * 1000  # Convert to ms

    return total_time, avg_epoch_time, epoch, epoch_stats


def train_mini_batch(model, data, optimizer, conf, logger, device,
                     ckpt_manager=None, resume_from_ckpt=False, epoch_refine_callback=None):
    """Mini-batch training for large graphs."""
    logger.info("=" * 60)
    logger.info("Training Mode: Mini-batch (NeighborLoader)")
    logger.info("=" * 60)

    num_layers = conf.get('num_layers', 1)
    fan_out = conf.get('fan_out', 10)
    batch_size = conf.get('batch_size', 1024)
    num_workers = conf.get('num_workers', 0)

    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Fan-out: [{fan_out}] × {num_layers} layers")

    train_loader = NeighborLoader(
        data,
        input_nodes=None,
        num_neighbors=[fan_out] * num_layers,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    epochs = conf.get('epochs', 200)
    patience = conf.get('patience', 50)
    save_every = conf.get('save_every', None)  # None means no periodic epoch saves
    save_every_batch = conf.get('save_every_batch', None)  # Save every N batches

    # Resume from checkpoint if requested
    start_epoch = 1
    start_batch = 0
    best_loss = float('inf')
    patience_counter = 0

    if resume_from_ckpt and ckpt_manager is not None:
        checkpoint = ckpt_manager.load_checkpoint(
            model, optimizer, load_best=False, device=device
        )
        if checkpoint is not None:
            start_epoch = checkpoint['epoch']
            start_batch = checkpoint.get('batch_idx', 0)
            best_loss = checkpoint.get('best_loss', float('inf'))
            patience_counter = checkpoint.get('patience_counter', 0)

            # If we have intra-epoch checkpoint but save_every_batch is None,
            # we can't resume from mid-epoch, so start from next epoch
            if start_batch > 0 and save_every_batch is None:
                logger.info(
                    f"Warning: Found intra-epoch checkpoint (batch {start_batch}) "
                    f"but save_every_batch is disabled. Starting from next epoch."
                )
                start_epoch += 1
                start_batch = 0

            # If we finished the epoch, move to next epoch
            if start_batch >= len(train_loader):
                start_epoch += 1
                start_batch = 0

            if start_batch > 0:
                logger.info(f"Resuming training from epoch {start_epoch}, batch {start_batch}")
            else:
                logger.info(f"Resuming training from epoch {start_epoch}")

    epoch_times = []
    epoch_stats = []
    start_time = time.time()

    epoch = start_epoch - 1
    # Choose training strategy based on save_every_batch
    if save_every_batch is None or save_every_batch <= 0:
        # Simple case: Use model.train_batch() directly
        logger.info("Using standard epoch-level training (no intra-epoch checkpoints)")

        for epoch in range(start_epoch, epochs + 1):
            epoch_start_time = time.time()

            # Use the model's built-in train_batch method
            avg_loss = model.train_batch(
                train_loader, optimizer, epoch,
                verbose=(epoch == 1 or epoch % 10 == 0),
                current_epoch=epoch,
            )

            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            if getattr(model, 'last_epoch_loss_stats', None) is not None:
                epoch_stats.append(model.last_epoch_loss_stats)
                components = model.last_epoch_loss_stats.get('components', {})
                effective_extra_lambda = float(components.get('effective_extra_lambda', 0.0))
                positive_loss = float(components.get('positive_loss', components.get('extra', 0.0)))
                extra_pair_count = float(components.get('extra_pair_count', components.get('extra_pairs', 0.0)))
                active_flag = bool(effective_extra_lambda > 0 and extra_pair_count > 0)
                logger.info(
                    "LCSR_EPOCH_AUDIT "
                    f"epoch={epoch} "
                    f"effective_extra_lambda={effective_extra_lambda:.6f} "
                    f"positive_loss={positive_loss:.6f} "
                    f"extra_pair_count={extra_pair_count:.2f} "
                    f"active_flag={int(active_flag)}"
                )

            # Determine if this is the best model
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch_refine_callback is not None:
                epoch_refine_callback(model=model, epoch=epoch, logger=logger)

            # Save checkpoint based on configuration
            if ckpt_manager is not None:
                # Always save if it's the best or last epoch
                should_save = is_best or (epoch == epochs)

                # Save periodically if save_every is specified
                if save_every is not None and save_every > 0:
                    should_save = should_save or (epoch % save_every == 0)

                if should_save:
                    ckpt_manager.save_checkpoint(
                        model, optimizer, epoch, avg_loss, is_best=is_best,
                        additional_info={'patience_counter': patience_counter}
                    )

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            if hasattr(model, 'should_stop_training_early') and model.should_stop_training_early():
                logger.info("Stopping early after collecting the requested LCSR audit/profile batches.")
                break

    else:
        # Complex case: Save checkpoints at batch level (for very large datasets)
        logger.info(f"Using intra-epoch checkpointing (save every {save_every_batch} batches)")

        for epoch in range(start_epoch, epochs + 1):
            model.train()

            # Determine starting batch for current epoch
            batch_start = start_batch if epoch == start_epoch else 0

            # Setup progress bar for first epoch and every 10 epochs
            if batch_start == 0 and (epoch == 1 or epoch % 10 == 0):
                from tqdm import tqdm
                pbar = tqdm(total=len(train_loader) * batch_size)
                pbar.set_description(f'Epoch {epoch:03d}')
            else:
                pbar = None

            epoch_start_time = time.time()
            total_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                # Skip batches if resuming from checkpoint
                if batch_idx < batch_start:
                    if pbar is not None:
                        pbar.update(batch.batch_size)
                    continue

                batch = batch.to(device)
                optimizer.zero_grad()

                loss_output = model.loss_batch(batch, current_epoch=epoch)

                # Handle both single Tensor and LossOutput returns
                if hasattr(loss_output, 'total'):
                    loss = loss_output.total
                else:
                    loss = loss_output

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                if pbar is not None:
                    pbar.update(batch.batch_size)

                # Save checkpoint every N batches
                if (batch_idx + 1) % save_every_batch == 0:
                    avg_loss = total_loss / num_batches
                    is_best = avg_loss < best_loss
                    if is_best:
                        best_loss = avg_loss
                        patience_counter = 0

                    if ckpt_manager is not None:
                        ckpt_manager.save_checkpoint(
                            model, optimizer, epoch, avg_loss,
                            is_best=is_best, batch_idx=batch_idx + 1,
                            additional_info={'patience_counter': patience_counter}
                        )

            if pbar is not None:
                pbar.close()

            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)

            # Compute average loss for the epoch
            avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')

            # Log epoch results
            if epoch == 1 or epoch % 10 == 0:
                logger.info(f"Epoch: {epoch:03d} Loss: {avg_loss:.4f}")
            epoch_stats.append({
                'loss': float(avg_loss),
                'components': {
                    name: value / num_batches for name, value in components_sum.items()
                },
            })
            components = epoch_stats[-1].get('components', {})
            effective_extra_lambda = float(components.get('effective_extra_lambda', 0.0))
            positive_loss = float(components.get('positive_loss', components.get('extra', 0.0)))
            extra_pair_count = float(components.get('extra_pair_count', components.get('extra_pairs', 0.0)))
            active_flag = bool(effective_extra_lambda > 0 and extra_pair_count > 0)
            logger.info(
                "LCSR_EPOCH_AUDIT "
                f"epoch={epoch} "
                f"effective_extra_lambda={effective_extra_lambda:.6f} "
                f"positive_loss={positive_loss:.6f} "
                f"extra_pair_count={extra_pair_count:.2f} "
                f"active_flag={int(active_flag)}"
            )

            # Determine if this is the best model
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch_refine_callback is not None:
                epoch_refine_callback(model=model, epoch=epoch, logger=logger)

            # Save checkpoint at end of epoch
            if ckpt_manager is not None:
                # Always save if it's the best or last epoch
                should_save = is_best or (epoch == epochs)

                # Save periodically if save_every is specified
                if save_every is not None and save_every > 0:
                    should_save = should_save or (epoch % save_every == 0)

                if should_save:
                    ckpt_manager.save_checkpoint(
                        model, optimizer, epoch, avg_loss, is_best=is_best,
                        batch_idx=len(train_loader),  # Mark as end of epoch
                        additional_info={'patience_counter': patience_counter}
                    )

            # Reset start_batch for subsequent epochs
            if epoch == start_epoch:
                start_batch = 0

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    total_time = time.time() - start_time
    avg_epoch_time = np.mean(epoch_times) * 1000 if epoch_times else 0.0  # Convert to ms

    return total_time, avg_epoch_time, epoch, epoch_stats


def summarize_epoch_stats(epoch_stats, extra_lambda):
    if not epoch_stats:
        return None
    component_keys = set()
    for item in epoch_stats:
        component_keys.update(item.get('components', {}).keys())

    component_means = {}
    for key in sorted(component_keys):
        values = [item['components'][key] for item in epoch_stats if key in item.get('components', {})]
        if values:
            component_means[key] = float(mean(values))

    start_extra_sim = None
    end_extra_sim = None
    extra_sim_values = [
        item['components']['extra_sim']
        for item in epoch_stats
        if 'extra_sim' in item.get('components', {})
    ]
    if extra_sim_values:
        start_extra_sim = float(extra_sim_values[0])
        end_extra_sim = float(extra_sim_values[-1])

    return {
        'loss_mean': float(mean(item['loss'] for item in epoch_stats)),
        'components_mean': component_means,
        'extra_lambda': float(extra_lambda),
        'extra_sim_start': start_extra_sim,
        'extra_sim_end': end_extra_sim,
    }


def _fmt_optional_float(value):
    if value is None:
        return "nan"
    return f"{value:.6f}"


def _resolved_filter_k(args, conf):
    return int(args.fcrs_filter_k if args.fcrs_filter_k is not None else conf.get('num_hops', 2))


def _collect_resolved_config(args, conf, effective_extra_loss):
    resolved = {
        'dataset': args.dataset,
        'mini_batch': bool(conf.get('mini_batch', True)),
        'gnn_type': conf.get('gnn_type', 'sage').lower(),
        'cached': bool(conf.get('cached', True)),
        'num_layers': int(conf.get('num_layers', 1)),
        'fan_out': int(conf.get('fan_out', -1)),
        'batch_size': int(conf.get('batch_size', -1)),
        'epochs': int(conf.get('epochs', 0)),
        'fcrs_filter_k': _resolved_filter_k(args, conf),
        'num_hops': int(conf.get('num_hops', _resolved_filter_k(args, conf))),
        'fcrs_extra_loss': bool(effective_extra_loss),
        'fcrs_extra_lambda': float(args.fcrs_extra_lambda),
        'fcrs_extra_warmup': int(args.fcrs_extra_warmup),
        'fcrs_batch_local_admission': False,
        'fcrs_batch_local_semantics': args.lcsr_batch_local_semantics,
        'lcsr_support_source': args.lcsr_support_source,
        'lcsr_positive_mode': args.lcsr_positive_mode or args.lcsr_positive_loss,
        'lcsr_margin': float(args.lcsr_margin),
        'lcsr_rho': float(args.lcsr_rho),
        'lcsr_kmax': int(args.lcsr_kmax),
        'lcsr_candidate_pool_size': int(args.lcsr_candidate_pool_size),
        'lcsr_candidate_bank_size': int(args.lcsr_candidate_bank_size),
        'lcsr_budget_match': bool(args.lcsr_budget_match),
        'lcsr_use_gate': False,
        'lcsr_release': bool(args.lcsr_release),
        'lcsr_complete': bool(args.lcsr_complete),
        'lcsr_weighted_complete': bool(args.lcsr_weighted_complete),
    }
    return resolved


def _log_resolved_config(logger, resolved_config):
    logger.info("RESOLVED_CONFIG_BEGIN")
    for key, value in resolved_config.items():
        logger.info(f"RESOLVED_CONFIG {key}={value}")
    logger.info("RESOLVED_CONFIG_END")


def _validate_minibatch_gcn_cached_guard(conf):
    if (
        bool(conf.get('mini_batch', True))
        and str(conf.get('gnn_type', 'sage')).lower() == 'gcn'
        and bool(conf.get('cached', True))
    ):
        raise ValueError(
            "Invalid config: mini_batch=true with gnn_type=gcn requires cached=false. "
            "NeighborLoader mini-batch GCN cannot reuse cached normalization safely."
        )


def _warn_if_lcsr_never_activates(logger, args, conf, effective_extra_loss):
    if (
        args.dataset == 'Physics'
        and effective_extra_loss
        and int(conf.get('epochs', 0)) == 50
        and int(args.fcrs_extra_warmup) >= 50
    ):
        logger.warning("LCSR never becomes active under the current epoch budget.")


def _summarize_pair_same_ratio(y, left, right, scores=None, q=0.2):
    if y is None or left is None or right is None:
        return {
            'same_ratio': None,
            'top20_same': None,
            'bottom20_same': None,
            'gap': None,
        }
    left = torch.as_tensor(left, dtype=torch.long)
    right = torch.as_tensor(right, dtype=torch.long)
    if left.numel() == 0 or right.numel() == 0:
        return {
            'same_ratio': None,
            'top20_same': None,
            'bottom20_same': None,
            'gap': None,
        }
    y_cpu = y.detach().cpu()
    valid = torch.ones(left.numel(), dtype=torch.bool)
    if torch.is_floating_point(y_cpu):
        valid = (~torch.isnan(y_cpu[left])) & (~torch.isnan(y_cpu[right]))
    if int(valid.sum().item()) == 0:
        return {
            'same_ratio': None,
            'top20_same': None,
            'bottom20_same': None,
            'gap': None,
        }
    same = (y_cpu[left][valid] == y_cpu[right][valid]).float()
    result = {
        'same_ratio': float(same.mean().item()),
        'top20_same': None,
        'bottom20_same': None,
        'gap': None,
    }
    if scores is not None:
        score_t = torch.as_tensor(scores, dtype=torch.float32)
        score_t = score_t[valid]
        if score_t.numel() > 0:
            k = max(1, int(np.ceil(score_t.numel() * q)))
            order = torch.argsort(score_t)
            low = float(same[order[:k]].mean().item())
            high = float(same[order[-k:]].mean().item())
            result['top20_same'] = high
            result['bottom20_same'] = low
            result['gap'] = high - low
    return result


def _build_physics_label_only_diagnostic(model, data, y):
    result = {
        'random_nonedge_same_label_ratio': None,
        'candidate_bank_same_label_ratio': None,
        'post_margin_pair_same_label_ratio': None,
        'admitted_a_plus_same_label_ratio': None,
        'observed_edge_same_label_ratio': None,
        'admitted_score_top20_same_label_ratio': None,
        'admitted_score_bottom20_same_label_ratio': None,
        'admitted_score_same_label_gap': None,
    }
    if y is None:
        return result

    edge_src = data.edge_index[0].detach().cpu().long()
    edge_dst = data.edge_index[1].detach().cpu().long()
    observed = _summarize_pair_same_ratio(y, edge_src, edge_dst)
    result['observed_edge_same_label_ratio'] = observed['same_ratio']
    num_nodes = int(data.x.size(0))
    undirected_edges = set()
    for u, v in zip(edge_src.tolist(), edge_dst.tolist()):
        if u == v:
            continue
        a = int(u) if int(u) < int(v) else int(v)
        b = int(v) if int(u) < int(v) else int(u)
        undirected_edges.add((a, b))
    rng = np.random.default_rng(0)
    random_left = []
    random_right = []
    max_trials = 20000
    target = min(4096, max(num_nodes // 4, 512))
    trials = 0
    while len(random_left) < target and trials < max_trials:
        left = int(rng.integers(0, num_nodes))
        right = int(rng.integers(0, num_nodes))
        trials += 1
        if left == right:
            continue
        a = left if left < right else right
        b = right if left < right else left
        if (a, b) in undirected_edges:
            continue
        random_left.append(left)
        random_right.append(right)
    random_diag = _summarize_pair_same_ratio(y, random_left, random_right)
    result['random_nonedge_same_label_ratio'] = random_diag['same_ratio']

    bank = getattr(model, 'fcrs_candidate_bank', None)
    if bank is not None and bank.numel() > 0:
        anchor = torch.arange(bank.size(0), dtype=torch.long).unsqueeze(1).expand_as(bank)
        valid = bank >= 0
        bank_diag = _summarize_pair_same_ratio(
            y,
            anchor[valid].detach().cpu(),
            bank[valid].detach().cpu(),
        )
        result['candidate_bank_same_label_ratio'] = bank_diag['same_ratio']

    route_rows = list(getattr(model, 'fcrs_route_audit_rows', []))
    if route_rows:
        margin_left = []
        margin_right = []
        admitted_left = []
        admitted_right = []
        admitted_scores = []
        for row in route_rows:
            margin_pairs = row.get('post_margin_global_pairs', [])
            admitted_pairs = row.get('admitted_global_pairs', [])
            if margin_pairs:
                margin_left.extend([int(p[0]) for p in margin_pairs])
                margin_right.extend([int(p[1]) for p in margin_pairs])
            if admitted_pairs:
                admitted_left.extend([int(p[0]) for p in admitted_pairs])
                admitted_right.extend([int(p[1]) for p in admitted_pairs])
            admitted_scores.extend([float(v) for v in row.get('admitted_scores', [])])
        margin_diag = _summarize_pair_same_ratio(y, margin_left, margin_right)
        admitted_diag = _summarize_pair_same_ratio(y, admitted_left, admitted_right, scores=admitted_scores)
        result['post_margin_pair_same_label_ratio'] = margin_diag['same_ratio']
        result['admitted_a_plus_same_label_ratio'] = admitted_diag['same_ratio']
        result['admitted_score_top20_same_label_ratio'] = admitted_diag['top20_same']
        result['admitted_score_bottom20_same_label_ratio'] = admitted_diag['bottom20_same']
        result['admitted_score_same_label_gap'] = admitted_diag['gap']

    return result


def _write_physics_audit_reports(json_path, md_path, report):
    json_path = Path(json_path)
    md_path = Path(md_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')

    lines = [
        "# Physics Audit Report",
        "",
        "## Resolved Config",
        "",
    ]
    for key, value in report.get('resolved_config', {}).items():
        lines.append(f"- `{key}`: {value}")
    lines.extend([
        "",
        "## Route Audit",
        "",
    ])
    for key, value in report.get('route_audit', {}).items():
        if key == 'admission_audit_rows':
            lines.append(f"- `admission_audit_rows`: {len(value)} rows")
        else:
            lines.append(f"- `{key}`: {value}")
    lines.extend([
        "",
        "## Effective Lambda History",
        "",
        "| Epoch | effective_extra_lambda | positive_loss | extra_pair_count | active_flag |",
        "| ---: | ---: | ---: | ---: | --- |",
    ])
    for row in report.get('effective_lambda_history', []):
        lines.append(
            f"| {row['epoch']} | {row['effective_extra_lambda']:.6f} | {row['positive_loss']:.6f} | "
            f"{row['extra_pair_count']:.2f} | {int(row['active_flag'])} |"
        )
    lines.extend([
        "",
        "## Candidate Quality Diagnostic",
        "",
    ])
    for key, value in report.get('candidate_quality_diagnostic', {}).items():
        lines.append(f"- `{key}`: {value}")
    lines.extend([
        "",
        "## Final Eight Metrics",
        "",
        "| Metric | Mean | Std |",
        "| --- | ---: | ---: |",
    ])
    for metric_name, metric_info in report.get('final_metrics', {}).items():
        lines.append(f"| {metric_name} | {metric_info['mean']:.4f} | {metric_info['std']:.4f} |")
    md_path.write_text("\n".join(lines) + "\n", encoding='utf-8')


def _write_lcsr_runtime_profile_report(path, args, conf, summary):
    if path is None or summary is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ranked = summary["ranked"]
    meta = summary["meta"]
    modules = summary["modules"]
    lines = [
        f"# LCSR ArXiv Runtime Profile",
        "",
        f"- Dataset: {args.dataset}",
        f"- Profiled effective batches: {meta['profiled_batches']}",
        f"- Batch size mean: {meta['batch_size_mean']:.2f}",
        f"- Subgraph nodes mean: {meta['num_nodes_mean']:.2f}",
        f"- Subgraph edges mean: {meta['num_edges_mean']:.2f}",
        f"- Extra pairs mean: {meta['extra_pairs_mean']:.2f}",
        f"- Candidate pool size: {args.lcsr_candidate_pool_size}",
        f"- Row block: {meta.get('row_block', 'nan')}",
        f"- Col block: {meta.get('col_block', 'nan')}",
        f"- Dense score elements per batch: {meta.get('dense_score_elements_per_batch', 'nan')}",
        "",
        "## Module Breakdown",
        "",
        "| Module | Mean Time (s) | Share |",
        "| --- | ---: | ---: |",
    ]
    for item in ranked:
        lines.append(f"| {item['module']} | {item['mean_s']:.6f} | {item['share'] * 100:.2f}% |")
    lines.extend([
        "",
        "## Top-3 Slowest",
        "",
        "| Rank | Module | Mean Time (s) | Share |",
        "| --- | --- | ---: | ---: |",
    ])
    for idx, item in enumerate(ranked[:3], 1):
        lines.append(f"| {idx} | {item['module']} | {item['mean_s']:.6f} | {item['share'] * 100:.2f}% |")
    lines.extend([
        "",
        "## Code-Path Audit",
        "",
        "- Batch-local path uses block-dense seed-by-subgraph scoring, not pre-pruned BxP candidate-only scoring.",
        "- Candidate assembly is still top-k based inside each block merge; no full-graph pair package is constructed.",
        "- Profiling mode only adds timing synchronization and does not change admission, loss, or message-passing semantics.",
        "",
        "## Raw Summary",
        "",
        "```json",
        json.dumps({
            "args": {
                "fcrs_extra_lambda": args.fcrs_extra_lambda,
                "fcrs_extra_k": args.fcrs_extra_k,
                "fcrs_extra_warmup": args.fcrs_extra_warmup,
                "lcsr_support_source": args.lcsr_support_source,
                "lcsr_positive_loss": args.lcsr_positive_mode or args.lcsr_positive_loss,
                "lcsr_candidate_pool_size": args.lcsr_candidate_pool_size,
                "lcsr_margin": args.lcsr_margin,
                "lcsr_rho": args.lcsr_rho,
                "lcsr_kmax": args.lcsr_kmax,
            },
            "meta": meta,
            "modules": modules,
            "top3": ranked[:3],
        }, indent=2, ensure_ascii=False),
        "```",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_lcsr_admission_audit_reports(markdown_path, json_path, args, summary):
    if summary is None:
        return
    if json_path is not None:
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if markdown_path is None:
        return
    markdown_path = Path(markdown_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    k = int(summary["config"]["fcrs_extra_k"])
    lines = [
        "# LCSR Batch-Local Admission Audit",
        "",
        f"- Dataset: {args.dataset}",
        f"- Effective batches audited: {summary['effective_batches']}",
        f"- `fcrs_extra_k`: {summary['config']['fcrs_extra_k']}",
        f"- `fcrs_extra_warmup`: {summary['config']['fcrs_extra_warmup']}",
        f"- `support_source`: {summary['config']['fcrs_extra_source']}",
        f"- `risk_budget`: {int(summary['config']['fcrs_risk_budget'])}",
        "",
        "## Aggregate Admission Fractions",
        "",
        "| Admitted per anchor | Fraction |",
        "| ---: | ---: |",
    ]
    for key, value in summary["aggregate"]["admitted_fraction_by_count"].items():
        lines.append(f"| {key} | {value:.6f} |")
    lines.extend([
        "",
        "## Aggregate Admitted Score Quantiles",
        "",
        "| Quantile | Score |",
        "| --- | ---: |",
    ])
    for key, value in summary["aggregate"]["admitted_score_quantiles"].items():
        lines.append(f"| {key} | {value:.6f} |")
    lines.extend([
        "",
        "## Per-Batch Summary",
        "",
        "| Batch | Epoch | Batch size | Subgraph nodes | Threshold type | Threshold value | Mean candidate count | Mean pass count | Mean admitted count | zero ratio | mean dyn budget | degree source | cache hit | bank shape | score shape | frac(0) | frac(1) | frac(2) | frac(k) | q50 | q90 | q99 |",
        "| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in summary["batches"]:
        cand = row["candidate_count_per_anchor"]
        passed = row["pass_count_per_anchor"] or []
        admitted = row["admitted_count_per_anchor"]
        frac = row["admitted_fraction_by_count"]
        q = row["admitted_score_quantiles"]
        mean_cand = sum(cand) / max(len(cand), 1)
        mean_pass = (sum(passed) / max(len(passed), 1)) if passed else float("nan")
        mean_admit = sum(admitted) / max(len(admitted), 1)
        lines.append(
            f"| {row['batch_index']} | {row['epoch']} | {row['batch_size']} | {row['num_nodes']} | "
            f"{row['threshold_type']} | {row['threshold_value'] if row['threshold_value'] is not None else 'nan'} | "
            f"{mean_cand:.2f} | {mean_pass:.2f} | {mean_admit:.2f} | "
            f"{row.get('zero_admission_ratio', float('nan')):.6f} | {row.get('mean_dynamic_budget', float('nan')):.6f} | "
            f"{row.get('budget_degree_source', 'unknown')} | {int(bool(row.get('candidate_bank_cache_hit', False)))} | "
            f"{row.get('candidate_bank_shape', '[]')} | {row.get('online_score_shape', '[]')} | "
            f"{frac.get('0', 0.0):.6f} | {frac.get('1', 0.0):.6f} | {frac.get('2', 0.0):.6f} | "
            f"{frac.get(str(k), 0.0):.6f} | {q.get('q50', float('nan')):.6f} | {q.get('q90', float('nan')):.6f} | {q.get('q99', float('nan')):.6f} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pair_same_ratio_from_index(y, pair_index):
    if y is None or pair_index is None or pair_index.numel() == 0:
        return None
    y_cpu = y.detach().cpu()
    left = pair_index[0].detach().cpu().long()
    right = pair_index[1].detach().cpu().long()
    return float((y_cpu[left] == y_cpu[right]).float().mean().item())


def _select_release_subset(drop_pair_index, drop_pair_scores, fraction: float):
    if drop_pair_index is None or drop_pair_index.numel() == 0:
        return drop_pair_index, drop_pair_scores
    fraction = float(max(0.0, min(1.0, fraction)))
    if fraction >= 1.0:
        return drop_pair_index, drop_pair_scores
    count = int(round(drop_pair_index.size(1) * fraction))
    if count <= 0:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float32)
    order = torch.argsort(drop_pair_scores, descending=False)
    keep = order[:count]
    return drop_pair_index[:, keep], drop_pair_scores[keep]


def _build_soft_release_weights(drop_pair_scores, beta: float):
    if drop_pair_scores is None or drop_pair_scores.numel() == 0:
        return drop_pair_scores
    suspiciousness = 1.0 - drop_pair_scores.float()
    min_s = suspiciousness.min()
    max_s = suspiciousness.max()
    if float(max_s - min_s) <= 1e-12:
        normalized = torch.zeros_like(suspiciousness)
    else:
        normalized = (suspiciousness - min_s) / (max_s - min_s)
    weights = 1.0 - float(beta) * normalized
    return weights.clamp_(0.3, 1.0)


def _build_score_weighted_release_weights(drop_pair_scores, weight_floor: float):
    if drop_pair_scores is None or drop_pair_scores.numel() == 0:
        return drop_pair_scores
    floor = float(max(0.0, min(1.0, weight_floor)))
    scores = drop_pair_scores.float().clamp_(0.0, 1.0)
    return floor + (1.0 - floor) * scores


def _normalize_positive_weights(weights: torch.Tensor) -> torch.Tensor:
    weights = weights.float().clamp_min(1e-6)
    return weights / weights.mean().clamp_min(1e-12)


def _build_lcsr_plus_weights(add_pair_scores, mode: str, floor: float):
    if add_pair_scores is None or add_pair_scores.numel() == 0:
        return None
    score = add_pair_scores.detach().cpu().float().clamp(0.0, 1.0)
    if mode == 'uniform':
        raw_weight = torch.ones_like(score)
    elif mode == 'score':
        raw_weight = score
    elif mode == 'score_floor':
        floor = float(max(0.0, min(1.0, floor)))
        raw_weight = floor + (1.0 - floor) * score
    else:
        raise ValueError(f"Unknown lcsr-plus-weight mode: {mode}")
    return _normalize_positive_weights(raw_weight)


@torch.no_grad()
def summarize_selected_extra_pairs(top_values, top_indices, y):
    selected_mask = (top_indices >= 0) & torch.isfinite(top_values)
    flat_scores = top_values[selected_mask]
    selected = int(selected_mask.sum().item())
    mean_score = float(flat_scores.mean().item()) if flat_scores.numel() > 0 else float("nan")

    y_cpu = y.detach().cpu()
    valid_nodes = torch.ones(y_cpu.size(0), dtype=torch.bool)
    if torch.is_floating_point(y_cpu):
        valid_nodes = ~torch.isnan(y_cpu)

    anchor_ids = torch.arange(y_cpu.size(0)).unsqueeze(1).expand_as(top_indices)
    flat_anchor = anchor_ids[selected_mask]
    flat_partner = top_indices[selected_mask]
    pair_valid = valid_nodes[flat_anchor] & valid_nodes[flat_partner]
    if pair_valid.any():
        same = (y_cpu[flat_anchor][pair_valid] == y_cpu[flat_partner][pair_valid]).float()
        same_ratio = float(same.mean().item())
        valid_scores = flat_scores[pair_valid]
        q = max(1, int(np.ceil(valid_scores.numel() * 0.2)))
        order = torch.argsort(valid_scores)
        bottom20_same = float(same[order[:q]].mean().item())
        top20_same = float(same[order[-q:]].mean().item())
        gap = top20_same - bottom20_same
    else:
        same_ratio = float("nan")
        top20_same = float("nan")
        bottom20_same = float("nan")
        gap = float("nan")

    return {
        'selected': selected,
        'mean_score': mean_score,
        'same_ratio': same_ratio,
        'top20_same': top20_same,
        'bottom20_same': bottom20_same,
        'gap': gap,
    }


@torch.no_grad()
def inference_embeddings(model, data, conf, logger, device, labeled_indices=None):
    """Generate embeddings based on mini_batch configuration with automatic fallback.

    Fallback priority:
    1. GPU full-batch
    2. CPU full-batch (if GPU OOM and allowed)
    3. GPU mini-batch

    Args:
        model: The trained model
        data: Full graph data
        conf: Configuration dictionary
        logger: Logger instance
        device: Device to use
        labeled_indices: Optional tensor of node indices to compute embeddings for.
                        If provided, only compute embeddings for these nodes.
    """
    model.eval()
    logger.info("=" * 60)
    logger.info("Inference Stage")
    logger.info("=" * 60)

    # Check configuration
    mini_batch_config = conf.get('mini_batch', True)
    force_full_batch_inference = conf.get('force_full_batch_inference', False)
    allow_cpu_fallback = conf.get('allow_cpu_fallback', True)

    original_device = device
    start_time = time.time()

    if labeled_indices is not None:
        logger.info(f"Computing embeddings only for {len(labeled_indices):,} selected nodes")
    else:
        logger.info(f"Computing embeddings for all nodes")

    # Try full-batch inference first if forced or not configured as mini_batch
    if force_full_batch_inference or not mini_batch_config:
        # Try GPU full-batch first
        logger.info(f"Attempting full-batch inference on {device}...")

        try:
            z = model.infer_full(data.to(device))
            z = z[labeled_indices] if labeled_indices is not None else z

            inference_time = time.time() - start_time
            logger.info(f"✓ Full-batch inference completed on {device} in {inference_time:.2f}s")
            return z.to(original_device), inference_time

        except RuntimeError as e:
            if 'out of memory' in str(e).lower() or 'oom' in str(e).lower():
                logger.warning(f"Full-batch inference failed on {device} due to OOM")

                # Clear CUDA cache if on GPU
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

                # Try CPU full-batch if allowed
                if allow_cpu_fallback and device.type == 'cuda':
                    logger.info("Attempting full-batch inference on CPU...")

                    try:
                        # Move model to CPU
                        model_device = next(model.parameters()).device
                        model.cpu()

                        cpu_start = time.time()
                        z = model.infer_full(data.to('cpu'))
                        z = z[labeled_indices] if labeled_indices is not None else z
                        cpu_time = time.time() - cpu_start

                        # Move model back to original device
                        model.to(model_device)

                        inference_time = time.time() - start_time
                        logger.info(f"✓ Full-batch inference completed on CPU in {cpu_time:.2f}s "
                                    f"(total with device transfer: {inference_time:.2f}s)")

                        return z.to(original_device), inference_time

                    except RuntimeError as cpu_e:
                        logger.warning(f"CPU full-batch inference also failed: {cpu_e}")
                        # Move model back to original device
                        model.to(model_device)
                        logger.info("Falling back to GPU mini-batch inference...")
                else:
                    logger.info("CPU fallback disabled, falling back to mini-batch inference...")
            else:
                # Re-raise if it's not a memory error
                raise e

    # GPU mini-batch inference (either configured or fallback from full-batch)
    logger.info(f"Using mini-batch inference on {device}...")

    num_layers = conf.get('num_layers', 1)

    # Use separate inference batch size if specified, otherwise use training batch size
    infer_batch_size = conf.get('infer_batch_size', conf.get('batch_size', 1024))

    # If infer_batch_size is still causing OOM, try to automatically reduce it
    original_infer_batch_size = infer_batch_size

    num_workers = conf.get('num_workers', 0)
    infer_fan_out = conf.get('infer_fan_out', -1)  # -1 means sample all neighbors

    logger.info(f"Inference batch size: {infer_batch_size}")
    logger.info(f"Inference fan-out: [{infer_fan_out}] × {num_layers} layers")
    logger.info(f"Num workers: {num_workers}")

    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            inference_loader = NeighborLoader(
                data,
                input_nodes=labeled_indices,  # Only sample from labeled nodes
                num_neighbors=[infer_fan_out] * num_layers,
                batch_size=infer_batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

            z = model.infer_batch(inference_loader, verbose=True)
            inference_time = time.time() - start_time

            if infer_batch_size < original_infer_batch_size:
                logger.info(f"✓ Mini-batch inference completed with reduced batch size "
                            f"({infer_batch_size}) in {inference_time:.2f}s")
            else:
                logger.info(f"✓ Mini-batch inference completed in {inference_time:.2f}s")

            return z, inference_time

        except RuntimeError as e:
            if 'out of memory' in str(e).lower() or 'oom' in str(e).lower():
                retry_count += 1

                if retry_count >= max_retries:
                    logger.error(f"Mini-batch inference failed after {max_retries} retries")
                    raise e

                # Reduce batch size by half
                infer_batch_size = max(infer_batch_size // 2, 32)

                logger.warning(f"Mini-batch inference OOM (attempt {retry_count}/{max_retries})")
                logger.info(f"Reducing inference batch size to {infer_batch_size} and retrying...")

                # Clear CUDA cache
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            else:
                # Re-raise if it's not a memory error
                raise e

    # Should not reach here
    raise RuntimeError("Inference failed after all retries")

@torch.no_grad()
def log_fcrs_label_diagnostics(model, data, y, logger, device, q=0.2):
    """Use labels only for post-hoc diagnostics. Labels are not used for training."""
    if not hasattr(model, "_update_fcrs_diagnostics"):
        logger.info("[FCRS label diag] Model has no FCRS diagnostics.")
        return

    # Recompute diagnostics on the clean full graph.
    was_training = model.training
    model.eval()
    model._update_fcrs_diagnostics(data.x.to(device), data.edge_index.to(device))
    if was_training:
        model.train()

    diag = getattr(model, "last_fcrs_diagnostics", None)
    if not diag:
        logger.info("[FCRS label diag] No diagnostics found.")
        return

    pair_index = diag.get("pair_index", None)
    if pair_index is None:
        logger.info("[FCRS label diag] No pair_index found.")
        return

    y_cpu = y.detach().cpu()
    src = pair_index[0].detach().cpu()
    dst = pair_index[1].detach().cpu()

    if torch.is_floating_point(y_cpu):
        valid = (~torch.isnan(y_cpu[src])) & (~torch.isnan(y_cpu[dst]))
    else:
        valid = torch.ones(src.numel(), dtype=torch.bool)

    if valid.sum().item() == 0:
        logger.info("[FCRS label diag] No valid labeled pairs.")
        return

    same = (y_cpu[src][valid] == y_cpu[dst][valid]).float()

    def report(name, score):
        score = score.detach().cpu()[valid]
        n = score.numel()
        k = max(1, int(n * q))

        order = torch.argsort(score)
        low_idx = order[:k]
        high_idx = order[-k:]

        low_same = same[low_idx].mean().item()
        high_same = same[high_idx].mean().item()

        logger.info(
            f"[FCRS label diag] {name}: "
            f"top{int(q * 100)}% same={high_same:.4f}, "
            f"bottom{int(q * 100)}% same={low_same:.4f}, "
            f"gap={high_same - low_same:.4f}, "
            f"score={score.mean().item():.4f}±{score.std().item():.4f}"
        )

    logger.info("=" * 60)
    logger.info("FCRS Pair-Level Label Diagnostics")
    logger.info("=" * 60)

    report("reliability", diag["reliability"])
    report("pseudo_risk", diag["pseudo_risk"])
    report("var", diag["var"])
    report("mu", diag["mu"])

def clustering_evaluation(z, y, edge_index, args, conf, logger, labeled_subgraph=None, labeled_indices=None):
    """Perform clustering and evaluate metrics.

    Args:
        z: Node embeddings. If labeled_indices is provided, z corresponds to embeddings
           of only those nodes. Otherwise, z contains embeddings for all nodes.
        y: Full label vector (for all nodes)
        edge_index: Full graph edge index
        args: Command line arguments
        conf: Configuration dictionary
        logger: Logger instance
        labeled_subgraph: Optional subgraph information for structure metrics
        labeled_indices: Optional indices of labeled nodes in the original graph
    """
    logger.info("=" * 60)
    logger.info("Clustering Stage")
    logger.info("=" * 60)

    # Normalize embeddings (optional, can be controlled by config)
    if conf.get('normalize_embeddings', True):
        z = torch.nn.functional.normalize(z, p=2, dim=1)
        logger.info("Embeddings normalized using L2 normalization")

    # Determine which nodes we're working with
    if labeled_indices is not None:
        # z corresponds to labeled nodes only
        y = y[labeled_indices]
        logger.info(f"Working with {len(labeled_indices):,} labeled nodes")

    valid_mask = ~torch.isnan(y)
    n_clusters = int(y[valid_mask].max().item()) + 1

    logger.info(f"Number of clusters: {n_clusters}")
    logger.info(f"Valid nodes for evaluation: {valid_mask.sum().item()} / {len(y)}")

    # Get K-Means parameters from config
    kmeans_backend = conf.get('kmeans_backend', 'torch')
    kmeans_n_init = conf.get('kmeans_n_init', 10)
    kmeans_max_iter = conf.get('kmeans_max_iter', 300)

    logger.info(f"K-Means backend: {kmeans_backend}")
    logger.info(f"K-Means n_init: {kmeans_n_init}")
    logger.info(f"K-Means max_iter: {kmeans_max_iter}")

    # Get metrics from config
    label_metric_names = tuple(conf.get('label_metrics', ['NMI', 'ARI', 'ACC', 'F1']))
    struct_metric_names = tuple(conf.get('struct_metrics', ['Mod', 'Cond']))

    logger.info(f"Label metrics: {', '.join(label_metric_names)}")
    logger.info(f"Structure metrics: {', '.join(struct_metric_names)}")

    # Setup for structure metrics
    if labeled_subgraph is not None:
        logger.info(f"Using labeled subgraph for structure metrics:")
        logger.info(f"  Subgraph nodes: {labeled_subgraph['num_nodes']:,}")
        logger.info(f"  Subgraph edges: {labeled_subgraph['edge_index'].shape[1]:,}")
        struct_edge_index = labeled_subgraph['edge_index']
    else:
        struct_edge_index = edge_index

    all_results = []
    clustering_times = []
    metrics_times = []

    start_time = time.time()

    for run in range(args.runs):
        run_start = time.time()

        # K-Means clustering with configurable parameters
        kmeans_start = time.time()
        kmeans = KMeansClusterHead(
            n_clusters=n_clusters,
            backend=kmeans_backend,
            n_init=kmeans_n_init,
            max_iter=kmeans_max_iter,
            random_state=args.seed + 42 * run
        )
        pred = kmeans.fit_predict(z)
        clustering_time = time.time() - kmeans_start
        clustering_times.append(clustering_time)

        # Compute metrics
        metrics_start = time.time()

        # Compute label-based metrics
        label_results = label_metrics(
            y[valid_mask],
            pred[valid_mask],
            metrics=label_metric_names
        )
        label_metrics_end = time.time()

        # Compute structure-based metrics
        struct_results = structure_metrics(
            struct_edge_index,
            pred,
            metrics=struct_metric_names
        )
        struct_metrics_end = time.time()

        label_metrics_time = label_metrics_end - metrics_start
        struct_metrics_time = struct_metrics_end - label_metrics_end
        metrics_time = struct_metrics_end - metrics_start
        metrics_times.append(metrics_time)

        total_run_time = time.time() - run_start

        # Merge results
        run_results = {**label_results, **struct_results}
        all_results.append(run_results)

        # Build log string dynamically
        metric_strs = []
        for name in label_metric_names + struct_metric_names:
            value = run_results[name] * 100
            metric_strs.append(f"{name}={value:.2f}")

        logger.info(
            f'Run {run + 1}/{args.runs}: '
            f'{", ".join(metric_strs)} '
            f'[Cluster: {clustering_time:.2f}s,'
            f' Metrics: {label_metrics_time:.2f}+{struct_metrics_time:.2f}={metrics_time:.2f}s,'
            f' Total: {total_run_time:.2f}s]'
        )

    total_time = time.time() - start_time
    avg_clustering_time = np.mean(clustering_times)
    avg_metrics_time = np.mean(metrics_times)
    avg_total_time = avg_clustering_time + avg_metrics_time

    # Compute statistics across runs
    all_metric_names = label_metric_names + struct_metric_names
    mean_results = {}
    std_results = {}

    for metric_name in all_metric_names:
        values = [result[metric_name] * 100 for result in all_results]
        mean_results[metric_name] = np.mean(values)
        std_results[metric_name] = np.std(values)

    # Log timing summary
    logger.info("=" * 60)
    logger.info("Clustering Timing Summary")
    logger.info("=" * 60)
    logger.info(f"Average clustering time:  {avg_clustering_time:.2f}s")
    logger.info(f"Average metrics time:     {avg_metrics_time:.2f}s")
    logger.info(f"Average total time:       {avg_total_time:.2f}s")
    logger.info(f"Total time ({args.runs} runs): {total_time:.2f}s")

    return mean_results, std_results, avg_clustering_time, avg_metrics_time, total_time


def log_system_info(model, data, device, logger, conf):
    """Log dataset and model information."""
    logger.info("=" * 60)
    logger.info("System Information")
    logger.info("=" * 60)

    # Dataset info
    logger.info(f"Dataset: {data}")
    logger.info(f"Nodes: {data.num_nodes:,}")
    logger.info(f"Edges: {data.num_edges:,}")
    logger.info(f"Features: {data.num_features}")
    logger.info(f"Avg degree: {data.num_edges / data.num_nodes:.2f}")

    # Training mode info
    mini_batch = conf.get('mini_batch', True)
    gnn_type = conf.get('gnn_type', 'sage')
    logger.info(f"Training mode: {'Mini-batch' if mini_batch else 'Full-batch'}")
    logger.info(f"GNN type: {gnn_type.upper()}")

    # Model info
    logger.info(f"Model: {model}")
    if hasattr(model, 'encoder'):
        logger.info(f"Encoder: {model.encoder}")

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Total parameters: {total_params / 1e6:.3f}M")
    logger.info(f"Trainable parameters: {trainable_params / 1e6:.3f}M")

    # Device info
    logger.info(f"Device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(device)}")
        logger.info(f"CUDA version: {torch.version.cuda}")


def log_final_results(mean_results, std_results, train_time, inference_time,
                     clustering_time, metrics_time, logger):
    """Log final results in a formatted table."""
    logger.info("=" * 60)
    logger.info("Final Results Summary")
    logger.info("=" * 60)

    # Clustering metrics
    logger.info("Clustering Metrics (mean ± std):")
    for metric_name in mean_results.keys():
        mean_val = mean_results[metric_name]
        std_val = std_results[metric_name]
        logger.info(f"  {metric_name:6s}: {mean_val:6.2f} ± {std_val:.2f}")

    # Time statistics
    logger.info("Time Statistics:")
    logger.info(f"  Training time:   {train_time:8.2f}s")
    logger.info(f"  Inference time:  {inference_time:8.2f}s")
    logger.info(f"  Clustering time: {clustering_time:8.2f}s")
    logger.info(f"  Metrics time:    {metrics_time:8.2f}s")
    total_eval_time = clustering_time + metrics_time
    logger.info(f"  Total eval time: {total_eval_time:8.2f}s")
    logger.info(f"  Total time:      {train_time + inference_time + total_eval_time:8.2f}s")
    logger.info(f"  Train+Infer+Cluster time: {train_time + inference_time + clustering_time:8.2f}s")

    logger.info("=" * 60 + "\n")


def _fmt_param_bool(value):
    return "true" if bool(value) else "false"


def _fmt_param_float(value, decimals):
    if value is None:
        return "None"
    return f"{float(value):.{decimals}f}"


def _resolve_csv_path_name(args, conf, effective_extra_loss, use_lcsr_large_batch_lite):
    if args.fcrs_mode in {'lcsr', 'lcsr_v2', 'lcsr_v3'}:
        if conf.get('mini_batch', True) and use_lcsr_large_batch_lite:
            return args.lcsr_batch_local_semantics
        return "full_batch_precomputed_pair_package"
    if (
        conf.get('mini_batch', True)
        and effective_extra_loss
        and args.fcrs_extra_mode == 'plain'
    ):
        return args.lcsr_batch_local_semantics
    if args.fcrs_extra_mode == 'weighted':
        return "weighted_extra_pair_package"
    return "full_batch_precomputed_pair_package"


def _build_lcsr_runs5_param_str(args, conf, effective_positive_loss, effective_extra_loss, use_lcsr_large_batch_lite):
    csv_path_name = _resolve_csv_path_name(
        args=args,
        conf=conf,
        effective_extra_loss=effective_extra_loss,
        use_lcsr_large_batch_lite=use_lcsr_large_batch_lite,
    )
    positive_quantile = args.lcsr_positive_quantile if effective_positive_loss in {'quantile_hinge', 'saturation_gate'} else None
    candidate_bank_size = args.lcsr_candidate_bank_size if csv_path_name == 'candidate_bank_v2' else None
    parts = [
        f"dataset={args.dataset}",
        f"seed={args.seed}",
        f"runs={args.runs}",
        f"mini_batch={_fmt_param_bool(conf.get('mini_batch', True))}",
        f"src={args.lcsr_support_source}",
        f"mode={effective_positive_loss}",
        f"q={_fmt_param_float(positive_quantile, 2) if positive_quantile is not None else 'None'}",
        f"lambda={_fmt_param_float(args.fcrs_extra_lambda, 3)}",
        f"warmup={args.fcrs_extra_warmup}",
        f"margin={_fmt_param_float(args.lcsr_margin, 2) if args.lcsr_margin is not None else 'None'}",
        f"rho={_fmt_param_float(args.lcsr_rho, 2) if args.lcsr_rho is not None else 'None'}",
        f"kmax={args.lcsr_kmax}",
        f"pool={args.lcsr_candidate_pool_size}",
        f"bank={candidate_bank_size if candidate_bank_size is not None else 'None'}",
        f"gnn={conf.get('gnn_type', 'sage')}",
        f"hidden={conf.get('hidden_channels', 'None')}",
        f"layers={conf.get('num_layers', 'None')}",
        f"epochs={conf.get('epochs', 'None')}",
        f"p_fm1={_fmt_param_float(conf.get('p_fm1', None), 2) if conf.get('p_fm1', None) is not None else 'None'}",
        f"p_ed1={_fmt_param_float(conf.get('p_ed1', None), 2) if conf.get('p_ed1', None) is not None else 'None'}",
        f"p_fm2={_fmt_param_float(conf.get('p_fm2', None), 2) if conf.get('p_fm2', None) is not None else 'None'}",
        f"p_ed2={_fmt_param_float(conf.get('p_ed2', None), 2) if conf.get('p_ed2', None) is not None else 'None'}",
        f"path={csv_path_name}",
    ]
    return "|".join(parts)


def _fmt_metric_cell(mean_value, std_value):
    return f"{float(mean_value):.2f}±{float(std_value):.2f}"


def append_lcsr_runs5_csv(args, conf, mean_results, std_results, logger, effective_positive_loss,
                          effective_extra_loss, use_lcsr_large_batch_lite):
    required_metrics = ('NMI', 'ARI', 'ACC', 'F1', 'Homo', 'Comp', 'Mod', 'Cond')
    missing = [name for name in required_metrics if name not in mean_results or name not in std_results]
    if missing:
        raise ValueError(
            "Missing required metrics for CSV append: "
            + ", ".join(missing)
            + ". Ensure label_metrics includes Homo/Comp and struct_metrics includes Mod/Cond."
        )

    csv_path = (
        Path(args.lcsr_csv_path)
        if args.lcsr_csv_path is not None
        else Path(__file__).resolve().parents[2] / "results" / "lcsr_runs5_search.csv"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    row = [
        _build_lcsr_runs5_param_str(
            args=args,
            conf=conf,
            effective_positive_loss=effective_positive_loss,
            effective_extra_loss=effective_extra_loss,
            use_lcsr_large_batch_lite=use_lcsr_large_batch_lite,
        ),
        *[_fmt_metric_cell(mean_results[name], std_results[name]) for name in required_metrics],
    ]
    with csv_path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['param_str', 'NMI', 'ARI', 'ACC', 'F1', 'Homo', 'Comp', 'Mod', 'Cond'])
        writer.writerow(row)
    logger.info(f"[LCSR CSV] Appended successful result to {csv_path}")


# Keep a clean UTF-8 visible symbol for CSV metric cells even if an older
# definition above was saved under a legacy code page.
def _fmt_metric_cell(mean_value, std_value):
    return f"{float(mean_value):.2f}\u00b1{float(std_value):.2f}"


def main():
    parser = argparse.ArgumentParser(description='LCSR for node clustering')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (cuda:0, cuda:1, cpu, etc.)')
    parser.add_argument('--root', type=str, default='../data',
                        help='Root path of dataset')
    parser.add_argument('--dataset', type=str, default='Cora',
                        choices=['Cora', 'Photo', 'Physics', 'HM', 'Flickr',
                                 'ArXiv', 'Reddit', 'MAG', 'Pokec', 'Products', 'WebTopic', 'Papers100M'],
                        help='Dataset name')
    parser.add_argument('--gnn_type', type=str, default=None,
                        choices=['gcn', 'sage', 'graphsage', 'gat', 'gatv2', 'gin', 'pna', 'edgecnn'],
                        help='Optional override for the encoder backbone type')
    parser.add_argument('--hidden_channels', type=int, default=None,
                        help='Optional override for encoder hidden width')
    parser.add_argument('--num_layers', type=int, default=None,
                        help='Optional override for encoder depth')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Optional override for the number of training epochs')
    parser.add_argument('--p_fm1', type=float, default=None,
                        help='Optional override for feature masking probability of view 1')
    parser.add_argument('--p_ed1', type=float, default=None,
                        help='Optional override for edge dropping probability of view 1')
    parser.add_argument('--p_fm2', type=float, default=None,
                        help='Optional override for feature masking probability of view 2')
    parser.add_argument('--p_ed2', type=float, default=None,
                        help='Optional override for edge dropping probability of view 2')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save logs')
    parser.add_argument('--ckpt_dir', type=str, default='ckpts',
                        help='Directory to save checkpoints')
    parser.add_argument('--load_ckpt', action='store_true',
                        help='Whether to load existing checkpoint for inference only')
    parser.add_argument('--resume', action='store_true',
                        help='Whether to resume training from last checkpoint')
    parser.add_argument('--runs', type=int, default=5,
                        help='Number of evaluation runs for stability')
    parser.add_argument('--lcsr-csv-path', type=str, default=None,
                        help='Optional CSV output path for successful run summaries')
    parser.add_argument('--fcrs-nonedge-diag', action='store_true',
                        help='Run the no-grad FCRS non-edge diagnostic after evaluation')
    parser.add_argument('--fcrs-topks', type=int, nargs='+', default=[1, 5, 10, 20],
                        help='Top-k values for the FCRS non-edge diagnostic')
    parser.add_argument('--fcrs-filter-k', type=int, default=None,
                        help='Propagation hop count K used by the FCRS non-edge diagnostic')
    parser.add_argument('--fcrs-pair-source-compare', action='store_true',
                        help='Run pair-source comparison after evaluation')
    parser.add_argument('--fcrs-pair-topks', type=int, nargs='+', default=[1, 5, 10, 20],
                        help='Top-k values for pair-source comparison')
    parser.add_argument('--fcrs-current-consensus-diag', action='store_true',
                        help='Run a post-hoc diagnostic for the current semantic-frequency consensus operator')
    parser.add_argument('--lcsr-mode', type=str, default=None,
                        choices=['none', 'sfpa', 'add_only', 'complete', 'weighted_complete'],
                        help='Public LCSR entry mode. Default behavior is add_only.')
    parser.add_argument('--fcrs-mode', type=str, default='lcsr_v3',
                        choices=['none', 'sfpa', 'lcsr', 'lcsr_v2', 'lcsr_v3'],
                        help='Internal compatibility mode. Public LCSR runs should prefer --lcsr-mode.')
    parser.add_argument('--fcrs-extra-loss', action='store_true',
                        help='Enable fixed extra positive loss on selected non-edge pairs')
    parser.add_argument('--fcrs-extra-mode', type=str, default='plain',
                        choices=['plain', 'weighted'],
                        help='Extra positive loss mode: plain mean or weighted candidate-confidence decoupling')
    parser.add_argument('--fcrs-extra-source', type=str, default='lcsr_mu',
                        choices=['raw', 'lowpass', 'fcrs_mu', 'lcsr_mu', 'semantic_frequency', 'raw_mul_fcrs', 'raw_mul_lcsr'],
                        help='Pre-embedding pair source for fixed extra positive pairs')
    parser.add_argument('--fcrs-candidate-source', type=str, default='raw',
                        choices=['raw', 'lowpass'],
                        help='Candidate non-edge pair source for weighted extra loss mode')
    parser.add_argument('--fcrs-weight-source', type=str, default='none',
                        choices=['none', 'fcrs_mu', 'lcsr_mu', 'fcrs_lcb', 'lcsr_lcb'],
                        help='Confidence weight source for weighted extra loss mode')
    parser.add_argument('--fcrs-lcb-rho', type=float, default=1.0,
                        help='Rho used in lcsr_lcb = mu - rho * std')
    parser.add_argument('--fcrs-extra-k', type=int, default=1,
                        help='Top-k non-edge pairs per node for the fixed extra positive set')
    parser.add_argument('--fcrs-extra-lambda', type=float, default=0.0,
                        help='Weight of the extra positive loss')
    parser.add_argument('--fcrs-extra-warmup', type=int, default=0,
                        help='Number of warmup epochs before enabling the extra positive loss')
    parser.add_argument('--fcrs-extra-ramp-epochs', type=int, default=0,
                        help='Linear ramp epochs after warmup for the extra positive loss weight')
    parser.add_argument('--fcrs-consensus', type=str, default='mean',
                        choices=list(LCSR_CONSENSUS_CHOICES),
                        help='Consensus operator used by semantic-frequency admission')
    parser.add_argument('--lcsr-candidate-pool-size', type=int, default=32,
                        help='Top candidate pool size used only for LCSR non-edge retrieval')
    parser.add_argument('--lcsr-margin', type=float, default=0.0,
                        help='Margin used by LCSR-v2 swap condition P_add > P_drop + margin')
    parser.add_argument('--lcsr-rho', type=float, default=1.0,
                        help='Node-wise editing budget ratio for LCSR-v2')
    parser.add_argument('--lcsr-kmax', type=int, default=32,
                        help='Node-wise editing budget hard cap for LCSR-v2')
    parser.add_argument('--lcsr-release', action='store_true',
                        help='Enable LCSR release: remove A- from observed positive pair supervision')
    parser.add_argument('--lcsr-complete', action='store_true',
                        help='Enable complete LCSR supervision rectification: remove A- from positive supervision and add A+ as positive support while keeping message passing graph unchanged')
    parser.add_argument('--lcsr-weighted-complete', action='store_true',
                        help='Enable weighted complete mode: keep A- in positive supervision but down-weight it by calibrated drop reliability')
    parser.add_argument('--lcsr-release-fraction', type=float, default=1.0,
                        help='Fraction of A- released from observed positive supervision in complete mode')
    parser.add_argument('--lcsr-release-beta', type=float, default=0.0,
                        help='Soft-release beta for down-weighting A- positive supervision in complete mode')
    parser.add_argument('--lcsr-drop-weight-floor', type=float, default=0.5,
                        help='Weight floor for score-weighted A- positive down-weighting')
    parser.add_argument('--lcsr-budget-match', action='store_true',
                        help='Enable canonical undirected budget matching so |A+| ~= |A-|')
    parser.add_argument('--lcsr-reliability-mode', type=str, default='full',
                        choices=['full', 'identity_only', 'low_only'],
                        help='Reliability source used by LCSR-v2 pair scoring')
    parser.add_argument('--lcsr-support-source', type=str, default='freq',
                        choices=['raw', 'freq', 'mu', 'raw_mul_mu', 'source_adaptive'],
                        help='Support source used by LCSR add-only admission')
    parser.add_argument('--lcsr-risk-budget', action='store_true',
                        help='Reduce add budget when source-adaptive confidence is weak')
    parser.add_argument('--lcsr-force-batch-local', action='store_true',
                        help='Force LCSR add-only to use the batch-local NeighborLoader admission path')
    parser.add_argument('--lcsr-batch-local-semantics', type=str, default='legacy_topk',
                        choices=['aligned', 'legacy_topk', 'candidate_bank_v2'],
                        help='Batch-local LCSR admission semantics for mini-batch LCSR training; full-batch datasets still use the precomputed pair-package path')
    parser.add_argument('--lcsr-candidate-bank-size', type=int, default=64,
                        help='Static candidate-bank width used by batch-local candidate_bank_v2 semantics')
    parser.add_argument('--lcsr-positive-loss', type=str, default='linear',
                        choices=['linear', 'hinge', 'softplus_hinge', 'quantile_hinge', 'saturation_gate'],
                        help='Positive loss used on A+ latent support pairs')
    parser.add_argument('--lcsr-positive-mode', type=str, default=None,
                        choices=['linear', 'hinge', 'softplus_hinge', 'quantile_hinge', 'saturation_gate'],
                        help='Alias of lcsr-positive-loss; useful for mode-specific sweeps')
    parser.add_argument('--lcsr-positive-margin', type=float, default=0.8,
                        help='Margin used by hinge-style A+ positive loss')
    parser.add_argument('--lcsr-positive-temperature', type=float, default=0.05,
                        help='Temperature used by softplus_hinge A+ positive loss')
    parser.add_argument('--lcsr-positive-quantile', type=float, default=0.2,
                        help='Quantile used by quantile_hinge to derive the activation margin')
    parser.add_argument('--lcsr-saturation-tau', type=float, default=0.80,
                        help='Mean-A+ similarity center used by saturation_gate')
    parser.add_argument('--lcsr-saturation-temp', type=float, default=0.03,
                        help='Temperature used by saturation_gate')
    parser.add_argument('--lcsr-disable-local-calibration', action='store_true',
                        help='Disable local observed-support calibration and use raw reliability directly')
    parser.add_argument('--lcsr-disable-mutual', action='store_true',
                        help='Disable mutual conservative calibration and use average symmetric score')
    parser.add_argument('--lcsr-v3-variant', type=str, default='decoupled_global_add',
                        choices=['decoupled_global_add', 'degree_shrink'],
                        help='LCSR-v3 rectification variant')
    parser.add_argument('--lcsr-add-score', type=str, default='global',
                        choices=['global', 'shrink'],
                        help='Add-side score used by LCSR-v3 degree-shrink calibration')
    parser.add_argument('--lcsr-plus-weight-mode', type=str, default='uniform',
                        choices=['uniform', 'score', 'score_floor'],
                        help='Weighting mode for A+ latent positive loss only')
    parser.add_argument('--lcsr-plus-weight-floor', type=float, default=0.5,
                        help='Floor used when lcsr-plus-weight-mode=score_floor')
    parser.add_argument('--lcsr-plus-dropout', type=float, default=0.0,
                        help='Drop probability applied only to A+ latent positive pairs during training')
    parser.add_argument('--lcsr-refine-with-emb', action='store_true',
                        help='Rebuild A+ after warmup using current encoder embeddings')
    parser.add_argument('--lcsr-refine-mode', type=str, default='blend',
                        choices=['blend', 'geom', 'filter'],
                        help='Warmup embedding refinement mode for A+ admission')
    parser.add_argument('--lcsr-refine-alpha', type=float, default=0.2,
                        help='Blend weight for embedding-refined A+ admission')
    parser.add_argument('--lcsr-refine-emb-floor', type=float, default=0.7,
                        help='Embedding percentile floor used by filter refinement mode')
    parser.add_argument('--lcsr-refine-rebuild-epoch', type=int, default=-1,
                        help='Epoch to rebuild refined A+; default uses warmup epoch')
    parser.add_argument('--lcsr-runtime-profile', action='store_true',
                        help='Profile the batch-local LCSR runtime on the first effective mini-batches')
    parser.add_argument('--lcsr-runtime-profile-batches', type=int, default=10,
                        help='Number of effective mini-batches to profile')
    parser.add_argument('--lcsr-runtime-profile-only', action='store_true',
                        help='Stop training once runtime profiling has collected enough effective mini-batches')
    parser.add_argument('--lcsr-runtime-profile-out', type=str, default=None,
                        help='Optional markdown output path for runtime profiling summary')
    parser.add_argument('--lcsr-admission-audit', action='store_true',
                        help='Audit batch-local LCSR admission counts without changing selection semantics')
    parser.add_argument('--lcsr-admission-audit-batches', type=int, default=20,
                        help='Number of effective batch-local admission batches to audit')
    parser.add_argument('--lcsr-admission-audit-only', action='store_true',
                        help='Stop after collecting the requested admission-audit batches')
    parser.add_argument('--lcsr-admission-audit-md', type=str, default=None,
                        help='Markdown output path for the LCSR batch-local admission audit report')
    parser.add_argument('--lcsr-admission-audit-json', type=str, default=None,
                        help='JSON output path for the LCSR batch-local admission audit summary')
    parser.add_argument('--physics-audit', action='store_true',
                        help='Run the Physics mini-batch correctness/mechanism audit and emit JSON/Markdown reports')
    args = parser.parse_args()

    if args.lcsr_mode is not None:
        if args.lcsr_mode == 'none':
            args.fcrs_mode = 'none'
        elif args.lcsr_mode == 'sfpa':
            args.fcrs_mode = 'sfpa'
        else:
            args.fcrs_mode = 'lcsr_v3'
            if args.lcsr_mode == 'complete':
                args.lcsr_complete = True
            elif args.lcsr_mode == 'weighted_complete':
                args.lcsr_complete = True
                args.lcsr_weighted_complete = True
    if args.physics_audit:
        args.dataset = 'Physics'
        args.seed = 0
        args.runs = 5
        args.lcsr_admission_audit = True
        args.lcsr_admission_audit_batches = min(int(args.lcsr_admission_audit_batches), 5) if args.lcsr_admission_audit_batches > 0 else 5
        if args.lcsr_admission_audit_md is None:
            args.lcsr_admission_audit_md = str(Path(__file__).resolve().parents[2] / "results" / "physics_admission_audit.md")
        if args.lcsr_admission_audit_json is None:
            args.lcsr_admission_audit_json = str(Path(__file__).resolve().parents[2] / "results" / "physics_admission_audit.json")

    # Setup device
    if args.device.startswith('cuda'):
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    # Load configuration.
    # Dataset config should provide the default LCSR/NS4GC values, while
    # explicit CLI arguments should override only when the user actually
    # changed them from the parser defaults.
    dataset_conf = get_training_config(args.dataset, config_path='train.conf.yaml')
    parsed_args = vars(args).copy()
    conf = dict(dataset_conf)
    for key, value in parsed_args.items():
        if not hasattr(args, key):
            continue
        try:
            default_value = parser.get_default(key)
        except Exception:
            default_value = None
        if key not in conf:
            conf[key] = value
        elif value != default_value:
            conf[key] = value

    # Push resolved config values back into args so downstream code that still
    # reads args.* sees the dataset-specific LCSR settings.
    for key, value in conf.items():
        if hasattr(args, key):
            setattr(args, key, value)

    mode_extra_loss = args.fcrs_mode in {'sfpa', 'lcsr', 'lcsr_v2', 'lcsr_v3'}
    effective_extra_loss = args.fcrs_extra_loss or mode_extra_loss
    effective_extra_source = 'semantic_frequency' if args.fcrs_mode == 'sfpa' else args.fcrs_extra_source
    effective_consensus = 'mean' if args.fcrs_mode in {'lcsr', 'lcsr_v2', 'lcsr_v3'} else args.fcrs_consensus
    effective_positive_loss = args.lcsr_positive_mode or args.lcsr_positive_loss
    requested_extra_warmup = int(args.fcrs_extra_warmup)
    args.fcrs_extra_warmup = requested_extra_warmup

    # Set to 0 for CPU to avoid multiprocessing issues
    if device.type == 'cpu':
        conf['num_workers'] = 0

    # Setup logging
    gnn_type = conf.get('gnn_type', 'sage').lower()
    mini_batch_mode = 'mini' if conf.get('mini_batch', True) else 'full'
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(
        args.log_dir,
        args.dataset,
        f'{gnn_type}_{mini_batch_mode}'
    )
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger(os.path.join(log_dir, f'seed{args.seed}_{timestamp}.log'))

    # Log configuration
    logger.info("=" * 60)
    logger.info("Configuration")
    logger.info("=" * 60)
    for key, value in sorted(conf.items()):
        logger.info(f"  {key}: {value}")
    _validate_minibatch_gcn_cached_guard(conf)
    if effective_extra_loss and requested_extra_warmup >= conf.get('epochs', 0):
        logger.warning(
            f"[LCSR warning] dataset={args.dataset} requested warmup={requested_extra_warmup} "
            f">= epochs={conf.get('epochs', 0)}; LCSR will stay inactive unless the epoch budget is increased."
        )
    _warn_if_lcsr_never_activates(logger, args, conf, effective_extra_loss)

    """Load Dataset"""
    set_seed(args.seed)
    logger.info(f"Loading dataset: {args.dataset}...")

    is_papers100m = args.dataset.lower() in ['papers100m', 'ogbn-papers100m']
    if is_papers100m:
        # Load with splits and subgraph
        x, edge_index, y, train_idx, valid_idx, test_idx, labeled_subgraph = get_dataset(
            args.dataset, root=args.root, return_splits=True
        )
        labeled_indices = torch.cat([train_idx, valid_idx, test_idx])
    else:
        x, edge_index, y = get_dataset(args.dataset, root=args.root, return_splits=False)
        labeled_subgraph = None
        labeled_indices = None
    data = Data(x=x, edge_index=edge_index)
    large_graph_threshold = 200000
    use_lcsr_large_batch_lite = (
        effective_extra_loss
        and args.fcrs_mode in {'lcsr', 'lcsr_v2', 'lcsr_v3'}
        and conf.get('mini_batch', True)
        and (
            args.lcsr_force_batch_local
            or
            args.dataset == 'Physics'
            or
            args.dataset == 'WebTopic'
            or int(x.size(0)) > large_graph_threshold
        )
    )
    if use_lcsr_large_batch_lite:
        effective_extra_source = {
            'raw': 'raw',
            'freq': 'semantic_frequency',
            'mu': 'lcsr_mu',
            'raw_mul_mu': 'raw_mul_lcsr',
            'source_adaptive': 'source_adaptive',
        }[args.lcsr_support_source]

    """Create Model"""
    set_seed(args.seed)

    # Get training mode and GNN type from config
    mini_batch_training = conf.get('mini_batch', True)
    gnn_type = conf.get('gnn_type', 'sage').lower()

    # Create data augmentation transforms
    transform1 = GSSLTransform(
        p_feat_mask=conf.get('p_fm1', 0.1),
        p_edge_drop=conf.get('p_ed1', 0.1)
    )
    transform2 = GSSLTransform(
        p_feat_mask=conf.get('p_fm2', 0.1),
        p_edge_drop=conf.get('p_ed2', 0.1)
    )

    # Create encoder
    encoder = create_tuned_gnn(
        gnn_type=gnn_type,
        in_channels=data.num_features,
        hidden_channels=conf.get('hidden_channels', 512),
        num_layers=conf.get('num_layers', 2),
        out_channels=None,
        dropout=conf.get('dropout', 0.0),
        act=conf.get('act', 'relu'),
        act_first=conf.get('act_first', False),
        act_last=conf.get('act_last', False),
        norm=conf.get('norm', None) if conf.get('norm') != 'none' else None,
        residual=conf.get('residual', False),
        pre_linear=conf.get('pre_linear', False),
        jk=conf.get('jk', None),
        add_self_loops=conf.get('add_self_loops', None),
        normalize=conf.get('normalize', True),
        improved=conf.get('improved', False),
        cached=conf.get('cached', True),
        aggr=conf.get('aggr', 'mean'),
        project=conf.get('project', False),
        heads=conf.get('heads', 1),
        concat=conf.get('concat', True),
        negative_slope=conf.get('negative_slope', 0.2),
        train_eps=conf.get('train_eps', False),
        bias=conf.get('bias', True),
    )

    # Create model
    model = LCSR(
        encoder=encoder,
        transform1=transform1,
        transform2=transform2,
        s=conf.get('s', 0.6),
        tau=conf.get('tau', 0.1),
        lam=conf.get('lam', 1.0),
        gam=conf.get('gam', 1.0),
        fcrs_extra_loss=effective_extra_loss,
        fcrs_extra_mode=args.fcrs_extra_mode,
        fcrs_extra_source=effective_extra_source,
        fcrs_candidate_source=args.fcrs_candidate_source,
        fcrs_weight_source=args.fcrs_weight_source,
        fcrs_lcb_rho=args.fcrs_lcb_rho,
        fcrs_extra_k=args.fcrs_extra_k,
        fcrs_extra_lambda=args.fcrs_extra_lambda,
        fcrs_extra_warmup=args.fcrs_extra_warmup,
        fcrs_extra_ramp_epochs=args.fcrs_extra_ramp_epochs,
        fcrs_consensus=effective_consensus,
        fcrs_filter_k=args.fcrs_filter_k if args.fcrs_filter_k is not None else conf.get('num_hops', 2),
        fcrs_risk_budget=args.lcsr_risk_budget,
        fcrs_positive_loss=effective_positive_loss,
        fcrs_positive_margin=args.lcsr_positive_margin,
        fcrs_positive_temperature=args.lcsr_positive_temperature,
        fcrs_positive_quantile=args.lcsr_positive_quantile,
        fcrs_saturation_tau=args.lcsr_saturation_tau,
        fcrs_saturation_temp=args.lcsr_saturation_temp,
        fcrs_batch_local_admission=(
            (
                args.fcrs_mode not in {'lcsr', 'lcsr_v2', 'lcsr_v3'}
                or use_lcsr_large_batch_lite
            )
            and conf.get('mini_batch', True)
            and effective_extra_loss
            and args.fcrs_extra_mode == 'plain'
            and (
                effective_extra_source in {
                    'semantic_frequency',
                    'raw_mul_fcrs',
                    'raw_mul_lcsr',
                    'raw',
                    'fcrs_mu',
                    'lcsr_mu',
                    'source_adaptive',
                }
                or (use_lcsr_large_batch_lite and args.fcrs_mode in {'lcsr', 'lcsr_v2', 'lcsr_v3'})
            )
        ),
        fcrs_complete_mode=(args.lcsr_complete or args.lcsr_weighted_complete),
        fcrs_plus_dropout=args.lcsr_plus_dropout,
        fcrs_runtime_profile=args.lcsr_runtime_profile,
        fcrs_runtime_profile_batches=args.lcsr_runtime_profile_batches,
        fcrs_runtime_profile_only=args.lcsr_runtime_profile_only,
        fcrs_admission_audit=args.lcsr_admission_audit,
        fcrs_admission_audit_batches=args.lcsr_admission_audit_batches,
        fcrs_admission_audit_only=args.lcsr_admission_audit_only,
        fcrs_batch_local_semantics=args.lcsr_batch_local_semantics,
        fcrs_lcsr_candidate_pool_size=args.lcsr_candidate_pool_size,
        fcrs_lcsr_margin=args.lcsr_margin,
        fcrs_lcsr_rho=args.lcsr_rho,
        fcrs_lcsr_kmax=args.lcsr_kmax,
        fcrs_lcsr_budget_match=args.lcsr_budget_match,
        fcrs_lcsr_disable_local_calibration=args.lcsr_disable_local_calibration,
        fcrs_lcsr_disable_mutual=args.lcsr_disable_mutual,
    ).to(device)

    model.set_logger(logger)
    degree = torch.bincount(edge_index[0].detach().cpu(), minlength=int(data.x.size(0)))
    model.set_global_degree(degree)
    resolved_config = _collect_resolved_config(args=args, conf=conf, effective_extra_loss=effective_extra_loss)
    resolved_config['fcrs_batch_local_admission'] = bool(getattr(model, 'fcrs_batch_local_admission', False))
    _log_resolved_config(logger, resolved_config)
    filter_k = _resolved_filter_k(args, conf)
    if (
        effective_extra_loss
        and conf.get('mini_batch', True)
        and getattr(model, 'fcrs_batch_local_admission', False)
        and args.lcsr_batch_local_semantics == 'candidate_bank_v2'
    ):
        candidate_bank_cache_dir = Path(__file__).resolve().parents[2] / "cache" / "lcsr_candidate_banks"
        candidate_bank, candidate_bank_meta = build_or_load_lcsr_candidate_bank(
            x=data.x,
            edge_index=data.edge_index,
            dataset_name=args.dataset,
            support_source=args.lcsr_support_source,
            filter_k=filter_k,
            bank_size=args.lcsr_candidate_bank_size,
            cache_dir=candidate_bank_cache_dir,
            work_device=device,
            logger=logger,
        )
        model.set_batch_local_candidate_bank(candidate_bank, candidate_bank_meta)
    if args.dataset == 'Physics' and effective_extra_loss:
        if not conf.get('mini_batch', True):
            raise ValueError("Physics LCSR audit requires mini_batch=true.")
        if not getattr(model, 'fcrs_batch_local_admission', False):
            raise ValueError("Physics LCSR route invalid: fcrs_batch_local_admission must be true.")
        if args.lcsr_batch_local_semantics != 'candidate_bank_v2':
            raise ValueError("Physics LCSR route invalid: only candidate_bank_v2 is allowed.")
        if args.lcsr_complete or args.lcsr_weighted_complete or args.lcsr_release:
            raise ValueError("Physics LCSR route invalid: release/complete/weighted_complete are not allowed.")
        candidate_bank = getattr(model, 'fcrs_candidate_bank', None)
        if candidate_bank is None or candidate_bank.numel() == 0:
            raise ValueError("Physics LCSR route invalid: candidate bank is empty.")
        if candidate_bank.dim() != 2 or int(candidate_bank.size(0)) != int(data.x.size(0)) or int(candidate_bank.size(1)) <= 0:
            raise ValueError(
                f"Physics LCSR route invalid: candidate bank must have shape [N,R], got {tuple(candidate_bank.shape)}."
            )
        model.fcrs_route_audit_meta = {
            'route_name': 'candidate_bank_v2',
            'bank_shape': [int(candidate_bank.size(0)), int(candidate_bank.size(1))],
            'online_score_shape': ['B', int(candidate_bank.size(1))],
            'cache_hit': bool(getattr(model, 'fcrs_candidate_bank_meta', {}).get('cache_hit', False)),
            'observed_topology_unchanged': True,
            'budget_degree_source': 'global' if getattr(model, 'fcrs_global_degree', None) is not None else 'batch_local',
            'filter_k': int(filter_k),
            'candidate_bank_meta': dict(getattr(model, 'fcrs_candidate_bank_meta', {})),
        }
        logger.info(
            "[PHYSICS ROUTE] candidate_bank_v2 "
            f"bank_shape={model.fcrs_route_audit_meta['bank_shape']} "
            f"score_shape={model.fcrs_route_audit_meta['online_score_shape']} "
            f"cache_hit={int(model.fcrs_route_audit_meta['cache_hit'])} "
            "observed topology unchanged"
        )
    logger.info("LCSR training path keeps the NS4GC backbone and message-passing graph unchanged.")
    logger.info(
        "FCRS_WARMUP "
        f"dataset={args.dataset} "
        f"requested_warmup={requested_extra_warmup} "
        f"effective_warmup={args.fcrs_extra_warmup}"
    )
    lcsr_runtime = {
        "package": None,
        "refined": False,
        "refine_skipped": False,
    }

    def _log_lcsr_package(package, complete_variant="none"):
        logger.info(
            f"LCSR extra loss enabled: mode={package.mode_name}, "
            f"candidate_pool_size={args.lcsr_candidate_pool_size}, "
            f"lambda={args.fcrs_extra_lambda}, warmup={args.fcrs_extra_warmup}, filter_k={filter_k}, "
            f"ramp_epochs={args.fcrs_extra_ramp_epochs}, plus_dropout={args.lcsr_plus_dropout:.3f}, "
            f"margin={package.margin}, rho={package.rho}, kmax={package.kmax}, "
            f"budget_match={int(package.budget_match)}, release={int(package.release)}, "
            f"rectify_variant={package.rectify_variant}, "
            f"add_score={package.add_score_name}, "
            f"plus_weight_mode={args.lcsr_plus_weight_mode}, "
            f"reliability={package.reliability_mode}, "
            f"local_calibration={int(package.use_local_calibration)}, "
            f"mutual={int(package.use_mutual)}, "
            f"backbone_graph=observed_only, latent_graph=add_only, positive_supervision={'rectified' if (args.lcsr_complete or args.lcsr_weighted_complete) else 'observed_plus_extra'}, "
            f"complete_variant={complete_variant}"
        )
        logger.info(
            "LCSR_RECTIFY "
            f"mode={package.mode_name} "
            f"candidate_pool_size={package.candidate_pool_size} "
            f"margin={package.margin:.4f} "
            f"rho={package.rho:.4f} "
            f"kmax={package.kmax} "
            f"budget_match={int(package.budget_match)} "
            f"release={int(package.release)} "
            f"rectify_variant={package.rectify_variant} "
            f"add_score={package.add_score_name} "
            f"reliability={package.reliability_mode} "
            f"local_calibration={int(package.use_local_calibration)} "
            f"mutual={int(package.use_mutual)} "
            f"raw_swap_count={package.raw_swap_count} "
            f"raw_unique_add={package.raw_unique_add_count} "
            f"raw_unique_drop={package.raw_unique_drop_count} "
            f"matched_unique_add={package.matched_unique_add_count} "
            f"matched_unique_drop={package.matched_unique_drop_count} "
            f"add_ratio={package.add_ratio:.6f} "
            f"drop_ratio={package.drop_ratio:.6f} "
            f"dynamic_k_mean={package.dynamic_k_mean:.6f} "
            f"dynamic_k_std={package.dynamic_k_std:.6f} "
            f"dynamic_k_max={package.dynamic_k_max} "
            f"active_node_ratio={package.active_node_ratio:.6f} "
            f"mean_p_add={package.mean_p_add:.6f} "
            f"mean_p_drop={package.mean_p_drop:.6f} "
            f"support_gain={package.support_gain:.6f}"
        )
        logger.info(
            "LCSR_LABEL_DIAG "
            f"a_plus_same={_fmt_optional_float(package.a_plus_same_ratio)} "
            f"a_minus_cross={_fmt_optional_float(package.a_minus_cross_ratio)} "
            f"sfpa_same={_fmt_optional_float(package.sfpa_same_ratio)} "
            f"random_nonedge_same={_fmt_optional_float(package.random_nonedge_same_ratio)} "
            f"observed_edge_same={_fmt_optional_float(package.observed_edge_same_ratio)} "
            f"observed_edge_cross={_fmt_optional_float(package.observed_edge_cross_ratio)}"
        )
        logger.info(
            "LCSR_REFINE "
            f"refine_mode={package.refine_mode or 'none'} "
            f"refine_alpha={_fmt_optional_float(package.refine_alpha)} "
            f"refine_emb_floor={_fmt_optional_float(package.refine_emb_floor)} "
            f"mean_static_add_score={_fmt_optional_float(package.mean_static_add_score)} "
            f"mean_embedding_add_score={_fmt_optional_float(package.mean_embedding_add_score)} "
            f"changed_add_count_vs_static={package.changed_add_count_vs_static if package.changed_add_count_vs_static is not None else -1} "
            f"overlap_ratio_vs_static={_fmt_optional_float(package.overlap_ratio_vs_static)}"
        )
        logger.info(
            "LCSR_SOURCE_ADAPT "
            f"support_source={package.support_source} "
            f"gap_raw={_fmt_optional_float(package.source_gap_raw)} "
            f"gap_freq={_fmt_optional_float(package.source_gap_freq)} "
            f"gap_mu={_fmt_optional_float(package.source_gap_mu)} "
            f"gap_raw_mul_mu={_fmt_optional_float(package.source_gap_raw_mul_mu)} "
            f"weight_raw={_fmt_optional_float(package.source_weight_raw)} "
            f"weight_freq={_fmt_optional_float(package.source_weight_freq)} "
            f"weight_mu={_fmt_optional_float(package.source_weight_mu)} "
            f"weight_raw_mul_mu={_fmt_optional_float(package.source_weight_raw_mul_mu)} "
            f"risk_budget_scale={_fmt_optional_float(package.risk_budget_scale)}"
        )

    if effective_extra_loss:
        filter_k = args.fcrs_filter_k if args.fcrs_filter_k is not None else conf.get('num_hops', 2)
        if args.fcrs_mode in {'lcsr', 'lcsr_v2', 'lcsr_v3'} and use_lcsr_large_batch_lite:
            mode_name = (
                f"lcsr_large_batch_add_only"
                if args.fcrs_mode == 'lcsr_v3'
                else "lcsr_large_batch_add_only"
            )
            model.set_extra_pair_index(None, num_nodes=data.x.size(0))
            model.set_extra_pairs(None)
            model.set_release_pair_index(None, num_nodes=data.x.size(0))
            model.set_positive_pair_index(None, num_nodes=data.x.size(0))
            logger.info(
                "LCSR_LARGE_LITE "
                f"mode={mode_name} "
                f"dataset={args.dataset} "
                f"num_nodes={int(data.x.size(0))} "
                f"mini_batch={int(conf.get('mini_batch', True))} "
                f"batch_size={conf.get('batch_size', -1)} "
                f"fan_out={conf.get('fan_out', -1)} "
                f"lambda={args.fcrs_extra_lambda} "
                f"requested_warmup={requested_extra_warmup} "
                f"effective_warmup={args.fcrs_extra_warmup} "
                f"filter_k={filter_k} "
                f"support_source={args.lcsr_support_source} "
                f"risk_budget={int(args.lcsr_risk_budget)} "
                f"batch_local_semantics={args.lcsr_batch_local_semantics} "
                f"candidate_bank_size={args.lcsr_candidate_bank_size} "
                f"forced_batch_local={int(args.lcsr_force_batch_local)} "
                "admission=batch_local_neighborloader "
                "release=0"
            )
            logger.info(
                "LCSR_LARGE_LITE_SANITY "
                "verify_no_full_graph_pair_package_for_WebTopic=1 "
                "verify_no_full_graph_sparse_mm_for_WebTopic=1 "
                "verify_no_full_graph_pair_package=1 "
                "verify_no_full_graph_sparse_mm=1 "
                "verify_batch_local_Aplus_used=1 "
                "verify_A_minus_not_used_as_negative=1 "
                "verify_message_passing_follows_NS4GC_minibatch_path=1"
            )
            logger.info(
                "FCRS_EXTRA_PAIR_SETUP "
                "source=semantic_frequency consensus=mean "
                "selected=-1 mean_score=nan same_ratio=nan top20_same=nan bottom20_same=nan gap=nan"
            )
        elif args.fcrs_mode in {'lcsr', 'lcsr_v2', 'lcsr_v3'}:
            lcsr_package = build_lcsr_pair_package(
                x=data.x,
                edge_index=data.edge_index,
                y=y,
                filter_k=filter_k,
                candidate_pool_size=args.lcsr_candidate_pool_size,
                seed=args.seed,
                work_device=device,
                margin=0.0 if args.fcrs_mode == 'lcsr' else args.lcsr_margin,
                rho=1.0 if args.fcrs_mode == 'lcsr' else args.lcsr_rho,
                kmax=args.lcsr_candidate_pool_size if args.fcrs_mode == 'lcsr' else args.lcsr_kmax,
                release=(((args.fcrs_mode == 'lcsr_v2') and args.lcsr_release) or args.lcsr_complete),
                budget_match=((args.fcrs_mode in {'lcsr_v2', 'lcsr_v3'}) and args.lcsr_budget_match),
                reliability_mode='full' if args.fcrs_mode == 'lcsr' else args.lcsr_reliability_mode,
                support_source=args.lcsr_support_source,
                risk_budget=args.lcsr_risk_budget,
                use_local_calibration=not args.lcsr_disable_local_calibration,
                use_mutual=not args.lcsr_disable_mutual,
                mode_name=(
                    'lcsr_current' if args.fcrs_mode == 'lcsr' else (
                        'lcsr_v2_add_release' if args.fcrs_mode == 'lcsr_v2' and args.lcsr_release else (
                            'lcsr_v2_add_only' if args.fcrs_mode == 'lcsr_v2' else (
                                f"lcsr_v3_{args.lcsr_v3_variant}_{args.lcsr_add_score}_{'complete' if args.lcsr_complete else 'add_only'}"
                            )
                        )
                    )
                ),
                rectify_variant=(
                    'v2' if args.fcrs_mode in {'lcsr', 'lcsr_v2'} else args.lcsr_v3_variant
                ),
                add_score_name=(
                    'local' if args.fcrs_mode in {'lcsr', 'lcsr_v2'} else args.lcsr_add_score
                ),
            )
            add_pair_weights = _build_lcsr_plus_weights(
                lcsr_package.add_pair_scores,
                mode=args.lcsr_plus_weight_mode,
                floor=args.lcsr_plus_weight_floor,
            )
            model.set_extra_pair_index(
                lcsr_package.add_pair_index,
                num_nodes=data.x.size(0),
                pair_weights=add_pair_weights,
            )
            lcsr_runtime["package"] = lcsr_package
            release_pair_index = None
            release_pair_scores = None
            release_pair_weights = None
            hard_release = False
            if (args.fcrs_mode == 'lcsr_v2' and args.lcsr_release) or args.lcsr_complete or args.lcsr_weighted_complete:
                release_pair_index = lcsr_package.drop_pair_index
                release_pair_scores = lcsr_package.drop_pair_scores
                if args.lcsr_weighted_complete:
                    release_pair_weights = _build_score_weighted_release_weights(
                        release_pair_scores,
                        args.lcsr_drop_weight_floor,
                    )
                    hard_release = False
                elif args.lcsr_complete and args.lcsr_release_beta > 0:
                    release_pair_weights = _build_soft_release_weights(release_pair_scores, args.lcsr_release_beta)
                    hard_release = False
                else:
                    release_pair_index, release_pair_scores = _select_release_subset(
                        release_pair_index,
                        release_pair_scores,
                        args.lcsr_release_fraction if args.lcsr_complete else 1.0,
                    )
                    hard_release = True
                model.set_release_pair_index(
                    release_pair_index,
                    num_nodes=data.x.size(0),
                    pair_weights=release_pair_weights,
                    hard_release=hard_release,
                )
            else:
                model.set_release_pair_index(None, num_nodes=data.x.size(0))
            if args.lcsr_complete or args.lcsr_weighted_complete:
                model.set_positive_pair_index(lcsr_package.add_pair_index, num_nodes=data.x.size(0))
            else:
                model.set_positive_pair_index(None, num_nodes=data.x.size(0))
            complete_variant = "none"
            if args.lcsr_complete or args.lcsr_weighted_complete:
                if args.lcsr_weighted_complete:
                    complete_variant = f"weighted_floor_{args.lcsr_drop_weight_floor:.2f}"
                elif args.lcsr_release_beta > 0:
                    complete_variant = f"soft_beta_{args.lcsr_release_beta:.2f}"
                else:
                    complete_variant = f"selective_frac_{args.lcsr_release_fraction:.2f}"
            _log_lcsr_package(lcsr_package, complete_variant=complete_variant)
            mean_plus_weight = None
            min_plus_weight = None
            max_plus_weight = None
            if add_pair_weights is not None and add_pair_weights.numel() > 0:
                mean_plus_weight = float(add_pair_weights.mean().item())
                min_plus_weight = float(add_pair_weights.min().item())
                max_plus_weight = float(add_pair_weights.max().item())
            logger.info(
                "LCSR_PLUS_WEIGHT "
                f"mode={args.lcsr_plus_weight_mode} "
                f"floor={args.lcsr_plus_weight_floor:.6f} "
                f"positive_loss={effective_positive_loss} "
                f"positive_margin={_fmt_optional_float(args.lcsr_positive_margin)} "
                f"positive_temperature={_fmt_optional_float(args.lcsr_positive_temperature)} "
                f"positive_quantile={_fmt_optional_float(args.lcsr_positive_quantile)} "
                f"saturation_tau={_fmt_optional_float(args.lcsr_saturation_tau)} "
                f"saturation_temp={_fmt_optional_float(args.lcsr_saturation_temp)} "
                f"plus_dropout={args.lcsr_plus_dropout:.6f} "
                f"ramp_epochs={args.fcrs_extra_ramp_epochs} "
                f"mean_A_plus_weight={_fmt_optional_float(mean_plus_weight)} "
                f"min_A_plus_weight={_fmt_optional_float(min_plus_weight)} "
                f"max_A_plus_weight={_fmt_optional_float(max_plus_weight)} "
                f"verify_A_plus_weighted_loss_used={int(add_pair_weights is not None and add_pair_weights.numel() > 0 and args.lcsr_plus_weight_mode != 'uniform')}"
            )
            logger.info(
                "LCSR_MAIN_SANITY "
                "verify_message_passing_edges_unchanged=1 "
                "verify_A_minus_not_used_as_negative=1 "
                f"verify_budget_match={int(lcsr_package.matched_unique_add_count == lcsr_package.matched_unique_drop_count)}"
            )
            if args.lcsr_complete or args.lcsr_weighted_complete:
                observed_before = int(data.edge_index.size(1) // 2)
                released = int(release_pair_index.size(1)) if release_pair_index is not None else 0
                added = int(lcsr_package.matched_unique_add_count)
                observed_after = observed_before if args.lcsr_weighted_complete else max(observed_before - released, 0)
                final_positive = observed_before + added if (args.lcsr_weighted_complete or args.lcsr_release_beta > 0) else observed_after + added
                released_same = _pair_same_ratio_from_index(y, release_pair_index)
                released_cross = None if released_same is None else 1.0 - released_same
                mean_release_weight = None
                min_release_weight = None
                max_release_weight = None
                weighted_positive_mass = None
                if release_pair_weights is not None and release_pair_weights.numel() > 0:
                    mean_release_weight = float(release_pair_weights.mean().item())
                    min_release_weight = float(release_pair_weights.min().item())
                    max_release_weight = float(release_pair_weights.max().item())
                    weighted_positive_mass = float(observed_before - released + release_pair_weights.sum().item() + added)
                logger.info(
                    "LCSR_COMPLETE "
                    f"observed_positive_pairs_before={observed_before} "
                    f"released_positive_pairs_count={released} "
                    f"observed_positive_pairs_after={observed_after} "
                    f"added_positive_pairs_count={added} "
                    f"final_positive_pairs_count={final_positive} "
                    f"release_ratio={(released / observed_before) if observed_before > 0 else 0.0:.6f} "
                    f"released_A_minus_cross={_fmt_optional_float(released_cross)} "
                    f"soft_release_beta={args.lcsr_release_beta:.6f} "
                    f"weighted_A_minus_count={released} "
                    f"mean_A_minus_weight={_fmt_optional_float(mean_release_weight)} "
                    f"min_A_minus_weight={_fmt_optional_float(min_release_weight)} "
                    f"max_A_minus_weight={_fmt_optional_float(max_release_weight)} "
                    f"weight_floor={args.lcsr_drop_weight_floor:.6f} "
                    f"weighted_positive_pair_mass={_fmt_optional_float(weighted_positive_mass)} "
                    f"hard_release={int(hard_release)} "
                    "verify_message_passing_edges_unchanged=1 "
                    "verify_A_minus_not_used_as_negative=1 "
                    f"verify_positive_release_applied={int(released > 0)} "
                    f"verify_A_minus_not_removed={int(args.lcsr_weighted_complete)} "
                    f"verify_weighted_loss_used={int(args.lcsr_weighted_complete and release_pair_weights is not None)} "
                    f"verify_budget_match={int(lcsr_package.matched_unique_add_count == lcsr_package.matched_unique_drop_count)}"
                )
        elif args.fcrs_extra_mode == 'weighted':
            extra_pair_package = build_lcsr_extra_pair_package(
                x=data.x,
                edge_index=data.edge_index,
                candidate_source=args.fcrs_candidate_source,
                weight_source=args.fcrs_weight_source,
                topk=args.fcrs_extra_k,
                filter_k=filter_k,
                rho=args.fcrs_lcb_rho,
                work_device=device,
            )
            model.set_extra_pairs(extra_pair_package.top_indices, extra_pair_package.pair_weights)
            logger.info(
                f"FCRS weighted extra loss enabled: candidate={args.fcrs_candidate_source}, "
                f"weight={args.fcrs_weight_source}, rho={args.fcrs_lcb_rho}, "
                f"k={args.fcrs_extra_k}, lambda={args.fcrs_extra_lambda}, "
                f"warmup={args.fcrs_extra_warmup}, filter_k={filter_k}, "
                f"consensus={effective_consensus}"
            )
        else:
            source_name = {
                'raw': 'raw',
                'lowpass': 'lowpass',
                'fcrs_mu': 'lcsr_mu',
                'lcsr_mu': 'lcsr_mu',
                'semantic_frequency': 'semantic_frequency',
                'raw_mul_fcrs': 'raw_mul_lcsr',
                'raw_mul_lcsr': 'raw_mul_lcsr',
            }[effective_extra_source]
            display_source = 'semantic_frequency' if effective_extra_source in {'semantic_frequency', 'raw_mul_fcrs', 'raw_mul_lcsr'} else effective_extra_source
            use_batch_local_admission = (
                args.fcrs_mode not in {'lcsr', 'lcsr_v2', 'lcsr_v3'}
                and
                conf.get('mini_batch', True)
                and display_source == 'semantic_frequency'
            )
            if use_batch_local_admission:
                model.set_extra_pairs(None)
                logger.info(
                    f"FCRS extra loss enabled: source={display_source}, "
                    f"k={args.fcrs_extra_k}, lambda={args.fcrs_extra_lambda}, "
                    f"warmup={args.fcrs_extra_warmup}, filter_k={filter_k}, "
                    f"consensus={effective_consensus}, admission=batch_local_neighborloader"
                )
                logger.info(
                    "FCRS_EXTRA_PAIR_SETUP "
                    f"source={display_source} consensus={effective_consensus} "
                    "selected=-1 mean_score=nan same_ratio=nan top20_same=nan bottom20_same=nan gap=nan"
                )
            else:
                extra_top_values, extra_top_indices = select_nonedge_topk_by_source(
                    source_name=source_name,
                    x=data.x,
                    edge_index=data.edge_index,
                    embeddings=None,
                    filter_k=filter_k,
                    topk=args.fcrs_extra_k,
                    seed=args.seed,
                    consensus=effective_consensus,
                    work_device=device,
                )
                model.set_extra_pairs(extra_top_indices)
                extra_setup = summarize_selected_extra_pairs(extra_top_values, extra_top_indices, y)
                logger.info(
                    f"FCRS extra loss enabled: source={display_source}, "
                    f"k={args.fcrs_extra_k}, lambda={args.fcrs_extra_lambda}, "
                    f"warmup={args.fcrs_extra_warmup}, filter_k={filter_k}, "
                    f"consensus={effective_consensus}"
                )
                logger.info(
                    "FCRS_EXTRA_PAIR_SETUP "
                    f"source={display_source} "
                    f"consensus={effective_consensus} "
                    f"selected={extra_setup['selected']} "
                    f"mean_score={extra_setup['mean_score']:.6f} "
                    f"same_ratio={extra_setup['same_ratio']:.6f} "
                    f"top20_same={extra_setup['top20_same']:.6f} "
                    f"bottom20_same={extra_setup['bottom20_same']:.6f} "
                    f"gap={extra_setup['gap']:.6f}"
                )

    # Log system information
    log_system_info(model, data, device, logger, conf)

    """Setup Checkpoint Manager"""
    ckpt_dir = os.path.join(args.ckpt_dir, args.dataset, f'{gnn_type}_{mini_batch_mode}')
    ckpt_name = f"seed{args.seed}"
    ckpt_manager = CheckpointManager(ckpt_dir, ckpt_name, logger)

    """Training"""
    # Check if we should skip training
    skip_training = args.load_ckpt and ckpt_manager.has_checkpoint(load_best=True)
    epoch_stats = []
    epoch_refine_callback = None

    if (
        effective_extra_loss
        and args.fcrs_mode == 'lcsr_v3'
        and args.lcsr_refine_with_emb
        and not args.lcsr_complete
        and not args.lcsr_weighted_complete
    ):
        rebuild_epoch = args.lcsr_refine_rebuild_epoch if args.lcsr_refine_rebuild_epoch >= 0 else args.fcrs_extra_warmup

        def epoch_refine_callback(model, epoch, logger):
            if lcsr_runtime["refined"] or lcsr_runtime["refine_skipped"] or epoch < rebuild_epoch:
                return
            with torch.no_grad():
                was_training = model.training
                model.eval()
                full_x = data.x.to(device)
                full_edge_index = data.edge_index.to(device)
                embeddings = model.embed(full_x, full_edge_index).detach().cpu()
                if was_training:
                    model.train()
            refined_package = build_lcsr_pair_package_with_embedding_refinement(
                x=data.x,
                edge_index=data.edge_index,
                y=y,
                embeddings=embeddings,
                filter_k=filter_k,
                candidate_pool_size=args.lcsr_candidate_pool_size,
                seed=args.seed,
                work_device=device,
                margin=args.lcsr_margin,
                rho=args.lcsr_rho,
                kmax=args.lcsr_kmax,
                budget_match=args.lcsr_budget_match,
                reliability_mode=args.lcsr_reliability_mode,
                mode_name=f"lcsr_v3_{args.lcsr_v3_variant}_{args.lcsr_add_score}_refined_{args.lcsr_refine_mode}",
                rectify_variant=args.lcsr_v3_variant,
                add_score_name=args.lcsr_add_score,
                refine_mode=args.lcsr_refine_mode,
                refine_alpha=args.lcsr_refine_alpha,
                refine_emb_floor=args.lcsr_refine_emb_floor,
            )
            if refined_package.a_plus_same_ratio is not None and lcsr_runtime["package"] is not None:
                static_same = lcsr_runtime["package"].a_plus_same_ratio
                if static_same is not None and refined_package.a_plus_same_ratio < static_same:
                    logger.info(
                        "LCSR_REFINE_SKIP "
                        f"reason=lower_a_plus_same "
                        f"static_same={static_same:.6f} "
                        f"refined_same={refined_package.a_plus_same_ratio:.6f}"
                    )
                    lcsr_runtime["refine_skipped"] = True
                    return
            refined_add_weights = _build_lcsr_plus_weights(
                refined_package.add_pair_scores,
                mode=args.lcsr_plus_weight_mode,
                floor=args.lcsr_plus_weight_floor,
            )
            model.set_extra_pair_index(
                refined_package.add_pair_index,
                num_nodes=data.x.size(0),
                pair_weights=refined_add_weights,
            )
            lcsr_runtime["package"] = refined_package
            lcsr_runtime["refined"] = True
            _log_lcsr_package(refined_package, complete_variant="none")
            logger.info(
                "LCSR_REFINE_REBUILD "
                f"epoch={epoch} "
                f"refine_mode={args.lcsr_refine_mode} "
                f"refine_alpha={args.lcsr_refine_alpha:.6f} "
                f"refine_emb_floor={args.lcsr_refine_emb_floor:.6f}"
            )

    if skip_training:
        logger.info("=" * 60)
        logger.info("Skipping training - loading best checkpoint")
        logger.info("=" * 60)
        ckpt_manager.load_checkpoint(model, optimizer=None, load_best=True, device=device)
        train_time = 0.0
        avg_epoch_time = 0.0
        final_epoch = 0
    else:
        set_seed(args.seed)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=conf.get('lr', 0.001),
            weight_decay=conf.get('wd', 0.0)
        )

        # Choose training mode based on configuration
        if mini_batch_training:
            train_time, avg_epoch_time, final_epoch, epoch_stats = train_mini_batch(
                model, data, optimizer, conf, logger, device,
                ckpt_manager=ckpt_manager, resume_from_ckpt=args.resume,
                epoch_refine_callback=epoch_refine_callback,
            )
        else:
            train_time, avg_epoch_time, final_epoch, epoch_stats = train_full_batch(
                model, data, optimizer, conf, logger, device,
                ckpt_manager=ckpt_manager, resume_from_ckpt=args.resume,
                epoch_refine_callback=epoch_refine_callback,
            )

        # Log training statistics
        logger.info("=" * 60)
        logger.info("Training Statistics")
        logger.info("=" * 60)
        logger.info(f"Total epochs: {final_epoch}")
        logger.info(f"Total training time: {train_time:.2f}s")
        logger.info(f"Average epoch time: {avg_epoch_time:.2f}ms")

        if device.type == 'cuda':
            mem_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
            mem_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            logger.info(f"Peak GPU memory reserved: {mem_reserved:.1f} MB")
            logger.info(f"Peak GPU memory allocated: {mem_allocated:.1f} MB")
    train_monitor_summary = summarize_epoch_stats(
        epoch_stats if not skip_training else [],
        extra_lambda=args.fcrs_extra_lambda,
    )

    """Inference"""
    set_seed(args.seed)

    # Load best checkpoint for inference
    ckpt_manager.load_checkpoint(model, optimizer=None, load_best=True, device=device)

    # For Papers100M, only infer embeddings for labeled nodes
    z, inference_time = inference_embeddings(
        model, data, conf, logger, device,
        labeled_indices=labeled_indices
    )

    logger.info(f"Embedding shape: {z.shape}")

    if labeled_indices is not None:
        logger.info(f"Embeddings correspond to {len(labeled_indices):,} labeled nodes")

    """Clustering"""
    set_seed(args.seed)
    mean_results, std_results, avg_clustering_time, avg_metrics_time, total_clustering_time = clustering_evaluation(
        z, y, edge_index, args, conf, logger,
        labeled_subgraph=labeled_subgraph,
        labeled_indices=labeled_indices
    )

    if train_monitor_summary is not None:
        components = train_monitor_summary['components_mean']
        logger.info(
            "FCRS_TRAIN_MONITOR "
            f"ns4gc_mean={_fmt_optional_float(components.get('ns4gc', float('nan')))} "
            f"extra_mean={_fmt_optional_float(components.get('extra', float('nan')))} "
            f"extra_ratio_mean={_fmt_optional_float(components.get('extra_ratio', float('nan')))} "
            f"active_pair_ratio_mean={_fmt_optional_float(components.get('active_pair_ratio', float('nan')))} "
            f"positive_margin_value_mean={_fmt_optional_float(components.get('positive_margin_value', float('nan')))} "
            f"saturation_gamma_mean={_fmt_optional_float(components.get('saturation_gamma', float('nan')))} "
            f"extra_pairs_mean={_fmt_optional_float(components.get('extra_pairs', float('nan')))} "
            f"admit_score_mean={_fmt_optional_float(components.get('admit_score', float('nan')))} "
            f"gap_raw_mean={_fmt_optional_float(components.get('gap_raw', float('nan')))} "
            f"gap_freq_mean={_fmt_optional_float(components.get('gap_freq', float('nan')))} "
            f"gap_mu_mean={_fmt_optional_float(components.get('gap_mu', float('nan')))} "
            f"gap_raw_mul_mu_mean={_fmt_optional_float(components.get('gap_raw_mul_mu', float('nan')))} "
            f"weight_raw_mean={_fmt_optional_float(components.get('weight_raw', float('nan')))} "
            f"weight_freq_mean={_fmt_optional_float(components.get('weight_freq', float('nan')))} "
            f"weight_mu_mean={_fmt_optional_float(components.get('weight_mu', float('nan')))} "
            f"weight_raw_mul_mu_mean={_fmt_optional_float(components.get('weight_raw_mul_mu', float('nan')))} "
            f"risk_budget_scale_mean={_fmt_optional_float(components.get('risk_budget_scale', float('nan')))} "
            f"extra_sim_start={_fmt_optional_float(train_monitor_summary.get('extra_sim_start'))} "
            f"extra_sim_end={_fmt_optional_float(train_monitor_summary.get('extra_sim_end'))}"
        )
    runtime_profile_summary = model.get_runtime_profile_summary() if hasattr(model, 'get_runtime_profile_summary') else None
    if runtime_profile_summary is not None:
        for item in runtime_profile_summary["ranked"][:3]:
            logger.info(
                "LCSR_RUNTIME_PROFILE "
                f"module={item['module']} "
                f"mean_s={item['mean_s']:.6f} "
                f"share={item['share']:.6f}"
            )
        _write_lcsr_runtime_profile_report(
            path=args.lcsr_runtime_profile_out,
            args=args,
            conf=conf,
            summary=runtime_profile_summary,
        )
    admission_audit_summary = model.get_admission_audit_summary() if hasattr(model, 'get_admission_audit_summary') else None
    if admission_audit_summary is not None:
        _write_lcsr_admission_audit_reports(
            markdown_path=args.lcsr_admission_audit_md,
            json_path=args.lcsr_admission_audit_json,
            args=args,
            summary=admission_audit_summary,
        )
    if args.dataset == 'Physics' and y is not None:
        physics_label_diag = _build_physics_label_only_diagnostic(model=model, data=data, y=y)
        logger.info(
            "PHYSICS_LABEL_DIAG "
            f"random_nonedge_same_label_ratio={_fmt_optional_float(physics_label_diag['random_nonedge_same_label_ratio'])} "
            f"candidate_bank_same_label_ratio={_fmt_optional_float(physics_label_diag['candidate_bank_same_label_ratio'])} "
            f"post_margin_pair_same_label_ratio={_fmt_optional_float(physics_label_diag['post_margin_pair_same_label_ratio'])} "
            f"admitted_a_plus_same_label_ratio={_fmt_optional_float(physics_label_diag['admitted_a_plus_same_label_ratio'])} "
            f"observed_edge_same_label_ratio={_fmt_optional_float(physics_label_diag['observed_edge_same_label_ratio'])} "
            f"admitted_score_top20_same_label_ratio={_fmt_optional_float(physics_label_diag['admitted_score_top20_same_label_ratio'])} "
            f"admitted_score_bottom20_same_label_ratio={_fmt_optional_float(physics_label_diag['admitted_score_bottom20_same_label_ratio'])} "
            f"admitted_score_same_label_gap={_fmt_optional_float(physics_label_diag['admitted_score_same_label_gap'])}"
        )
        if args.physics_audit:
            physics_report = {
                'resolved_config': resolved_config,
                'route_audit': {
                    **dict(getattr(model, 'fcrs_route_audit_meta', {})),
                    'descriptor_norm_audit': list(getattr(model, 'fcrs_descriptor_audit_rows', [])),
                    'admission_audit_rows': list(getattr(model, 'fcrs_route_audit_rows', [])),
                },
                'effective_lambda_history': [
                    {
                        'epoch': idx + 1,
                        'effective_extra_lambda': float(item.get('components', {}).get('effective_extra_lambda', 0.0)),
                        'positive_loss': float(item.get('components', {}).get('positive_loss', item.get('components', {}).get('extra', 0.0))),
                        'extra_pair_count': float(item.get('components', {}).get('extra_pair_count', item.get('components', {}).get('extra_pairs', 0.0))),
                        'active_flag': bool(
                            float(item.get('components', {}).get('effective_extra_lambda', 0.0)) > 0
                            and float(item.get('components', {}).get('extra_pair_count', item.get('components', {}).get('extra_pairs', 0.0))) > 0
                        ),
                    }
                    for idx, item in enumerate(epoch_stats)
                ],
                'candidate_quality_diagnostic': physics_label_diag,
                'final_metrics': {
                    name: {'mean': float(mean_results[name]), 'std': float(std_results[name])}
                    for name in mean_results.keys()
                },
            }
            physics_report_root = Path(__file__).resolve().parents[2] / "results"
            _write_physics_audit_reports(
                json_path=physics_report_root / "physics_audit_report.json",
                md_path=physics_report_root / "physics_audit_report.md",
                report=physics_report,
            )

    if args.fcrs_nonedge_diag:
        filter_k = args.fcrs_filter_k if args.fcrs_filter_k is not None else conf.get('num_hops', 2)
        diag_start_time = time.time()
        diag_peak_allocated_mb = None
        diag_peak_reserved_mb = None
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        diag_result = run_lcsr_nonedge_diagnostic(
            x=data.x,
            edge_index=data.edge_index,
            y=y,
            dataset_name=args.dataset,
            topks=args.fcrs_topks,
            filter_k=filter_k,
            logger=logger,
            seed=args.seed,
            work_device=device,
        )
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            diag_peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            diag_peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        diag_runtime = time.time() - diag_start_time
        logger.info(
            f"[FCRS non-edge diag] runtime={diag_runtime:.2f}s "
            f"peak_allocated_mb={diag_peak_allocated_mb:.1f} "
            f"peak_reserved_mb={diag_peak_reserved_mb:.1f}"
            if diag_peak_allocated_mb is not None
            else f"[FCRS non-edge diag] runtime={diag_runtime:.2f}s"
        )
        logger.info(f"[FCRS non-edge diag] completed dataset={diag_result.dataset} filter_k={diag_result.filter_k}")

    if args.fcrs_pair_source_compare:
        filter_k = args.fcrs_filter_k if args.fcrs_filter_k is not None else conf.get('num_hops', 2)
        compare_result = run_pair_source_comparison(
            x=data.x,
            edge_index=data.edge_index,
            y=y,
            embeddings=z,
            dataset_name=args.dataset,
            topks=args.fcrs_pair_topks,
            filter_k=filter_k,
            logger=logger,
            seed=args.seed,
            work_device=device,
        )
        logger.info(
            f"[FCRS pair source] completed dataset={compare_result.dataset} "
            f"filter_k={compare_result.filter_k}"
        )

    if args.fcrs_current_consensus_diag:
        filter_k = args.fcrs_filter_k if args.fcrs_filter_k is not None else conf.get('num_hops', 2)
        diag_metrics = evaluate_nonedge_source(
            source_name='semantic_frequency',
            x=data.x,
            edge_index=data.edge_index,
            y=y,
            embeddings=None,
            filter_k=filter_k,
            topks=[args.fcrs_extra_k],
            seed=args.seed,
            consensus=args.fcrs_consensus,
            work_device=device,
        )[args.fcrs_extra_k]
        logger.info(
            "FCRS_CURRENT_CONSENSUS_DIAG "
            f"source=semantic_frequency consensus={args.fcrs_consensus} k={args.fcrs_extra_k} "
            f"selected={diag_metrics['selected']} coverage={diag_metrics['coverage']} "
            f"mean_score={diag_metrics['mean_score']:.6f} "
            f"same_ratio={diag_metrics['same']:.6f} random_same={diag_metrics['random']:.6f} "
            f"edge_same={diag_metrics['edge_same']:.6f} top20_same={diag_metrics['top20']:.6f} "
            f"bottom20_same={diag_metrics['bottom20']:.6f} gap={diag_metrics['gap']:.6f}"
        )

    """Final Results"""
    # Note: We only count one clustering run for total time
    log_final_results(
        mean_results, std_results, train_time, inference_time,
        avg_clustering_time, avg_metrics_time, logger
    )
    append_lcsr_runs5_csv(
        args=args,
        conf=conf,
        mean_results=mean_results,
        std_results=std_results,
        logger=logger,
        effective_positive_loss=effective_positive_loss,
        effective_extra_loss=effective_extra_loss,
        use_lcsr_large_batch_lite=use_lcsr_large_batch_lite,
    )

    # Additional compact format for easy parsing
    logger.info("Compact Results:")

    # Build metric string dynamically
    metric_strs = []
    for metric_name in mean_results.keys():
        mean_val = mean_results[metric_name]
        std_val = std_results[metric_name]
        metric_strs.append(f"{metric_name}={mean_val:.2f}±{std_val:.2f}")

    logger.info(", ".join(metric_strs))
    total_eval_time = avg_clustering_time + avg_metrics_time
    logger.info(
        f"Time: Train={train_time:.2f}s, Infer={inference_time:.2f}s, "
        f"Cluster={avg_clustering_time:.2f}s, Metrics={avg_metrics_time:.2f}s, "
        f"Total={train_time + inference_time + total_eval_time:.2f}s, "
        f"Train+Infer+Clustering={train_time + inference_time + avg_clustering_time:.2f}s"
    )


if __name__ == '__main__':
    main()
