import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .base import TrainableModel


class CompactLearning(nn.Module):
    """Low-rank compactness module adapted from the upstream CoCo release."""

    def __init__(self, k: int = 64, stage_num: int = 10, beta: float = 1.0, lamd: float = 1.0):
        super().__init__()
        self.k = k
        self.stage_num = stage_num
        self.beta = beta
        self.lamd = lamd

        mu = torch.randn(1, k) * math.sqrt(2.0 / k)
        mu = mu / (1e-6 + mu.norm(dim=0, keepdim=True))
        self.register_buffer("mu", mu)

    def forward(self, embeddings: Tensor) -> Tensor:
        num_points, feature_dim = embeddings.size()
        mu = self.mu.repeat(num_points, 1).to(embeddings.device)

        # Keep the original implementation's column-wise normalization semantics.
        embd_norm = embeddings / (1e-6 + embeddings.norm(dim=0, keepdim=True))
        compact_embeddings = embd_norm

        with torch.no_grad():
            for _ in range(self.stage_num):
                compact_t = compact_embeddings.transpose(0, 1)
                assignment = torch.mm(compact_t, mu) / self.lamd
                assignment = F.softmax(assignment, dim=1)
                assignment = assignment / (1e-6 + assignment.sum(dim=0, keepdim=True))
                mu = torch.mm(compact_embeddings, assignment)
                mu = mu / (1e-6 + mu.norm(dim=0, keepdim=True))

        compact_embeddings = torch.mm(mu, assignment.transpose(0, 1))
        return self.beta * compact_embeddings + embd_norm


class SampleSimilarities(nn.Module):
    def __init__(
        self,
        feats_dim: int,
        queue_size: int,
        temperature: float,
        sample_size: int = 128,
    ):
        super().__init__()
        self.queue_size = queue_size
        self.temperature = temperature
        self.sample_size = sample_size
        self.index = 0

        stdv = 1.0 / math.sqrt(feats_dim / 3.0)
        memory = torch.rand(queue_size, feats_dim).mul_(2 * stdv).add_(-stdv)
        self.register_buffer("memory", memory)

    def forward(self, q: Tensor, update: bool = True) -> Tensor:
        num_nodes = q.size(0)
        sample_size = min(self.sample_size, num_nodes)

        if sample_size == num_nodes:
            batch = q
        else:
            indices = torch.randperm(num_nodes, device=q.device)[:sample_size]
            batch = q.index_select(0, indices)

        batch_size = batch.size(0)
        # Clone the detached queue before the matrix multiply so the later
        # in-place memory-bank update does not invalidate autograd versioning.
        queue = self.memory.detach().clone()
        out = torch.mm(queue, batch.transpose(0, 1)).transpose(0, 1)
        out = out / self.temperature

        if update:
            with torch.no_grad():
                out_ids = torch.arange(batch_size, device=q.device, dtype=torch.long) + self.index
                out_ids = torch.remainder(out_ids, self.queue_size)
                self.memory.index_copy_(0, out_ids, batch.detach())
                self.index = (self.index + batch_size) % self.queue_size

        return out


class KLConsistency(nn.Module):
    def forward(self, targets: Tensor, inputs: Tensor) -> Tensor:
        targets = F.softmax(targets, dim=1)
        inputs = F.log_softmax(inputs, dim=1)
        return F.kl_div(inputs, targets, reduction="batchmean")


class CoCo(TrainableModel):
    """Compactness and Consistency for Deep Graph Clustering."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        activation: str = "ident",
        k: int = 64,
        stage_num: int = 10,
        beta: float = 1.0,
        consistency_memory_size: int = 32768,
        consistency_temperature: float = 0.02,
        consistency_sample_size: int = 128,
    ):
        super().__init__()
        self.local_proj = nn.Linear(input_dim, hidden_dim)
        self.global_proj = nn.Linear(input_dim, hidden_dim)
        self.compact = CompactLearning(k=k, stage_num=stage_num, beta=beta)
        self.local_similarities = SampleSimilarities(
            feats_dim=hidden_dim,
            queue_size=consistency_memory_size,
            temperature=consistency_temperature,
            sample_size=consistency_sample_size,
        )
        self.global_similarities = SampleSimilarities(
            feats_dim=hidden_dim,
            queue_size=consistency_memory_size,
            temperature=consistency_temperature,
            sample_size=consistency_sample_size,
        )
        self.consistency = KLConsistency()
        self.activation_name = activation
        self.activation = self._make_activation(activation)

    @staticmethod
    def _make_activation(name: str):
        if name == "ident":
            return nn.Identity()
        if name == "sigmoid":
            return nn.Sigmoid()
        if name == "relu":
            return nn.ReLU()
        if name == "leakyrelu":
            return nn.LeakyReLU(0.2, inplace=False)
        raise ValueError(f"Unsupported activation: {name}")

    def reset_parameters(self):
        self.local_proj.reset_parameters()
        self.global_proj.reset_parameters()

    def forward(self, local_x: Tensor, global_x: Tensor) -> tuple[Tensor, Tensor]:
        z1 = self.activation(self.local_proj(local_x))
        z2 = self.activation(self.global_proj(global_x))

        all_embeddings = torch.cat((z1, z2), dim=0)
        all_embeddings = self.compact(all_embeddings)

        z1, z2 = torch.split(all_embeddings, all_embeddings.size(0) // 2, dim=0)
        z1 = F.normalize(z1, dim=1, p=2)
        z2 = F.normalize(z2, dim=1, p=2)
        return z1, z2

    def embed(self, local_x: Tensor, global_x: Tensor) -> Tensor:
        z1, z2 = self.forward(local_x, global_x)
        return 0.5 * (z1 + z2)

    def loss(self, local_x: Tensor, global_x: Tensor) -> Tensor:
        z1, z2 = self.forward(local_x, global_x)
        similarities_local = self.local_similarities(z1, update=True)
        similarities_global = self.global_similarities(z2, update=True)
        loss = 0.5 * (
            self.consistency(similarities_local, similarities_global) +
            self.consistency(similarities_global, similarities_local)
        )
        return loss
