import math

import torch
import torch.nn.functional as F
from torch import nn

from pyagc.models.base import TrainableModel


def select_activation(name):
    if name == 'elu':
        return F.elu
    if name == 'relu':
        return F.relu
    if name == 'sigmoid':
        return F.sigmoid
    if name == 'tanh':
        return F.tanh
    if name == 'leaky_relu':
        return F.leaky_relu
    if name == 'gelu':
        return F.gelu
    if name is None:
        return None
    raise NotImplementedError(f'Unsupported activation: {name}')


def sample_gumbel(shape, device, eps=1e-20):
    u = torch.rand(shape, device=device)
    return -torch.log(-torch.log(u + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1.0):
    y = logits + sample_gumbel(logits.size(), logits.device)
    return torch.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=0.2, hard=False):
    y = gumbel_softmax_sample(logits, temperature)
    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y


def graph_top_k(dense_adj, k):
    if k >= dense_adj.shape[-1]:
        raise ValueError(f'k={k} must be smaller than number of nodes {dense_adj.shape[-1]}.')
    _, indices = dense_adj.topk(k=k + 1, dim=-1)
    mask = torch.zeros(dense_adj.shape, dtype=torch.bool, device=dense_adj.device)
    mask[torch.arange(dense_adj.shape[0], device=dense_adj.device)[:, None], indices] = True
    mask[torch.arange(dense_adj.shape[0], device=dense_adj.device),
         torch.arange(dense_adj.shape[0], device=dense_adj.device)] = False
    return torch.masked_fill(dense_adj, ~mask, value=0.0).to_sparse_coo()


class Lorentz:
    def __init__(self, k=1.0):
        self.k = float(k)

    def inner(self, x, y, keepdim=False):
        prod = -x[..., :1] * y[..., :1] + (x[..., 1:] * y[..., 1:]).sum(dim=-1, keepdim=True)
        return prod if keepdim else prod.squeeze(-1)

    def cinner(self, x, y):
        x = x.clone()
        x[..., :1] *= -1
        return x @ y.transpose(-1, -2)

    def expmap0(self, u, eps=1e-8):
        spatial = u[..., 1:]
        norm = spatial.norm(dim=-1, keepdim=True).clamp_min(eps)
        time = torch.cosh(norm)
        space = torch.sinh(norm) * spatial / norm
        zero_mask = (spatial.abs().sum(dim=-1, keepdim=True) == 0)
        time = torch.where(zero_mask, torch.ones_like(time), time)
        space = torch.where(zero_mask.expand_as(space), torch.zeros_like(space), space)
        return torch.cat([time, space], dim=-1)

    def dist(self, x, y, eps=1e-8):
        value = (-self.inner(x, y, keepdim=False)).clamp_min(1.0 + eps)
        return torch.acosh(value)

    def frechet_mean(self, x, weights=None, keepdim=False):
        if weights is None:
            z = torch.sum(x, dim=0, keepdim=True)
        else:
            z = torch.sum(x * weights, dim=0, keepdim=keepdim)
        denorm = self.inner(z, z, keepdim=keepdim).abs().clamp_min(1e-8).sqrt()
        return z / denorm


def edge_softmax_by_src(src, score, num_nodes):
    max_per_src = torch.full(
        (num_nodes,),
        -torch.inf,
        dtype=score.dtype,
        device=score.device,
    )
    max_per_src.scatter_reduce_(0, src, score, reduce='amax', include_self=True)
    exp_score = torch.exp(score - max_per_src[src])
    sum_per_src = torch.zeros(num_nodes, dtype=score.dtype, device=score.device)
    sum_per_src.scatter_add_(0, src, exp_score)
    return exp_score / sum_per_src[src].clamp_min(1e-12)


class LorentzGraphConvolution(nn.Module):
    def __init__(self, manifold, in_dim, out_dim, use_bias, dropout, use_att, nonlin=None):
        super().__init__()
        self.linear = LorentzLinear(manifold, in_dim, out_dim, use_bias, dropout, nonlin=nonlin)
        self.agg = LorentzAgg(manifold, out_dim, dropout, use_att)

    def forward(self, x, adj):
        h = self.linear(x)
        return self.agg(h, adj)


class LorentzLinear(nn.Module):
    def __init__(self, manifold, in_dim, out_dim, bias=True, dropout=0.1, scale=10, fixscale=False,
                 nonlin=None):
        super().__init__()
        self.manifold = manifold
        self.nonlin = nonlin
        self.in_features = in_dim
        self.out_features = out_dim
        self.bias = bias
        self.weight = nn.Linear(self.in_features, self.out_features, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(()) * math.log(scale), requires_grad=not fixscale)
        self.reset_parameters()

    def forward(self, x):
        if self.nonlin is not None:
            x = self.nonlin(x)
        x = self.weight(self.dropout(x))
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)
        time = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp() + 1.1
        scale = (time * time - 1) / (x_narrow * x_narrow).sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return torch.cat([time, x_narrow * scale.sqrt()], dim=-1)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out_features)
        nn.init.uniform_(self.weight.weight, -stdv, stdv)
        with torch.no_grad():
            self.weight.weight[:, 0] = 0
        if self.bias:
            nn.init.constant_(self.weight.bias, 0)


class LorentzAgg(nn.Module):
    def __init__(self, manifold, in_dim, dropout, use_att):
        super().__init__()
        self.manifold = manifold
        self.in_features = in_dim
        self.dropout = dropout
        self.use_att = use_att
        if self.use_att:
            self.key_linear = LorentzLinear(manifold, in_dim, in_dim)
            self.query_linear = LorentzLinear(manifold, in_dim, in_dim)
            self.bias = nn.Parameter(torch.zeros(()) + 20)
            self.scale = nn.Parameter(torch.zeros(()) + math.sqrt(in_dim))

    def forward(self, x, adj):
        if self.use_att:
            query = self.query_linear(x)
            key = self.key_linear(x)
            att_adj = 2 + 2 * self.manifold.cinner(query, key)
            att_adj = att_adj / self.scale + self.bias
            att_adj = torch.sigmoid(att_adj)
            att_adj = torch.mul(adj.to_dense(), att_adj)
            support_t = att_adj @ x
        else:
            support_t = torch.sparse.mm(adj, x)
        denorm = (-self.manifold.inner(support_t, support_t, keepdim=True)).abs().clamp_min(1e-8).sqrt()
        return support_t / denorm


class LorentzAssignment(nn.Module):
    def __init__(self, manifold, in_dim, hid_dim, num_assign, dropout, bias=False, temperature=0.2):
        super().__init__()
        self.manifold = manifold
        self.num_assign = num_assign
        self.assign_linear = nn.Linear(in_dim - 1, num_assign, bias=bias)
        nn.init.xavier_normal_(self.assign_linear.weight)
        self.temperature = temperature
        self.key_linear = LorentzLinear(manifold, in_dim, hid_dim, bias=False)
        self.query_linear = LorentzLinear(manifold, in_dim, hid_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        ass = self.assign_linear(x[..., 1:]).softmax(-1)
        q = self.query_linear(x)
        k = self.key_linear(x)
        edge_index = adj.coalesce().indices()
        src, dst = edge_index[0], edge_index[1]
        score = self.manifold.dist(q[src], k[dst])
        score = edge_softmax_by_src(src, -score, x.shape[0])
        att = torch.sparse_coo_tensor(edge_index, score, size=(x.shape[0], x.shape[0]), device=x.device)
        ass = torch.sparse.mm(att, ass)
        return gumbel_softmax(torch.log(ass + 1e-6), temperature=self.temperature)


class LSENetLayer(nn.Module):
    def __init__(self, manifold, in_dim, hid_dim, num_assign, dropout, bias=False, use_att=False,
                 nonlin=None, temperature=0.2):
        super().__init__()
        self.manifold = manifold
        self.assigner = LorentzAssignment(manifold, hid_dim, hid_dim, num_assign, dropout, bias, temperature)

    def forward(self, x, adj):
        ass = self.assigner(x, adj)
        support_t = ass.t() @ x
        denorm = (-self.manifold.inner(support_t, support_t, keepdim=True)).abs().clamp_min(1e-8).sqrt()
        x_par = support_t / denorm

        adj_dense = adj.to_dense()
        adj_par = ass.t() @ adj_dense @ ass
        idx = adj_par.nonzero().t()
        adj_par = torch.sparse_coo_tensor(idx, adj_par[idx[0], idx[1]], size=adj_par.shape, device=x.device)
        return x_par, adj_par, ass, x


class LorentzBoost(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        self.beta = nn.Parameter(torch.randn(in_dim - 1) * 0.01)

    def forward(self, x):
        beta = torch.tanh(self.beta) * 0.5
        beta_norm_sq = (beta ** 2).sum().clamp_max(1 - 1e-6)
        gamma = 1.0 / torch.sqrt(1.0 - beta_norm_sq + 1e-8)

        boost = torch.eye(self.in_dim, device=x.device, dtype=x.dtype)
        boost[0, 0] = gamma
        boost[0, 1:] = -gamma * beta
        boost[1:, 0] = -gamma * beta
        boost[1:, 1:] += (gamma - 1) * torch.outer(beta, beta) / (beta_norm_sq + 1e-8)
        return torch.einsum('ij,...j->...i', boost, x)


class LSENet(nn.Module):
    def __init__(self, manifold, in_dim, hid_dim, max_nums, temperature=0.2, dropout=0.5,
                 nonlin_str='relu'):
        super().__init__()
        self.manifold = manifold
        self.max_nums = max_nums
        self.height = len(max_nums) + 1
        self.input_proj = LorentzGraphConvolution(
            manifold, in_dim + 1, hid_dim + 1, True, dropout, False, select_activation(nonlin_str)
        )
        self.input_proj2 = LorentzGraphConvolution(
            manifold, hid_dim + 1, hid_dim + 1, True, dropout, False, select_activation(nonlin_str)
        )
        self.layers = nn.ModuleList()
        curr_dim = hid_dim + 1
        for i in range(self.height - 1):
            self.layers.append(
                LSENetLayer(
                    manifold,
                    curr_dim,
                    hid_dim + 1,
                    max_nums[i],
                    dropout=dropout,
                    temperature=temperature,
                    nonlin=select_activation(nonlin_str),
                )
            )
            curr_dim = hid_dim + 1

    def embed_leaf(self, x, adj):
        o = torch.zeros_like(x[:, :1])
        x = torch.cat([o, x], dim=1)
        x = self.manifold.expmap0(x)
        x = self.input_proj(x, adj)
        x = self.input_proj2(x, adj)
        return x

    def forward(self, x, adj):
        z = self.embed_leaf(x, adj)
        tree_coord_dict = {self.height: z}
        ass_dict = {}
        adj_dict = {self.height: adj}

        current_z = z
        current_adj = adj
        for i, layer in enumerate(self.layers):
            z_par, adj_par, ass, _ = layer(current_z, current_adj)
            level_curr = self.height - i
            level_par = self.height - i - 1
            tree_coord_dict[level_par] = z_par
            ass_dict[level_curr] = ass
            adj_dict[level_par] = adj_par
            current_z = z_par
            current_adj = adj_par

        root = self.manifold.frechet_mean(current_z)
        tree_coord_dict[0] = root
        ass_dict[1] = torch.ones(current_z.size(0), 1, device=x.device, dtype=x.dtype)
        return tree_coord_dict, ass_dict, adj_dict


class ASIL(TrainableModel):
    def __init__(self, in_dim, hid_dim, num_nodes, max_nums, temperature=0.2, dropout=0.5,
                 nonlin='relu', tau=1.0, alpha=0.01, knn=8):
        super().__init__()
        self.num_nodes = num_nodes
        self.height = len(max_nums) + 1
        self.manifold = Lorentz()
        self.encoder = LSENet(self.manifold, in_dim, hid_dim, max_nums, temperature, dropout, nonlin)
        self.lorentz_proj = LorentzBoost(hid_dim + 1)
        self.temperature = temperature
        self.tau = tau
        self.alpha = alpha
        self.knn = knn

    def embed(self, x, edge_index=None, adj=None):
        if adj is None:
            raise ValueError('ASIL.embed requires a normalized sparse adjacency via `adj`.')
        coord_dict, _, _ = self.encoder(x, adj)
        return coord_dict[self.height]

    def forward_tree(self, x, adj):
        return self.encoder(x, adj)

    def get_cluster_results(self, x, adj):
        coord_dict, ass_dict, _ = self.encoder(x, adj)
        embed_dict = {height: z.detach() for height, z in coord_dict.items()}
        clu_mat_dict = {self.height: torch.eye(self.num_nodes, device=x.device, dtype=x.dtype)}
        for k in range(self.height - 1, 0, -1):
            clu_mat_dict[k] = clu_mat_dict[k + 1] @ ass_dict[k + 1]
        for k, value in list(clu_mat_dict.items()):
            idx = value.max(1)[1]
            t = torch.zeros_like(value)
            t[torch.arange(t.shape[0], device=t.device), idx] = 1.0
            clu_mat_dict[k] = t
        return embed_dict, clu_mat_dict

    def fix_cluster_results(self, clu_res_mat, embed_dict, eps_int=7):
        clu_nums = clu_res_mat.sum(0)
        clu_res = clu_res_mat.argmax(1)
        corr_idx = clu_nums > eps_int
        if torch.all(corr_idx):
            return clu_res

        idx = torch.arange(clu_res_mat.shape[1], device=clu_res.device)
        idx = idx[corr_idx]
        err_idx = torch.where(clu_res_mat[:, clu_nums <= eps_int] == 1.0)[0]
        node = embed_dict[self.height]
        parent = embed_dict[1]
        error_node = node[err_idx]
        fixed_parent = parent[corr_idx]
        score = torch.log_softmax(2 + 2 * self.manifold.cinner(error_node, fixed_parent), dim=-1)
        fixed_res = gumbel_softmax(score, self.temperature)
        fixed_res = idx[fixed_res.argmax(1)]
        clu_res[err_idx] = fixed_res
        return clu_res

    def loss(self, x, adj, eps=1e-6):
        z_leaf = self.encoder.embed_leaf(x, adj)
        z_leaf = self.lorentz_proj(z_leaf)
        neg_dist2 = 2 + 2 * self.manifold.cinner(z_leaf, z_leaf)
        adj_aug = graph_top_k(torch.softmax(neg_dist2 / self.tau, dim=-1), k=self.knn)
        _, ass_aug_dict, adj_aug_dict = self.encoder(x, self.alpha * adj_aug + adj)
        return self._si_loss(ass_aug_dict, adj_aug_dict, eps)

    def _si_loss(self, ass_dict, adj_dict, eps=1e-6):
        se_loss = 0
        vol_g = adj_dict[self.height].sum()

        for k in range(self.height, 0, -1):
            adj_dense = adj_dict[k].to_dense()
            degree = adj_dense.sum(dim=1)
            diag = adj_dense.diag()
            if k == 1:
                vol_parent = vol_g
            else:
                vol_parent = adj_dict[k - 1].to_dense().sum(dim=-1)
                vol_parent = torch.einsum('ij,j->i', ass_dict[k], vol_parent)
            delta_vol = degree - diag
            log_vol_ratio = torch.log2((degree + eps) / (vol_parent + eps))
            se_loss += torch.sum(delta_vol * log_vol_ratio)
        return -se_loss / vol_g
