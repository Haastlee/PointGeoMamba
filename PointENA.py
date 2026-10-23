import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    """
    x: (B, N, 3)
    return: (B, N, k_eff)
    """
    B, N, _ = x.shape
    k_eff = min(k, N)
    dist = torch.cdist(x, x, p=2)
    return dist.topk(k=k_eff, dim=-1, largest=False, sorted=True).indices


def index_points(points, idx):
    """
    points: (B, N, C)
    idx:    (B, N, K)
    return: (B, N, K, C)
    """
    B, N, C = points.shape
    K = idx.shape[-1]
    batch_offset = (torch.arange(B, device=points.device) * N).view(B, 1, 1)
    idx_flat = (idx + batch_offset).reshape(-1)
    return points.reshape(B * N, C)[idx_flat].reshape(B, N, K, C)


class PointENAblock(nn.Module):
    def __init__(self, channels, knn_k=8, gamma_init=0.86):
        super().__init__()
        if not 0.0 < gamma_init < 1.0:
            raise ValueError("gamma_init must be in (0, 1).")

        self.C = channels
        self.k = knn_k

        # Q/K/V projections and output projection
        self.q_proj = nn.Linear(channels, channels, bias=False)
        self.k_proj = nn.Linear(channels, channels, bias=False)
        self.v_proj = nn.Linear(channels, channels, bias=False)
        self.out_proj = nn.Linear(channels, channels, bias=False)

        # Learnable unconstrained parameter eta:
        # gamma = sigmoid(eta) is always in (0, 1).
        eta_init = torch.logit(torch.tensor(float(gamma_init), dtype=torch.float32))
        self.eta = nn.Parameter(eta_init)

    @property
    def gamma(self):
        return torch.sigmoid(self.eta)

    def forward(self, feats, coords):
        """
        feats:  (B, N, C)
        coords: (B, N, 3)
        """
        B, N, C = feats.shape

        q = self.q_proj(feats)                         # (B, N, C)
        key = self.k_proj(feats)                       # (B, N, C)
        v = self.v_proj(feats)                         # (B, N, C)

        idx = knn(coords, self.k)                      # (B, N, K)
        key_neigh = index_points(key, idx)             # (B, N, K, C)
        v_neigh = index_points(v, idx)                 # (B, N, K, C)
        coord_neigh = index_points(coords, idx)        # (B, N, K, 3)

        # Euclidean distances d_ij
        dist = torch.norm(
            coords.unsqueeze(2) - coord_neigh, dim=-1
        )                                               # (B, N, K)

        # Content similarity: Q_i^T K_j / sqrt(C)
        content_scores = (
            q.unsqueeze(2) * key_neigh
        ).sum(dim=-1) / (C ** 0.5)                     # (B, N, K)

        # Additive distance bias:
        # b_ij = log(w_ij) = d_ij * log(gamma)
        # F.logsigmoid(eta) computes log(sigmoid(eta)) stably.
        log_gamma = F.logsigmoid(self.eta)
        distance_bias = dist * log_gamma               # (B, N, K)

        # Paper formulation: content similarity + distance bias
        scores = content_scores + distance_bias

        # Neighborhood attention and feature aggregation
        weights = F.softmax(scores, dim=2)             # (B, N, K)
        agg = (weights.unsqueeze(-1) * v_neigh).sum(dim=2)

        return self.out_proj(agg)                      # (B, N, C)


class PointENA(nn.Module):
    def __init__(self, channels, knn_k=8, gamma_init=0.86, activation="relu"):
        super().__init__()
        self.attn = PointENAblock(channels, knn_k, gamma_init)
        self.act = getattr(F, activation)

    def forward(self, x_feats, x_coords):
        """
        x_feats:  (B, C, N)
        x_coords: (B, N, 3)
        """
        feats = x_feats.transpose(1, 2).contiguous()   # (B, N, C)
        out = self.attn(feats, x_coords)                # (B, N, C)
        out = out.transpose(1, 2).contiguous()         # (B, C, N)
        return self.act(out + x_feats)
