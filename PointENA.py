import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(x, k):
    B, N, _ = x.shape
    k_eff = min(k, N)
    dist = torch.cdist(x, x, p=2)                      # (B, N, N)
    _, idx = torch.topk(dist, k=k_eff, largest=False)  # (B, N, k_eff)
    return idx

def index_points(points, idx):
    B, N, C = points.shape
    k_eff = idx.shape[-1]
    batch_offset = (torch.arange(B, device=points.device) * N) \
                       .view(B,1,1)
    idx_flat = (idx + batch_offset).view(-1)
    return points.view(B*N, C)[idx_flat].view(B, N, k_eff, C)

class PointENAblock(nn.Module):
    def __init__(self, channels, knn_k=8, gamma_init=0.92):
        super().__init__()
        self.C  = channels
        self.k  = knn_k

        # Q/K/V & 输出投射
        self.q_proj   = nn.Linear(channels, channels, bias=False)
        self.k_proj   = nn.Linear(channels, channels, bias=False)
        self.v_proj   = nn.Linear(channels, channels, bias=False)
        self.out_proj = nn.Linear(channels, channels, bias=False)

        # 可训练 gamma：存 log_gamma
        self.log_gamma = nn.Parameter(torch.log(torch.tensor(gamma_init, dtype=torch.float32)))

    def forward(self, feats, coords):
        """
        feats:  (B, N, C)
        coords: (B, N, 3)
        """
        B, N, C = feats.shape

        q = self.q_proj(feats)           # (B, N, C)
        k = self.k_proj(feats)
        v = self.v_proj(feats)

        idx        = knn(coords, self.k)           # (B, N, k_eff)
        k_neigh    = index_points(k, idx)          # (B, N, k_eff, C)
        v_neigh    = index_points(v, idx)          # (B, N, k_eff, C)
        coord_neigh= index_points(coords, idx)     # (B, N, k_eff, 3)

        dist = torch.norm(coords.unsqueeze(2) - coord_neigh, dim=-1)  # (B, N, k_eff)

        gamma_w = torch.exp(self.log_gamma * dist)                    # (B, N, k_eff)

        scores = (q.unsqueeze(2) * k_neigh).sum(-1) / (C ** 0.5)      # (B, N, k_eff)
        scores = scores * gamma_w

        weights = F.softmax(scores, dim=2)                           # (B, N, k_eff)
        agg     = (weights.unsqueeze(-1) * v_neigh).sum(2)           # (B, N, C)

        # 6) 输出
        out = self.out_proj(agg)                                     # (B, N, C)
        return out

class PointENA(nn.Module):
    def __init__(self, channels, knn_k=8, gamma_init=0.92, activation='relu'):
        super().__init__()
        self.attn = PointENAblock(channels, knn_k, gamma_init)
        self.act  = getattr(F, activation)

    def forward(self, x_feats, x_coords):
        """
        x_feats: (B, C, N)
        x_coords:(B, N, 3)
        """
        feats = x_feats.permute(0,2,1).contiguous()  # (B, N, C)
        out   = self.attn(feats, x_coords)           # (B, N, C)
        out   = out.permute(0,2,1).contiguous()       # (B, C, N)
        return self.act(out + x_feats)
