import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# -------------------------
# Helpers
# -------------------------
def index_points(points, idx):
    """
    points: (B, N, C)
    idx:    (B, S, k)
    returns: (B, S, k, C)
    """
    device = points.device
    B, N, C = points.shape
    S, k = idx.shape[1], idx.shape[2]
    batch_idx = torch.arange(B, dtype=torch.long, device=device).view(B, 1, 1)
    batch_idx = batch_idx.repeat(1, S, k)  # (B, S, k)
    return points[batch_idx, idx, :]       # (B, S, k, C)

def knn_indices_from_coords(coords, k, chunk_size=None):
    """
    coords: (B, N, D)
    returns idx: (B, N, k)
    chunk_size: if set, compute cdist on query chunks of size chunk_size
    """
    B, N, D = coords.shape
    device = coords.device
    if chunk_size is None or N <= chunk_size:
        # full cdist (B, N, N)
        dist = torch.cdist(coords, coords, p=2)  # (B, N, N)
        _, idx = dist.topk(k=k, dim=-1, largest=False, sorted=True)
        return idx
    else:
        idx_list = []
        for start in range(0, N, chunk_size):
            end = min(N, start + chunk_size)
            # dist between queries coords[:, start:end, :] and refs coords
            dist_chunk = torch.cdist(coords[:, start:end, :], coords, p=2)  # (B, q, N)
            _, idx_chunk = dist_chunk.topk(k=k, dim=-1, largest=False, sorted=True)  # (B, q, k)
            idx_list.append(idx_chunk)
        return torch.cat(idx_list, dim=1)  # (B, N, k)

# -------------------------
# Simple SE channel module
# -------------------------
class SE_Channel(nn.Module):
    def __init__(self, channels, r=4):
        super().__init__()
        hidden = max(1, channels // r)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):  # x: (B, C, N)
        B, C, N = x.shape
        s = x.mean(dim=2)  # (B, C)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s)).view(B, C, 1)
        return x * s

# -------------------------
# Chunked GeoAttnUpConv
# -------------------------
class GeoAttnUpConv(nn.Module):
    """
    Chunked GeoAttnUpConv (memory-friendly)
    Args:
        in_channels (int)
        out_channels (int)
        k (int): neighbors
        hidden_channels (int): hidden dim for edge_mlp
        activation (str): 'relu' or 'gelu'
        use_se (bool)
        dynamic (bool): recompute knn each forward (pos preferred)
        knn_dim (int|None): if set, project features to knn_dim before knn
        knn_chunk_size (int|None): chunk size used IN knn cdist computation
        query_chunk (int): chunk size for processing queries (reduces peak mem)
        use_checkpoint (bool): use activation checkpoint on edge_mlp within each chunk
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 k=16,
                 hidden_channels=None,
                 activation='relu',
                 use_se=True,
                 dynamic=False,
                 knn_dim=32,
                 knn_chunk_size=1024,
                 query_chunk=256,
                 use_checkpoint=False):
        super().__init__()
        self.k = k
        self.dynamic = dynamic
        self.knn_dim = knn_dim
        self.knn_chunk_size = knn_chunk_size
        self.query_chunk = query_chunk
        self.use_checkpoint = use_checkpoint

        if hidden_channels is None:
            hidden_channels = max(in_channels, out_channels) // 2

        if activation.lower() == 'gelu':
            act_layer = nn.GELU
        else:
            act_layer = lambda: nn.ReLU(inplace=True)

        # edge MLP: input channels = 2 * in_channels -> hidden_channels
        # uses Conv2d treating (neighbor dim) as last spatial dim
        self.edge_mlp = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            act_layer()
        )

        # attention per neighbor -> scalar
        self.att_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=False)

        # fuse aggregated hidden -> out_channels
        self.fuse = nn.Sequential(
            nn.Conv1d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        self.act = act_layer()

        self.use_se = use_se
        if use_se:
            self.se = SE_Channel(out_channels, r=4)

        # residual mapping if needed
        self.residual = (in_channels == out_channels)
        if not self.residual:
            self.res_conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )

        # optional small projection for knn computation
        if self.knn_dim is not None and self.knn_dim > 0:
            self.knn_proj = nn.Conv1d(in_channels, self.knn_dim, kernel_size=1, bias=False)
        else:
            self.knn_proj = None

        # store dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

    def _compute_knn_idx(self, x, pos):
        """
        Compute knn indices (B, N, k) using either pos (geom) or projected features.
        x: (B, C, N)
        pos: (B, N, D) or None
        """
        B, C, N = x.shape
        device = x.device
        if self.dynamic:
            if pos is not None:
                coords = pos  # (B, N, D)
                idx = knn_indices_from_coords(coords, k=self.k, chunk_size=self.knn_chunk_size)
                return idx
            else:
                if self.knn_proj is not None:
                    proj = self.knn_proj(x)           # (B, knn_dim, N)
                    proj = proj.transpose(2, 1)       # (B, N, knn_dim)
                else:
                    proj = x.transpose(2, 1)          # (B, N, C)
                idx = knn_indices_from_coords(proj, k=self.k, chunk_size=self.knn_chunk_size)
                return idx
        else:
            # static: compute from input features once (with chunking to avoid full N*N if large)
            x_t = x.transpose(2, 1)  # (B, N, C)
            idx = knn_indices_from_coords(x_t, k=self.k, chunk_size=self.knn_chunk_size)
            return idx

    def forward(self, x, pos=None):
        """
        x: (B, C, N) - input point features
        pos: (B, N, D) optional geometry to compute knn on (preferred if dynamic True)
        returns: (B, out_channels, N)
        """
        B, C, N = x.shape
        device = x.device

        # compute knn indices (chunked inside if necessary)
        idx = self._compute_knn_idx(x, pos)  # (B, N, k)

        # prepare output tensor
        out = x.new_zeros((B, self.out_channels, N))

        # transpose once for indexing
        x_t_all = x.transpose(2, 1).contiguous()  # (B, N, C)

        # process queries in chunks to reduce peak memory
        for start in range(0, N, self.query_chunk):
            end = min(N, start + self.query_chunk)
            q = end - start  # chunk size

            # indices for this chunk: (B, q, k)
            idx_chunk = idx[:, start:end, :].contiguous()

            # gather neighbor features and center features
            neigh = index_points(x_t_all, idx_chunk)          # (B, q, k, C)
            center = x_t_all[:, start:end, :].unsqueeze(2).expand(-1, -1, self.k, -1)  # (B, q, k, C)

            # edge features: [neigh - center, center] -> (B, q, k, 2C)
            edge_chunk = torch.cat((neigh - center, center), dim=-1)  # (B, q, k, 2C)

            # permute to (B, 2C, q, k) for Conv2d
            edge_chunk = edge_chunk.permute(0, 3, 1, 2).contiguous()  # (B, 2C, q, k)

            # forward through edge_mlp (optionally checkpointed)
            if self.use_checkpoint:
                h = checkpoint(self.edge_mlp, edge_chunk)  # (B, hidden, q, k)
            else:
                h = self.edge_mlp(edge_chunk)               # (B, hidden, q, k)

            # attention across neighbors
            att = self.att_conv(h).squeeze(1)              # (B, q, k)
            att = F.softmax(att, dim=-1).unsqueeze(1)      # (B, 1, q, k)

            # weighted sum over neighbors -> (B, hidden, q)
            agg = torch.sum(h * att, dim=-1)               # (B, hidden, q)

            # fuse and activation -> (B, out, q)
            out_chunk = self.fuse(agg)                     # (B, out, q)
            out_chunk = self.act(out_chunk)

            if self.use_se:
                out_chunk = self.se(out_chunk)

            # residual: add original x chunk (mapped if needed)
            x_chunk = x[:, :, start:end]                   # (B, C, q)
            if self.residual:
                out_chunk = out_chunk + x_chunk
            else:
                out_chunk = out_chunk + self.res_conv(x_chunk)

            # write out
            out[:, :, start:end] = out_chunk

        return out  # (B, out, N)


class GeoExtractionWithCoords(nn.Module):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1, bias=True, activation='relu'):
        """
        支持坐标信息的GeoExtraction版本
        """
        super(GeoExtractionWithCoords, self).__init__()
        self.channels = channels
        self.blocks = blocks

        # 创建GeoAttnUpConv模块
        self.geo_attn = GeoAttnUpConv(
            in_channels=channels,
            out_channels=channels,
            k=16,
            hidden_channels=max(channels // 4, 32),
            activation=activation,
            use_se=True,
            dynamic=True,
            knn_dim=16,
            knn_chunk_size=1024,
            query_chunk=256,
            use_checkpoint=False
        )

        # 激活函数
        if activation.lower() == 'gelu':
            self.act = nn.GELU()
        elif activation.lower() == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x, coords=None):
        """
        增强的前向传播，支持坐标信息
        x: [B, C, N] - 特征
        coords: [B, N, 3] - 坐标信息（可选）
        返回: [B, C, N] - 处理后的特征
        """
        # 如果提供了坐标信息，使用坐标计算几何关系
        # 如果没有提供坐标，GeoAttnUpConv会使用特征本身计算KNN
        out = self.geo_attn(x, pos=coords)

        return self.act(out)