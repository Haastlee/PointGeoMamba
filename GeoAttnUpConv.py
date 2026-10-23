import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class GeoAttnUpConv(nn.Module):
    """
    Geometry-Aware KAT Attention Module (GKAM geometric-attention stage).

    Args:
        in_channels: Input feature channels.
        out_channels: Output feature channels.
        k: Number of KNN neighbors.
        hidden_channels: Edge MLP hidden channels.
        activation: 'relu' or 'gelu'.
        use_se: Whether to use global channel attention.
        dynamic: Use coordinates for KNN when coords are provided.
        knn_dim: Feature projection dimension when coordinates are absent.
        knn_chunk_size: Query chunk size for KNN computation.
        query_chunk: Query chunk size for edge feature processing.
        use_checkpoint: Whether to checkpoint Edge MLP during training.

    Inputs:
        x: (B, C, N) point features.
        coords: (B, N, 3) corresponding point coordinates, optional.

    Output:
        (B, out_channels, N)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        k=16,
        hidden_channels=None,
        activation="relu",
        use_se=True,
        dynamic=True,
        knn_dim=32,
        knn_chunk_size=1024,
        query_chunk=256,
        use_checkpoint=False,
    ):
        super().__init__()

        if k < 1 or query_chunk < 1:
            raise ValueError("k and query_chunk must be positive.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.dynamic = dynamic
        self.knn_chunk_size = knn_chunk_size
        self.query_chunk = query_chunk
        self.use_checkpoint = use_checkpoint
        self.use_se = use_se

        if hidden_channels is None:
            hidden_channels = max(in_channels, out_channels) // 2

        def make_activation():
            if activation.lower() == "gelu":
                return nn.GELU()
            return nn.ReLU(inplace=True)

        # Edge feature: [x_i, x_j - x_i, p_j - p_i], dimension = 2C + 3.
        self.edge_mlp = nn.Sequential(
            nn.Conv2d(2 * in_channels + 3, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            make_activation(),
        )

        # Neighbor attention.
        self.att_conv = nn.Conv2d(hidden_channels, 1, 1, bias=False)

        # Map aggregated edge features to output channels.
        self.fuse = nn.Sequential(
            nn.Conv1d(hidden_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
        )

        # SE channel attention.
        if use_se:
            se_hidden = max(1, out_channels // 4)
            self.se_fc1 = nn.Linear(out_channels, se_hidden)
            self.se_fc2 = nn.Linear(se_hidden, out_channels)

        # Residual channel alignment.
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.residual = nn.Identity()

        # Optional feature projection for KNN when coordinates are unavailable.
        if knn_dim is not None and knn_dim > 0:
            self.knn_proj = nn.Conv1d(in_channels, knn_dim, 1, bias=False)
        else:
            self.knn_proj = None

        self.out_act = make_activation()

    @staticmethod
    def _index_points(points, idx):
        """
        points: (B, N, C)
        idx:    (B, S, K)
        return: (B, S, K, C)
        """
        B = points.shape[0]
        batch_idx = torch.arange(
            B, device=points.device
        ).view(B, 1, 1).expand_as(idx)
        return points[batch_idx, idx, :]

    def _knn_indices(self, refs, k):
        """
        refs: (B, N, D)
        return: (B, N, K)
        """
        B, N, _ = refs.shape
        k = min(k, N)
        chunk_size = self.knn_chunk_size

        if chunk_size is None or N <= chunk_size:
            dist = torch.cdist(refs.float(), refs.float(), p=2)
            return dist.topk(
                k=k, dim=-1, largest=False, sorted=True
            ).indices

        indices = []
        refs_float = refs.float()

        for start in range(0, N, chunk_size):
            end = min(N, start + chunk_size)
            dist = torch.cdist(
                refs_float[:, start:end, :], refs_float, p=2
            )
            idx_chunk = dist.topk(
                k=k, dim=-1, largest=False, sorted=True
            ).indices
            indices.append(idx_chunk)

        return torch.cat(indices, dim=1)

    def _compute_knn(self, x, coords):
        if self.dynamic and coords is not None:
            refs = coords
        elif self.knn_proj is not None:
            refs = self.knn_proj(x).transpose(1, 2).contiguous()
        else:
            refs = x.transpose(1, 2).contiguous()

        return self._knn_indices(refs, self.k)

    def _se_attention(self, x):
        # x: (B, C, N)
        s = x.mean(dim=-1)
        s = F.relu(self.se_fc1(s))
        s = torch.sigmoid(self.se_fc2(s)).unsqueeze(-1)
        return x * s

    def forward(self, x, coords=None):
        """
        x:     (B, C, N)
        coords: (B, N, 3), must correspond to the points in x.
        """
        B, C, N = x.shape

        if C != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} feature channels, got {C}."
            )

        if coords is not None:
            if coords.shape != (B, N, 3):
                raise ValueError(
                    f"coords must have shape {(B, N, 3)}, "
                    f"got {tuple(coords.shape)}."
                )
            coords = coords.to(device=x.device, dtype=x.dtype)

        # Dynamic spatial KNN when coordinates are available.
        idx = self._compute_knn(x, coords)
        K = idx.shape[-1]

        x_t = x.transpose(1, 2).contiguous()
        coords_t = coords.contiguous() if coords is not None else None
        out_chunks = []

        for start in range(0, N, self.query_chunk):
            end = min(N, start + self.query_chunk)

            idx_chunk = idx[:, start:end, :]
            center = x_t[:, start:end, :].unsqueeze(2)
            center = center.expand(-1, -1, K, -1)
            neigh = self._index_points(x_t, idx_chunk)

            # Semantic feature difference: x_j - x_i.
            feature_diff = neigh - center

            # Geometric displacement: p_j - p_i.
            if coords_t is not None:
                pos_center = coords_t[:, start:end, :].unsqueeze(2)
                pos_neigh = self._index_points(coords_t, idx_chunk)
                pos_diff = pos_neigh - pos_center
            else:
                # Geometry is unavailable; retain input compatibility.
                pos_diff = x.new_zeros((B, end - start, K, 3))

            # [x_i, x_j - x_i, p_j - p_i]: (B, 2C+3, Q, K).
            edge = torch.cat(
                (center, feature_diff, pos_diff), dim=-1
            ).permute(0, 3, 1, 2).contiguous()

            if self.use_checkpoint and self.training:
                h = checkpoint(
                    self.edge_mlp, edge, use_reentrant=False
                )
            else:
                h = self.edge_mlp(edge)

            # Attention normalization over each point's K neighbors.
            att = F.softmax(self.att_conv(h).squeeze(1), dim=-1)

            # Weighted aggregation: (B, hidden_channels, Q).
            agg = (h * att.unsqueeze(1)).sum(dim=-1)

            out_chunk = self.out_act(self.fuse(agg))
            out_chunks.append(out_chunk)

        # Concatenate chunks before global SE attention.
        out = torch.cat(out_chunks, dim=-1)

        if self.use_se:
            out = self._se_attention(out)

        # Residual fusion and output activation.
        out = out + self.residual(x)
        return self.out_act(out)
