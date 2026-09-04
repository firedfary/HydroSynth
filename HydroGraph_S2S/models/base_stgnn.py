import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class DiffusionGraphConv(nn.Module):
    """
    Diffusion Graph Convolution layer over multiple graph transition matrices (P_f, P_b, A_adapt).
    """
    def __init__(self, in_channels: int, out_channels: int, num_supports: int = 2, max_diffusion_step: int = 2):
        super().__init__()
        self.num_matrices = num_supports * max_diffusion_step + 1
        self.weight = nn.Parameter(torch.FloatTensor(self.num_matrices * in_channels, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.max_diffusion_step = max_diffusion_step
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, supports: List[torch.Tensor]) -> torch.Tensor:
        """
        x: (B, T, N, C_in) or (B, N, C_in)
        supports: list of (N, N) transition/adjacency matrices
        Output: (B, T, N, C_out) or (B, N, C_out)
        """
        has_time_dim = (x.dim() == 4)
        if has_time_dim:
            B, T, N, C = x.shape
            x_reshaped = x.view(B * T, N, C)
        else:
            B, N, C = x.shape
            x_reshaped = x

        x_list = [x_reshaped]
        for adj in supports:
            x_k = x_reshaped
            for _ in range(self.max_diffusion_step):
                x_k = torch.matmul(adj, x_k)
                x_list.append(x_k)

        # Concatenate along channel dimension: (B*T, N, num_matrices * C)
        x_concat = torch.cat(x_list, dim=-1)
        out = torch.matmul(x_concat, self.weight) + self.bias

        if has_time_dim:
            out = out.view(B, T, N, -1)
        return out


class AdaptiveAdjacency(nn.Module):
    """
    Learns data-driven hidden teleconnections and dynamic adjacency matrix:
    A_adp = Softmax(ReLU(E_1 @ E_2^T))
    """
    def __init__(self, num_nodes: int, embed_dim: int = 16):
        super().__init__()
        self.node_vec1 = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.node_vec2 = nn.Parameter(torch.randn(num_nodes, embed_dim))
        nn.init.xavier_uniform_(self.node_vec1)
        nn.init.xavier_uniform_(self.node_vec2)

    def forward(self) -> torch.Tensor:
        adj = F.softmax(F.relu(torch.matmul(self.node_vec1, self.node_vec2.t())), dim=-1)
        return adj


class TemporalGatedConv(nn.Module):
    """
    Dilated Temporal Convolutional Network (TCN) with Gated Linear Unit (GLU):
    out = tanh(Conv1(x)) * sigmoid(Conv2(x))
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation  # Causal padding
        
        self.conv_filter = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1),
            dilation=(dilation, 1)
        )
        self.conv_gate = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1),
            dilation=(dilation, 1)
        )
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T, N)
        Output: (B, C_out, T, N)
        """
        x_pad = F.pad(x, (0, 0, self.padding, 0))  # Pad along temporal dimension only
        
        filter_out = torch.tanh(self.conv_filter(x_pad))
        gate_out = torch.sigmoid(self.conv_gate(x_pad))
        out = filter_out * gate_out
        
        res = self.residual_conv(x)
        return out + res


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM):
    Injects macro-scale dynamical model climate forcing (Z) into micro-scale graph representations (H):
    FiLM(H) = gamma(Z) * H + beta(Z)
    """
    def __init__(self, macro_dim: int, hidden_dim: int):
        super().__init__()
        self.scale_net = nn.Sequential(
            nn.Linear(macro_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.shift_net = nn.Sequential(
            nn.Linear(macro_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, h: torch.Tensor, macro_z: torch.Tensor) -> torch.Tensor:
        """
        h: (B, T, N, hidden_dim)
        macro_z: (B, N, macro_dim)
        Output: (B, T, N, hidden_dim)
        """
        gamma = self.scale_net(macro_z)  # (B, N, hidden_dim)
        beta = self.shift_net(macro_z)   # (B, N, hidden_dim)
        
        gamma = gamma.unsqueeze(1)       # (B, 1, N, hidden_dim)
        beta = beta.unsqueeze(1)         # (B, 1, N, hidden_dim)
        
        return (1.0 + gamma) * h + beta


class SpatialTemporalBlock(nn.Module):
    """
    ST-GNN Core Building Block:
    Temporal Dilated Conv -> Diffusion Graph Conv -> FiLM Macro Modulation -> LayerNorm / Residual
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        macro_dim: int,
        kernel_size: int = 3,
        dilation: int = 1,
        num_supports: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        self.tcn = TemporalGatedConv(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
        self.gcn = DiffusionGraphConv(out_channels, out_channels, num_supports=num_supports)
        self.film = FiLMLayer(macro_dim, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        self.res_proj = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, supports: List[torch.Tensor], macro_z: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, N, in_channels)
        supports: list of graph transition matrices
        macro_z: (B, N, macro_dim)
        Output: (B, T, N, out_channels)
        """
        B, T, N, C = x.shape
        res = self.res_proj(x)
        
        # TCN expects (B, C, T, N)
        x_tcn = x.permute(0, 3, 1, 2)
        h_tcn = self.tcn(x_tcn).permute(0, 2, 3, 1)  # (B, T, N, out_channels)
        
        # Spatial Graph Conv
        h_gcn = self.gcn(h_tcn, supports)
        
        # Macro climate conditioning
        h_film = self.film(h_gcn, macro_z)
        
        out = self.norm(h_film + res)
        out = self.dropout(out)
        return out


class BaseSTGNN(nn.Module):
    """
    Base Spatio-Temporal Graph Neural Network with multi-scale blocks.
    """
    def __init__(
        self,
        num_nodes: int,
        in_dim: int,
        hidden_dim: int,
        macro_dim: int,
        num_layers: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.adaptive_adj = AdaptiveAdjacency(num_nodes)
        
        self.input_fc = nn.Linear(in_dim, hidden_dim)
        
        self.st_blocks = nn.ModuleList([
            SpatialTemporalBlock(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                macro_dim=macro_dim,
                dilation=2**i,
                num_supports=3,  # P_f, P_b, A_adp
                dropout=dropout
            )
            for i in range(num_layers)
        ])

    def extract_features(
        self,
        x: torch.Tensor,
        supports: List[torch.Tensor],
        macro_z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, N, in_dim)
        supports: list of precomputed transition matrices
        macro_z: (B, N, macro_dim)
        Returns:
            h: (B, T, N, hidden_dim)
            adp: (N, N) learned adaptive adjacency
        """
        adp = self.adaptive_adj()
        all_supports = list(supports) + [adp]
        
        h = self.input_fc(x)
        for block in self.st_blocks:
            h = block(h, all_supports, macro_z)
            
        return h, adp
