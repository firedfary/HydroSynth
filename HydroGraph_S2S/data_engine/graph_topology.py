import os
import numpy as np
import scipy.sparse as sp
import torch
from typing import Tuple, List, Optional, Dict


class GraphTopologyBuilder:
    """
    Constructs multi-relational spatial graph topologies for 2371 stations:
    1. A_geo: Geographic distance graph via Haversine formula + Gaussian kernel + k-NN
    2. A_dem: Topographic elevation difference barrier graph
    3. A_corr: Historical precipitation teleconnection correlation graph
    4. Transition matrices (Forward/Backward) for Diffusion Graph Convolution
    """
    def __init__(
        self,
        coords: np.ndarray,
        elevations: np.ndarray,
        knn_k: int = 12,
        geo_sigma: float = 200.0,   # km
        dem_sigma: float = 500.0,   # meters
        cache_dir: str = "./cache"
    ):
        self.coords = coords          # (N, 2) [lat, lon]
        self.elevations = elevations  # (N,)
        self.num_nodes = coords.shape[0]
        self.knn_k = knn_k
        self.geo_sigma = geo_sigma
        self.dem_sigma = dem_sigma
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.cache_file = os.path.join(self.cache_dir, f"graph_adj_N{self.num_nodes}_k{knn_k}.npz")

    @staticmethod
    def haversine_distance_matrix(coords: np.ndarray) -> np.ndarray:
        """
        Compute pairwise great-circle distance (km) between coordinates [lat, lon].
        """
        R = 6371.0  # Earth radius in km
        lat = np.radians(coords[:, 0])
        lon = np.radians(coords[:, 1])
        
        dlat = lat[:, None] - lat[None, :]
        dlon = lon[:, None] - lon[None, :]
        
        a = np.sin(dlat / 2.0)**2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2.0)**2
        a = np.clip(a, 0.0, 1.0)
        c = 2.0 * np.arcsin(np.sqrt(a))
        return R * c

    def build_geo_adj(self, dist_matrix: np.ndarray) -> np.ndarray:
        """Construct Gaussian kernel geographic distance graph with k-NN sparsification."""
        N = self.num_nodes
        adj = np.zeros((N, N), dtype=np.float32)
        
        for i in range(N):
            # Sort distances
            dists = dist_matrix[i]
            knn_indices = np.argsort(dists)[1:self.knn_k + 1]  # exclude self
            for j in knn_indices:
                w = np.exp(- (dists[j] ** 2) / (self.geo_sigma ** 2))
                adj[i, j] = w
                adj[j, i] = w  # symmetric
        np.fill_diagonal(adj, 1.0)
        return adj

    def build_dem_adj(self, geo_adj: np.ndarray) -> np.ndarray:
        """Construct topographic elevation barrier graph."""
        elev_diff = np.abs(self.elevations[:, None] - self.elevations[None, :])
        dem_decay = np.exp(- elev_diff / self.dem_sigma)
        dem_adj = geo_adj * dem_decay
        np.fill_diagonal(dem_adj, 1.0)
        return dem_adj.astype(np.float32)

    def build_corr_adj(self, historical_precip: Optional[np.ndarray], threshold: float = 0.35) -> np.ndarray:
        """Construct historical teleconnection correlation graph."""
        N = self.num_nodes
        if historical_precip is None:
            return np.eye(N, dtype=np.float32)
            
        corr = np.corrcoef(historical_precip.T)
        corr = np.nan_to_num(corr, nan=0.0)
        
        adj_corr = np.where(corr >= threshold, corr, 0.0).astype(np.float32)
        np.fill_diagonal(adj_corr, 1.0)
        return adj_corr

    @staticmethod
    def compute_transition_matrices(adj: np.ndarray) -> List[np.ndarray]:
        """
        Compute forward and backward random walk transition matrices for Diffusion GCN:
        P_f = D_out^{-1} A
        P_b = D_in^{-1} A^T
        """
        adj = np.maximum(adj, 0.0)
        # Add self-loop
        adj = adj + np.eye(adj.shape[0], dtype=np.float32)
        
        # Out-degree normalization
        d_out = np.sum(adj, axis=1)
        d_out_inv = np.divide(1.0, d_out, out=np.zeros_like(d_out), where=d_out > 0)
        p_f = d_out_inv[:, None] * adj
        
        # In-degree normalization
        d_in = np.sum(adj, axis=0)
        d_in_inv = np.divide(1.0, d_in, out=np.zeros_like(d_in), where=d_in > 0)
        p_b = d_in_inv[None, :] * adj.T
        
        return [p_f.astype(np.float32), p_b.astype(np.float32)]

    def get_all_topologies(
        self,
        historical_precip: Optional[np.ndarray] = None,
        force_recompute: bool = False
    ) -> Dict[str, np.ndarray]:
        """Build or load all adjacency topologies and transition matrices."""
        if not force_recompute and os.path.exists(self.cache_file):
            print(f"[GraphTopologyBuilder] Loading cached adjacency topologies from {self.cache_file}...")
            data = np.load(self.cache_file)
            return {key: data[key] for key in data.files}

        print(f"[GraphTopologyBuilder] Computing distance matrix for {self.num_nodes} stations...")
        dist_mat = self.haversine_distance_matrix(self.coords)
        
        print("[GraphTopologyBuilder] Building Geographic and Topographic Adjacency Graphs...")
        adj_geo = self.build_geo_adj(dist_mat)
        adj_dem = self.build_dem_adj(adj_geo)
        adj_corr = self.build_corr_adj(historical_precip)
        
        trans_geo = self.compute_transition_matrices(adj_geo)
        trans_dem = self.compute_transition_matrices(adj_dem)
        
        topologies = {
            "adj_geo": adj_geo,
            "adj_dem": adj_dem,
            "adj_corr": adj_corr,
            "pf_geo": trans_geo[0],
            "pb_geo": trans_geo[1],
            "pf_dem": trans_dem[0],
            "pb_dem": trans_dem[1]
        }
        
        np.savez_compressed(self.cache_file, **topologies)
        print(f"[GraphTopologyBuilder] Saved adjacency topologies to {self.cache_file}.")
        return topologies
