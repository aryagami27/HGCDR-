import numpy as np
from sklearn.neighbors import NearestNeighbors

class EdgePruner:
    def __init__(self, k_neighbors=10, contamination=0.05):
        """
        Args:
            k_neighbors (int): 'v' in the paper, number of neighbors to consider.
            contamination (float): Top % of items to prune (e.g., 0.05 for 5%).
        """
        self.k = k_neighbors
        self.contamination = contamination

    def fit_transform(self, item_embeddings):
        """
        Applies HSGE Edge Pruning.
        
        Args:
            item_embeddings (np.array or torch.Tensor): Matrix of item vectors [N, D]
            
        Returns:
            keep_mask (np.array): Boolean array of shape [N], True = Keep.
        """
        if hasattr(item_embeddings, 'cpu'):
            item_embeddings = item_embeddings.cpu().numpy()
            
        N = item_embeddings.shape[0]
        if N < self.k + 1:
            print(f"Warning: Not enough items ({N}) for k-NN ({self.k}). Skipping pruning.")
            return np.ones(N, dtype=bool)

        # 1. Compute k-NN distances and indices
        nbrs = NearestNeighbors(n_neighbors=self.k + 1).fit(item_embeddings)
        distances, indices = nbrs.kneighbors(item_embeddings)
        
        # distances[:, 0] is distance to self (0.0). use 1 to k.
        k_distances = distances[:, 1:] 
        k_indices = indices[:, 1:]
        
        # 2. Compute Reachability Distance (Eq 11)
        # Reach-dist_k(i, o) = max(k-distance(o), dist(i, o))
        # k-distance(o) is the distance of o to its k-th neighbor (distances[:, -1])
        
        k_dist_o = k_distances[:, -1] # [N] (The 'k-distance' of every point)
        
        # We need Reach-dist for every neighbor o of i
        # This is computationally heavy to map fully. 
        # Simplified vectorized approach:
        # reach_dist[i, j] (j-th neighbor) = max(k_dist[neighbor_index], dist[i, neighbor])
        
        # Gather k-distances of neighbors
        # k_indices is [N, k]
        neighbor_k_dists = k_dist_o[k_indices] # [N, k]
        
        # dist(i, o) is k_distances
        reach_dists = np.maximum(neighbor_k_dists, k_distances) # [N, k]
        
        # 3. Compute LRD (Eq 10)
        # LRD(i) = 1 / (mean(Reach-dist of k neighbors) + eps)
        avg_reach_dist = np.mean(reach_dists, axis=1) # [N]
        lrd = 1.0 / (avg_reach_dist + 1e-10) # [N]
        
        # 4. Compute IOF (Eq 12)
        # IOF(i) = mean(LRD of neighbors) / LRD(i)
        neighbor_lrds = lrd[k_indices] # [N, k]
        avg_neighbor_lrd = np.mean(neighbor_lrds, axis=1) # [N]
        
        iof = avg_neighbor_lrd / (lrd + 1e-10) # [N]
        
        # 5. Thresholding
        threshold = np.percentile(iof, 100 * (1 - self.contamination))
        
        keep_mask = iof <= threshold
        
        pruned_count = N - np.sum(keep_mask)
        print(f"Edge Pruner: IOF Threshold={threshold:.4f}. Pruned {pruned_count} items ({pruned_count/N:.2%}).")
        
        return keep_mask
