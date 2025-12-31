
try:
    import faiss
except ImportError:
    faiss = None
import numpy as np

class ANNItemRetriever:
    """
    Approximate Nearest Neighbor (ANN) Retrieval using FAISS.
    Retrieves top-K candidate items for a given user embedding using L2 distance.
    """
    def __init__(self, emb_dim: int, nlist: int = 100):
        """
        Args:
            emb_dim: Dimension of embeddings.
            nlist: Number of clusters (Voronoi cells) for IVFFlat index.
        """
        if faiss is None:
            raise ImportError("FAISS is not installed. Please install it via 'pip install faiss-cpu' or 'faiss-gpu'.")
            
        self.emb_dim = emb_dim
        self.nlist = nlist
        self.is_trained = False
        
        # We use IndexIVFFlat for faster search on large datasets
        # Quantizer is a flat L2 index
        quantizer = faiss.IndexFlatL2(emb_dim)
        self.index = faiss.IndexIVFFlat(quantizer, emb_dim, nlist, faiss.METRIC_L2)

    def build(self, item_embeddings: np.ndarray):
        """
        Builds the FAISS index from item embeddings.
        Args:
            item_embeddings: [num_items, emb_dim] numpy array.
        """
        if item_embeddings.shape[1] != self.emb_dim:
            raise ValueError(f"Embedding dim mismatch. Expected {self.emb_dim}, got {item_embeddings.shape[1]}")
            
        # FAISS expects float32
        item_embeddings = item_embeddings.astype('float32')
        
        num_items = item_embeddings.shape[0]
        
        # Heuristic: If items < 39 * nlist, use Flat index to avoid clustering errors/warnings
        if not self.index.is_trained:
            if num_items < self.nlist * 39:
                print(f"Dataset size ({num_items}) is too small for IVF{self.nlist}. Switching to IndexFlatL2.")
                self.index = faiss.IndexFlatL2(self.emb_dim) # Replace with simple flat index
                self.is_trained = True # Flat index doesn't need training
            else:
                 self.index.train(item_embeddings)
                 self.is_trained = True
            
        self.index.add(item_embeddings)

    def retrieve(self, user_embs: np.ndarray, k: int = 1000):
        """
        Retrieves top-K nearest items for a batch of user embeddings.
        Args:
            user_embs: [batch_size, emb_dim] numpy array.
            k: Number of candidates to retrieve.
            
        Returns:
            indices: [batch_size, k] Item IDs of retrieved candidates.
        """
        if not self.is_trained:
             raise RuntimeError("Index not trained. Call build() first.")
             
        user_embs = user_embs.astype('float32')
        
        # Perform search
        # D: distances, I: indices
        D, I = self.index.search(user_embs, k)
        return I, D
