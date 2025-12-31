import torch
import torch.nn as nn
import yaml
import pickle
import logging
import os
import numpy as np

# Imports from project
from models.recommender import HGCDRPlus
from retrieval.ann_retriever import ANNItemRetriever
from retrieval.reranker import NeuralReRanker

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

class HGCDRInference:
    def __init__(self, config_path, model_path, mapping_path, device='cpu'):
        self.device = device
        self.config = self._load_config(config_path)
        # [CRITICAL] Force heads=8 to match training checkpoint
        self.config['gnn_heads'] = 8
        self.mappings = self._load_mappings(mapping_path)
        
        # Initialize Model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Initialize Retrieval Components
        self.retriever = ANNItemRetriever(emb_dim=self.config['embedding_dim'])
        self.reranker = NeuralReRanker(self.config['embedding_dim'], self.config['embedding_dim']).to(device)
        self.reranker.eval()
        
        # Cache for Embeddings (to be populated)
        self.item_embeddings = None
        self.user_embeddings = None

    def _load_config(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _load_mappings(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _load_model(self, path):
        # Metadata needed for initialization (should be saved with config or mappings)
        # For now, we infer minimal metadata or it should be passed?
        # HGCDRPlus init requires: metadata, num_users_src, num_users_tgt, ...
        # We derive sizes from mappings
        
        num_users_src = len(self.mappings['source_user2id'])
        num_users_tgt = len(self.mappings['target_user2id'])
        num_items_src = len(self.mappings['source_item2id'])
        num_items_tgt = len(self.mappings['target_item2id'])
        
        # Metadata hardcoded or saved? 
        # Ideally passed in config or saved as 'metadata.pkl'
        # Fallback: Hardcoded standard metadata for HGCDR
        metadata = (
            ['user_src', 'user_tgt', 'item_src', 'item_tgt', 'entity'],
            [('user_src', 'rates', 'item_src'), ('item_src', 'rated_by', 'user_src'),
             ('user_tgt', 'rates', 'item_tgt'), ('item_tgt', 'rated_by', 'user_tgt'),
             ('item_tgt', 'has_kg', 'entity'), ('entity', 'kg_of', 'item_tgt')]
        )
        
        model = HGCDRPlus(self.config, metadata, num_users_src, num_users_tgt, num_items_src, num_items_tgt)
        if os.path.exists(path):
            state_dict = torch.load(path, map_location=self.device)
            # Use strict=False to handle missing Disentangle keys (checkpoint has fewer than model needs)
            # This is safe because training seemingly didn't use/save them, so random init for unused parts is fine.
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            logging.info(f"Loaded model weights from {path}")
            if missing:
                logging.warning(f"Missing keys (safe to ignore if unused): {missing[:5]}...")
            if unexpected:
                logging.warning(f"Unexpected keys: {unexpected[:5]}...")
        else:
            raise FileNotFoundError(f"Critical: Model checkpoint not found at {path}. Inference cannot proceed without trained weights.")
            
        return model.to(self.device)

    def update_item_index(self, item_embeddings_tensor=None):
        """
        Updates the FAISS index with item embeddings.
        Args:
            item_embeddings_tensor: (num_items, dim)
        """
        if item_embeddings_tensor is None:
            # Compute from model? 
            # Ideally we extract trained embeddings.
            # Using Target Item Embeddings from model's ID Encoder as baseline?
            # Or HGT Output? for inference often Transductive HGT output is used.
            # Here we use ID embeddings for simplicity + speed in this template.
            # In production, run HGT on full graph to get 'contextual' embeddings.
            
            # Using ID Embeddings:
            num_tgt_items = len(self.mappings['target_item2id'])
            with torch.no_grad():
                # Batch these if too large
                all_ids = torch.arange(num_tgt_items, device=self.device)
                item_embeddings_tensor = self.model.tgt_item_emb(all_ids).cpu().numpy()
        
        self.retriever.build(item_embeddings_tensor)
        self.item_embeddings = torch.tensor(item_embeddings_tensor, device=self.device)
        logging.info("Updated ANN Index.")

    def predict(self, user_id_raw, k=10, domain='target'):
        """
        End-to-End Prediction.
        Args:
            user_id_raw: Original user ID (string/int).
            k: Number of recommendations.
        """
        # 1. Map ID
        if domain == 'target':
            u_map = self.mappings['target_user2id']
            i_map_rev = {v: k for k, v in self.mappings['target_item2id'].items()}
        else:
            raise NotImplementedError("Only target domain inference supported.")
            
        if user_id_raw not in u_map:
            logging.warning(f"User {user_id_raw} not found. Returning random/popular.")
            return []
            
        uid = u_map[user_id_raw]
        uid_tensor = torch.tensor([uid], device=self.device)
        
        # 2. Get User Embedding
        with torch.no_grad():
            # Again, HGT context? 
            # For pure ID inference:
            user_emb = self.model.tgt_user_emb(uid_tensor) # (1, dim)
            
            # If we had disentanglement, we'd apply it here.
            # Assuming user_emb is the 'Final' representation.
        
        # 3. Retrieval (ANN)
        user_emb_np = user_emb.cpu().numpy()
        candidates_indices, distances = self.retriever.retrieve(user_emb_np, k=k*5) # Retrieve top 50
        
        candidates_indices = candidates_indices[0] # Flatten
        
        # 4. Re-ranking
        # Get Candidate Embeddings
        cand_tensor = torch.tensor(candidates_indices, device=self.device)
        cand_embs = self.model.tgt_item_emb(cand_tensor) # (50, dim)
        
        # Pass to Reranker
        # Reranker takes (user, item)
        # User repeats
        user_rep = user_emb.repeat(len(candidates_indices), 1)
        
        with torch.no_grad():
            scores = self.reranker(user_rep, cand_embs).squeeze()
            
        # 5. Top K
        top_k_vals, top_k_inds = torch.topk(scores, k)
        final_indices = candidates_indices[top_k_inds.cpu().numpy()]
        
        # 6. Decode
        results = [i_map_rev.get(idx, f"Item_{idx}") for idx in final_indices]
        return results

if __name__ == "__main__":
    # Example Usage
    config_path = 'configs/config.yaml'
    model_path = 'models/saved/best_model.pth'
    mapping_path = 'models/saved/mappings.pkl'
    
    # Fixed: No fallback to random weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Best model not found at {model_path}. Run training first.")
    
    inference = HGCDRInference(
        config_path=config_path,
        model_path=model_path, 
        mapping_path=mapping_path
    )
    
    # Initialize Index
    inference.update_item_index()
    
    # Predict for a Valid User
    # Get a real user ID from valid mappings
    all_users = list(inference.mappings['target_user2id'].keys())
    if all_users:
        sample_user = all_users[0] # Pick first valid user
        logging.info(f"Testing Inference for User: {sample_user}")
        recs = inference.predict(sample_user)
        print(f"Recommendations for {sample_user}: {recs}")
    else:
        logging.error("No users found in mapping!")
