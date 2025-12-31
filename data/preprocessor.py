import torch
from torch_geometric.data import HeteroData
import pandas as pd
import numpy as np

class Preprocessor:
    def __init__(self, source_df, target_df, user_mapping, item_mapping_src=None, item_mapping_tgt=None):
        """
        Args:
            source_df: DataFrame with columns [user_id, item_id, rating, ...]
            target_df: DataFrame with columns [user_id, item_id, rating, ...]
            user_mapping: Dict mapping raw user IDs to integers (shared space or separate)
            item_mapping_src: Dict mapping raw source item IDs to integers
            item_mapping_tgt: Dict mapping raw target item IDs to integers
        """
        self.source_df = source_df
        self.target_df = target_df
        self.user_mapping = user_mapping
        self.item_mapping_src = item_mapping_src if item_mapping_src else self._create_mapping(source_df['item_id'])
        self.item_mapping_tgt = item_mapping_tgt if item_mapping_tgt else self._create_mapping(target_df['item_id'])

    def _create_mapping(self, series):
        unique_ids = series.unique()
        return {uid: i for i, uid in enumerate(unique_ids)}

    def process_graph(self, embedding_dim=128, item_emb_src=None, item_emb_tgt=None, prune_edges=False, pruner_params=None):
        """
        Constructs a HeteroData object from the dataframes.
        Args:
            embedding_dim: Dimension for random embeddings if not provided.
            item_emb_src: Pre-computed source item embeddings (Tensor).
            item_emb_tgt: Pre-computed target item embeddings (Tensor).
            prune_edges: Whether to apply HSGE edge pruning on source domain.
            pruner_params: Dict with 'k_neighbors' and 'contamination'.
        """
        data = HeteroData()

        # 1. Create Nodes
        num_users_src = len(self.source_df['user_id'].unique())
        num_users_tgt = len(self.target_df['user_id'].unique())
        num_items_src = len(self.item_mapping_src)
        num_items_tgt = len(self.item_mapping_tgt)
        
        # Initialize node features
        data['user_src'].x = torch.randn(num_users_src, embedding_dim)
        data['user_tgt'].x = torch.randn(num_users_tgt, embedding_dim)
        
        if item_emb_src is not None:
            data['item_src'].x = item_emb_src
        else:
            data['item_src'].x = torch.randn(num_items_src, embedding_dim)
            
        if item_emb_tgt is not None:
            data['item_tgt'].x = item_emb_tgt
        else:
            data['item_tgt'].x = torch.randn(num_items_tgt, embedding_dim)
        
        # 2. Edge Pruning (HSGE)
        # Only applied to Source Domain as per paper (to remove noise irrelevant to target)
        # Or applied to both? Paper says "Inter-domain relatedness-based edge pruning".
        # Usually we prune source items that are outliers relative to the structure.
        # Let's apply to source items based on the prompt "pass Amazon item embeddings into this".
        
        valid_src_items = None
        if prune_edges and item_emb_src is not None:
            from data.edge_pruner import EdgePruner
            params = pruner_params if pruner_params else {}
            pruner = EdgePruner(**params)
            
            # fit_transform returns mask of items to KEEP
            keep_mask = pruner.fit_transform(item_emb_src)
            
            # We need to filter interactions where item_id is in the keep set
            # keep_mask is size [num_items_src]
            # We can create a set of valid item IDs
            valid_src_items = set(np.where(keep_mask)[0])
            print(f"Edge Pruning: Kept {len(valid_src_items)}/{num_items_src} source items.")

        # 3. Create Edges
        # Source Domain
        src_u = self.source_df['user_id'].values
        src_i = self.source_df['item_id'].values
        
        if valid_src_items is not None:
            # Filter edges
            mask = [i in valid_src_items for i in src_i]
            src_u = src_u[mask]
            src_i = src_i[mask]
            
        data['user_src', 'rates', 'item_src'].edge_index = torch.stack([
            torch.tensor(src_u, dtype=torch.long), 
            torch.tensor(src_i, dtype=torch.long)
        ], dim=0)
        
        # Reverse edge
        data['item_src', 'rated_by', 'user_src'].edge_index = torch.stack([
            torch.tensor(src_i, dtype=torch.long), 
            torch.tensor(src_u, dtype=torch.long)
        ], dim=0)
        
        # Target Domain
        tgt_u = torch.tensor(self.target_df['user_id'].values, dtype=torch.long)
        tgt_i = torch.tensor(self.target_df['item_id'].values, dtype=torch.long)
        
        data['user_tgt', 'rates', 'item_tgt'].edge_index = torch.stack([tgt_u, tgt_i], dim=0)
        data['item_tgt', 'rated_by', 'user_tgt'].edge_index = torch.stack([tgt_i, tgt_u], dim=0)

        return data

    def process_graph_with_kg(self, kg_df, embedding_dim=128, item_emb_src=None, item_emb_tgt=None, prune_edges=False, pruner_params=None, valid_asin_map=None):
        """
        Extended process_graph to include KG.
        """
        # First build base graph
        data = self.process_graph(embedding_dim, item_emb_src, item_emb_tgt, prune_edges, pruner_params)
        
        if kg_df is None:
            return data
            
        print("Integrating Knowledge Graph...")
        
        # 1. Filter KG triples where head is a valid target item
        # Use provided valid_asin_map (ASIN -> DenseID) if available
        if valid_asin_map is not None:
             tgt_map = {str(k): v for k, v in valid_asin_map.items()}
             valid_asins = set(tgt_map.keys())
        else:
             print("Warning: valid_asin_map missing, fallback to internal map (likely encoded).")
             tgt_map = {str(k): v for k, v in self.item_mapping_tgt.items()}
             valid_asins = set(tgt_map.keys())

        # Create Entity Mapping (External Entities)
        # We only care about tails from valid heads
        # Force string conversion for robust matching
        kg_df['head_id'] = kg_df['head_id'].astype(str)
        
        # DEBUG: Intersection check
        # common = valid_asins.intersection(set(kg_df['head_id'].unique()))
        # print(f"DEBUG: Common Items found in KG: {len(common)}")
        
        kg_valid = kg_df[kg_df['head_id'].isin(valid_asins)]
        print(f"DEBUG: KG Valid Length: {len(kg_valid)}")
        
        if len(kg_valid) == 0:
            print("Warning: No matching KG entities found for target items.")
            return data

        unique_tails = kg_valid['tail_id'].unique()
        entity_mapping = {eid: i for i, eid in enumerate(unique_tails)}
        
        # Prepare mapped edge indices
        item_indices = []
        entity_indices = []
        
        for _, row in kg_valid.iterrows():
            # Force string comparison
            h_id = str(row['head_id'])
            t_id = row['tail_id']
            
            if h_id in tgt_map and t_id in entity_mapping:
                item_indices.append(tgt_map[h_id])
                entity_indices.append(entity_mapping[t_id])
                
        # Create Entity Nodes
        num_entities = len(entity_mapping)
        
        # USER REQUIRED FIX: KG Node Features (Sentence-BERT)
        try:
             from sentence_transformers import SentenceTransformer
             import gc
             
             print(f"Generating embeddings for {num_entities} KG entities...")
             
             # [CRITICAL] Memory Optimization for Full Scale
             # Clear unused memory before loading new model
             gc.collect() 
             if torch.backends.mps.is_available():
                 torch.mps.empty_cache()
             
             # Initialize model on CPU to avoid MPS contention/OOM during heavy data loading phase
             model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
             
             # Extract texts from node IDs (e.g. "cat:Fiction" -> "Fiction")
             # Sort by ID to match node index
             sorted_entities = [eid for eid, idx in sorted(entity_mapping.items(), key=lambda x: x[1])]
             cleaned_texts = [eid.replace('cat:', '').replace('res:', '').replace('_', ' ') for eid in sorted_entities]
             
             # Encode on CPU
             embeddings = model.encode(cleaned_texts, convert_to_tensor=True, show_progress_bar=False, device='cpu', batch_size=32)
             
             # S-BERT returns inference tensors; detach to interact with Autograd
             data['entity'].x = embeddings.clone().detach().to(torch.float)
             
             # Cleanup S-BERT model immediately to free memory
             del model
             del embeddings
             gc.collect()
             
             print("KG Node Embeddings generated successfully.")
             
        except ImportError:
             print("Warning: sentence_transformers not installed. Using random embeddings.")
             data['entity'].x = torch.randn(num_entities, embedding_dim)
        except Exception as e:
             print(f"Warning: Failed to generate KG embeddings: {e}. Using random.")
             data['entity'].x = torch.randn(num_entities, embedding_dim)
        
        # Add KG Edges: (item_tgt, has_knowledge, entity)
        data['item_tgt', 'has_kg', 'entity'].edge_index = torch.stack([
            torch.tensor(item_indices, dtype=torch.long),
            torch.tensor(entity_indices, dtype=torch.long)
        ], dim=0)
        
        # Reverse KG Edges
        data['entity', 'kg_of', 'item_tgt'].edge_index = torch.stack([
            torch.tensor(entity_indices, dtype=torch.long),
            torch.tensor(item_indices, dtype=torch.long)
        ], dim=0)
        
        print(f"KG Integrated: {num_entities} entities, {len(item_indices)} edges.")
        
        return data
