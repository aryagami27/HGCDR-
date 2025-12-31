import os
import pandas as pd
import torch

class KGLoader:
    def __init__(self, data_dir, domain='Books'):
        """
        Loads Amazon-KG dataset.
        Args:
            data_dir: Root directory of the dataset (e.g., .../Amazon-KG-v2.0-dataset-main)
            domain: Domain name (e.g., 'Books', 'Movies') matches folder naming.
        """
        self.data_dir = data_dir
        self.domain = domain
        self.dataset_path = os.path.join(data_dir, f"Amazon-KG-5core-{domain}")
        self.kg_file = os.path.join(self.dataset_path, f"Amazon-KG-5core-{domain}.kg")
        self.inter_file = os.path.join(self.dataset_path, f"Amazon-KG-5core-{domain}.inter")

    def load_kg(self):
        """
        Loads Knowledge Graph triples.
        Returns:
            pd.DataFrame: DataFrame with [head_id, relation_id, tail_id]
        """
        if not os.path.exists(self.kg_file):
            print(f"Warning: KG file not found at {self.kg_file}")
            return None
        
        # Determine separator: usually tab-separated for RecBole datasets
        try:
            kg_df = pd.read_csv(self.kg_file, sep='\t')
        except:
            kg_df = pd.read_csv(self.kg_file)
            
        # Normalize columns (remove :token suffix if present)
        kg_df.columns = [c.split(':')[0] for c in kg_df.columns]
        
        return kg_df

    def load_inter(self):
        """
        Loads User-Item interactions from the KG dataset.
        Returns:
            pd.DataFrame: DataFrame with [user_id, item_id, rating, etc.]
        """
        if not os.path.exists(self.inter_file):
            print(f"Warning: Inter file not found at {self.inter_file}")
            return None
            
        try:
            inter_df = pd.read_csv(self.inter_file, sep='\t')
        except:
            inter_df = pd.read_csv(self.inter_file)

        # Normalize columns
        inter_df.columns = [c.split(':')[0] for c in inter_df.columns]
        
        return inter_df

    def get_entity_mappings(self, kg_df):
        """
        Creates mappings for entities and relations.
        """
        if kg_df is None:
            return {}, {}
            
        # Collect all unique entities (heads and tails)
        entities = set(kg_df['head_id'].unique()) | set(kg_df['tail_id'].unique())
        relations = set(kg_df['relation_id'].unique())
        
        entity2id = {e: i for i, e in enumerate(entities)}
        relation2id = {r: i for i, r in enumerate(relations)}
        
        return entity2id, relation2id
