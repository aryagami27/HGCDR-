
import pandas as pd
import os
import argparse
import sys
import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np

def load_kg_genres(kg_path):
    print(f"Loading KG from {kg_path}...")
    # Load KG and extract Genre/Category nodes
    # We want nodes that are tails of 'dct:subject' or 'dbo:literaryGenre'
    df_kg = pd.read_csv(kg_path, sep='\t')
    
    # Relations of interest
    valid_relations = ['dct:subject', 'dbo:literaryGenre', 'dbo:genre']
    
    # Filter
    genre_edges = df_kg[df_kg['relation_id:token'].isin(valid_relations)].copy()
    
    # Extract unique tails (Genres/Categories)
    unique_genres = genre_edges['tail_id:token'].unique()
    
    # Create a cleaner lookup for semantic matching
    # e.g., "cat:Science_Fiction" -> "Science Fiction"
    genre_data = []
    for g in unique_genres:
        clean_name = g.replace('cat:', '').replace('res:', '').replace('_', ' ')
        genre_data.append({'kg_id': g, 'clean_name': clean_name})
        
    return pd.DataFrame(genre_data), genre_edges


def align_kg(data_dir, output_file, threshold=0.4):
    print(f"Starting Hybrid KG Alignment in {data_dir}")
    
    # Paths (Support Full or Sample)
    full_path = os.path.join(data_dir, "Amazon_dataset.csv")
    sample_path = os.path.join(data_dir, "amazon_sample_100k.csv")
    lasso_path = os.path.join(data_dir, "lasso_augmented_data.csv")
    
    # Search for KG dir - handles both Mini and Full structures
    # Usually: Datasets/Amazon-KG-v2.0-dataset-main/Amazon-KG-5core-Books
    kg_candidate_1 = os.path.join(data_dir, "Amazon-KG-v2.0-dataset-main", "Amazon-KG-5core-Books")
    # Sometimes it might be directly in data_dir if flattened
    kg_candidate_2 = os.path.join(data_dir, "Amazon-KG-5core-Books")
    
    if os.path.exists(kg_candidate_1):
        kg_root = kg_candidate_1
    elif os.path.exists(kg_candidate_2):
        kg_root = kg_candidate_2
    else:
        # Fallback for 'Datasets' root if not 'mini'
        # Maybe data_dir IS 'Datasets', but we need to check subfolders?
        kg_root = kg_candidate_1 # Default to expect standard structure
    
    print(f"Looking for KG in: {kg_root}")
    link_path = os.path.join(kg_root, "Amazon-KG-5core-Books.link")
    kg_path = os.path.join(kg_root, "Amazon-KG-5core-Books.kg")
    
    if not os.path.exists(link_path) or not os.path.exists(kg_path):
        print(f"Critical KG files missing at {kg_root}")
        return

    # 1. Strict Alignment (ID Matching)
    # Load Link File
    print("Loading Link File for Strict Alignment...")
    df_link = pd.read_csv(link_path, sep='\t')
    item_to_entity = dict(zip(df_link['item_id:token'], df_link['entity_id:token']))
    
    # Load Data Items (Full or Sample)
    dataset_items = set()
    
    # Check Full First
    if os.path.exists(full_path):
        print(f"Loading Full Amazon Dataset from: {full_path}")
        # Full usually has header 'user_id,item_id,...'
        try:
             df_full = pd.read_csv(full_path, dtype={'item_id': str})
             if 'item_id' in df_full.columns:
                 dataset_items.update(df_full['item_id'].unique())
             else:
                 # Try header=None
                 df_full = pd.read_csv(full_path, header=None, names=['user_id', 'item_id', 'rating', 'timestamp'], dtype={'item_id': str})
                 dataset_items.update(df_full['item_id'].unique())
        except Exception as e:
             print(f"Error loading full dataset: {e}")
             
    # Also check Sample (merge if both exist or just fallback)
    if os.path.exists(sample_path):
        print(f"Loading Sample Amazon Dataset from: {sample_path}")
        df_sample = pd.read_csv(sample_path, header=None, names=['user_id', 'item_id', 'rating', 'timestamp'], dtype={'item_id': str})
        dataset_items.update(df_sample['item_id'].unique())
        
    print(f"Loaded {len(dataset_items)} unique items from Datasets.")
    
    # Load KG Genres
    df_genres, df_genre_edges = load_kg_genres(kg_path)
    print(f"Loaded {len(df_genres)} Genre/Category nodes.")
    
    # Build Entity -> Genres Map (for expanding Strict mappings)
    # head -> list of tails
    entity_to_genres = df_genre_edges.groupby('head_id:token')['tail_id:token'].apply(list).to_dict()
    
    output_edges = []
    
    # Process Strict Alignment
    print("Processing Strict Alignment...")
    strict_matches = 0
    
    for item in dataset_items:
        if item in item_to_entity:
            entity = item_to_entity[item]
            # Link to Genres of that Entity (1-hop)
            if entity in entity_to_genres:
                genres = entity_to_genres[entity]
                for g in genres:
                    output_edges.append((item, "belongs_to", g))
                strict_matches += 1
            else:
                pass
                
    print(f"Strict Alignment Coverage: {strict_matches}/{len(dataset_items)} items.")
    
    # 2. Fuzzy Semantic Alignment (Lasso Items)
    if os.path.exists(lasso_path):
        print(f"Loading Lasso Data for Semantic Alignment from {lasso_path}...")
        try:
            # Try reading with header 'user_id,item_id,rating,text'
            with open(lasso_path, 'r') as f:
                first_line = f.readline()
            
            if 'text' in first_line:
                df_lasso = pd.read_csv(lasso_path, dtype={'item_id': str})
            else:
                df_lasso = pd.read_csv(lasso_path, header=None, names=['user_id', 'item_id', 'rating', 'text'], dtype={'item_id': str})
                
            items_for_fuzzy = []
            texts_for_fuzzy = []
            
            # Unique items df
            # Sort by text length to get good descriptions
            df_lasso['text_len'] = df_lasso['text'].astype(str).fillna('').apply(len)
            df_lasso = df_lasso.sort_values('text_len', ascending=False)
            df_lasso_unique = df_lasso.drop_duplicates(subset='item_id')
            
            for idx, row in df_lasso_unique.iterrows():
                item = row['item_id']
                text = str(row['text'])
                if item not in item_to_entity: # Only if NO strict map
                     items_for_fuzzy.append(item)
                     texts_for_fuzzy.append(text)
            
            print(f"Found {len(items_for_fuzzy)} items for Fuzzy Matching.")
            
            if items_for_fuzzy:
                print("Loading SentenceTransformer (all-MiniLM-L6-v2)...")
                model = SentenceTransformer('all-MiniLM-L6-v2')
                
                # Check device
                if torch.backends.mps.is_available():
                    model = model.to('mps')
                elif torch.cuda.is_available():
                    model = model.to('cuda')
                    
                print("Encoding Item Texts...")
                item_embeddings = model.encode(texts_for_fuzzy, show_progress_bar=True, convert_to_tensor=True)
                
                print("Encoding KG Genre Names...")
                genre_names = df_genres['clean_name'].tolist()
                genre_ids = df_genres['kg_id'].tolist()
                genre_embeddings = model.encode(genre_names, show_progress_bar=True, convert_to_tensor=True)
                
                print("Computing Similarity...")
                # Cosine Sim
                hits = util.semantic_search(item_embeddings, genre_embeddings, top_k=1)
                
                fuzzy_matches = 0
                for i, hit in enumerate(hits):
                    # hit is list of dicts {'corpus_id': ..., 'score': ...}
                    best_hit = hit[0]
                    score = best_hit['score']
                    
                    if score >= threshold:
                        item_id = items_for_fuzzy[i]
                        genre_idx = best_hit['corpus_id']
                        kg_node = genre_ids[genre_idx]
                        
                        output_edges.append((item_id, "belongs_to", kg_node))
                        fuzzy_matches += 1
                        
                print(f"Fuzzy Matches Found: {fuzzy_matches} (Threshold {threshold})")
            
        except Exception as e:
            print(f"Error processing Lasso data: {e}")
            import traceback
            traceback.print_exc()

    # Save
    if not output_edges:
        print("No alignment edges generated.")
        return

    out_df = pd.DataFrame(output_edges, columns=['item_id', 'relation', 'kg_node_id'])
    # Remove duplicates
    out_df = out_df.drop_duplicates()
    
    out_path = os.path.join(data_dir, output_file)
    out_df.to_csv(out_path, index=False)
    
    total_unique_items_mapped = out_df['item_id'].nunique()
    print("-" * 30)
    print(f"Total Aligned Items: {total_unique_items_mapped}")
    print(f"Total Edges: {len(out_df)}")
    print(f"Saved to {out_path}")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Align Amazon Items to Knowledge Graph')
    parser.add_argument('--data_dir', type=str, default='./Datasets_mini', help='Path to datasets directory')
    parser.add_argument('--threshold', type=float, default=0.6, help='Similarity threshold for fuzzy matching')
    
    args = parser.parse_args()
    
    align_kg(args.data_dir, "amazon_item_to_kg.csv", threshold=args.threshold)
