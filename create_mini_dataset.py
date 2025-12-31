
import pandas as pd
import os
import shutil
import csv

SOURCE_DIR = './Datasets'
TARGET_DIR = './Datasets_mini_v8(10000000)'
SAMPLE_SIZE = 10000000

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def sample_csv(source_path, target_path, sample_size, **kwargs):
    if not os.path.exists(source_path):
        print(f"Skipping {source_path}, not found.")
        return
    
    print(f"Sampling {source_path} to {target_path}...")
    df = pd.read_csv(source_path, **kwargs)
    
    # Random sample
    if len(df) > sample_size:
        df_sampled = df.sample(n=sample_size, random_state=42)
    else:
        df_sampled = df
        
    # Write back
    # If header was None in read_csv, we should probably write without header
    header = kwargs.get('header', 'infer')
    write_header = True
    if header is None:
        write_header = False
        
    df_sampled.to_csv(target_path, index=False, header=write_header)
    print(f"  Saved {len(df_sampled)} rows.")

def main():
    if os.path.exists(TARGET_DIR):
        print(f"Removing existing {TARGET_DIR}...")
        shutil.rmtree(TARGET_DIR)
    
    create_dir(TARGET_DIR)
    
    # 1. Yelp (Source)
    # Raw file path: /Volumes/PORTABLESSD/Code/Research/Datasets/yelp_academic_dataset_review.csv
    # main.py expects: user_id, business_id, stars, text
    raw_yelp_path = '/Volumes/PORTABLESSD/Code/Research/Datasets/yelp_academic_dataset_review.csv'
    target_yelp_path = os.path.join(TARGET_DIR, 'yelp_sample_100k.csv')
    
    if os.path.exists(raw_yelp_path):
        print(f"Sampling raw Yelp from {raw_yelp_path}...")
        # Read only necessary columns to save memory/time if file is huge
        # Also sample randomly. For a huge file, 'sample' might be slow if we read all.
        # Let's read a chunk.
        try:
            # chunksize=100000 to avoid loading everything
            # We just want a mini dataset of 1000.
            # Reading first chunk is enough for a 'mini' dataset, but random sample is better.
            # We'll read 2*SAMPLE_SIZE rows to get some randomness or just head if strictness not needed.
            df = pd.read_csv(raw_yelp_path, nrows=SAMPLE_SIZE*10, usecols=['user_id', 'business_id', 'stars', 'text'])
            df_sampled = df.sample(n=SAMPLE_SIZE, random_state=42)
            df_sampled.to_csv(target_yelp_path, index=False)
            print(f"  Saved {len(df_sampled)} rows to {target_yelp_path}")
        except Exception as e:
            print(f"Failed to sample raw Yelp: {e}")
    else:
        print(f"Raw Yelp file not found at {raw_yelp_path}")

    
    # 2. Amazon (Target)
    # main.py: target_df = pd.read_csv('./Datasets/amazon_sample_100k.csv', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
    # format: no header
    sample_csv(
        os.path.join(SOURCE_DIR, 'Amazon_dataset.csv'),
        os.path.join(TARGET_DIR, 'amazon_sample_100k.csv'),
        SAMPLE_SIZE,
        header=None
    )
    
    
    # 4. Douban
    # main.py: uses DoubanLoader which looks for 'douban_dataset(text information)/bookreviews_cleaned.txt'
    # format: tab separated, quoting=csv.QUOTE_ALL
    douban_src_dir = os.path.join(SOURCE_DIR, 'douban_dataset(text information)')
    douban_tgt_dir = os.path.join(TARGET_DIR, 'douban_dataset(text information)')
    create_dir(douban_tgt_dir)
    
    book_reviews = os.path.join(douban_src_dir, 'bookreviews_cleaned.txt')
    if os.path.exists(book_reviews):
        print(f"Sampling Douban {book_reviews}...")
        try:
            df = pd.read_csv(book_reviews, sep='\t', quoting=csv.QUOTE_ALL, on_bad_lines='skip', encoding='utf-8')
            if len(df) > SAMPLE_SIZE:
                df_sampled = df.sample(n=SAMPLE_SIZE, random_state=42)
            else:
                df_sampled = df
            
            # Write back using same format
            target_path = os.path.join(douban_tgt_dir, 'bookreviews_cleaned.txt')
            df_sampled.to_csv(target_path, sep='\t', quoting=csv.QUOTE_ALL, index=False)
            print(f"  Saved {len(df_sampled)} rows.")
        except Exception as e:
            print(f"Failed to process Douban: {e}")
            
    # 5. Knowledge Graph
    # Copy the folders
    kg_src = os.path.join(SOURCE_DIR, 'Amazon-KG-v2.0-dataset-main')
    kg_tgt = os.path.join(TARGET_DIR, 'Amazon-KG-v2.0-dataset-main')
    
    if os.path.exists(kg_src):
        print(f"Copying KG from {kg_src} to {kg_tgt}...")
        shutil.copytree(kg_src, kg_tgt)
        print("  KG Copied.")
    else:
        print("KG directory not found.")

    print("Mini dataset creation complete!")
    print(f"Created in {TARGET_DIR}")

if __name__ == "__main__":
    main()
