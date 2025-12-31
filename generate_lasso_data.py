import pandas as pd
import json
import os
import argparse
from tqdm import tqdm
import random

# Set Cache Dirs (Must be before imports that use them if possible, though mlx_lm uses hf_hub_download)
# User requested HF_HOME to be inside "Latest Implementation". 
# Assuming script is run from that directory:
os.environ['HF_HOME'] = os.path.abspath('./hf_cache')

try:
    from mlx_lm import load, generate
except ImportError:
    print("Error: mlx_lm not installed. Please run: pip install mlx-lm")
    exit(1)

# MLX CONFIG
# Using a 4-bit quantized Llama 3.2 3B Instruct model for speed/quality balance on Mac
MODEL_PATH = "mlx-community/Llama-3.2-3B-Instruct-4bit"

def generate_with_mlx(model, tokenizer, prompt):
    """
    Generates content using MLX.
    """
    # Create a chat-like prompt format if using Instruct model
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    response = generate(
        model, 
        tokenizer, 
        prompt=prompt, 
        max_tokens=512, 
        verbose=False
    )
    return response

def generate_synthetic_data(data_dir):
    print(f"Loading MLX model: {MODEL_PATH}...")
    print(f"Using HF_HOME: {os.environ['HF_HOME']}")
    
    model, tokenizer = load(MODEL_PATH)

    source_path = os.path.join(data_dir, 'douban_dataset(text information)', 'bookreviews_cleaned.txt')
    yelp_path = os.path.join(data_dir, 'yelp_sample_100k.csv') if os.path.exists(os.path.join(data_dir, 'yelp_sample_100k.csv')) else os.path.join(data_dir, 'yelp_academic_dataset_review.csv')
    amazon_path = os.path.join(data_dir, 'amazon_sample_100k.csv') if os.path.exists(os.path.join(data_dir, 'amazon_sample_100k.csv')) else os.path.join(data_dir, 'Amazon_dataset.csv')
    output_path = os.path.join(data_dir, 'lasso_augmented_data.csv')

    print(f"Loading Source from: {yelp_path}")
    if os.path.exists(yelp_path):
        source_df = pd.read_csv(yelp_path, usecols=['user_id', 'business_id', 'stars', 'text'])
        source_df.rename(columns={'business_id': 'item_id', 'stars': 'rating'}, inplace=True)
    else:
        print(f"Yelp data not found at {yelp_path}.")
        return

    print(f"Loading Target from: {amazon_path}")
    if os.path.exists(amazon_path):
        target_df = pd.read_csv(amazon_path, header=None, usecols=[0], names=['user_id'])
    else:
        # If headers exist or file missing
        try:
             target_df = pd.read_csv(amazon_path, usecols=['user_id'])
        except:
             print(f"Amazon data not found or invalid at {amazon_path}. Creating dummy target user set.")
             target_df = pd.DataFrame({'user_id': []})
    
    # Identify Source-Only Users
    source_users = set(source_df['user_id'].unique())
    target_users = set(target_df['user_id'].unique())
    overlap_users = source_users.intersection(target_users)
    source_only_users = list(source_users - overlap_users)
    
    print(f"Total Source Users: {len(source_users)}")
    print(f"Overlap Users: {len(overlap_users)}")
    print(f"Source-Only Users (Targets for Lasso): {len(source_only_users)}")
    
    # Process a larger subset
    batch_size = 2000
    users_to_process = source_only_users[:batch_size]
    print(f"Generating data for first {batch_size} users using MLX...")

    synthetic_data = []

    for user in tqdm(users_to_process):
        # Get user history
        history = source_df[source_df['user_id'] == user]
        if len(history) == 0:
            continue
            
        reviews = history['text'].sample(n=min(10, len(history))).tolist()
        
        # Construct Prompt
        prompt = f"""
        You are a simulator for a user who reviews businesses on Yelp. 
        Simulate their behavior on Amazon (Products) based on these reviews:
        
        {json.dumps(reviews)}
        
        Generate 10 synthetic Amazon reviews.
        Return ONLY a complete JSON list (no markdown, no extra text) of objects with keys: "item_id" (e.g. B00...), "rating" (1-5), "text".
        Example: [{{"item_id": "B001", "rating": 5, "text": "Good"}}]
        """
        
        content = generate_with_mlx(model, tokenizer, prompt)
        
        if content:
             try:
                # heuristic cleanup
                start_idx = content.find('[')
                end_idx = content.rfind(']')
                if start_idx != -1 and end_idx != -1:
                    json_str = content[start_idx:end_idx+1]
                    generated_items = json.loads(json_str)
                    
                    for item in generated_items:
                        synthetic_data.append({
                            'user_id': user, 
                            'item_id': item.get('item_id', f'SYNTH_{random.randint(1000,9999)}'),
                            'rating': item.get('rating', 3),
                            'text': item.get('text', '')
                        })
             except:
                 continue
            
    # Save Augmented Data
    if synthetic_data:
        aug_df = pd.DataFrame(synthetic_data)
        print(f"Generated {len(aug_df)} interactions.")
        aug_df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")
    else:
        print("No data generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./Datasets', help='Path to datasets')
    args = parser.parse_args()
    
    generate_synthetic_data(args.data_dir)
