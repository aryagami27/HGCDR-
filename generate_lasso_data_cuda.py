# If using uv (recommended based on your logs)
# uv pip install huggingface_hub
# uv pip install llama-cpp-python \
#   --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

import pandas as pd
import json
import os
import argparse
from tqdm import tqdm
import random
from huggingface_hub import hf_hub_download

# Set Cache Dirs
os.environ['HF_HOME'] = os.path.abspath('./hf_cache')

try:
    from llama_cpp import Llama
except ImportError:
    print("Error: llama_cpp not installed.")
    print("Please run: pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124")
    exit(1)

# CUDA / GGUF CONFIGURATION
# Using Llama 3.2 3B Instruct (Q8_0 for max quality, or Q4_K_M for speed)
# This matches the model size you were using in MLX but for CUDA.
REPO_ID = "bartowski/Llama-3.2-3B-Instruct-GGUF"
FILENAME = "Llama-3.2-3B-Instruct-Q8_0.gguf" 

def get_model_path():
    """Downloads model if not present and returns path."""
    print(f"Checking for model {FILENAME} in {REPO_ID}...")
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    return model_path

def generate_with_cuda(llm, prompt):
    """
    Generates content using Llama CPP with CUDA.
    """
    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful data generation assistant. Output only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
            temperature=0.7,
            # Enforce JSON output structure
            response_format={
                "type": "json_object"
            }
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        # Fallback for older library versions
        print(f"Generation Warning: {e}")
        return None

def generate_synthetic_data(data_dir):
    model_path = get_model_path()
    
    print(f"Loading GGUF model from: {model_path}")
    print("Initializing Llama with CUDA support (n_gpu_layers=-1)...")
    
    # Initialize Model with CUDA Offloading
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,      # -1 = Offload ALL layers to GPU
        n_ctx=8192,           # Context window (Llama 3 supports up to 128k, but 8k is safe/fast)
        n_batch=512,          # Batch processing size
        verbose=False         # Set True to debug CUDA loading
    )

    # ---------------------------------------------------------
    # DATA LOADING LOGIC (Preserved from original script)
    # ---------------------------------------------------------
    source_path = os.path.join(data_dir, 'douban_dataset(text information)', 'bookreviews_cleaned.txt')
    
    # Handle filename variations based on your provided script
    p1 = os.path.join(data_dir, 'yelp_sample_100k.csv')
    p2 = os.path.join(data_dir, 'yelp_academic_dataset_review.csv')
    yelp_path = p1 if os.path.exists(p1) else p2

    a1 = os.path.join(data_dir, 'amazon_sample_100k.csv')
    a2 = os.path.join(data_dir, 'Amazon_dataset.csv')
    amazon_path = a1 if os.path.exists(a1) else a2
    
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
        try:
            target_df = pd.read_csv(amazon_path, header=None, usecols=[0], names=['user_id'])
        except:
             target_df = pd.read_csv(amazon_path, usecols=['user_id'])
    else:
         print(f"Amazon data not found. Creating dummy target user set.")
         target_df = pd.DataFrame({'user_id': []})
    
    # Identify Source-Only Users
    source_users = set(source_df['user_id'].unique())
    target_users = set(target_df['user_id'].unique())
    overlap_users = source_users.intersection(target_users)
    source_only_users = list(source_users - overlap_users)
    
    print(f"Total Source Users: {len(source_users)}")
    print(f"Overlap Users: {len(overlap_users)}")
    print(f"Source-Only Users (Targets for Lasso): {len(source_only_users)}")
    
    # Process batch
    batch_size = 2000
    users_to_process = source_only_users[:batch_size]
    print(f"Generating data for first {batch_size} users using CUDA...")

    synthetic_data = []

    for user in tqdm(users_to_process, desc="Generating"):
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
        Return ONLY a complete JSON list of objects with keys: "item_id" (e.g. B00...), "rating" (1-5), "text".
        Example: [{{"item_id": "B001", "rating": 5, "text": "Good"}}]
        """
        
        content = generate_with_cuda(llm, prompt)
        
        if content:
             try:
                # Heuristic cleanup (handle potential markdown/extra text)
                clean_content = content.replace("```json", "").replace("```", "").strip()
                
                start_idx = clean_content.find('[')
                end_idx = clean_content.rfind(']')
                
                if start_idx != -1 and end_idx != -1:
                    json_str = clean_content[start_idx:end_idx+1]
                    generated_items = json.loads(json_str)
                    
                    for item in generated_items:
                        synthetic_data.append({
                            'user_id': user, 
                            'item_id': item.get('item_id', f'SYNTH_{random.randint(1000,9999)}'),
                            'rating': item.get('rating', 3),
                            'text': item.get('text', '')
                        })
             except json.JSONDecodeError:
                 continue
             except Exception:
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