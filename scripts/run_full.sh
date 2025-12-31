#!/bin/bash
# Full System Execution Script
set -e

# 1. Generate User Mapping (Full Data)
if [ ! -f ./Datasets/user_mapping.csv ]; then
    echo "User Mapping missing in target. Creating them..."
    uv run data/create_user_mapping.py --data_dir ./Datasets
fi

# 2. Generate Lasso Data
if [ ! -f ./Datasets/lasso_augmented_data.csv ]; then
    echo "Lasso data missing in target. Creating them..."
    uv run generate_lasso_data.py --data_dir ./Datasets
fi

# 3. Align KG
if [ ! -f ./Datasets/amazon_item_to_kg.csv ]; then
    echo "KG data missing in target. Creating them..."
    uv run scripts/align_kg.py --data_dir ./Datasets
fi

# 4. Train
echo "Starting Training..."
uv run main.py --data_dir ./Datasets --lasso_path ./Datasets/lasso_augmented_data.csv

# 5. Inference Demo
echo "Running Inference Demo..."
uv run inference.py
