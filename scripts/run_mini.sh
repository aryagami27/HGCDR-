#!/bin/bash
# Full System Execution Script
if [ ! -f ./Datasets_mini/user_mapping.csv ]; then
    echo "User Mapping missing in target. Creating them..."
    uv run data/create_user_mapping.py --data_dir ./Datasets_mini
fi

if [ ! -f ./Datasets_mini/lasso_augmented_data.csv ]; then
    echo "Lasso data missing in target. Creating them..."
    uv run generate_lasso_data.py --data_dir ./Datasets_mini
fi

if [ ! -f ./Datasets_mini/amazon_item_to_kg.csv ]; then
    echo "KG data missing in target. Creating them..."
    uv run scripts/align_kg.py --data_dir ./Datasets_mini
fi

echo "Starting Training..."
uv run main.py --data_dir ./Datasets_mini --lasso_path ./Datasets_mini/lasso_augmented_data.csv

echo "Running Inference Demo..."
uv run inference.py