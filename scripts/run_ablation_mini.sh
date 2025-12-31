#!/bin/bash
set -e

DATA_DIR="Datasets_mini_v2(10000)"
LASSO_PATH="Datasets/lasso_augmented_data.csv"
RESULTS_DIR="ablation_results_mini_v2(10000)"

# HGCDR++ Ablation Study Script (Interactive)

# Setup Data
if [ ! -f ./$DATA_DIR/user_mapping.csv ]; then
    echo "User Mapping missing in target. Creating them..."
    uv run data/create_user_mapping.py --data_dir ./$DATA_DIR
fi

# if [ ! -f ./Datasets_mini/lasso_augmented_data.csv ]; then
#     echo "Lasso data missing in target. Creating them..."
#     uv run generate_lasso_data.py --data_dir ./Datasets_mini
# fi

if [ ! -f ./$DATA_DIR/amazon_item_to_kg.csv ]; then
    echo "KG data missing in target. Creating them..."
    uv run scripts/align_kg.py --data_dir ./$DATA_DIR
fi


mkdir -p $RESULTS_DIR

COMMON_ARGS="--data_dir $DATA_DIR --lasso_path $LASSO_PATH --seed 42"

run_experiment() {
    ID=$1
    echo "----------------------------------------------------------------"
    case $ID in
        1)
            echo "[1/38] Running Full Proposed Model (Ours)..."
            uv run main.py $COMMON_ARGS > "$RESULTS_DIR/log_full_model.txt" 2>&1
            echo "Full Model Complete. Log: $RESULTS_DIR/log_full_model.txt"
            ;;
        2)
            echo "[2/38] Ablation: Removing Knowledge Graph..."
            uv run main.py $COMMON_ARGS --disable_kg > "$RESULTS_DIR/log_no_kg.txt" 2>&1
            echo "No KG Complete. Log: $RESULTS_DIR/log_no_kg.txt"
            ;;
        3)
            echo "[3/38] Ablation: Removing Causal Debiasing..."
            uv run main.py $COMMON_ARGS --disable_causal > "$RESULTS_DIR/log_no_causal.txt" 2>&1
            echo "No Causal Complete. Log: $RESULTS_DIR/log_no_causal.txt"
            ;;
        4)
            echo "[4/38] Ablation: Removing Meta-Learning..."
            uv run main.py $COMMON_ARGS --disable_meta > "$RESULTS_DIR/log_no_meta.txt" 2>&1
            echo "No Meta Complete. Log: $RESULTS_DIR/log_no_meta.txt"
            ;;
        5)
            echo "[5/38] Ablation: Removing Lasso Data..."
            uv run main.py $COMMON_ARGS --disable_lasso > "$RESULTS_DIR/log_no_lasso.txt" 2>&1
            echo "No Lasso Complete. Log: $RESULTS_DIR/log_no_lasso.txt"
            ;;
        6)
            echo "[6/38] Appendix: Scaled Architecture (3 Layers, 8 Heads)..."
            uv run main.py $COMMON_ARGS --gnn_layers 3 --gnn_heads 8 > "$RESULTS_DIR/log_scaled_appendix.txt" 2>&1
            echo "Scaled Run Complete. Log: $RESULTS_DIR/log_scaled_appendix.txt"
            ;;
        7)
            echo "[7/38] Ablation: KG Frozen (Structural Benefit Only)..."
            uv run main.py $COMMON_ARGS --freeze_kg_embeddings > "$RESULTS_DIR/log_kg_frozen.txt" 2>&1
            echo "KG Frozen Complete. Log: $RESULTS_DIR/log_kg_frozen.txt"
            ;;
        8)
            echo "[8/38] Ablation: Causal Loss without IPW..."
            uv run main.py $COMMON_ARGS --disable_ipw > "$RESULTS_DIR/log_causal_no_ipw.txt" 2>&1
            echo "Causal No IPW Complete. Log: $RESULTS_DIR/log_causal_no_ipw.txt"
            ;;
        9)
            echo "[9/38] Evaluation: Cold-Start Focus..."
            uv run main.py $COMMON_ARGS --eval_cold_only > "$RESULTS_DIR/log_cold_start.txt" 2>&1
            echo "Cold Start Eval Complete. Log: $RESULTS_DIR/log_cold_start.txt"
            ;;
        10)
            echo "[10/38] Benchmark: Latency & Throughput..."
            uv run main.py $COMMON_ARGS --benchmark_inference > "$RESULTS_DIR/log_benchmark.txt" 2>&1
            echo "Benchmark Complete. Log: $RESULTS_DIR/log_benchmark.txt"
            ;;
        11)
            echo "[11/38] Ablation: KG Alignment Noise (30% Shuffle)..."
            uv run main.py $COMMON_ARGS --kg_alignment_noise 0.3 > "$RESULTS_DIR/log_kg_noise_0.3.txt" 2>&1
            echo "KG Noise 0.3 Complete. Log: $RESULTS_DIR/log_kg_noise_0.3.txt"
            ;;
        12)
            echo "[12/38] Ablation: KG Alignment Drop (50% Drop)..."
            uv run main.py $COMMON_ARGS --kg_alignment_drop 0.5 > "$RESULTS_DIR/log_kg_drop_0.5.txt" 2>&1
            echo "KG Drop 0.5 Complete. Log: $RESULTS_DIR/log_kg_drop_0.5.txt"
            ;;
        13)
            echo "[13/38] Ablation: KG-Only (No Text Encoder)..."
            uv run main.py $COMMON_ARGS --disable_text_encoder > "$RESULTS_DIR/log_kg_only_no_text.txt" 2>&1
            echo "KG-Only Complete. Log: $RESULTS_DIR/log_kg_only_no_text.txt"
            ;;
        14)
            echo "[14/38] Ablation: Deep HGT (4 Layers, 8 Heads)..."
            uv run main.py $COMMON_ARGS --gnn_layers 4 --gnn_heads 8 > "$RESULTS_DIR/log_deep_gnn.txt" 2>&1
            echo "Deep GNN Complete. Log: $RESULTS_DIR/log_deep_gnn.txt"
            ;;
        15)
            echo "[15/38] Ablation: Text-Only Transfer (Baseline)..."
            uv run main.py $COMMON_ARGS --disable_kg --disable_meta --disable_causal --disable_disentangle --disable_contrast > "$RESULTS_DIR/log_text_only.txt" 2>&1
            echo "Text-Only Complete. Log: $RESULTS_DIR/log_text_only.txt"
            ;;
        16)
            echo "[16/38] Ablation: KG Randomization (Semantic Sanity)..."
            uv run main.py $COMMON_ARGS --kg_randomize_edges > "$RESULTS_DIR/log_kg_random.txt" 2>&1
            echo "KG Random Complete. Log: $RESULTS_DIR/log_kg_random.txt"
            ;;
        17)
            echo "[17/38] Ablation: Overlap 10%..."
            uv run main.py $COMMON_ARGS --overlap_ratio 0.1 > "$RESULTS_DIR/log_overlap_10.txt" 2>&1
            echo "Overlap 10% Complete. Log: $RESULTS_DIR/log_overlap_10.txt"
            ;;
        18)
            echo "[18/38] Ablation: Overlap 30%..."
            uv run main.py $COMMON_ARGS --overlap_ratio 0.3 > "$RESULTS_DIR/log_overlap_30.txt" 2>&1
            echo "Overlap 30% Complete. Log: $RESULTS_DIR/log_overlap_30.txt"
            ;;
        19)
            echo "[19/38] Ablation: Overlap 50%..."
            uv run main.py $COMMON_ARGS --overlap_ratio 0.5 > "$RESULTS_DIR/log_overlap_50.txt" 2>&1
            echo "Overlap 50% Complete. Log: $RESULTS_DIR/log_overlap_50.txt"
            ;;
        20)
            echo "[20/38] Ablation: HGT Grid (2L, 8H)..."
            uv run main.py $COMMON_ARGS --gnn_layers 2 --gnn_heads 8 > "$RESULTS_DIR/log_grid_2L_8H.txt" 2>&1
            echo "Grid (2L, 8H) Complete. Log: $RESULTS_DIR/log_grid_2L_8H.txt"
            ;;
        21)
            echo "[21/38] Ablation: HGT Grid (3L, 4H)..."
            uv run main.py $COMMON_ARGS --gnn_layers 3 --gnn_heads 4 > "$RESULTS_DIR/log_grid_3L_4H.txt" 2>&1
            echo "Grid (3L, 4H) Complete. Log: $RESULTS_DIR/log_grid_3L_4H.txt"
            ;;
        22)
            echo "[22/38] Ablation: Lambda Causal 0.0..."
            uv run main.py $COMMON_ARGS --lambda_causal 0.0 > "$RESULTS_DIR/log_lambda_0.0.txt" 2>&1
            echo "Lambda 0.0 Complete. Log: $RESULTS_DIR/log_lambda_0.0.txt"
            ;;
        23)
            echo "[23/38] Ablation: Lambda Causal 0.1..."
            uv run main.py $COMMON_ARGS --lambda_causal 0.1 > "$RESULTS_DIR/log_lambda_0.1.txt" 2>&1
            echo "Lambda 0.1 Complete. Log: $RESULTS_DIR/log_lambda_0.1.txt"
            ;;
        24)
            echo "[24/38] Ablation: Lambda Causal 1.0..."
            uv run main.py $COMMON_ARGS --lambda_causal 1.0 > "$RESULTS_DIR/log_lambda_1.0.txt" 2>&1
            echo "Lambda 1.0 Complete. Log: $RESULTS_DIR/log_lambda_1.0.txt"
            ;;
        25)
            echo "[25/38] Ablation: Exposure Randomization..."
            uv run main.py $COMMON_ARGS --randomize_propensity > "$RESULTS_DIR/log_random_prop.txt" 2>&1
            echo "Random Propensity Complete. Log: $RESULTS_DIR/log_random_prop.txt"
            ;;
        26)
            echo "[26/38] Ablation: Disable Curriculum..."
            uv run main.py $COMMON_ARGS --disable_curriculum > "$RESULTS_DIR/log_no_curriculum.txt" 2>&1
            echo "No Curriculum Complete. Log: $RESULTS_DIR/log_no_curriculum.txt"
            ;;
        27)
            echo "[27/38] Reliability: Reproducibility Check (Repeat Full Model)..."
            uv run main.py $COMMON_ARGS > "$RESULTS_DIR/log_reproducibility.txt" 2>&1
            echo "Reproducibility Run Complete. Log: $RESULTS_DIR/log_reproducibility.txt"
            ;;
        28)
            echo "[28/38] Ablation: No Disentanglement..."
            uv run main.py $COMMON_ARGS --disable_disentangle > "$RESULTS_DIR/log_no_disentangle.txt" 2>&1
            echo "No Disentanglement Complete. Log: $RESULTS_DIR/log_no_disentangle.txt"
            ;;
        29)
            echo "[29/38] Ablation: Lambda Orth 0.0..."
            uv run main.py $COMMON_ARGS --lambda_orth 0.0 > "$RESULTS_DIR/log_orth_0.0.txt" 2>&1
            echo "Orth 0.0 Complete. Log: $RESULTS_DIR/log_orth_0.0.txt"
            ;;
        30)
            echo "[30/38] Ablation: Lambda Orth 0.001..."
            uv run main.py $COMMON_ARGS --lambda_orth 0.001 > "$RESULTS_DIR/log_orth_0.001.txt" 2>&1
            echo "Orth 0.001 Complete. Log: $RESULTS_DIR/log_orth_0.001.txt"
            ;;
        31)
            echo "[31/38] Ablation: Lambda Orth 0.1..."
            uv run main.py $COMMON_ARGS --lambda_orth 0.1 > "$RESULTS_DIR/log_orth_0.1.txt" 2>&1
            echo "Orth 0.1 Complete. Log: $RESULTS_DIR/log_orth_0.1.txt"
            ;;
        32)
            echo "[32/38] Ablation: Text+KG w/o Graph Propagation..."
            uv run main.py $COMMON_ARGS --disable_hgt > "$RESULTS_DIR/log_no_graph.txt" 2>&1
            echo "No Graph Complete. Log: $RESULTS_DIR/log_no_graph.txt"
            ;;
        33)
            echo "[33/38] Ablation: Meta Inner Steps 1..."
            uv run main.py $COMMON_ARGS --meta_inner_steps 1 > "$RESULTS_DIR/log_meta_step_1.txt" 2>&1
            echo "Meta Step 1 Complete. Log: $RESULTS_DIR/log_meta_step_1.txt"
            ;;
        34)
            echo "[34/38] Ablation: Meta Inner Steps 3..."
            uv run main.py $COMMON_ARGS --meta_inner_steps 3 > "$RESULTS_DIR/log_meta_step_3.txt" 2>&1
            echo "Meta Step 3 Complete. Log: $RESULTS_DIR/log_meta_step_3.txt"
            ;;
        35)
            echo "[35/38] Ablation: Item Cold-Start Evaluation..."
            uv run main.py $COMMON_ARGS --eval_item_cold_only > "$RESULTS_DIR/log_item_cold.txt" 2>&1
            echo "Item Cold-Start Eval Complete. Log: $RESULTS_DIR/log_item_cold.txt"
            ;;
        36)
            echo "[36/38] Ablation: KG Relation Subset (High Freq)..."
            uv run main.py $COMMON_ARGS --kg_relation_subset high > "$RESULTS_DIR/log_kg_rel_high.txt" 2>&1
            echo "KG Rel High Complete. Log: $RESULTS_DIR/log_kg_rel_high.txt"
            ;;
        37)
            echo "[37/38] Ablation: KG Relation Subset (Low Freq)..."
            uv run main.py $COMMON_ARGS --kg_relation_subset low > "$RESULTS_DIR/log_kg_rel_low.txt" 2>&1
            echo "KG Rel Low Complete. Log: $RESULTS_DIR/log_kg_rel_low.txt"
            ;;
        38)
            echo "[38/38] Ablation: Reverse Transfer..."
            uv run main.py $COMMON_ARGS --reverse_transfer > "$RESULTS_DIR/log_reverse_transfer.txt" 2>&1
            echo "Reverse Transfer Complete. Log: $RESULTS_DIR/log_reverse_transfer.txt"
            ;;
        *)
            echo "Invalid Case ID: $1"
            ;;
    esac
}

echo "=========================================="
echo "Available Experiments:"
echo " 1. Full Proposed Model"
echo " 2. No KG"
echo " 3. No Causal Debiasing"
echo " 4. No Meta-Learning"
echo " 5. No Lasso Data"
echo " 6. Scaled Architecture (3L, 8H)"
echo " 7. KG Frozen"
echo " 8. Causal No IPW"
echo " 9. Cold-Start Eval"
echo "10. Benchmark Latency"
echo "11. KG Noise 30%"
echo "12. KG Drop 50%"
echo "13. KG-Only (No Text)"
echo "14. Deep HGT"
echo "15. Text-Only (No Graph)"
echo "16. KG Random"
echo "17-19. Overlap (10%, 30%, 50%)"
echo "20-21. Grid Search HGT"
echo "22-24. Lambda Causal Grid"
echo "25. Random Propensity"
echo "26. No Curriculum"
echo "27. Reproducibility Check"
echo "28. No Disentanglement"
echo "29-31. Lambda Orth Grid"
echo "32. No Graph Propagation"
echo "33-34. Meta Inner Steps"
echo "35. Item Cold-Start"
echo "36-37. KG Relation Subset"
echo "38. Reverse Transfer"
echo "=========================================="

read -p "Enter Case ID to run (0 for all): " CASE_ID

if [ "$CASE_ID" -eq 0 ]; then
    echo "Running ALL 38 Experiments..."
    for i in {1..38}; do
        run_experiment $i
    done
else
    # Allow comma separated? The user said "ask which case... if 0 run all". 
    # I'll support single ID or 0.
    run_experiment $CASE_ID
fi

echo "=========================================="
echo "Ablation Study Finished!"
echo "Check $RESULTS_DIR for logs."
