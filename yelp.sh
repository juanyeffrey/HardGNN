#!/bin/bash
set -e  # exit on any error

# Set GPU device
export CUDA_VISIBLE_DEVICES=0

# Activate venv if using one
# source /path/to/your/python37/venv/bin/activate

# Create log directory if it doesn't exist
mkdir -p logs

# Get current timestamp for log file
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="logs/yelp_${timestamp}.log"

echo "Starting experiment at $(date)" | tee -a "$log_file"
echo "Logging to: $log_file" | tee -a "$log_file"

# Run the Python script with arguments and redirect output to log
python3.7 main.py \
  --data yelp \
  --reg 1e-2 \
  --temp 0.1 \
  --ssl_reg 1e-7 \
  --save_path yelp12 \
  --epoch 150 \
  --batch 512 \
  --sslNum 40 \
  --graphNum 12 \
  --gnn_layer 3 \
  --att_layer 2 \
  --test True \
  --testSize 1000 \
  --ssldim 32 \
  --sampNum 40 \
  --use_hard_neg True \
  --hard_neg_top_k 5 \
  --contrastive_weight 0.1 2>&1 | tee -a "$log_file"

echo "Experiment completed at $(date)" | tee -a "$log_file"