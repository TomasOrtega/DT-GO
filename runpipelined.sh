#!/bin/bash

# ONLY WORKS WITH BASH SHELL

# Navigate to experiments folder
cd experiments || exit

echo "=================================================="
echo "Running Pipelined DT-GO (Evolving Topology)"
echo "=================================================="

# Define number of experiments
NUM_EXPERIMENTS=1000  # Increased example count
START_SEED=1
MAX_PARALLEL=100      # Limit concurrent jobs

# Calculate the end seed
END_SEED=$((START_SEED + NUM_EXPERIMENTS - 1))

# Create logs folder
mkdir -p logs

echo "Starting $NUM_EXPERIMENTS runs (Seeds $START_SEED to $END_SEED)"
echo "Concurrency limited to $MAX_PARALLEL processes."

for SEED in $(seq $START_SEED $END_SEED)
do
    # 1. Check active background jobs. 
    # If >= MAX_PARALLEL, wait a bit and check again.
    while [ "$(jobs -r | wc -l)" -ge "$MAX_PARALLEL" ]; do
        sleep 1
    done

    echo "Launching Experiment with Seed $SEED..."
    
    # 2. Run in background
    python experiment_pipelined.py \
        --n_agents 100 \
        --learning_rate 2 \
        --seed "$SEED" \
        --results_folder "results" > "logs/pipe_seed_${SEED}.log" 2>&1 &
        
done

echo "All jobs scheduled. Waiting for remaining processes to finish..."
wait

echo "All experiments finished. Generating Plots..."

# Run the plotting script
python plot_pipelined.py

echo "Done."
echo "Cost Plot: experiments/figures/pipelined_cost.pdf"
echo "Consensus Plot: experiments/figures/pipelined_consensus.pdf"