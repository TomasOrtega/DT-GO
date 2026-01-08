#!/bin/bash

# Navigate to experiments folder
cd experiments || exit

echo "=================================================="
echo "Running Pipelined DT-GO (Evolving Topology)"
echo "=================================================="

# Define number of parallel experiments
NUM_EXPERIMENTS=100
START_SEED=1

# Calculate the end seed based on number of experiments
END_SEED=$((START_SEED + NUM_EXPERIMENTS - 1))

# Create logs folder
mkdir -p logs

echo "Starting $NUM_EXPERIMENTS parallel runs (Seeds $START_SEED to $END_SEED)..."

# Use seq to generate the range of seeds
for SEED in $(seq $START_SEED $END_SEED)
do
    echo "Launching Experiment with Seed $SEED..."
    
    # Run in background
    python experiment_pipelined.py \
        --n_agents 100 \
        --learning_rate 2 \
        --seed "$SEED" \
        --results_folder "results" > "logs/pipe_seed_${SEED}.log" 2>&1 &
        
    # Optional: small sleep to avoid OS process spawn race conditions
    sleep 0.1
done

echo "All processes launched. Waiting for completion..."
wait

echo "All experiments finished. Generating Plots..."

# Run the updated plotting script
python plot_pipelined.py

echo "Done."
echo "Cost Plot: experiments/figures/pipelined_cost.pdf"
echo "Consensus Plot: experiments/figures/pipelined_consensus.pdf"