#!/bin/bash

# Navigate to experiments folder
cd experiments || exit

echo "=================================================="
echo "Running Pipelined DT-GO Experiment (Drifting Topology)"
echo "Comparing Standard vs Pipelined implementation"
echo "=================================================="

# Run the experiment
# We use slightly more rounds (1500) to clearly show the drift effect
# We run 20 experiments and average them to get smooth curves
python experiment_pipelined.py \
    --n_agents 100 \
    --n_rounds 1500 \
    --learning_rate 2.0 \
    --n_experiments 20 \
    --results_folder "results"

echo "Experiment finished. Generating Plot..."

# Plot the results
python plot_pipelined.py

echo "Done. Check 'experiments/figures/pipelined_comparison.pdf'"