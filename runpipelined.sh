#!/bin/bash

# Navigate to experiments folder
cd experiments || exit

echo "=================================================="
echo "Running Pipelined DT-GO (Evolving Topology)"
echo "=================================================="

# Run the experiment
python experiment_pipelined.py \
    --n_agents 100 \
    --learning_rate 2 \
    --n_experiments 1 \
    --results_folder "results"

echo "Experiment finished. Generating Plots..."

# Run the updated plotting script
python plot_pipelined.py

echo "Done."
echo "Cost Plot: experiments/figures/pipelined_cost.pdf"
echo "Consensus Plot: experiments/figures/pipelined_consensus.pdf"