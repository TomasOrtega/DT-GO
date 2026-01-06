#!/bin/bash

# Navigate to experiments folder
cd experiments || exit

# Create a directory to store the progress logs
mkdir -p logs

echo "Starting concurrent experiments..."
echo "Progress bars are being sent to the 'experiments/logs/' directory."

# 1. Run standalone experiments
python p_experiment.py > logs/p_experiment.log 2>&1 &
python lambda_experiment.py > logs/lambda_experiment.log 2>&1 &

# 2. Run Regression Loops

# Loop: Varying P (Fixed Lambda 0.3)
for p in 1.0 0.8 0.6 0.4 0.2; do 
    python experiment_regression.py --lam 0.3 --p "$p" --n_experiments 1000 > "logs/reg_lam0.3_p${p}.log" 2>&1 & 
done

# Loop: Varying Lambda (Fixed P 1.0)
# Removed 0.3 because it is already covered in the "Varying P" loop above
for lam in 0.1 0.2 0.4; do 
    python experiment_regression.py --lam "$lam" --p 1.0 --n_experiments 1000 > "logs/reg_p1.0_lam${lam}.log" 2>&1 & 
done

# Loop: Varying P (Fixed Lambda 0.0)
for p in 0.8 0.6 0.4 0.2; do 
    python experiment_regression.py --lam 0.0 --p "$p" --n_experiments 1000 > "logs/reg_lam0.0_p${p}.log" 2>&1 & 
done

# Loop: Time Varying Probability q
for q in 0.0 0.01 0.03; do 
    python experiment_regression.py --p 0.2 --lam 0.0 --time_varying --time_varying_prob "$q" --n_experiments 1 > "logs/reg_q${q}.log" 2>&1 & 
done

# Loop: Packet Error Probability (p_err)
for perr in 0.0 0.1 0.2; do 
    python experiment_regression.py --p 0.5 --time_varying --p_err "$perr" --n_experiments 1000 > "logs/reg_perr${perr}.log" 2>&1 & 
done

# 3. Wait for completion
echo "All processes started. You can watch a specific log using: tail -f logs/filename.log"
echo "Waiting for completion..."
wait

echo "All experiments finished. Generating plots..."

# 4. Generate Plots
python plot_p_experiment.py > logs/plot_p.log 2>&1
python plot_lambda_experiment.py > logs/plot_lambda.log 2>&1
python plot_experiments.py > logs/plot_experiments.log 2>&1
python plot_experiment_p_err.py > logs/plot_perr.log 2>&1

echo "Done! Check the 'experiments/results' folder for PDFs."