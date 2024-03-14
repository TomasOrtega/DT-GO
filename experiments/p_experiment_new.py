import numpy as np
from experiment_regression import Experiment
from log_reg_utils import get_args_as_obj
import scienceplots
import matplotlib
from matplotlib import pyplot as plt
import os
import sys


RESULTS_FOLDER = "results/p_experiments"


def get_results(args, experiment):
    runname = experiment.runname
    folder = f"{RESULTS_FOLDER}/{runname}"
    try:
        costs = np.load(f"{folder}/costs.npy")
        print(f"Loaded {folder}/costs.npy")
        params = np.load(f"{folder}/params.npy")
        print(f"Loaded {folder}/params.npy")
        to_means = np.load(f"{folder}/to_means.npy")
        print(f"Loaded {folder}/to_means.npy")
    except:
        experiment.run_experiment()
        costs = np.load(f"{folder}/costs.npy")
        print(f"Loaded {folder}/costs.npy")
        params = np.load(f"{folder}/params.npy")
        print(f"Loaded {folder}/params.npy")
        to_means = np.load(f"{folder}/to_means.npy")
    return costs, params, to_means


args = {
    "n_agents": 100,
    "n_rounds": 1000,
    "learning_rate": 2,
    "baseline_loss": 0.014484174216922262,
    "seed": 0,
    "results_folder": RESULTS_FOLDER,
    "verbose": False,
    "test_run": False,
    "warm_up_rounds": 1024,
    "lambdas": None,
    "experiment_type": "p",
    "n_experiments": 1000,
    "p_values": [1.0, 0.8, 0.6, 0.4, 0.2],
    "l2_strength": None,
    "n_gossip": 1
}

args = get_args_as_obj(args)

experiment = Experiment(args)

# get losses from folder if it exists
costs, params, to_means = get_results(args, experiment)

# plot
plt.style.use(['ieee', 'high-vis'])  # try bright instead of high-vis maybe
# Avoid Type 3 fonts for IEEE publications (switch to True)
matplotlib.rcParams['text.usetex'] = True

# Display results for cost suboptimality
plt.figure()

baseline_cost = args.baseline_loss
if args.experiment_type == "lambda":
    # Set the baseline cost as the cost for lambda = 0
    for param, cost in zip(params, costs):
        if param == 0.0:
            baseline_cost = cost
            break
elif args.experiment_type == "p":
    # Set the baseline cost as the cost for p = 1
    for param, cost in zip(reversed(params), reversed(costs)):
        if param == 1.0:
            baseline_cost = cost
            break

for param, cost, to_mean in zip(params, costs, to_means):
    if args.experiment_type == "lambda":
        if param == 0.0:
            continue
        plt.plot(
            np.arange(1, args.n_rounds + 1), cost - baseline_cost, label=f"$\lambda={param:.1f}$"
        )
    elif args.experiment_type == "p":
        if param == 1.0:
            continue
        plt.plot(
            np.arange(1, args.n_rounds + 1), cost - baseline_cost, label=f"$p={param:.1f}$"
        )
    else:
        plt.plot(
            np.arange(1, args.n_rounds + 1), cost - args.baseline_loss, label=f"f - f*"
        )

plt.xlabel("Round")
plt.ylabel("Cost suboptimality")
plt.yscale("log")
plt.legend()
plt.tight_layout()
# Make sure directory exists
if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

plt.savefig(f"{RESULTS_FOLDER}/{experiment.runname}/cost_suboptimality_{args.experiment_type}.pdf")


# Display results for consensus suboptimality
plt.figure()

for param, cost, to_mean in (zip(params, costs, to_means)):
    if args.experiment_type == "lambda":
        if param == 0.0:
            continue
        plt.plot(
            np.arange(1, args.n_rounds + 1), to_mean, label=f"$\lambda={param:.1f}$"
        )
    elif args.experiment_type == "p":
        if param == 1.0:
            continue
        plt.plot(
            np.arange(1, args.n_rounds + 1), to_mean, label=f"$p={param:.1f}$"
        )
    else:
        plt.plot(
            np.arange(1, args.n_rounds + 1), to_mean, label=f"Consensus"
        )

plt.xlabel("Round")
plt.ylabel("Consensus suboptimality")
plt.yscale("log")
plt.legend()
plt.tight_layout()
# Make sure directory exists
if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

plt.savefig(f"{RESULTS_FOLDER}/{experiment.runname}/consensus_suboptimality_{args.experiment_type}.pdf")
