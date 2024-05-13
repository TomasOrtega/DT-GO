import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from experiment_regression import Experiment, RECORD_KEYS
from log_reg_utils import get_args_as_obj, config_dict_to_str
import matplotlib
import scienceplots

RESULTS_FOLDER = "results"
N_EXP = 1000

# Plot settings
plt.style.use(['ieee', 'high-vis'])  # try bright instead of high-vis maybe
# Avoid Type 3 fonts for IEEE publications (switch to True)
matplotlib.rcParams['text.usetex'] = True


def get_results(args, n_exp=1):
    """
    Get the results of an experiment.

    Args:
        args (dict): The arguments for the experiment.
        n_exp (int, optional): The number of experiments to average the results over. Defaults to 1.

    Returns:
        tuple: A tuple containing the averaged costs and to_means arrays.
    """
    # Convert args to an object
    args = get_args_as_obj(args)

    # Generate runname and folder path
    runname = config_dict_to_str(args, record_keys=RECORD_KEYS)
    folder = os.path.join(RESULTS_FOLDER, runname)

    # Run the experiment if the folder does not exist
    if not os.path.exists(folder):
        experiment = Experiment(args)
        experiment.run_experiment()

    # Load the results and average for n_exp experiments
    costs = []
    to_means = []
    print(f"Loading results for {runname}...")
    for i in tqdm(range(n_exp)):
        exp_folder = os.path.join(folder, "experiments", str(i))
        costs.append(np.load(os.path.join(exp_folder, "cost.npy")))
        to_means.append(np.load(os.path.join(exp_folder, "to_mean.npy")))

    costs = np.mean(costs, axis=0)
    to_means = np.mean(to_means, axis=0)

    return costs, to_means


def generate_plot(dict_results, cost_or_consensus, baseline, filename, time_varying=False, p_err=False):
    """
    Generate a plot of cost or consensus suboptimality over rounds.

    Args:
        dict_results (dict): A dictionary containing the results of the experiment.
            The keys are tuples of (lambda, p) values, and the values are tuples of
            cost values and means values. If time_varying is true, the tuples are (q,p)
        cost_or_consensus (str): Specifies whether to plot cost suboptimality or consensus suboptimality.
            Must be either "cost" or "consensus".
        baseline (float): The baseline value to subtract from the cost or means values.
        filename (str): The name of the file to save the plot to.

    Returns:
        None
    """
    plt.figure()

    # For every product of lambda and p, plot the cost suboptimality
    if time_varying:
        for (q, p), (costs, to_means) in dict_results.items():
            yvals = costs - baseline if cost_or_consensus == "cost" else to_means - baseline
            n_rounds = len(yvals)
            if p_err:
                p_err_string = 'p_{err}'
                plt.plot(
                    np.arange(1, n_rounds + 1), yvals, label=f"${p_err_string}={q:.1f}, p={p:.1f}$"
                )
            else:
                plt.plot(
                    np.arange(1, n_rounds + 1), yvals, label=f"$q={q:.2f}, p={p:.1f}$"
                )
    else:
        for (lam, p), (costs, to_means) in dict_results.items():
            yvals = costs - baseline if cost_or_consensus == "cost" else to_means - baseline
            n_rounds = len(yvals)
            plt.plot(
                np.arange(1, n_rounds + 1), yvals, label=f"$\lambda={lam:.1f}, p={p:.1f}$"
            )

    plt.xlabel("Round")
    ylabel = "Cost suboptimality" if cost_or_consensus == "cost" else "Consensus suboptimality"
    plt.ylabel(ylabel)
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, filename))


# Arguments for the experiment
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
    "lam": 0.0,
    "p": 1.0,
    "n_experiments": N_EXP,
    "l2_strength": None,
    "n_gossip": 1,
    "time_varying": False,
    "time_varying_prob": 0,
    "p_err": 0.0,
}

args_baseline = args.copy()
args_baseline["lam"] = 0.0
args_baseline["p"] = 1.0
args_baseline["n_experiments"] = 1

# Get the baseline results
baseline_cost, baseline_to_means = get_results(args_baseline, 1)

# Plot mixed plots
lambdas = [0.3]
ps = [1.0, 0.8, 0.6, 0.4, 0.2]
dict_results = {}

# For every product of lambda and p, run the experiment or load the results if they exist
for lam in lambdas:
    for p in ps:
        args["lam"] = lam
        args["p"] = p
        costs, to_means = get_results(args, N_EXP)
        dict_results[(lam, p)] = (costs, to_means)

generate_plot(dict_results, "cost", baseline_cost,
              filename="cost_suboptimality_mixed.pdf")
generate_plot(dict_results, "consensus", baseline_to_means,
              filename="consensus_suboptimality_mixed.pdf")

# Plot lambda plots
lambdas = [0.1, 0.2, 0.3, 0.4]
dict_results = {}

# For every product of lambda and p, run the experiment or load the results if they exist
for lam in lambdas:
    args["lam"] = lam
    args["p"] = 1.0
    costs, to_means = get_results(args, N_EXP)
    dict_results[(lam, 1.0)] = (costs, to_means)

generate_plot(dict_results, "cost", baseline_cost,
              filename="cost_suboptimality_lambda.pdf")
generate_plot(dict_results, "consensus", baseline_to_means,
              filename="consensus_suboptimality_lambda.pdf")

# Plot p plots
ps = [0.8, 0.6, 0.4, 0.2]
dict_results = {}

for p in ps:
    args["lam"] = 0.0
    args["p"] = p
    costs, to_means = get_results(args, N_EXP)
    dict_results[(0.0, p)] = (costs, to_means)

generate_plot(dict_results, "cost", baseline_cost,
              filename="cost_suboptimality_p.pdf")
generate_plot(dict_results, "consensus", baseline_to_means,
              filename="consensus_suboptimality_p.pdf")


# Plot time-varying plots
qs = [0.0, 0.01, 0.03]
p = 0.2
args["p"] = p
args["lam"] = 0.0
args["time_varying"] = True
dict_results = {}

for q in qs:
    args["time_varying_prob"] = q
    costs, to_means = get_results(args, 1)
    dict_results[(q, p)] = (costs, to_means)

generate_plot(dict_results, "cost", baseline_cost,
              filename="cost_suboptimality_q.pdf", time_varying=True)
generate_plot(dict_results, "consensus", baseline_to_means,
              filename="consensus_suboptimality_q.pdf", time_varying=True)


# Plot time-varying plots
p_errs = [0.0, 0.1, 0.2]
args["p"] = 0.5
args["lam"] = 0.0
args["time_varying_prob"] = 0.0
args["time_varying"] = True
dict_results = {}

for p_err in p_errs:
    args["p_err"] = p_err
    costs, to_means = get_results(args, N_EXP)
    dict_results[(p_err, p)] = (costs, to_means)

generate_plot(dict_results, "cost", baseline_cost,
              filename="cost_suboptimality_p_err.pdf", time_varying=True, p_err=True)
generate_plot(dict_results, "consensus", baseline_to_means,
              filename="consensus_suboptimality_p_err.pdf", time_varying=True, p_err=True)
