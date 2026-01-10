import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import scienceplots
import matplotlib

RESULTS_DIR = "results/pipelined_evolving"
COST_OUT_FILE = "figures/pipelined_cost.pdf"
CONS_OUT_FILE = "figures/pipelined_consensus.pdf"
BASELINE_LOSS = 0.014484174216922262

plt.style.use(["ieee", "high-vis"])
matplotlib.rcParams["text.usetex"] = True


def load_and_aggregate(metric_name):
    """
    Scans the RESULTS_DIR for subfolders (e.g. run_1, run_2)
    loads the specific metric file, and computes the mean across all runs.
    """
    search_path = os.path.join(RESULTS_DIR, "run_*", f"{metric_name}.npy")
    files = glob.glob(search_path)

    if not files:
        print(f"No files found for metric: {metric_name} in {search_path}")
        return None

    data_list = []
    print(f"Aggregating {len(files)} experiments for {metric_name}...")

    for f in files:
        try:
            arr = np.load(f)
            data_list.append(arr)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not data_list:
        return None

    # Stack and Mean
    # Shape: (N_experiments, N_rounds)
    stacked = np.stack(data_list, axis=0)
    mean_data = np.mean(stacked, axis=0)

    return mean_data


def plot_metric(y_std, y_pipe, filename, ylabel, title=None, baseline_subopt=False):
    plt.figure()

    if baseline_subopt:
        y_std = np.maximum(y_std - BASELINE_LOSS, 1e-15)
        y_pipe = np.maximum(y_pipe - BASELINE_LOSS, 1e-15)

    rounds = np.arange(1, len(y_std) + 1)

    plt.plot(rounds, y_std, label=r"Standard", linestyle="--", linewidth=1.0)
    plt.plot(rounds, y_pipe, label=r"Pipelined (EMA)", linestyle="-", linewidth=1.0)

    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.yscale("log")

    if title:
        plt.title(title)

    plt.legend()
    plt.tight_layout()

    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.close()


def main():
    if not os.path.exists(RESULTS_DIR):
        print(f"Directory {RESULTS_DIR} not found.")
        return

    if not os.path.exists("figures"):
        os.makedirs("figures")

    # Load and Aggregate Data
    cost_std = load_and_aggregate("cost_standard")
    cost_pipe = load_and_aggregate("cost_pipelined")
    cons_std = load_and_aggregate("consensus_standard")
    cons_pipe = load_and_aggregate("consensus_pipelined")

    if any(x is None for x in [cost_std, cost_pipe, cons_std, cons_pipe]):
        print("Error: Missing data for one or more metrics.")
        return

    # Check for infinity and cleanup
    def cleanup(arr):
        arr = np.where(np.isinf(arr), np.nan, arr)
        arr = np.where(arr < 0, np.nan, arr)
        arr = np.clip(arr, 1e-15, 1e15)
        return arr

    cost_std = cleanup(cost_std)
    cost_pipe = cleanup(cost_pipe)
    cons_std = cleanup(cons_std)
    cons_pipe = cleanup(cons_pipe)

    # Plot Cost Suboptimality
    plot_metric(
        cost_std,
        cost_pipe,
        COST_OUT_FILE,
        r"Cost suboptimality",
        baseline_subopt=True,
    )

    # Plot Consensus Suboptimality
    plot_metric(
        cons_std,
        cons_pipe,
        CONS_OUT_FILE,
        r"Consensus suboptimality",
        baseline_subopt=False,
    )


if __name__ == "__main__":
    main()
