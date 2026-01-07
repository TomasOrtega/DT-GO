import numpy as np
import matplotlib.pyplot as plt
import os
import scienceplots
import matplotlib

# ==========================================
# CONFIGURATION
# ==========================================
RESULTS_DIR = "results/pipelined_drifting"
COST_OUT_FILE = "figures/pipelined_cost.pdf"
CONS_OUT_FILE = "figures/pipelined_consensus.pdf"
BASELINE_LOSS = 0.014484174216922262  # f* for Mushrooms dataset

# Apply style
plt.style.use(["ieee", "high-vis"])
matplotlib.rcParams["text.usetex"] = True


def plot_metric(y_std, y_pipe, filename, ylabel, title=None):
    plt.figure()

    rounds = np.arange(1, len(y_std) + 1)

    # Plot Standard
    plt.plot(rounds, y_std, label=r"Standard", linestyle="--", linewidth=0.8)

    # Plot Pipelined
    plt.plot(
        rounds,
        y_pipe,
        label=r"Pipelined (Dynamic $D$)",
        linestyle="-",
        linewidth=1.0,
    )

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
        print(f"Directory {RESULTS_DIR} not found. Run the experiment first.")
        return

    if not os.path.exists("figures"):
        os.makedirs("figures")

    try:
        # Load Cost Data
        cost_std = np.load(os.path.join(RESULTS_DIR, "cost_standard.npy"))
        cost_pipe = np.load(os.path.join(RESULTS_DIR, "cost_pipelined.npy"))

        # Load Consensus Data
        cons_std = np.load(os.path.join(RESULTS_DIR, "consensus_standard.npy"))
        cons_pipe = np.load(os.path.join(RESULTS_DIR, "consensus_pipelined.npy"))
    except FileNotFoundError:
        print("Data files not found. Please re-run experiment_pipelined.py")
        return

    # 1. Plot Cost Suboptimality
    subopt_std = cost_std - BASELINE_LOSS
    subopt_pipe = cost_pipe - BASELINE_LOSS
    plot_metric(subopt_std, subopt_pipe, COST_OUT_FILE, "Average Cost Suboptimality")

    # 2. Plot Consensus Suboptimality
    # Consensus metric is already 1/N * sum ||x_i - mean||^2, so baseline is 0
    plot_metric(cons_std, cons_pipe, CONS_OUT_FILE, "Average Consensus Suboptimality")


if __name__ == "__main__":
    main()
