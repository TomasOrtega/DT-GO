import numpy as np
import matplotlib.pyplot as plt
import os
import scienceplots
import matplotlib

RESULTS_DIR = "results/pipelined_evolving" 
COST_OUT_FILE = "figures/pipelined_cost.pdf"
CONS_OUT_FILE = "figures/pipelined_consensus.pdf"
BASELINE_LOSS = 0.014484174216922262

plt.style.use(["ieee", "high-vis"])
matplotlib.rcParams["text.usetex"] = True

def plot_metric(y_std, y_pipe, filename, ylabel, title=None, baseline_subopt=False):
    plt.figure()
    
    if baseline_subopt:
        y_std = np.maximum(y_std - BASELINE_LOSS, 1e-15)
        y_pipe = np.maximum(y_pipe - BASELINE_LOSS, 1e-15)

    rounds = np.arange(1, len(y_std) + 1)

    plt.plot(rounds, y_std, label=r"Standard", linestyle="--", linewidth=1.0)
    # Updated label to reflect EMA
    plt.plot(rounds, y_pipe, label=r"Pipelined (EMA)", linestyle="-", linewidth=1.0)

    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.yscale("log")

    if title:
        plt.title(title)

    plt.legend()
    # plt.tight_layout()

    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.close()

def main():
    if not os.path.exists(RESULTS_DIR):
        print(f"Directory {RESULTS_DIR} not found.")
        return
    
    if not os.path.exists("figures"):
        os.makedirs("figures")

    try:
        cost_std = np.load(os.path.join(RESULTS_DIR, "cost_standard.npy"))
        cost_pipe = np.load(os.path.join(RESULTS_DIR, "cost_pipelined.npy"))
        cons_std = np.load(os.path.join(RESULTS_DIR, "consensus_standard.npy"))
        cons_pipe = np.load(os.path.join(RESULTS_DIR, "consensus_pipelined.npy"))

        # Check for infinity and replace with NaN (Matplotlib ignores NaNs safely)
        cost_std = np.where(np.isinf(cost_std), np.nan, cost_std)
        cost_pipe = np.where(np.isinf(cost_pipe), np.nan, cost_pipe)
        cons_std = np.where(np.isinf(cons_std), np.nan, cons_std)
        cons_pipe = np.where(np.isinf(cons_pipe), np.nan, cons_pipe)

        # Remove negative values (if any) to avoid issues with log scale
        cost_std = np.where(cost_std < 0, np.nan, cost_std)
        cost_pipe = np.where(cost_pipe < 0, np.nan, cost_pipe)
        cons_std = np.where(cons_std < 0, np.nan, cons_std)
        cons_pipe = np.where(cons_pipe < 0, np.nan, cons_pipe)

        # clip extreme outliers for better visualization
        cost_std = np.clip(cost_std, 1e-15, 1e2)
        cost_pipe = np.clip(cost_pipe, 1e-15, 1e2)
        cons_std = np.clip(cons_std, 1e-15, 1e2)
        cons_pipe = np.clip(cons_pipe, 1e-15, 1e2)

    except FileNotFoundError:
        print("Data files not found.")
        return

    # Plot Cost Suboptimality
    plot_metric(cost_std, cost_pipe, COST_OUT_FILE, 
                r"Cost Suboptimality $f(x) - f^*$", 
                baseline_subopt=True)

    # Plot Consensus Suboptimality
    plot_metric(cons_std, cons_pipe, CONS_OUT_FILE, 
                r"Consensus Suboptimality (Variance)", 
                baseline_subopt=False)

if __name__ == "__main__":
    main()