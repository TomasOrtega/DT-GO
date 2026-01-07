import numpy as np
import matplotlib.pyplot as plt
import os
import scienceplots
import matplotlib

# ==========================================
# CONFIGURATION
# ==========================================
RESULTS_DIR = "results/pipelined_drifting"
OUT_FILE = "figures/pipelined_comparison.pdf"
BASELINE_LOSS = 0.014484174216922262  # f* for Mushrooms dataset

# Apply the same style as other experiment plots
plt.style.use(["ieee", "high-vis"])
matplotlib.rcParams["text.usetex"] = True


def main():
    if not os.path.exists(RESULTS_DIR):
        print(f"Directory {RESULTS_DIR} not found. Run the experiment first.")
        return

    try:
        cost_std = np.load(os.path.join(RESULTS_DIR, "cost_standard.npy"))
        cost_pipelined = np.load(os.path.join(RESULTS_DIR, "cost_pipelined.npy"))
    except FileNotFoundError:
        print("Data files not found in the results directory.")
        return

    # Calculate Suboptimality (f(x) - f*)
    subopt_std = cost_std - BASELINE_LOSS
    subopt_pipelined = cost_pipelined - BASELINE_LOSS

    n_rounds = len(cost_std)
    rounds = np.arange(1, n_rounds + 1)

    # ==========================================
    # PLOTTING
    # ==========================================
    plt.figure()

    # Plot Standard (Static) approach
    plt.plot(rounds, subopt_std, label=r"Standard", linestyle="--", linewidth=0.8)

    # Plot Pipelined (Dynamic) approach
    plt.plot(
        rounds,
        subopt_pipelined,
        label=r"Pipelined (Dynamic $D$)",
        linestyle="-",
        linewidth=1.0,
    )

    # Formatting matches plot_experiments.py
    plt.xlabel("Round")
    plt.ylabel("Cost suboptimality")
    plt.yscale("log")

    # Optional: If you want to highlight the drift, you can add text or lines
    # But usually, keeping it clean (ieee style) is better.

    plt.legend()
    plt.tight_layout()

    # Save output
    if not os.path.exists("figures"):
        os.makedirs("figures")

    plt.savefig(OUT_FILE)
    print(f"Plot saved to {OUT_FILE}")
    # plt.show() # Uncomment if running locally with a display


if __name__ == "__main__":
    main()
