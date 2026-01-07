import numpy as np
import matplotlib.pyplot as plt
import os
import scienceplots
import matplotlib

# Setup styles
plt.style.use(["ieee", "high-vis"])
matplotlib.rcParams["text.usetex"] = True


def main():
    results_dir = "results/pipelined_drifting"

    if not os.path.exists(results_dir):
        print(f"Directory {results_dir} not found. Run the experiment first.")
        return

    try:
        cost_std = np.load(os.path.join(results_dir, "cost_standard.npy"))
        cost_pipe = np.load(os.path.join(results_dir, "cost_pipelined.npy"))
    except FileNotFoundError:
        print("Data files not found.")
        return

    # Baseline optimal loss (from regression utils)
    f_star = 0.014484174216922262

    # Calculate Suboptimality
    subopt_std = cost_std - f_star
    subopt_pipe = cost_pipe - f_star

    rounds = np.arange(len(cost_std))

    plt.figure(figsize=(6, 4))

    # Plot Standard
    plt.plot(
        rounds, subopt_std, label=r"Standard \textsc{DT-GO}", linestyle="--", alpha=0.9
    )

    # Plot Pipelined
    plt.plot(
        rounds,
        subopt_pipe,
        label=r"Pipelined \textsc{DT-GO}",
        linestyle="-",
        linewidth=1.5,
    )

    plt.xlabel("Round (Topology Drifting $A \to B$)")
    plt.ylabel(r"$f(x) - f^\star$")
    plt.yscale("log")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.tight_layout()

    # Save
    if not os.path.exists("figures"):
        os.makedirs("figures")

    out_file = "figures/pipelined_comparison.pdf"
    plt.savefig(out_file)
    print(f"Plot saved to {out_file}")
    plt.show()


if __name__ == "__main__":
    main()
