import scienceplots  # Importing the scienceplots library for improved plot styling
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
import matplotlib  # Import Matplotlib for plotting
import numpy as np  # Import NumPy for numerical computing
import pickle  # Import pickle for saving experiment results

# Load the experiment results from the file
result_filename = "results/results_lambda_regression.pickle"
with open(result_filename, "rb") as file:
    experiment_results, baseline_cost, to_mean_baseline = pickle.load(file)

# Get the number of rounds from the baseline cost
n_rounds = len(baseline_cost)

# Apply specific style settings for IEEE publications
plt.style.use(["ieee", "high-vis"])

# Avoid Type 3 fonts for IEEE publications
matplotlib.rcParams["text.usetex"] = True

# Display results for cost suboptimality
plt.figure()

# Plot baseline for cost suboptimality
#plt.plot(
#    np.arange(1, n_rounds + 1), 0 * np.arange(1, n_rounds + 1), label="$\lambda=0.0$"
#)

baseline_cost = None

# Plot results for each value of p for cost suboptimality
for lam, cost, to_mean in experiment_results:
    if lam == 0.0:
        baseline_cost = cost
        continue
    plt.plot(
        np.arange(1, n_rounds + 1), cost - baseline_cost, label=f"$\lambda={lam:.1f}$"
    )

# Set labels for the axes for cost suboptimality
plt.xlabel("Round")
plt.ylabel("Cost suboptimality")

# Set logarithmic scale for the y-axis for cost suboptimality
plt.yscale("log")

# Set the xlimit to [0, 30] for visibility
# plt.xlim([0, 30])

# Add a legend to the cost suboptimality plot
plt.legend()

# Save the cost suboptimality plot to a PDF file
plt.tight_layout()
plt.savefig("figures/cost_delays_regression.pdf")

# Display results for consensus suboptimality
plt.figure()

# Plot baseline for consensus suboptimality
# plt.plot(
#     np.arange(1, n_rounds + 1), 0 * np.arange(1, n_rounds + 1), label="$\lambda=0.0$"
# )

# Plot results for each value of p for consensus suboptimality
for lam, cost, to_mean in experiment_results:
    if lam == 0.0:
        continue
    plt.plot(
        np.arange(1, n_rounds + 1),
        to_mean - to_mean_baseline,
        label=f"$\lambda={lam:.1f}$",
    )

# Set labels for the axes for consensus suboptimality
plt.xlabel("Round")
plt.ylabel("Consensus suboptimality")
plt.yscale("log")  # Set logarithmic scale for the y-axis for consensus suboptimality

# Set the xlimit to [0, 30] for visibility
#plt.xlim([0, 30])

# Add a legend to the consensus suboptimality plot
plt.legend()

# Save the consensus suboptimality plot to a PDF file
plt.tight_layout()
plt.savefig("figures/consensus_delays_regression.pdf")
