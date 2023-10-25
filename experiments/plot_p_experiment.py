import scienceplots  # Importing the scienceplots library for improved plot styling
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
import matplotlib  # Import Matplotlib for plotting
import numpy as np  # Import NumPy for numerical computing
import pickle  # Import pickle for saving experiment results

# Load the experiment results from the file
result_filename = "results/results_p.pickle"
with open(result_filename, "rb") as file:
    experiment_results, baseline_cost, to_mean_baseline = pickle.load(file)

# Get the number of rounds from the baseline cost
n_rounds = len(baseline_cost)

# Plot experiment results
# Apply specific style settings for IEEE publications
plt.style.use(["ieee", "high-vis"])

# Avoid Type 3 fonts for IEEE publications
matplotlib.rcParams["text.usetex"] = True

# Create a figure for displaying cost suboptimality results
plt.figure()

# Plot results for each value of p for cost suboptimality
for p, cost, to_mean in experiment_results:
    plt.plot(np.arange(1, n_rounds + 1), cost - baseline_cost, label=f"$p={p:.1f}$")

# Plot the baseline for cost suboptimality
plt.plot(np.arange(1, n_rounds + 1), 0 * np.arange(1, n_rounds + 1), label="$p=1.0$")

# Set labels for the axes for cost suboptimality
plt.xlabel("Round")
plt.ylabel("Cost suboptimality")

# Add a legend to the cost suboptimality plot
plt.legend()

# Save the cost suboptimality plot to a PDF file
plt.tight_layout()
plt.savefig("figures/cost.pdf")  # Save the plot as a PDF file

# Display results for consensus suboptimality
plt.figure()
# Plot results for each value of p for consensus suboptimality
for p, cost, to_mean in experiment_results:
    plt.plot(
        np.arange(1, n_rounds + 1), to_mean - to_mean_baseline, label=f"$p={p:.1f}$"
    )

# Plot baseline for consensus suboptimality
plt.plot(np.arange(1, n_rounds + 1), 0 * np.arange(1, n_rounds + 1), label="$p=1.0$")

# Set labels for the axes for consensus suboptimality
plt.xlabel("Round")
plt.ylabel("Consensus suboptimality")

# Add a legend to the consensus suboptimality plot
plt.legend()

# Save the consensus suboptimality plot to a PDF file
plt.tight_layout()
plt.savefig("figures/consensus.pdf")
