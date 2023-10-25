# Import necessary libraries
import scienceplots  # Importing the scienceplots library for improved plot styling
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
import matplotlib  # Import Matplotlib for plotting
from tqdm import tqdm  # Import tqdm for progress bars
import numpy as np  # Import NumPy for numerical operations
import networkx as nx  # Import NetworkX for graph operations

# Import a custom function to generate random digraphs
from generate_random_digraph import generate_random_digraph

# Define a function to run the experiment


def run_experiment(N, p, n_rounds):
    # Obtain a random connected digraph
    G = generate_random_digraph(N, p)

    # Get the adjacency matrix
    adj = nx.adjacency_matrix(G).todense()
    adj = np.array(adj)

    # Assert that all nodes have self-loops
    assert np.all(np.diag(adj) == 1)

    # Assert that the weight matrix is column-stochastic (we transpose it later)
    column_sum = np.sum(adj, axis=0)
    A = adj / column_sum
    assert np.all(np.abs(np.sum(A, axis=0) - 1) < 1e-6)

    # Obtain the adjustment vector of the graph
    Ainf = np.linalg.matrix_power(A, 1024)
    adjustment = (Ainf)[:, 0]
    assert np.all(np.abs(A.dot(adjustment) - adjustment) < 1e-6)

    # Use the transpose and change notation for optimization (W * X)
    W = A.T

    # Number of gossip iterations per round
    n_goss = 1

    # Initialize variables for optimization and tracking
    X = np.zeros(N)
    global_cost = np.zeros(n_rounds)
    to_mean = np.zeros(n_rounds)

    learning_step = 0.01

    # Main optimization loop
    for k in range(n_rounds):
        # Compute global cost and consensus
        global_cost[k] = np.mean(((np.arange(1, N + 1) - X) ** 2))
        to_mean[k] = np.mean((X - np.mean(X)) ** 2)

        # Compute derivative for optimization
        deriv = -2 * (np.arange(1, N + 1) - X)

        # Update X using the learning step and adjustment vector
        X -= learning_step * np.sqrt(N / (k + 1)) * deriv / (adjustment * N)

        # Perform Gossiping for n_goss iterations
        for k_goss in range(n_goss):
            X = np.dot(W, X)

    # Return the results of the experiment
    return global_cost, to_mean


# Initialize random seed for reproducibility
np.random.seed(0)

# Set the size of the graph
N = 100

# Number of algorithm rounds
n_rounds = 100

# Run the baseline experiment with p=1
baseline_cost, to_mean_baseline = run_experiment(N, 1, n_rounds)

# List to store experiment results
experiment_results = []

# Run experiments for different values of parameter p
for p in tqdm([0.1, 0.2, 0.3, 0.4, 0.6]):
    n_exp = 100  # Number of experiments to run for each value of p

    # Run experiments for n_exp iterations
    cost_all, to_mean_all = zip(*(run_experiment(N, p, n_rounds) for _ in range(n_exp)))

    # Average results over the experiments and store them
    experiment_results.append(
        (p, np.mean(cost_all, axis=0), np.mean(to_mean_all, axis=0))
    )

# Plot experiment results
# Apply specific style settings for IEEE publications
plt.style.use(['ieee', 'high-vis'])

# Avoid Type 3 fonts for IEEE publications
matplotlib.rcParams['text.usetex'] = True

# Create a figure for displaying cost suboptimality results
plt.figure()

# Plot results for each value of p for cost suboptimality
for p, cost, to_mean in experiment_results:
    plt.plot(np.arange(1, n_rounds + 1), cost -
             baseline_cost, label=f'$p={p:.1f}$')

# Plot the baseline for cost suboptimality
plt.plot(np.arange(1, n_rounds + 1), 0 *
         np.arange(1, n_rounds + 1), label='$p=1.0$')

# Set labels for the axes for cost suboptimality
plt.xlabel("Round")
plt.ylabel("Cost suboptimality")

# Add a legend to the cost suboptimality plot
plt.legend()

# Save the cost suboptimality plot to a PDF file
plt.tight_layout()
plt.savefig("cost.pdf")  # Save the plot as a PDF file

# Display results for consensus suboptimality
plt.figure()
# Plot results for each value of p for consensus suboptimality
for p, cost, to_mean in experiment_results:
    plt.plot(np.arange(1, n_rounds + 1), to_mean -
             to_mean_baseline, label=f'$p={p:.1f}$')

# Plot baseline for consensus suboptimality
plt.plot(np.arange(1, n_rounds + 1), 0 *
         np.arange(1, n_rounds + 1), label='$p=1.0$')

# Set labels for the axes for consensus suboptimality
plt.xlabel("Round")
plt.ylabel("Consensus suboptimality")

# Add a legend to the consensus suboptimality plot
plt.legend()

# Save the consensus suboptimality plot to a PDF file
plt.tight_layout()
plt.savefig("consensus.pdf")
