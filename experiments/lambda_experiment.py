# Import necessary libraries
from tqdm import tqdm  # For creating progress bars
import scienceplots  # Style for the plots
import matplotlib.pyplot as plt  # Library for creating plots
import matplotlib  # Import Matplotlib for plotting
import networkx as nx  # NetworkX for working with graphs
import numpy as np  # NumPy for numerical operations

# Custom function to generate random directed graphs
from generate_random_digraph import generate_random_digraph
from add_delays_to_graph import add_delays_to_graph


def run_experiment(N, lam, n_rounds):
    # Obtain a fully connected digraph
    G = generate_random_digraph(N, 1)

    # Add delays
    G, virtual_nodes = add_delays_to_graph(G, lam, N)

    # Get the adjacency matrix
    adj = nx.adjacency_matrix(G).todense()
    adj = np.array(adj)

    # Assert all self-loops exist for non-virtual nodes
    assert np.all(np.diag(adj[0:N]) == 1)

    # Assert weight matrix is column-stochastic (we transpose later)
    column_sum = np.sum(adj, axis=0)
    A = adj / column_sum
    assert np.all(np.abs(np.sum(A, axis=0) - 1) < 1e-6)

    # Add delays and get virtual_nodes
    # A, virtual_nodes = add_delays_to_graph_v2(A, lam, N)
    # assert np all(np.abs(np.sum(A, axis=0) - 1) < 1e-6)

    # Obtain adjustment vector of the graph
    Ainf = np.linalg.matrix_power(A, 1024)
    adjustment = (Ainf)[:, 0]
    assert np.all(np.abs(A.dot(adjustment) - adjustment) < 1e-6)

    # Use the transpose and change notation for optimization (W * X)
    W = A.T

    # Number of gossip iterations per round
    n_goss = 1

    # Initialize variables for optimization and tracking
    X = np.zeros(N + virtual_nodes)
    global_cost = np.zeros(n_rounds)
    to_mean = np.zeros(n_rounds)

    learning_step = 0.01

    # Main optimization loop
    for k in range(n_rounds):
        # Compute global cost and consensus
        global_cost[k] = np.mean(((np.arange(1, N + 1) - X[0:N]) ** 2))
        to_mean[k] = np.mean((X[0:N] - np.mean(X[0:N])) ** 2)

        # Compute derivative for optimization
        deriv = -2 * (np.arange(1, N + 1) - X[0:N])

        # Update X using the learning step and adjustment vector
        X[0:N] -= learning_step * np.sqrt(N / (k + 1)) * deriv / (adjustment[0:N] * N)

        # Perform Gossiping for n_goss iterations
        for k_goss in range(n_goss):
            X = np.dot(W, X)

    # Return the results of the experiment
    return global_cost, to_mean


# Initialize random seed for reproducibility
np.random.seed(0)

# Size of the graph
N = 100

# Number of algorithm rounds
n_rounds = 100

# Run the baseline experiment with lambda=0
baseline_cost, to_mean_baseline = run_experiment(N, 0, n_rounds)

# List to store experiment results
experiment_results = []

# Run experiments for different values of parameter lambda
for lam in tqdm([0.1, 0.2, 0.3, 0.4]):
    n_exp = 100  # Number of experiments to run for each value of lambda

    # Run experiments for n_exp iterations
    cost_all, to_mean_all = zip(
        *(run_experiment(N, lam, n_rounds) for _ in range(n_exp))
    )

    # Average results over the experiments and store them
    experiment_results.append(
        (lam, np.mean(cost_all, axis=0), np.mean(to_mean_all, axis=0))
    )

# Apply specific style settings for IEEE publications
plt.style.use(['ieee', 'high-vis'])

# Avoid Type 3 fonts for IEEE publications
matplotlib.rcParams['text.usetex'] = True

# Display results for cost suboptimality
plt.figure()

# Plot baseline for cost suboptimality
plt.plot(np.arange(1, n_rounds + 1), 0 *
         np.arange(1, n_rounds + 1), label='$\lambda=0.0$')

# Plot results for each value of p for cost suboptimality
for lam, cost, to_mean in experiment_results:
    plt.plot(np.arange(1, n_rounds + 1), cost -
             baseline_cost, label=f'$\lambda={lam:.1f}$')

# Set labels for the axes for cost suboptimality
plt.xlabel("Round")
plt.ylabel("Cost suboptimality")

# Set the xlimit to [0, 30] for visibility
plt.xlim([0, 30])

# Add a legend to the cost suboptimality plot
plt.legend()

# Save the cost suboptimality plot to a PDF file
plt.tight_layout()
plt.savefig("cost_delays.pdf")

# Display results for consensus suboptimality
plt.figure()

# Plot baseline for consensus suboptimality
plt.plot(np.arange(1, n_rounds + 1), 0 *
         np.arange(1, n_rounds + 1), label='$\lambda=0.0$')

# Plot results for each value of p for consensus suboptimality
for lam, cost, to_mean in experiment_results:
    plt.plot(np.arange(1, n_rounds + 1), to_mean -
             to_mean_baseline, label=f'$\lambda={lam:.1f}$')

# Set labels for the axes for consensus suboptimality
plt.xlabel("Round")
plt.ylabel("Consensus suboptimality")

# Set the xlimit to [0, 30] for visibility
plt.xlim([0, 30])

# Add a legend to the consensus suboptimality plot
plt.legend()

# Save the consensus suboptimality plot to a PDF file
plt.tight_layout()
plt.savefig("consensus_delays.pdf")
