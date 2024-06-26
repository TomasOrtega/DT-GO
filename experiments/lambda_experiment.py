# Import necessary libraries
from tqdm import tqdm  # For creating progress bars
import networkx as nx  # NetworkX for working with graphs
import numpy as np  # NumPy for numerical operations
import pickle  # For saving experiment results

# Custom function to generate random directed graphs
from graph_utils import generate_random_digraph, add_delays_to_graph



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

# Save the experiment results to a file
result_filename = "results/results_lambda.pickle"
with open(result_filename, "wb") as file:
    pickle.dump((experiment_results, baseline_cost, to_mean_baseline), file)
