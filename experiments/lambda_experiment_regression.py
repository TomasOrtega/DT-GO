# Import necessary libraries
from tqdm import tqdm  # For creating progress bars
import networkx as nx  # NetworkX for working with graphs
import numpy as np  # NumPy for numerical operations
import pickle  # For saving experiment results
import urllib.request  # For downloading files
from sklearn.datasets import load_svmlight_file  # For loading the dataset
import scipy.optimize  # For optimizing the loss function
from log_reg_utils import loss, loss_grad, OPTIMAL_WEIGHTS  # For logistic regression

# Custom function to generate random directed graphs
from generate_random_digraph import generate_random_digraph
from add_delays_to_graph import add_delays_to_graph


def run_experiment(N, lam, n_rounds, initial_model):
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
    # X is a matrix with rows all equal to the initial model
    X = np.tile(initial_model, (N + virtual_nodes, 1))
    global_cost = np.zeros(n_rounds)
    to_mean = np.zeros(n_rounds)

    learning_step = 2

    # Main optimization loop
    for k in range(n_rounds):
        # Compute global cost and consensus
        cost_mean = 0
        for i in range(N):
            cost_mean += loss(X[i], data_clients[i], labels_clients[i], l2_strength)
        global_cost[k] = cost_mean / N # np.mean(((np.arange(1, N + 1) - X[0:N]) ** 2))
        to_mean_cost = 0
        mean_models = np.mean(X[0:N, :], axis=0)
        for i in range(N):
            to_mean_cost += np.sum((mean_models - X[i]) ** 2)
        to_mean[k] = to_mean_cost / N

        # Optimize at every client
        for i in range(N):
            # Compute derivative for optimization
            deriv = loss_grad(X[i], data_clients[i], labels_clients[i], l2_strength)

            # Update X using the learning step and adjustment vector
            # X[i] -= learning_step * np.sqrt(N / (k + 1)) * deriv / (adjustment[i] * N)
            X[i] -= learning_step * deriv / (adjustment[i] * N)

        # Perform Gossiping for n_goss iterations
        for _ in range(n_goss):
            X = np.dot(W, X)

    # Return the results of the experiment
    return global_cost, to_mean


# Initialize random seed for reproducibility
np.random.seed(0)

# Size of the graph
N = 100

# Number of algorithm rounds
n_rounds = 1000

# Get dataset
# Download the LIBSVM mushroom dataset from the URL
url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/mushrooms"
urllib.request.urlretrieve(url, "mushroom.libsvm")

# Load the downloaded dataset
data, target = load_svmlight_file("mushroom.libsvm")

# Convert the sparse data to a dense matrix
data = data.toarray()

# Bring target to 0,1
target = target - 1

# Get problem dimensions
n, d = data.shape

# Set the L2 regularization strength
l2_strength = 1.0 / n

# Add a bias term (intercept) to the data
data = np.hstack((np.ones((n, 1)), data))

# Define a global model
global_model = np.zeros(d + 1)

# Set the baseline loss
# Use a black-box optimizer to find the baseline loss
baseline_loss = scipy.optimize.minimize(
    loss, OPTIMAL_WEIGHTS,
    args=(data, target, l2_strength),
    options={"disp": True}
).fun

# Split the dataset into N parts for clients
data_clients = np.array_split(data, N)
labels_clients = np.array_split(target, N)

# Run the baseline experiment with lambda=0
baseline_cost, to_mean_baseline = run_experiment(N, 0, n_rounds, global_model.copy())
baseline_cost = np.tile(baseline_loss, n_rounds)

# List to store experiment results
experiment_results = []

# Run experiments for different values of parameter lambda
for lam in tqdm([0.0, 0.1, 0.2, 0.3, 0.4]):
    n_exp = 10#0  # Number of experiments to run for each value of lambda

    # Run experiments for n_exp iterations
    cost_all, to_mean_all = zip(
        *(run_experiment(N, lam, n_rounds, global_model.copy()) for _ in range(n_exp))
    )

    # Average results over the experiments and store them
    experiment_results.append(
        (lam, np.mean(cost_all, axis=0), np.mean(to_mean_all, axis=0))
    )

# Save the experiment results to a file
result_filename = "results/results_lambda_regression.pickle"
with open(result_filename, "wb") as file:
    pickle.dump((experiment_results, baseline_cost, to_mean_baseline), file)
