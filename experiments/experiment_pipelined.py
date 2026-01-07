import argparse
import os
import numpy as np
import networkx as nx
from tqdm import tqdm
from sklearn.datasets import load_svmlight_file
import urllib.request
from log_reg_utils import loss, loss_grad

# Reuse graph utils
from graph_utils import generate_random_digraph, adj_to_W


def get_drifting_topology(Adj_A, Adj_B, alpha, n_agents):
    """
    Generates a weight matrix W based on a linear interpolation
    of edge probabilities between Graph A and Graph B.
    """
    # Interpolate probability of edge existence
    Prob_Matrix = (1 - alpha) * Adj_A + alpha * Adj_B

    # Sample actual edges for this round based on probabilities
    # We use random float comparison against the probability matrix
    random_draw = np.random.rand(n_agents, n_agents)
    Current_Adj = (random_draw < Prob_Matrix).astype(float)

    # Ensure self-loops always exist
    np.fill_diagonal(Current_Adj, 1.0)

    # Convert to row-stochastic weight matrix
    # Handle isolated nodes (row sum 0) by making them self-loops if necessary
    row_sums = Current_Adj.sum(axis=1)
    Current_Adj[row_sums == 0, :] = 0
    np.fill_diagonal(Current_Adj, 1.0)  # Re-ensure diagonal

    return adj_to_W(Current_Adj)


class PipelinedExperiment:
    def __init__(self, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_rounds = args.n_rounds
        self.learning_rate = args.learning_rate
        self.n_experiments = args.n_experiments
        self.save_dir = os.path.join(args.results_folder, "pipelined_drifting")

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Load Data (Mushrooms)
        if not os.path.exists("mushroom.libsvm"):
            url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/mushrooms"
            urllib.request.urlretrieve(url, "mushroom.libsvm")

        data, target = load_svmlight_file("mushroom.libsvm")
        data = data.toarray()
        target = target - 1
        n, d = data.shape
        self.l2_strength = 1.0 / n
        data = np.hstack((np.ones((n, 1)), data))  # Bias term

        self.data_agents = np.array_split(data, self.n_agents)
        self.labels_agents = np.array_split(target, self.n_agents)
        self.d = d + 1

        # Baseline optimal loss for plotting
        # (Approximate optimal value for Mushrooms with this Reg)
        self.baseline_loss = 0.014484174216922262

    def run_single_experiment(self):
        """
        Runs a single simulation of topology drift.
        Returns (cost_std, cost_pipe, cons_std, cons_pipe) for this run.
        """
        # 1. Generate Start (A) and End (B) Graphs
        # We use sparse graphs (p=0.05) so centrality drift is significant
        G_A = generate_random_digraph(self.n_agents, p=0.05)
        G_B = generate_random_digraph(self.n_agents, p=0.05)

        Adj_A = nx.to_numpy_array(G_A)
        Adj_B = nx.to_numpy_array(G_B)

        # Ensure self-loops
        np.fill_diagonal(Adj_A, 1.0)
        np.fill_diagonal(Adj_B, 1.0)

        # --- SETUP STANDARD DT-GO (Static Warm-up) ---
        # Standard DT-GO warms up on Graph A and fixes D matrix
        W_A = adj_to_W(Adj_A)
        W_inf_A = np.linalg.matrix_power(W_A, 1000)  # Simulate warm-up
        pi_static = W_inf_A[0, :]  # Extract stationary dist
        D_static = 1.0 / (self.n_agents * pi_static)

        X_std = np.zeros((self.n_agents, self.d))

        # --- SETUP PIPELINED DT-GO ---
        # Pipelined runs an Estimation Stream (Matrix V) alongside optimization
        V_pipe = np.eye(self.n_agents)  # Initial dictionaries {id_n: 1}
        X_pipe = np.zeros((self.n_agents, self.d))

        # Metrics
        cost_std_hist = []
        cost_pipe_hist = []
        cons_std_hist = []
        cons_pipe_hist = []

        # Run rounds
        for k in range(self.n_rounds):
            # 1. Topology Drift
            alpha = k / self.n_rounds
            W_k = get_drifting_topology(Adj_A, Adj_B, alpha, self.n_agents)

            # --- Update Standard Algorithm ---
            self.update_step(X_std, W_k, D_static)

            # --- Update Pipelined Algorithm ---
            # 1. Estimation Stream
            V_pipe = np.matmul(W_k, V_pipe)
            # 2. Extract Pi estimate
            pi_est = np.diag(V_pipe)
            # 3. Optimization Stream (Dynamic D)
            D_dynamic = 1.0 / (self.n_agents * pi_est + 1e-12)
            self.update_step(X_pipe, W_k, D_dynamic)

            # --- Record Metrics ---
            cost_std_hist.append(self.compute_cost(X_std))
            cost_pipe_hist.append(self.compute_cost(X_pipe))
            cons_std_hist.append(self.compute_consensus(X_std))
            cons_pipe_hist.append(self.compute_consensus(X_pipe))

        return (
            np.array(cost_std_hist),
            np.array(cost_pipe_hist),
            np.array(cons_std_hist),
            np.array(cons_pipe_hist),
        )

    def run(self):
        print(
            f"Initializing Drifting Topology Experiment ({self.n_experiments} runs)..."
        )

        all_cost_std = []
        all_cost_pipe = []
        all_cons_std = []
        all_cons_pipe = []

        # Loop over number of experiments
        for i in tqdm(range(self.n_experiments), desc="Running Experiments"):
            c_std, c_pipe, cn_std, cn_pipe = self.run_single_experiment()
            all_cost_std.append(c_std)
            all_cost_pipe.append(c_pipe)
            all_cons_std.append(cn_std)
            all_cons_pipe.append(cn_pipe)

        # Average results
        avg_cost_std = np.mean(all_cost_std, axis=0)
        avg_cost_pipe = np.mean(all_cost_pipe, axis=0)
        avg_cons_std = np.mean(all_cons_std, axis=0)
        avg_cons_pipe = np.mean(all_cons_pipe, axis=0)

        # Save results
        print(f"Saving averaged results to {self.save_dir}...")
        np.save(os.path.join(self.save_dir, "cost_standard.npy"), avg_cost_std)
        np.save(os.path.join(self.save_dir, "cost_pipelined.npy"), avg_cost_pipe)
        np.save(os.path.join(self.save_dir, "consensus_standard.npy"), avg_cons_std)
        np.save(os.path.join(self.save_dir, "consensus_pipelined.npy"), avg_cons_pipe)
        print("Experiment Complete.")

    def update_step(self, X, W, D):
        """
        Performs one round of DT-GO: Descent -> Correct -> Mix
        X is modified in-place.
        """
        # 1. Local Descent
        Y = np.zeros_like(X)
        for i in range(self.n_agents):
            grad = loss_grad(
                X[i], self.data_agents[i], self.labels_agents[i], self.l2_strength
            )
            Y[i] = X[i] - self.learning_rate * grad

        # 2. Correction Step
        D_col = D.reshape(-1, 1)
        Z = X + D_col * (Y - X)

        # 3. Mixing Step
        X[:] = np.matmul(W, Z)

    def compute_cost(self, X):
        cost = 0
        for i in range(self.n_agents):
            cost += loss(
                X[i], self.data_agents[i], self.labels_agents[i], self.l2_strength
            )
        return cost / self.n_agents

    def compute_consensus(self, X):
        """
        Computes the mean squared deviation from the average model (Variance).
        """
        bar_x = np.mean(X, axis=0)
        diff = X - bar_x
        # 1/N * sum ||x_i - bar_x||^2
        return np.sum(diff**2) / self.n_agents


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_agents", type=int, default=100)
    parser.add_argument("--n_rounds", type=int, default=1500)
    parser.add_argument("--learning_rate", type=float, default=2.0)
    parser.add_argument(
        "--n_experiments", type=int, default=20, help="Number of runs to average"
    )
    parser.add_argument("--results_folder", type=str, default="results")
    args = parser.parse_args()

    exp = PipelinedExperiment(args)
    exp.run()
