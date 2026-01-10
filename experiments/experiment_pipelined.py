import argparse
import os
import numpy as np
import networkx as nx
from tqdm import tqdm
from sklearn.datasets import load_svmlight_file
import urllib.request
from log_reg_utils import loss, loss_grad

# Reuse graph utils from the provided context
from graph_utils import generate_random_digraph, adj_to_W


class PipelinedExperiment:
    def __init__(self, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_rounds = args.n_rounds
        self.learning_rate = args.learning_rate
        self.seed = args.seed
        self.d_clip_max = args.d_clip_max

        # Set random seed for reproducibility
        np.random.seed(self.seed)

        # Unique directory for this specific seed run
        self.save_dir = os.path.join(
            args.results_folder, "pipelined_evolving", f"run_{self.seed}"
        )

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Load Data
        if not os.path.exists("mushroom.libsvm"):
            url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/mushrooms"
            urllib.request.urlretrieve(url, "mushroom.libsvm")

        data, target = load_svmlight_file("mushroom.libsvm")
        data = data.toarray()
        target = target - 1
        n, d = data.shape
        self.l2_strength = 1.0 / n
        data = np.hstack((np.ones((n, 1)), data))

        self.data_agents = np.array_split(data, self.n_agents)
        self.labels_agents = np.array_split(target, self.n_agents)
        self.d = d + 1

    def run_single_experiment(self):
        # --- 1. INITIAL TOPOLOGY ---
        # Start with a random graph (e.g., density p=0.05)
        G = generate_random_digraph(self.n_agents, p=0.05)
        Adj = nx.to_numpy_array(G)
        np.fill_diagonal(Adj, 1.0)  # Ensure self-loops

        # Calculate Initial Weight Matrix
        W_curr = adj_to_W(Adj)

        # --- PRE-CALCULATE STATIC D (Baseline) ---
        # The Standard Algorithm calibrates ONLY on the initial graph (G_0).
        W_inf_init = np.linalg.matrix_power(W_curr, 100)
        pi_init = W_inf_init[0, :]
        D_static = 1.0 / (self.n_agents * pi_init + 1e-10)

        # --- INITIALIZATION ---
        X_std = np.zeros((self.n_agents, self.d))
        X_pipe = np.zeros((self.n_agents, self.d))

        # --- PIPELINED ESTIMATION VARIABLES ---
        # Shadow variable to track the changing eigenvectors
        V_shadow = np.eye(self.n_agents)

        # Default initialization (uniform) - will be overwritten after warm-up
        pi_active = np.ones(self.n_agents) / self.n_agents

        # Network Size Detection State (Bootstrap with known N)
        n_detected = float(self.n_agents)

        # Handover / Fading Logic
        T_est = 30  # Estimation window length
        T_fade = 10  # Fading transition length
        fade_counter = 0
        pi_old = pi_active.copy()
        pi_target = pi_active.copy()

        # Perturbation Probability
        p_perturb = 1e-4

        # Metrics
        cost_std_hist = []
        cost_pipe_hist = []
        cons_std_hist = []
        cons_pipe_hist = []

        # ==========================
        # PHASE 0: WARM-UP (Fixed Topology)
        # ==========================
        # Run estimation for T_est rounds on the FIXED initial topology.
        # This effectively calibrates the pipelined estimator to G0.

        for _ in range(T_est):
            # NO Topology Evolution here.
            # We assume the network is static during the "calibration/warm-up" phase.

            # Pipeline Estimation (V_shadow update only)
            V_shadow = np.matmul(W_curr, V_shadow)

        # --- FORCE HANDOVER AFTER WARM-UP ---
        # Use the result of the warm-up to set the initial D for the main loop.
        pi_warmup = np.diag(V_shadow)
        pi_active = pi_warmup.copy()
        pi_target = pi_warmup.copy()
        pi_old = pi_warmup.copy()

        # Reset shadow for the actual experiment
        V_shadow = np.eye(self.n_agents)

        # ==========================
        # PHASE 1: MAIN EXPERIMENT
        # ==========================
        for k in range(self.n_rounds):

            # 1. Evolve Topology
            # Now the graph starts changing.
            Adj, W_curr = self.evolve_topology(Adj, W_curr, p_perturb)

            # 2. STANDARD ALGORITHM
            # Uses D_static (calibrated to G0)
            self.update_step(X_std, W_curr, D_static)

            # 3. PIPELINED ALGORITHM

            # --- A) Shadow Estimation Update ---
            V_shadow = np.matmul(W_curr, V_shadow)

            # --- B) Check for Reset / Handover ---
            if (k + 1) % T_est == 0:
                pi_new_estimate = np.diag(V_shadow)
                pi_old = pi_active.copy()
                pi_target = pi_new_estimate.copy()
                fade_counter = T_fade
                V_shadow = np.eye(self.n_agents)

            # --- C) Apply Fading ---
            if fade_counter > 0:
                alpha = 1.0 - (fade_counter / T_fade)
                pi_active = (1 - alpha) * pi_old + alpha * pi_target
                fade_counter -= 1
            else:
                pi_active = pi_target

            # --- D) Optimization Step with Dynamic D ---
            # Use n_detected (Network Size Estimation)
            D_dynamic = 1.0 / (n_detected * pi_active + 1e-10)
            D_clipped = np.clip(D_dynamic, 0, self.d_clip_max)

            # --- E) DETECT AGENTS FOR NEXT ROUND ---
            non_zero_mask = D_clipped > 1e-6
            if np.any(non_zero_mask):
                n_new_est = np.sum(1.0 / D_clipped[non_zero_mask])
                n_detected = n_new_est

            self.update_step(X_pipe, W_curr, D_clipped)

            # 4. RECORD METRICS
            cost_std_hist.append(self.compute_cost(X_std))
            cons_std_hist.append(self.compute_consensus(X_std))
            cost_pipe_hist.append(self.compute_cost(X_pipe))
            cons_pipe_hist.append(self.compute_consensus(X_pipe))

        return (
            np.array(cost_std_hist),
            np.array(cost_pipe_hist),
            np.array(cons_std_hist),
            np.array(cons_pipe_hist),
        )

    def evolve_topology(self, Adj, W_curr, p_perturb):
        """Helper to handle random edge flips."""
        flip_mask = np.random.rand(self.n_agents, self.n_agents) < p_perturb
        np.fill_diagonal(flip_mask, 0)

        if np.any(flip_mask):
            Adj_candidate = np.abs(Adj - flip_mask)
            G_cand = nx.from_numpy_array(Adj_candidate, create_using=nx.DiGraph)
            if nx.is_strongly_connected(G_cand):
                Adj = Adj_candidate
                W_curr = adj_to_W(Adj)

        return Adj, W_curr

    def run(self):
        # Define expected output files
        expected_files = [
            "cost_standard.npy",
            "cost_pipelined.npy",
            "consensus_standard.npy",
            "consensus_pipelined.npy",
        ]

        # Check if all files already exist
        if all(os.path.exists(os.path.join(self.save_dir, f)) for f in expected_files):
            print(
                f"Results for seed {self.seed} already exist in {self.save_dir}. Skipping."
            )
            return

        print(f"Running Pipelined Experiment (Seed: {self.seed})...")
        c_std, c_pipe, cn_std, cn_pipe = self.run_single_experiment()

        # Save individual run results
        np.save(os.path.join(self.save_dir, "cost_standard.npy"), c_std)
        np.save(os.path.join(self.save_dir, "cost_pipelined.npy"), c_pipe)
        np.save(os.path.join(self.save_dir, "consensus_standard.npy"), cn_std)
        np.save(os.path.join(self.save_dir, "consensus_pipelined.npy"), cn_pipe)

        print(f"Results for seed {self.seed} saved to: {self.save_dir}")

    def update_step(self, X, W, D):
        # Calculate Local Gradients
        Y = np.zeros_like(X)
        for i in range(self.n_agents):
            grad = loss_grad(
                X[i], self.data_agents[i], self.labels_agents[i], self.l2_strength
            )
            Y[i] = X[i] - self.learning_rate * grad

        # Apply Correction (DT-GO Step)
        D_col = D.reshape(-1, 1)
        Z = X + D_col * (Y - X)

        # Gossip (Mix on current topology)
        X[:] = np.matmul(W, Z)

    def compute_cost(self, X):
        cost = 0
        for i in range(self.n_agents):
            cost += loss(
                X[i], self.data_agents[i], self.labels_agents[i], self.l2_strength
            )
        return cost / self.n_agents

    def compute_consensus(self, X):
        bar_x = np.mean(X, axis=0)
        diff = X - bar_x
        return np.sum(diff**2) / self.n_agents


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_agents", type=int, default=100)
    parser.add_argument("--n_rounds", type=int, default=2500)
    parser.add_argument("--learning_rate", type=float, default=2)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for this run")
    parser.add_argument("--results_folder", type=str, default="results")
    parser.add_argument(
        "--d_clip_max",
        type=float,
        default=5.0,
        help="Maximum value for the correction factor D to prevent exploding gradients.",
    )

    args = parser.parse_args()

    exp = PipelinedExperiment(args)
    exp.run()
