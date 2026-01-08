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
        self.n_experiments = args.n_experiments
        self.save_dir = os.path.join(args.results_folder, "pipelined_evolving")

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
        np.fill_diagonal(Adj, 1.0) # Ensure self-loops
        
        # Calculate Initial Weight Matrix
        W_curr = adj_to_W(Adj)

        # --- PRE-CALCULATE STATIC D (Baseline) ---
        # The Standard Algorithm calibrates ONLY on the initial graph (G_0).
        # It assumes the topology never changes.
        W_inf_init = np.linalg.matrix_power(W_curr, 100)
        pi_init = W_inf_init[0, :]
        D_static = 1.0 / (self.n_agents * pi_init + 1e-10)

        # --- INITIALIZATION ---
        X_std = np.zeros((self.n_agents, self.d))
        X_pipe = np.zeros((self.n_agents, self.d))

        # --- PIPELINED ESTIMATION VARIABLES ---
        # Shadow variable to track the changing eigenvectors
        V_shadow = np.eye(self.n_agents)
        
        # Current estimate of the stationary distribution
        pi_active = np.ones(self.n_agents) / self.n_agents 
        
        # Handover / Fading Logic
        T_est = 30          # Estimation window length
        T_fade = 10         # Fading transition length
        fade_counter = 0    
        pi_old = pi_active.copy()
        pi_target = pi_active.copy()
        
        # Metrics
        cost_std_hist = []
        cost_pipe_hist = []
        cons_std_hist = []
        cons_pipe_hist = []

        # Perturbation Probability
        # Probability that any given edge flips status (exists <-> not exists) per round.
        # 1e-4 with 100 agents (~10,000 edges) means ~1 edge change(s) per round.
        p_perturb = 1e-4

        for k in range(self.n_rounds):
            
            # ==========================
            # 0. EVOLVING TOPOLOGY
            # ==========================
            # Generate a random mask of edges to flip
            # We strictly avoid modifying the diagonal (self-loops)
            flip_mask = (np.random.rand(self.n_agents, self.n_agents) < p_perturb)
            np.fill_diagonal(flip_mask, 0)

            if np.any(flip_mask):
                # Apply changes: 1 becomes 0, 0 becomes 1 (Absolute difference mimics XOR)
                Adj = np.abs(Adj - flip_mask)
                
                # Re-generate the stochastic weight matrix based on the new topology
                W_curr = adj_to_W(Adj)

            # ==========================
            # 1. STANDARD ALGORITHM
            # ==========================
            # Mixing happens on the CURRENT graph (W_curr), but correction uses OLD weights (D_static)
            self.update_step(X_std, W_curr, D_static)
            
            # ==========================
            # 2. PIPELINED ALGORITHM
            # ==========================
            
            # --- A) Shadow Estimation Update (Tracking the topology) ---
            V_shadow = np.matmul(W_curr, V_shadow)
            
            # --- B) Check for Reset / Handover ---
            if (k + 1) % T_est == 0:
                # Harvest estimate from diagonal of shadow matrix
                pi_new_estimate = np.diag(V_shadow)
                
                # Setup Fading
                pi_old = pi_active.copy()
                pi_target = pi_new_estimate.copy()
                fade_counter = T_fade
                
                # Reset Shadow Estimator
                V_shadow = np.eye(self.n_agents)

            # --- C) Apply Fading ---
            if fade_counter > 0:
                alpha = 1.0 - (fade_counter / T_fade)
                pi_active = (1 - alpha) * pi_old + alpha * pi_target
                fade_counter -= 1
            else:
                pi_active = pi_target

            # --- D) Optimization Step with Dynamic D ---
            # Updates correction weights based on the active estimate of the topology
            D_dynamic = 1.0 / (self.n_agents * pi_active + 1e-10)
            self.update_step(X_pipe, W_curr, D_dynamic)

            # ==========================
            # RECORD METRICS
            # ==========================
            cost_std_hist.append(self.compute_cost(X_std))
            cons_std_hist.append(self.compute_consensus(X_std))
            
            cost_pipe_hist.append(self.compute_cost(X_pipe))
            cons_pipe_hist.append(self.compute_consensus(X_pipe))

        return (
            np.array(cost_std_hist), 
            np.array(cost_pipe_hist),
            np.array(cons_std_hist),
            np.array(cons_pipe_hist)
        )

    def run(self):
        print(f"Running Evolving Topology Experiment ({self.n_experiments} runs)...")

        all_cost_std, all_cost_pipe = [], []
        all_cons_std, all_cons_pipe = [], []

        for i in tqdm(range(self.n_experiments)):
            c_std, c_pipe, cn_std, cn_pipe = self.run_single_experiment()
            all_cost_std.append(c_std)
            all_cost_pipe.append(c_pipe)
            all_cons_std.append(cn_std)
            all_cons_pipe.append(cn_pipe)

        # Average over runs
        avg_cost_std = np.mean(all_cost_std, axis=0)
        avg_cost_pipe = np.mean(all_cost_pipe, axis=0)
        avg_cons_std = np.mean(all_cons_std, axis=0)
        avg_cons_pipe = np.mean(all_cons_pipe, axis=0)

        np.save(os.path.join(self.save_dir, "cost_standard.npy"), avg_cost_std)
        np.save(os.path.join(self.save_dir, "cost_pipelined.npy"), avg_cost_pipe)
        np.save(os.path.join(self.save_dir, "consensus_standard.npy"), avg_cons_std)
        np.save(os.path.join(self.save_dir, "consensus_pipelined.npy"), avg_cons_pipe)
        print("Done. Results saved to:", self.save_dir)

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
    parser.add_argument("--n_rounds", type=int, default=3000) 
    parser.add_argument("--learning_rate", type=float, default=2)
    parser.add_argument("--n_experiments", type=int, default=1)
    parser.add_argument("--results_folder", type=str, default="results")
    args = parser.parse_args()

    exp = PipelinedExperiment(args)
    exp.run()