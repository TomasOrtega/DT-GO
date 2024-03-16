import argparse
import os
import sys
import matplotlib.pyplot as plt
import yaml
import pickle
from absl import app
from absl.flags import argparse_flags
from tqdm import tqdm  # For creating progress bars
import networkx as nx  # NetworkX for working with graphs
import numpy as np  # NumPy for numerical operations
import urllib.request  # For downloading files
from sklearn.datasets import load_svmlight_file  # For loading the dataset
import scipy.optimize  # For optimizing the loss function
from log_reg_utils import loss, loss_grad, OPTIMAL_WEIGHTS  # For logistic regression


# Custom function to generate random directed graphs
from generate_random_digraph import generate_random_digraph
from add_delays_to_graph import add_delays_to_graph


class Experiment:
    def __init__(self, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_rounds = args.n_rounds
        self.l2_strength = args.l2_strength
        self.learning_rate = args.learning_rate
        self.lam = args.lam
        self.p = args.p
        self.n_experiments = args.n_experiments
        self.results_folder = args.results_folder
        self.baseline_loss = args.baseline_loss
        self.test_run = args.test_run
        self.n_gossip = args.n_gossip
        self.warm_up_rounds = args.warm_up_rounds

        # Set the current time for asynchronous training
        np.random.seed(args.seed)

        ##################### BEGIN: Good old bookkeeping #########################
        self.runname = self.get_runname()
        self.save_dir = os.path.join(args.results_folder, self.runname)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        # TODO -- use a logger
        with open(os.path.join(self.save_dir, f"{self.runname}_args.yaml"), "w") as f:
            yaml.dump(vars(args), f)
        print(f"Created experiment {self.runname}")
        print(f"Will save to {self.save_dir}")
        ##################### END: Good old bookkeeping #########################

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
        self.l2_strength = 1.0 / n

        # Add a bias term (intercept) to the data
        data = np.hstack((np.ones((n, 1)), data))

        # Define a global model
        self.global_model = np.zeros(d + 1)

        # Set the baseline loss
        self.baseline_loss = args.baseline_loss
        if self.baseline_loss is None:
            # Use a black-box optimizer to find the baseline loss
            self.baseline_loss = scipy.optimize.minimize(
                loss, OPTIMAL_WEIGHTS,
                args=(data, target, self.l2_strength),
                options={"disp": True}
            ).fun

        # Split the dataset
        self.data_agents = np.array_split(data, self.n_agents)
        self.labels_agents = np.array_split(target, self.n_agents)

        self.data, self.target = data, target

        self.experiment_results = None

    def get_runname(self):
        from log_reg_utils import config_dict_to_str
        args = self.args
        runname = config_dict_to_str(vars(args), record_keys=(
            'p', 'lam', 'n_rounds', 'learning_rate', 'n_experiments'))
        return runname

    def save_results(self, save_dir=None):
        if save_dir is None:
            save_dir = self.save_dir
        # Unpack experiment results and save them to disk
        cost, to_mean = self.experiment_results
        np.save(f"{save_dir}/cost.npy", cost)
        np.save(f"{save_dir}/to_mean.npy", to_mean)

    def save_one_round_results(self, cost, to_mean, G, it, save_dir=None):
        if save_dir is None:
            save_dir = self.save_dir
        # Save the results of the experiment
        iteration_folder = f"{save_dir}/experiments/{it}"
        if not os.path.exists(iteration_folder):
            os.makedirs(iteration_folder)
        np.save(f"{iteration_folder}/cost.npy", cost)
        np.save(f"{iteration_folder}/to_mean.npy", to_mean)
        # Save the graph as a pickle file
        pickle.dump(G, open(f"{iteration_folder}/graph.pkl", "wb"))
    
    def load_one_round_results(self, it, save_dir=None):
        if save_dir is None:
            save_dir = self.save_dir
        # Load the results of the experiment
        iteration_folder = f"{save_dir}/experiments/{it}"
        cost = np.load(f"{iteration_folder}/cost.npy")
        to_mean = np.load(f"{iteration_folder}/to_mean.npy")
        # Load the graph as a pickle file
        G = pickle.load(open(f"{iteration_folder}/graph.pkl", "rb"))
        return cost, to_mean, G

    def run_one_graph(self, initial_model):

        # Obtain a strongly connected digraph
        G = generate_random_digraph(self.n_agents, self.p)
        # Add delays
        G, virtual_nodes = add_delays_to_graph(G, self.lam, self.n_agents)

        # Get the adjacency matrix
        adj = nx.adjacency_matrix(G).todense()
        adj = np.array(adj)

        # Assert all self-loops exist for non-virtual nodes
        assert np.all(np.diag(adj[0:self.n_agents]) == 1)

        # Assert weight matrix is column-stochastic (we transpose later)
        column_sum = np.sum(adj, axis=0)
        A = adj / column_sum
        assert np.all(np.abs(np.sum(A, axis=0) - 1) < 1e-6)

        # Add delays and get virtual_nodes
        # A, virtual_nodes = add_delays_to_graph_v2(A, lam, self.n_agents)
        # assert np all(np.abs(np.sum(A, axis=0) - 1) < 1e-6)

        # Obtain adjustment vector of the graph
        Ainf = np.linalg.matrix_power(A, self.warm_up_rounds)
        adjustment = (Ainf)[:, 0]
        assert np.all(np.abs(A.dot(adjustment) - adjustment) < 1e-6)

        # Use the transpose and change notation for optimization (W * X)
        W = A.T

        # Initialize variables for optimization and tracking
        # X is a matrix with rows all equal to the initial model
        X = np.tile(initial_model, (self.n_agents + virtual_nodes, 1))
        global_cost = np.zeros(self.n_rounds)
        to_mean = np.zeros(self.n_rounds)

        learning_step = self.learning_rate

        # Main optimization loop
        for k in range(self.n_rounds):
            # Compute global cost and consensus
            cost_mean = 0
            for i in range(self.n_agents):
                cost_mean += loss(X[i], self.data_agents[i],
                                  self.labels_agents[i], self.l2_strength)
            global_cost[k] = cost_mean / self.n_agents

            to_mean_cost = 0
            mean_models = np.mean(X[0:self.n_agents, :], axis=0)
            for i in range(self.n_agents):
                to_mean_cost += np.sum((mean_models - X[i]) ** 2)
            to_mean[k] = to_mean_cost / self.n_agents

            # Optimize at every agent
            for i in range(self.n_agents):
                # Compute derivative for optimization
                deriv = loss_grad(X[i], self.data_agents[i],
                                  self.labels_agents[i], self.l2_strength)

                # Update X using the learning step and adjustment vector
                # X[i] -= learning_step * np.sqrt(self.n_agents / (k + 1)) * deriv / (adjustment[i] * self.n_agents)
                X[i] -= learning_step * deriv / (adjustment[i] * self.n_agents)

            # Perform Gossiping for n_gossip iterations
            X = np.dot(np.linalg.matrix_power(W, self.n_gossip), X)

        # Return the results of the experiment
        return global_cost, to_mean, G

    def plot_results(self):
        # Display results for cost suboptimality
        plt.figure()
        cost, mean = self.experiment_results
            
        plt.plot(
            np.arange(1, self.n_rounds + 1), cost - self.baseline_loss, label=f"f - f*"
        )

        plt.xlabel("Round")
        plt.ylabel("Cost suboptimality")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        # Make sure directory exists
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        plt.savefig(f"{self.save_dir}/{self.runname}_cost_suboptimality.pdf")
        if self.args.verbose:
            plt.show()

    def run_experiment(self):
        print(f"Running experiment {self.runname}...")
        
        cost_all = []
        to_mean_all = []
        for it in tqdm(range(self.n_experiments)):
            # Run the experiment for the current parameter and iteration, and store the results
            cost, to_mean, G = self.run_one_graph(self.global_model.copy())
            if not self.test_run:
                self.save_one_round_results(cost, to_mean, G, it)
            cost_all.append(cost)
            to_mean_all.append(to_mean)

        # Average results over the experiments and store them
        self.experiment_results = (np.mean(cost_all, axis=0), np.mean(to_mean_all, axis=0))

        if self.test_run:
            print("Test run complete. Exiting.")
            sys.exit(0)
        else:
            self.save_results()
            print(f"Results saved to {self.save_dir}")

        self.plot_results()

        return self.experiment_results


def parse_args(argv):
    """Parses command line arguments."""
    parser = argparse_flags.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # High-level options.
    parser.add_argument("--verbose", "-V", action="store_true",
                        help="Plot progress and metrics after training.")
    parser.add_argument("--seed", type=int, default=0)

    # Specifying dataset
    # TO-DO: Add support for other datasets

    # Experiment specific args
    parser.add_argument("--n_agents", type=int, default=100,
                        help="Number of agents (i.e. agents, or computing nodes) to use for the experiment")
    parser.add_argument("--n_rounds", type=int, default=1000,
                        help="Number of algorithm rounds to use for the experiment")
    parser.add_argument("--l2_strength", type=float, default=None,
                        help="L2 regularization strength to use for the experiment, defaults to None, set to 1/n, where n is the number of samples")
    parser.add_argument("--n_gossip", type=int, default=1,
                        help="Number of gossip iterations per round")
    parser.add_argument("--warm_up_rounds", type=int, default=1024,
                        help="Number of warm-up rounds to use for the algorithm")
    parser.add_argument("--learning_rate", type=float,
                        default=2, help="Learning rate for the agent")
    parser.add_argument("--lam", type=float, default=0.0,
                        help="Lambda value, controls the distribution of delays in the graph")
    parser.add_argument("--p", type=float, default=1.0,
                        help="p value, controls the probability of an edge existing in the graph")
    parser.add_argument("--n_experiments", type=int, default=1,
                        help="Number of experiments to run for the given parameters")
    parser.add_argument("--results_folder", type=str, default="./results",
                        help="Folder to save the results of the experiment")
    parser.add_argument("--baseline_loss", type=float,
                        default=0.014484174216922262, help="Baseline loss for the experiment")
    parser.add_argument("--test_run", default=False, action='store_true',
                        help="Whether to run a test run of the experiment")

    # Parse arguments.
    args = parser.parse_args(argv[1:])

    return args


def main(args):
    experiment = Experiment(args)
    experiment_results = experiment.run_experiment()


if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
