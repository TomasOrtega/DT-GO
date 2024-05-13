import numpy as np
import networkx as nx


def add_delays_to_graph(G, lam, N):
    """
    Add delays to a graph's edges.

    Args:
        G (networkx.Graph): Graph object.
        lam (float): Poisson parameter for delay sampling.
        N (int): Number of nodes in the original graph.

    Returns:
        numpy.ndarray: Updated matrix with delays added.
        int: Number of virtual nodes in the graph.
    """

    # Generate the delay counts for all edges at once
    num_delays_matrix = np.random.poisson(lam, size=(N, N))

    # Set the delays to zero for edges that do not exist
    num_delays_matrix[~np.array(nx.to_numpy_array(G), dtype=bool)] = 0

    # Set the main diagonal to 0
    num_delays_matrix[np.eye(N, dtype=bool)] = 0

    # Add new virtual nodes
    total_virtual_nodes = np.sum(num_delays_matrix)
    G.add_nodes_from(range(N, N + total_virtual_nodes))

    # Add virtual nodes and corresponding edges for delays
    virtual_nodes_row_indices, virtual_nodes_col_indices = np.where(
        num_delays_matrix > 0
    )

    virtual_nodes = 0  # Initialize the count of virtual nodes

    for i, j in zip(virtual_nodes_row_indices, virtual_nodes_col_indices):
        num_delays = num_delays_matrix[i][j]
        G.remove_edge(i, j)
        new_nodes = [N + virtual_nodes + k for k in range(num_delays)]
        G.add_edges_from(zip([i] + new_nodes, new_nodes + [j]))
        virtual_nodes += num_delays

    return G, total_virtual_nodes


# Example usage:
# A, virtual_nodes = add_delays_to_graph(A, lam, N)


def generate_random_digraph(n, p):
    """
    Generates a random directed graph with n nodes and edge probability p.

    Args:
        n (int): The number of nodes in the graph.
        p (float): The probability of an edge between any pair of nodes.

    Returns:
        nx.DiGraph: A randomly generated directed graph.

    This function generates a directed graph by first creating an adjacency matrix A
    with edges randomly determined based on the given probability p. It ensures that
    all nodes have self-loops. If the resulting graph is strongly connected, it is
    returned. If not, the function recursively generates a new graph until it is
    strongly connected.
    """

    # Generate an adjacency matrix A for the directed graph with edge probability p
    A = np.random.rand(n, n) < p

    # Ensure all nodes have a self-loop by setting diagonal elements to 1
    np.fill_diagonal(A, True)

    # Create a directed graph object (DiGraph) from the adjacency matrix
    G = nx.DiGraph(A)

    # Check if the graph is strongly connected
    graph_connected = len(list(nx.strongly_connected_components(G))) == 1

    # If the graph is strongly connected, return it
    if graph_connected:
        return G

    # If the graph is not strongly connected, recursively generate a new one
    return generate_random_digraph(n, p)


def graph_to_W(G, n_agents=None):
    """
    Converts a graph to a weight matrix for optimization.

    Args:
        G (networkx.Graph): Graph object.
        n_agents (int): Number of non-virtual agents in the graph.

    Returns:
        numpy.ndarray: Weight matrix for optimization.
    """
    # Get the adjacency matrix
    adj = nx.adjacency_matrix(G).todense()
    adj = np.array(adj)

    n_agents = n_agents if n_agents is not None else G.number_of_nodes()

    # Assert all self-loops exist for non-virtual nodes
    assert np.all(np.diag(adj[0:n_agents]) == 1)

    # Make weight matrix is column-stochastic (we transpose later)
    column_sum = np.sum(adj, axis=0)
    A = adj / column_sum

    # Use the transpose and change notation for optimization (W * X)
    W = A.T
    return W

def change_graph(G, q):
    N = G.number_of_nodes()
    random_matrix = np.random.rand(N, N) < q

    # make sure that the diagonal is all zeros
    np.fill_diagonal(random_matrix, False)

    # iterate through all indices of True values in the random matrix
    for i, j in zip(*np.where(random_matrix)):
        # if edge exists, remove it
        if G.has_edge(i, j):
            G.remove_edge(i, j)
        else:
            G.add_edge(i, j)
    
    return G

def graph_with_errs(G, p_err):
    """
    Add errors to a graph's edges.

    Args:
        G (networkx.Graph): Graph object.
        p_err (float): Probability of an error in an edge.

    Returns:
        networkx.Graph: Graph object with errors added.
    """
    N = G.number_of_nodes()
    random_matrix = np.random.rand(N, N) < p_err

    # make sure that the diagonal is all zeros
    np.fill_diagonal(random_matrix, False)

    # iterate through all indices of True values in the random matrix
    for i, j in zip(*np.where(random_matrix)):
        # if edge exists, remove it
        if G.has_edge(i, j):
            G.remove_edge(i, j)
    
    return G