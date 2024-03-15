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
