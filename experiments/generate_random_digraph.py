import numpy as np
import networkx as nx


def generate_random_digraph(n, p):
    """Generates a random directed graph with n nodes and edge probability p.

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
