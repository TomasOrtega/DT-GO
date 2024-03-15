import argparse
import os
import sys
import matplotlib.pyplot as plt
import yaml
from absl import app
from absl.flags import argparse_flags
from tqdm import tqdm  # For creating progress bars
import networkx as nx  # NetworkX for working with graphs
import numpy as np  # NumPy for numerical operations



# Custom function to generate random directed graphs
from generate_random_digraph import generate_random_digraph
from add_delays_to_graph import add_delays_to_graph

n_agents = 4
p = 0.2
lam = 0.6

# Obtain a strongly connected digraph
G = generate_random_digraph(n_agents, p)

pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, nodelist=range(n_agents), node_color='r', node_shape='o', node_size=100)
nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
# Add labels to each node
labels = {node: node for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=12)
plt.show()

# Add delays
Gp, virtual_nodes = add_delays_to_graph(G, lam, n_agents)

# Plot the graph, with delay nodes as squares and agents as circles
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, nodelist=range(n_agents), node_color='r', node_shape='o', node_size=100)
nx.draw_networkx_nodes(G, pos, nodelist=range(n_agents, n_agents + virtual_nodes), node_color='b', node_shape='s', node_size=200)
nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
# Add labels to each node
labels = {node: node for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=12)
plt.show()

