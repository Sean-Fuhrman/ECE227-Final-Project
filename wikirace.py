#%%
import sys
import random
import argparse
from pathlib import Path

import networkx as nx
from tqdm import tqdm  # optional, for a progress bar

def test_strategy(
    G: nx.Graph,
    strategy: str = "random",
    n_tests: int = 10
) -> None:
    """
    Test a strategy for wikiracing.

    Parameters:
    - G: The graph to test.
    - strategy: The strategy to use (e.g., "random").
    - n_tests: Number of tests to run.
    """
    node_list = list(G.nodes())
    N = len(node_list)

    for i in range(n_tests):
        print(f"\nTest {i+1}/{n_tests} using strategy '{strategy}' "
              f"on graph with {G.number_of_nodes():,} nodes "
              f"and {G.number_of_edges():,} edges.")

        # pick start & goal by index
        start_id = node_list[i % N]
        goal_id  = node_list[(i + N//2) % N]

        # **HERE** is the fix: look up the attr dict on G.nodes[…]
        start_label = G.nodes[start_id].get("title", start_id)
        goal_label  = G.nodes[goal_id].get("title",  goal_id)

        print(f"  → Starting wikirace from node: {start_label!r}")
        print(f"  → Goal node for this test:   {goal_label!r}")

test_strategy(
    G=nx.read_graphml("data/combined_wikilink_subset.graphml"),
    strategy="random",
    n_tests=10
)
# %%
