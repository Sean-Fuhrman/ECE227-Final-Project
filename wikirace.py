#!/usr/bin/env python3
"""
Light-weight **wikiracing** benchmark for a directed Wikipedia link graph.

Key features
------------
* Works natively with ``nx.DiGraph``                           (successors = clicks)
* Re-usable “strategy” functions registered in a dict          (add your own!)
* Clean command-line interface + progress bar via **tqdm**
* Deterministic randomness with ``--seed`` for reproducibility
* Results written once to a CSV (one row = one test)            (easy to analyse)
"""

from __future__ import annotations
import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import networkx as nx
from tqdm import tqdm   # type: ignore

###############################################################################
# Strategy helpers
###############################################################################


def random_pick(
    G: nx.DiGraph,
    *,
    current: int,
    previous: int | None,
    goal: int,
    rng: random.Random,
) -> int:
    """
    Choose a successor uniformly at random, but occasionally let
    the “player” click the browser’s *Back* button and return to
    the previous page.
    """
    neighbors: List[int] = list(G.successors(current))  # outgoing links

    # allow going “backwards” once in a while
    if previous is not None:
        neighbors.append(previous)

    # Greedy win: click the goal if we see it
    if goal in neighbors:
        return goal

    return rng.choice(neighbors)


###############################################################################
# Strategy registry
###############################################################################

Strategy = Callable[
    [nx.DiGraph, int, int | None, int, random.Random], int
]  # (G, current, previous, goal, RNG) → next

STRATEGIES: Dict[str, Strategy] = {
    "random": random_pick,
    # plug-in more strategies here, e.g. "bfs": bfs_pick
}

###############################################################################
# Core benchmark
###############################################################################


def run_wikirace(
    G: nx.DiGraph,
    *,
    strategy_name: str,
    n_tests: int,
    max_hops: int,
    rng: random.Random,
) -> List[int]:
    """
    Play *n_tests* games and return the hop counts.
    """
    if strategy_name not in STRATEGIES:
        sys.exit(f"❌  Unknown strategy '{strategy_name}'. "
                 f"Available: {', '.join(STRATEGIES)}")

    pick_next = STRATEGIES[strategy_name]
    node_list: Sequence[int] = list(G.nodes)
    N = len(node_list)

    hop_counts: List[int] = []

    for i in tqdm(range(n_tests), desc="Tests", unit="game"):
        start_id = "859791" # Fixed start from center of BFS
        #grab start node from the list
        start = G.nodes[start_id]
        
        #some fixed random goal 
        goal  = node_list[((i * 14342)+ N // 2) % N]

        cur, prev = start, None
        hops = 0

        while cur != goal and hops < max_hops:
            nxt = pick_next(G, current=cur, previous=prev, goal=goal, rng=rng)
            prev, cur = cur, nxt
            hops += 1

        hop_counts.append(hops if cur == goal else max_hops)

    return hop_counts


###############################################################################
# CLI
###############################################################################


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wikiracing benchmark on a DiGraph")
    p.add_argument("--graphml", type=Path, default="data/wiki_slice_subset.graphml",
                   help="Path to the GraphML file (DiGraph)")
    p.add_argument("-s", "--strategy", default="random",
                   help="Navigation strategy to test "
                        f"({', '.join(STRATEGIES)})")
    p.add_argument("-n", "--n-tests", type=int, default=10,
                   help="Number of start/goal pairs to test")
    p.add_argument("--max-hops", type=int, default=1_000,
                   help="Safety cap to stop endless walks")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for reproducible runs")
    p.add_argument("-o", "--out", type=Path, default="wikirace_results.csv",
                   help="CSV file to store results")
    return p.parse_args()


###############################################################################
# Entry-point
###############################################################################


def main() -> None:
    args = parse_args()

    rng = random.Random(args.seed)

    print("📖  Loading graph …")
    G: nx.DiGraph = nx.read_graphml(args.graphml)          # always DiGraph
    print(f"   → {G.number_of_nodes():,} nodes / "
          f"{G.number_of_edges():,} edges")

    hops = run_wikirace(
        G,
        strategy_name=args.strategy,
        n_tests=args.n_tests,
        max_hops=args.max_hops,
        rng=rng,
    )

    avg = sum(hops) / len(hops)
    print(f"\n✅  Finished {args.n_tests} tests. "
          f"Average hops: {avg:.2f}")

    # ------------------------------------------------------------------ CSV --
    with args.out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "n_tests", "avg_hops", "all_hops"])
        writer.writerow([args.strategy, args.n_tests, f"{avg:.2f}",
                         " ".join(map(str, hops))])

    print(f"💾  Results saved to {args.out.resolve()}")


if __name__ == "__main__":
    main()
