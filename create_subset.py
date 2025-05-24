#!/usr/bin/env python3
"""
Make a 100 000-node subgraph from a (possibly large) directed GraphML,
preserving the average out-degree ≈ E/N of the original.
"""
import sys, random, argparse
from pathlib import Path
import networkx as nx

import random
import networkx as nx


def sample_connected_subgraph(
    G: nx.DiGraph,
    target_nodes: int = 100_000,
    strong: bool = False,
    max_retries: int = 50,
    seed: int = 42,
) -> nx.DiGraph:
    """
    Return a subgraph with exactly `target_nodes` nodes that is either
    weakly or strongly connected **and** whose edge-to-node ratio matches
    the original graph in expectation.

    strong=False  → weakly connected (recommended for Wikirace)
    strong=True   → strongly connected (may take many retries)
    """
    rng = random.Random(seed)
    orig_density = G.number_of_edges() / G.number_of_nodes()
    target_edges = int(round(orig_density * target_nodes))

    # Pre-index edges for quick random access
    all_edges = list(G.edges())

    for attempt in range(1, max_retries + 1):
        # ── 1.  Pick a random seed node and snowball outwards ───────────────
        seed_node = rng.choice(tuple(G.nodes()))
        seen = {seed_node}
        frontier = [seed_node]

        while frontier and len(seen) < target_nodes:
            print(f"↺  Attempt {attempt}: expanding frontier of {len(frontier):,} nodes", end="\r")
            new_frontier = []
            for u in frontier:
                # Expand neighbours *ignoring* direction for coverage speed
                for nbr in G.successors(u):
                    if nbr not in seen:
                        seen.add(nbr)
                        new_frontier.append(nbr)
                        if len(seen) == target_nodes:
                            break
                if len(seen) == target_nodes:
                    break
            frontier = new_frontier

        if len(seen) < target_nodes:
            # Rare – ran out of reachable nodes; try a new seed
            continue

        # ── 2.  Build an *induced* subgraph on that node set ────────────────
        H = G.subgraph(seen).copy()

        # ── 3.  Top-up edges to match target density if needed ──────────────
        if H.number_of_edges() < target_edges:
            rng.shuffle(all_edges)
            for u, v in all_edges:
                if u in H and v in H and not H.has_edge(u, v):
                    H.add_edge(u, v)
                    if H.number_of_edges() >= target_edges:
                        break

        # ── 4.  Connectivity check ─────────────────────────────────────────
        is_conn = (
            nx.is_strongly_connected(H) if strong else nx.is_weakly_connected(H)
        )
        if is_conn:
            print(
                f"✅  Connected subgraph found on attempt {attempt} "
                f"({H.number_of_nodes():,} nodes, {H.number_of_edges():,} edges)"
            )
            return H

        print(f"↺  Attempt {attempt}: subgraph not {'strongly' if strong else 'weakly'} connected")

    raise RuntimeError(
        f"Failed to find a {'strongly' if strong else 'weakly'} connected "
        f"subgraph of {target_nodes:,} nodes after {max_retries} attempts"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, help="Source GraphML (directed)")
    ap.add_argument("-o", "--output", type=Path,
                    default=Path("data/combined_wikilink_subset.graphml"))
    ap.add_argument("-n", "--nodes", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    G = nx.read_graphml(args.input)
    if not G.is_directed():
        G = G.to_directed()

    print(
        f"📥  Loaded {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges "
        f"(avg out-deg ≈ {G.number_of_edges()/G.number_of_nodes():.2f})",
        file=sys.stderr,
    )

    if G.number_of_nodes() < args.nodes:
        sys.exit("Graph too small for requested sample size.")

    H = sample_connected_subgraph(
        G,
        target_nodes=args.nodes,
        strong=False,          # set True if you really need strong SCC
        seed=args.seed)

    nx.write_graphml(H, args.output)
    print(
        f"📦  Subgraph: {H.number_of_nodes():,} nodes, {H.number_of_edges():,} edges "
        f"(avg out-deg ≈ {H.number_of_edges()/H.number_of_nodes():.2f}) → {args.output}",
        file=sys.stderr,
    )

if __name__ == "__main__":
    main()
