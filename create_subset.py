#!/usr/bin/env python
"""
extract_slice.py  –  Build a Wiki-race-friendly sub-graph with titles
Compatible with NetworKit 8.x → 10.x (Windows wheels included).

Example:
    python extract_slice.py --size 100000 --radius 3
"""

import argparse, random, time, math
from pathlib import Path

import networkit as nk
from networkit import distance
import pandas as pd
import networkx as nx


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--graph",  default="data/enwiki_2018.nkbg",
                   help="NetworKit .nkbg file")
    p.add_argument("--titles", default="data/node_titles.parquet",
                   help="Parquet with id ↔ title")
    p.add_argument("--size",   type=int, default=100_000,
                   help="Target #nodes in the slice (approx.)")
    p.add_argument("--radius", type=int, default=3,
                   help="Directed BFS depth from the seed page")
    p.add_argument("--seed",   type=int, default=42,
                   help="RNG seed (-1 = random each run)")
    p.add_argument("--out",    default="wiki_slice_subset.graphml",
                   help="Output GraphML file")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def out_deg(G: nk.graph.Graph, u: int) -> int:
    """True out-degree on any NetworKit version."""
    return G.degreeOut(u)


def prune_all_sinks(G: nk.graph.Graph):
    """Iteratively remove nodes whose **out-degree** becomes 0."""
    total = 0
    while True:
        sinks = [u for u in G.iterNodes() if out_deg(G, u) == 0 or G.degreeIn(u) == 0]
        if not sinks:
            break
        for u in sinks:
            G.removeNode(u)
        total += len(sinks)
        print(f"  • removed {len(sinks):,} sinks; "
              f"{G.numberOfNodes():,} nodes remain")
    print(f"Finished pruning ({total:,} total nodes removed).")


def k_ball(G: nk.graph.Graph, src: int, k: int):
    """Directed k-hop neighbourhood of `src` (node IDs are original)."""
    bfs = distance.BFS(G, src, True, True)
    bfs.run()
    dists = bfs.getDistances()
    return [v for v, d in enumerate(dists) if d <= k]


def clean(val):
    """Return a GraphML-safe primitive value."""
    if val is None:
        return ""
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return ""
    return str(val)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    rng  = random.Random(None if args.seed < 0 else args.seed)
    print(f"Using RNG seed: {args.seed}   (random={args.seed < 0})")

    # 1) Load full graph -----------------------------------------------------
    t0 = time.time()
    nkG = nk.graphio.readGraph(args.graph, nk.Format.NetworkitBinary)
    print(f"Loaded full graph: {nkG.numberOfNodes():,} nodes / "
          f"{nkG.numberOfEdges():,} edges   [{time.time() - t0:.1f}s]")

    # 2) Prune every sink ----------------------------------------------------
    print("Pruning sink nodes (out-degree 0) …")
    prune_all_sinks(nkG)

    # 3) Pick seed & collect radius-k ball ----------------------------------
    attempt = 0
    while True:
        seed = rng.randrange(nkG.numberOfNodes())
        if out_deg(nkG, seed) == 0:          # still a sink? resample
            continue
        ball = k_ball(nkG, seed, args.radius)
        print(f"Attempt #{attempt+1}: seed {seed} → "
              f"{len(ball):,} nodes in radius-{args.radius} ball")
        if args.size // 2 <= len(ball) <= args.size * 2:
            break
        attempt += 1
        if attempt > 25:
            raise RuntimeError("Couldn’t hit requested size ±2×; "
                               "increase --radius or --size.")

    print(f"Seed page id: {seed}")
    print(f"Slice size  : {len(ball):,} nodes (target ≈{args.size:,})")

    # 4) Build the slice directly in NetworkX (IDs stay original) -----------
    G = nx.DiGraph()
    G.add_nodes_from(ball)

    ball_set = set(ball)          # for fast membership
    for u in ball:
        for v in nkG.iterNeighbors(u):
            if v in ball_set:
                G.add_edge(u, v)

    print("Converted to NetworkX (no renumbering).")

    # 5) Attach titles -------------------------------------------------------
    titles_raw = pd.read_parquet(args.titles).set_index("id")["title"]
    titles = {int(k): str(v) for k, v in titles_raw.items()}

    nx.set_node_attributes(G,
        {n: {"title": titles.get(n, "")} for n in G.nodes})

    # ensure every attribute is GraphML-safe
    for _, data in G.nodes(data=True):
        for k, v in data.items():
            data[k] = clean(v)

    # 6) Save ---------------------------------------------------------------
    nx.write_graphml(G, args.out)
    print(f"✅  Saved slice → {args.out}  "
          f"({G.number_of_nodes():,} nodes / {G.number_of_edges():,} edges)")


if __name__ == "__main__":
    main()
