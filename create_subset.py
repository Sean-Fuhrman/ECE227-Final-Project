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

import random
import networkit as nk

def stratified_degree_sample(G: nk.Graph, frac: float = 0.1,
                             seed: int | None = None) -> nk.Graph:
    """
    Stratified degree sampling for NetworKit graphs.

    Parameters
    ----------
    G     : nk.Graph
        The original graph (directed or undirected).
    frac  : float, default 0.1
        Fraction of nodes to keep **per degree value** (0 < frac ≤ 1).
    seed  : int | None, default None
        RNG seed for reproducibility.

    Returns
    -------
    H : nk.Graph
        Induced sub-graph whose degree distribution mirrors G
        up to a constant scaling factor.
    """
    if not (0 < frac <= 1):
        raise ValueError("`frac` must be in (0, 1].")

    rng = random.Random(seed)

    # 1) bucket nodes by (total) degree
    buckets: dict[int, list[int]] = {}
    deg_fun = (lambda u: G.degreeOut(u) + G.degreeIn(u))
    for u in G.iterNodes():
        d = deg_fun(u)
        buckets.setdefault(d, []).append(u)

    # 2) sample the same fraction from every bucket
    keep: list[int] = []
    for nodes in buckets.values():
        k = max(1, round(frac * len(nodes)))
        k = min(k, len(nodes))        # just in case
        keep.extend(rng.sample(nodes, k))

    # # 4) Build the slice directly in NetworkX (IDs stay original) -----------
    H = nx.DiGraph()
    H.add_nodes_from(keep)

    ball_set = set(keep)          # for fast membership
    for u in keep:
        for v in G.iterNeighbors(u):
            if v in ball_set:
                H.add_edge(u, v)

    return H


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
    
    #Save the pruned graph to a new file
    # pruned_graph_path = args.graph.replace(".nkbg", "_pruned.nkbg")
    # nk.graphio.writeGraph(nkG, pruned_graph_path, nk.Format.NetworkitBinary)
    # print(f"Pruned graph saved to: {pruned_graph_path}")
    
    #Save the pruned graph to a GraphML file
    # pruned_graphml_path = args.graph.replace(".nkbg", "_pruned.graphml")
    # nxG = nx.DiGraph()
    # nxG.add_nodes_from(nkG.iterNodes())
    # for u in nkG.iterNodes():
    #     for v in nkG.iterNeighbors(u):
    #         nxG.add_edge(u, v)
    # nx.write_graphml(nxG, pruned_graphml_path)
    # print(f"Pruned graph saved to: {pruned_graphml_path}")
    
    # # 3) Pick seed & collect radius-k ball ----------------------------------
    # attempt = 0
    # while True:
    #     seed = rng.randrange(nkG.numberOfNodes())
    #     if out_deg(nkG, seed) == 0:          # still a sink? resample
    #         continue
    #     ball = k_ball(nkG, seed, args.radius)
    #     print(f"Attempt #{attempt+1}: seed {seed} → "
    #           f"{len(ball):,} nodes in radius-{args.radius} ball")
    #     if args.size // 2 <= len(ball) <= args.size * 2:
    #         break
    #     attempt += 1
    #     if attempt > 25:
    #         raise RuntimeError("Couldn’t hit requested size ±2×; "
    #                            "increase --radius or --size.")

    # print(f"Seed page id: {seed}")
    # print(f"Slice size  : {len(ball):,} nodes (target ≈{args.size:,})")
    
    # stratified sampling
    # print("Performing stratified degree sampling…")
    # nkG = stratified_degree_sample(nkG, frac=0.1, seed=args.seed)
    # print(f"Stratified sample size: {nkG.number_of_nodes():,} nodes "
    #         f"({nkG.number_of_edges():,} edges)")
    # convert to undirected
    print("Converting to undirected graph…")
    nkG = nk.graphtools.toUndirected(nkG)
    print("running Louvain community detection…")
    louvain = nk.community.PLM(nkG, turbo=False )  # gamma≈1 → “standard” Louvain
    louvain.run()
    part = louvain.getPartition()                              # nk.structures.Partition

    try:
        cid_iter = part.subsetSizeMap().keys()        # NetworKit ≥10
    except AttributeError:                            # older fallback
        cid_iter = (
            cid for cid in range(part.upperBound())
            if part.subsetSize(cid) > 0
        )

    comm_dict: dict[int, list[int]] = {}
    for cid in cid_iter:
        members = list(part.getMembers(cid))          # C++ → Python list
        comm_dict[int(cid)] = members                 # ensure JSON-serialisable

    out_file = args.out.replace(".graphml", "_louvain.json")
    import json
    # --- 3. Write to disk ----------------------------------------------------
    out_path = Path(out_file)
    out_path.write_text(json.dumps(comm_dict))

    print(f"✔ Louvain communities written to {out_path.resolve()}")

    # # print("Converted to NetworkX (no renumbering).")

 

    # Graph the in / out degree distribution on log-log scale as line graph on two axes
    # print("Graphing in / out degree distribution…")
    # import matplotlib.pyplot as plt
    # in_degrees = [nkG.degreeIn(u) for u in nkG.iterNodes()]
    # out_degrees = [nkG.degreeOut(u) for u in nkG.iterNodes()]
    # plt.figure(figsize=(10, 6))
    # plt.loglog(sorted(set(in_degrees)), 
    #            [in_degrees.count(d) for d in sorted(set(in_degrees))], 
    #            label='In-Degree', marker='o')
    # plt.loglog(sorted(set(out_degrees)),
    #             [out_degrees.count(d) for d in sorted(set(out_degrees))], 
    #             label='Out-Degree', marker='x')
    # plt.title("In / Out Degree Distribution")
    # plt.xlabel("Degree")
    # plt.ylabel("Count")
    # plt.legend()
    # plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    # plt.savefig("stratified degree_distribution.png")
    # print("Graph saved as degree_distribution.png")
    
    
    #  # 5) Attach titles -------------------------------------------------------
    # titles_raw = pd.read_parquet(args.titles).set_index("id")["title"]
    # titles = {int(k): str(v) for k, v in titles_raw.items()}

    # nx.set_node_attributes(G,
    #     {n: {"title": titles.get(n, "")} for n in G.nodes})

    # # ensure every attribute is GraphML-safe
    # for _, data in G.nodes(data=True):
    #     for k, v in data.items():
    #         data[k] = clean(v)

    # # 6) Save ---------------------------------------------------------------
    # nx.write_graphml(G, args.out)
    # print(f"✅  Saved slice → {args.out}  "
    #       f"({G.number_of_nodes():,} nodes / {G.number_of_edges():,} edges)")

if __name__ == "__main__":
    main()
