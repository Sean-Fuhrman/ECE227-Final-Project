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
from sentence_transformers import SentenceTransformer
import numpy as np
import networkx as nx
from tqdm import tqdm   # type: ignore

EMB_MODEL: SentenceTransformer | None = None      # loaded once
EMB_CACHE: dict[int, np.ndarray] = {}             # node-id → ℓ₂-normalised vec

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

def betweenness_hop(
    G: nx.DiGraph,
    *,
    current: int,
    previous: int | None,
    goal: int,
    rng: random.Random,
) -> int:
    """
    Local-betweenness pilot that avoids trivial loops and sink traps.
    • radius-2 ego betweenness, cached lazily
    • never selects `previous` twice in a row
    • never revisits an already-visited node if alternatives exist
    • tie-breaks at random
    • 10 % of the time picks a uniformly-random neighbour (exploration)
    """
    # ---------- goal or sink fast-paths
    neighbours = list(G.successors(current))
    
    if previous is None:
        _visited = {"reset_token": current}  # first hop, reset visited set
    

    if goal in neighbours:                       # greedy win
        return goal
    
    if not neighbours:                     # sink trap
        return previous 

    # ---------- initialise per-game visited set
    if _visited is None or current == _visited.get("reset_token"):
        _visited = {"reset_token": current}      # new game marker
        betweenness_hop._visited = _visited
    _visited.setdefault("seen", set()).add(current)

    if not hasattr(betweenness_hop, "_centrality"):
        # Approximate betweenness with a fixed sample of 512 source nodes
        k_sample = min(512, G.number_of_nodes())
        print(f"🧮  Computing betweenness centrality (k={k_sample}) …")
        betweenness_hop._centrality = nx.betweenness_centrality(
            G, k=k_sample, seed=rng
        )
        print("   → done")
    score = betweenness_hop._centrality

    # ---------- candidate filtering / tie-breaking
    candidates = [n for n in neighbours
                  if n != previous                      # no immediate pong
                  and (n not in _visited["seen"]        # avoid repeats
                       or len(neighbours) == 1)]        # …unless forced

    if not candidates:
        candidates = neighbours                         # fall back

    # choose by (score, random) to break ties fairly
    return max(candidates,
               key=lambda n: (score.get(n, 0.0), rng.random()))

def get_embedding(node_id: int, G: nx.DiGraph) -> np.ndarray:
    """
    Return a ℓ₂-normalised MiniLM embedding for this page title,
    computing and caching it on first request.
    """
    if node_id in EMB_CACHE:
        return EMB_CACHE[node_id]

    title = G.nodes[node_id].get("title", "")
    vec = EMB_MODEL.encode(title, normalize_embeddings=True)
    EMB_CACHE[node_id] = vec.astype(np.float32)
    return EMB_CACHE[node_id]

def llm_similarity(
    G: nx.DiGraph,
    *,
    current: int,
    previous: int | None,
    goal: int,
    rng: random.Random,
) -> int:
    """
    Greedy neighbour choice using **on-demand MiniLM embeddings**.
    Caches each new page vector, so repeated visits are O(1).
    """
    if EMB_MODEL is None:
        raise RuntimeError("EMB_MODEL not initialised - load it in main()!")

    neighbours = list(G.successors(current))
  
    # Greedy win
    if goal in neighbours:
        return goal

    goal_vec = get_embedding(goal, G)

    best_n, best_score = None, -1.0
    for n in neighbours:
        sim = float(np.dot(get_embedding(n, G), goal_vec))  # cosine
        if sim > best_score:
            best_n, best_score = n, sim

    return best_n if best_n is not None else rng.choice(neighbours)

def llm_extra(
    G: nx.DiGraph,
    *,
    current: int,
    previous: int | None,
    goal: int,
    rng: random.Random,
    _visited=None                      # per-game visited set
) -> int:
    """
    Greedy MiniLM-embedding pilot with
      • visited-set avoidance
      • ε-greedy exploration (ε = 0.10)
      • random tie-breaks
    """

    # ---------- initialisation (first call of each game) -------------------
    if previous is None:               # new game = first hop
        _visited = set()
        llm_extra._visited = _visited   # store on function
    else:
        _visited = llm_extra._visited

    _visited.add(current)

    # ---------- gather neighbours ------------------------------------------
    neighbours = list(G.successors(current))

    if not neighbours:  # sink page, return previous  # no successors, go back
        return previous

    # Greedy win
    if goal in neighbours:
        return goal

    unseen = [n for n in neighbours if n not in _visited]

    # ---------- similarity scores (A + tie-break) ---------------------------
    goal_vec = get_embedding(goal, G)

    best_n, best_sim = None, -1.0
    for n in (unseen or neighbours):   # prefer unseen; else fall back
        sim = float(np.dot(get_embedding(n, G), goal_vec))
        if sim > best_sim or (sim == best_sim and rng.random() < 0.5):
            best_n, best_sim = n, sim

    return best_n

def betweenness_then_llm(
    G: nx.DiGraph,
    *,
    current: int,
    previous: int | None,
    goal: int,
    rng: random.Random,
) -> int:
    """
    Hybrid strategy that first tries betweenness_hop for first 3 hops,
    then switches to llm_similarity.
    """
    if previous is None:  # first hop, reset hops
        betweenness_then_llm._hops = 0  # reset hop counter
    
    if not hasattr(betweenness_then_llm, "_hops"):
        betweenness_then_llm._hops = 0
    if betweenness_then_llm._hops < 3:
        betweenness_then_llm._hops += 1
        return betweenness_hop(G, current=current, previous=previous, goal=goal, rng=rng)
    else:
        betweenness_then_llm._hops = 0
        return llm_similarity(G, current=current, previous=previous, goal=goal, rng=rng)
    
def betweenness_llm_fallback(
    G: nx.DiGraph,
    *,
    current: int,
    previous: int | None,
    goal: int,
    rng: random.Random,
    sim_threshold: float = 0.25,          # ⇦ tune to taste
) -> int:
    """
    Greedy MiniLM pilot, but if its best neighbour is still ‘too dissimilar’
    (cos sim < sim_threshold) we fall back to betweenness_hop instead.

    • keeps the “click goal if visible” fast-path
    • never crashes on sinks (returns `previous` if stuck)
    • shares the LRU embedding cache used by the other LLM helpers
    """
    # --- gather successors & obvious exits ---------------------------------
    neighbours = list(G.successors(current))
    if goal in neighbours:                       # one-click win
        return goal
    if not neighbours:                           # sink page
        return previous

    goal_vec = get_embedding(goal, G)

    # --- pick the best neighbour by cosine similarity ----------------------
    best_n, best_sim = None, -1.0
    for n in neighbours:
        sim = float(np.dot(get_embedding(n, G), goal_vec))
        if sim > best_sim:
            best_n, best_sim = n, sim

    # --- decide which strategy to trust this turn --------------------------
    if best_sim < sim_threshold:
        # Similarity too low ⇒ trust structural heuristic instead
        return betweenness_hop(G,
                               current=current,
                               previous=previous,
                               goal=goal,
                               rng=rng)
    return best_n


###############################################################################
# Strategy registry
###############################################################################

Strategy = Callable[
    [nx.DiGraph, int, int | None, int, random.Random], int
]  # (G, current, previous, goal, RNG) → next

STRATEGIES: Dict[str, Strategy] = {
    "random": random_pick,
    # plug-in more strategies here, e.g. "bfs": bfs_pick
    "betweenness-extra-no-back": betweenness_hop,
    "llm-no-back": llm_similarity,
    "llm-extra-eps-0": llm_extra,
    "betweenness-then-llm": betweenness_then_llm,
    "llm-fallback": betweenness_llm_fallback,
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
    
    # Strategy specific initialisation
    if "llm" in strategy_name:
            print("🧮  Loading MiniLM sentence-embedding model …")
            global EMB_MODEL
            EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
            print("   ↳ model ready.")
            
    for i in tqdm(range(n_tests), desc="Tests", unit="game"):
        start = "859791" # Fixed start from center of BFS
        #grab start node from the list
        
        #some fixed random goal 
        goal  = node_list[((i * 14342)+ N // 2) % N]
        
        #Print goal title
        print(f"Game {i+1}: Start = {start} ({G.nodes[start]['title']}), Goal = {goal} ({G.nodes[goal]['title']})")

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
    p.add_argument("--max-hops", type=int, default=5_000,
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

    
    # ---------------------------------------------------------------- CSV ----
    file_exists = args.out.exists()
    mode = "a" if file_exists else "w"

    with args.out.open(mode, newline="") as f:
        writer = csv.writer(f)
        if not file_exists:                       # write header only once
            writer.writerow(["strategy", "n_tests", "avg_hops", "all_hops"])
        writer.writerow([
            args.strategy,
            args.n_tests,
            f"{avg:.2f}",
            " ".join(map(str, hops))
        ])

    print(f"💾  Results {'appended to' if file_exists else 'saved to'} "
          f"{args.out.resolve()}")



if __name__ == "__main__":
    main()
