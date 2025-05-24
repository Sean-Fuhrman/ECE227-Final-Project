#!/usr/bin/env python3
"""
find_shortest_paths.py

Compute all-pairs shortest paths for a GraphML graph.

Usage examples
--------------
# 1) Stream results to the terminal (unweighted):
python find_shortest_paths.py data/combined_wikilink.graphml

# 2) Save to a file, using the edge attribute "weight" as the distance metric:
python find_shortest_paths.py data/combined_wikilink.graphml \
    --output paths.tsv --weight weight
"""
import sys
import argparse
import networkx as nx
from pathlib import Path
from typing import Optional, TextIO
from tqdm import tqdm

def write_paths(
    G: nx.Graph,
    fh: TextIO,
    weight: Optional[str] = None,
) -> None:
    """
    Iterate over every source->target pair and dump one shortest path per line.

    Each line:  source<TAB>target<TAB>node1,node2,...,nodeK
    """
    for src, paths in tqdm(nx.all_pairs_shortest_path(G)):
        for tgt, path in paths.items():
            # Skip trivial 1-node paths (src == tgt) if you don’t want them:
            if src == tgt: continue
            fh.write(f"{src}\t{tgt}\t{','.join(path)}\n")

def main(input_path: str, output_path: Optional[str], weight: Optional[str]) -> None:
    if not Path(input_path).exists():
        sys.exit(f"❌  File not found: {input_path}")

    print(f"📥  Loading graph from {input_path} ...", file=sys.stderr)
    G = nx.read_graphml(input_path)
    print(f"   → {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges", file=sys.stderr)

    # Open an output filehandle (stdout by default)
    handle: TextIO = open(output_path, "w") if output_path else sys.stdout
    try:
        write_paths(G, handle, weight)
    finally:
        if handle is not sys.stdout:
            handle.close()
            print(f"💾  Results written to {output_path}", file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute all-pairs shortest paths for a GraphML graph.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="Path to the .graphml file")
    parser.add_argument(
        "-o", "--output", metavar="FILE.tsv",
        help="Write results to TSV instead of printing to stdout",
    )
    parser.add_argument(
        "-w", "--weight", metavar="EDGE_ATTR",
        help="Edge attribute to treat as weight (omit for unweighted BFS)",
    )
    args = parser.parse_args()

    main(args.input, args.output, args.weight)
