import glob
import networkx as nx
import os

chunks_dir = "data/chunks/"
output_file = "data/combined_wikilink_graph.graphml"
os.makedirs(chunks_dir, exist_ok=True)

def combine_chunks(output_dir, final_output_file):
    chunk_files = sorted(glob.glob(f"{output_dir}/chunk_*.graphml"))
    combined_graph = nx.DiGraph()

    # Merge graphs
    for chunk_file in chunk_files:
        G = nx.read_graphml(chunk_file)
        combined_graph = nx.compose(combined_graph, G)

        print(f"Combined {chunk_file}")

    nx.write_graphml(combined_graph, final_output_file)

if __name__ == '__main__':
    combine_chunks(chunks_dir, output_file)