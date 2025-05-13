import pandas as pd
import networkx as nx
import gzip
import os

input_file = "data/enwiki.wikilink_graph.2018-03-01.csv.gz"
output_dir = "data/chunks/"
os.makedirs(output_dir, exist_ok=True)

start_chunk = 0
chunksize = 100000

def clean_chunk(chunk):
    return chunk.applymap(lambda x: x.strip() if isinstance(x, str) else x)

if __name__ == '__main__':
    with gzip.open(input_file, 'rt', encoding='utf-8') as f:
        reader = pd.read_csv(
            f,
            sep='\t',
            header=None,
            names=["from_page_id", "from_page_title", "to_page_id", "to_page_title"],
            engine='python',
            chunksize=chunksize,
            on_bad_lines='skip'
        )

        for chunk_num, chunk in enumerate(reader):
            if chunk_num < start_chunk: continue  # Skip already processed chunks
    
            chunk = clean_chunk(chunk)
            G = nx.DiGraph()

            for _, row in chunk.iterrows():
                G.add_node(row["from_page_id"], title=row["from_page_title"])
                G.add_node(row["to_page_id"], title=row["to_page_title"])
                G.add_edge(row["from_page_id"], row["to_page_id"])

            # Save each chunk as a separate file
            chunk_file = f"{output_dir}/chunk_{chunk_num}.graphml"
            nx.write_graphml(G, chunk_file)
            print(f"Saved chunk {chunk_num} to {chunk_file}")