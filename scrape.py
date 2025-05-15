import pandas as pd
import networkx as nx
import gzip
import os

input_file = "data/enwiki.wikilink_graph.2018-03-01.csv.gz"
output_dir = "data/chunks/"
os.makedirs(output_dir, exist_ok=True)

chunk_size = 1000
chunk_limit = 5

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
            chunksize=chunk_size,
            on_bad_lines='skip'
        )

        for chunk_num, chunk in enumerate(reader):
            if chunk_num > chunk_limit: break
    
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