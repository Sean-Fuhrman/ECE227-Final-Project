import networkx as nx

# Load the GraphML file
graph_file = "data/combined_wikilink_graph_0_100.graphml"
G = nx.read_graphml(graph_file)

print(f"Loaded graph with {len(G.nodes)} nodes and {len(G.edges)} edges")