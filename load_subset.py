#%%
import networkx as nx

G = nx.read_graphml("data/degree_stratified_slice.graphml", node_type=int)

print(f"Loaded {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")


# print titles with top 10 edges
for node, data in G.nodes(data=True):
    print(f"Node {node}: {data.get('title', 'No title')}")
    #%%
# Create log-log degree distribution plot of in and out degrees
import matplotlib.pyplot as plt
in_degrees = [G.in_degree(n) for n in G.nodes()]
out_degrees = [G.out_degree(n) for n in G.nodes()]
plt.figure(figsize=(10, 6))
plt.loglog(sorted(set(in_degrees)), 
           [in_degrees.count(d) for d in sorted(set(in_degrees))], 
           label='In-Degree', marker='o')
plt.loglog(sorted(set(out_degrees)),
              [out_degrees.count(d) for d in sorted(set(out_degrees))], 
              label='Out-Degree', marker='x')
plt.title("In / Out Degree Distribution")
plt.xlabel("Degree")
plt.ylabel("Count")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.savefig("stratified_degree_distribution.png")
# %%
