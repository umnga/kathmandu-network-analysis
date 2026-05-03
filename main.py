import osmnx as ox
import matplotlib.pyplot as plt
import networkx as nx

# Download Kathmandu's road network
G = ox.graph_from_place("Kathmandu, Nepal", network_type="drive")

# Basic info
print(f"Number of intersections: {G.number_of_nodes()}")
print(f"Number of road segments: {G.number_of_edges()}")

# Plot it
# ox.plot_graph(G)
# plt.show()

# Convert to undirected for analysis
G_undirected = G.to_undirected()

# Get the largest connected component
largest_component = max(nx.connected_components(G_undirected), key=len)
G_main = G_undirected.subgraph(largest_component).copy()

print(f"\nAfter isolating largest connected component:")
print(f"Intersections: {G_main.number_of_nodes()}")
print(f"Road segments: {G_main.number_of_edges()}")

# Find articulation points
articulation_points = list(nx.articulation_points(G_main))
print(f"\nNumber of articulation points: {len(articulation_points)}")

# Find bridges
bridges = list(nx.bridges(G_main))
print(f"Number of bridges: {len(bridges)}")

# Get positions of all nodes
node_colors = []
for node in G_main.nodes():
    if node in articulation_points:
        node_colors.append("red")
    else:
        node_colors.append("white")

# Plot with articulation points highlighted in red
ox.plot_graph(G_main, node_color=node_colors, node_size=5, 
              edge_linewidth=0.5, bgcolor="black")