import osmnx as ox
import matplotlib.pyplot as plt

G = ox.graph_from_place("Kathmandu, Nepal", network_type="drive")

print(f"Number of intersections: {G.number_of_nodes()}")
print(f"Number of road segments: {G.number_of_edges()}")

ox.plot_graph(G)
plt.show()