import sys
from pathlib import Path

try:
    import osmnx as ox
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parent
    preferred = (
        project_root
        / ".venv"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    candidates = [preferred]
    candidates.extend(sorted((project_root / ".venv" / "lib").glob("python*/site-packages")))
    venv_site = next((p for p in candidates if p.exists()), None)
    if venv_site is not None:
        sys.path.insert(0, str(venv_site))
        import osmnx as ox
    else:
        raise

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import networkx as nx
import random
import numpy as np
import pandas as pd

# ============================================================
# PHASE 1: Graph Preparation
# ============================================================

ox.settings.use_cache = True
ox.settings.log_console = False

print("=" * 60)
print("Kathmandu Road Vulnerability Analysis")
print("=" * 60)
print("\n[Phase 1] Downloading Kathmandu road network...")

G = ox.graph_from_place("Kathmandu, Nepal", network_type="drive")
G_undirected = G.to_undirected()
largest_component_nodes = max(nx.connected_components(G_undirected), key=len)
G_main = G_undirected.subgraph(largest_component_nodes).copy()
G_simple = nx.Graph(G_main)

print(f"  Nodes: {G_main.number_of_nodes()}")
print(f"  Edges: {G_main.number_of_edges()}")

PLACE_NAME = "Kathmandu, Nepal"
FRACTION_TO_REMOVE = 0.20
STEP_SIZE = max(1, int(G_main.number_of_nodes() * 0.01))

# ============================================================
# PHASE 2: Attack Simulations
# ============================================================

print("\n[Phase 2] Running attack simulations...")


def simulate_attack(graph, nodes_to_remove, step_size):
    g = graph.copy()
    initial_lcc_size = float(g.number_of_nodes())
    initial_efficiency = nx.global_efficiency(g)
    lcc_curve = [1.0]
    efficiency_curve = [1.0]
    for idx in range(0, len(nodes_to_remove), step_size):
        chunk = nodes_to_remove[idx: idx + step_size]
        for node in chunk:
            if g.has_node(node):
                g.remove_node(node)
        if g.number_of_nodes() > 0:
            lcc_size = len(max(nx.connected_components(g), key=len))
            current_efficiency = nx.global_efficiency(g)
        else:
            lcc_size = 0
            current_efficiency = 0.0
        lcc_curve.append(lcc_size / initial_lcc_size)
        efficiency_curve.append(
            current_efficiency / initial_efficiency if initial_efficiency > 0 else 0.0
        )
    return lcc_curve, efficiency_curve


def static_degree_attack(graph, fraction=0.20, step_size=84):
    print("  Simulating static degree attack...")
    nodes_by_degree = sorted(graph.degree(), key=lambda x: x[1], reverse=True)
    num_to_remove = int(graph.number_of_nodes() * fraction)
    nodes_to_remove = [n for n, _ in nodes_by_degree[:num_to_remove]]
    return simulate_attack(graph, nodes_to_remove, step_size)


def sequential_degree_attack(graph, fraction=0.20, step_size=84):
    print("  Simulating sequential degree attack...")
    g = graph.copy()
    initial_lcc_size = float(g.number_of_nodes())
    initial_efficiency = nx.global_efficiency(g)
    lcc_curve = [1.0]
    efficiency_curve = [1.0]
    num_to_remove = int(g.number_of_nodes() * fraction)
    steps_needed = int(np.ceil(num_to_remove / step_size))
    for _ in range(steps_needed):
        if g.number_of_nodes() == 0:
            break
        for _ in range(step_size):
            if g.number_of_nodes() == 0:
                break
            node_to_remove = max(g.degree(), key=lambda x: x[1])[0]
            g.remove_node(node_to_remove)
        if g.number_of_nodes() > 0:
            lcc_size = len(max(nx.connected_components(g), key=len))
            current_efficiency = nx.global_efficiency(g)
        else:
            lcc_size = 0
            current_efficiency = 0.0
        lcc_curve.append(lcc_size / initial_lcc_size)
        efficiency_curve.append(
            current_efficiency / initial_efficiency if initial_efficiency > 0 else 0.0
        )
    return lcc_curve, efficiency_curve


def sequential_betweenness_attack(graph, fraction=0.20, step_size=84, k_approx=100):
    print(f"  Simulating sequential betweenness attack (k={k_approx}) — this is slow...")
    g = graph.copy()
    initial_lcc_size = float(g.number_of_nodes())
    initial_efficiency = nx.global_efficiency(g)
    lcc_curve = [1.0]
    efficiency_curve = [1.0]
    critical_nodes_removed = []
    num_to_remove = int(g.number_of_nodes() * fraction)
    steps_needed = int(np.ceil(num_to_remove / step_size))
    for step in range(steps_needed):
        if g.number_of_nodes() < 2:
            break
        print(f"    Step {step + 1}/{steps_needed}", end="\r")
        for _ in range(step_size):
            if g.number_of_nodes() < 2:
                break
            betweenness = nx.betweenness_centrality(g, k=k_approx, normalized=True)
            node_to_remove = max(betweenness, key=betweenness.get)
            if len(critical_nodes_removed) < 10:
                critical_nodes_removed.append(node_to_remove)
            g.remove_node(node_to_remove)
        if g.number_of_nodes() > 0:
            lcc_size = len(max(nx.connected_components(g), key=len))
            current_efficiency = nx.global_efficiency(g)
        else:
            lcc_size = 0
            current_efficiency = 0.0
        lcc_curve.append(lcc_size / initial_lcc_size)
        efficiency_curve.append(
            current_efficiency / initial_efficiency if initial_efficiency > 0 else 0.0
        )
    print()
    return lcc_curve, efficiency_curve, critical_nodes_removed


def random_attack(graph, fraction=0.20, step_size=84, runs=5):
    print(f"  Simulating random failure ({runs} runs)...")
    num_to_remove = int(graph.number_of_nodes() * fraction)
    all_lcc_runs = []
    all_eff_runs = []
    for i in range(runs):
        print(f"    Run {i + 1}/{runs}", end="\r")
        g_nodes = list(graph.nodes())
        random.shuffle(g_nodes)
        nodes_to_remove = g_nodes[:num_to_remove]
        lcc_curve, eff_curve = simulate_attack(graph, nodes_to_remove, step_size)
        all_lcc_runs.append(lcc_curve)
        all_eff_runs.append(eff_curve)
    print()
    avg_lcc = np.mean(np.array(all_lcc_runs), axis=0)
    avg_eff = np.mean(np.array(all_eff_runs), axis=0)
    return avg_lcc.tolist(), avg_eff.tolist()


lcc_static_degree, eff_static_degree = static_degree_attack(
    G_main, fraction=FRACTION_TO_REMOVE, step_size=STEP_SIZE
)
lcc_seq_degree, eff_seq_degree = sequential_degree_attack(
    G_main, fraction=FRACTION_TO_REMOVE, step_size=STEP_SIZE
)
lcc_seq_betweenness, eff_seq_betweenness, sim_critical_nodes = sequential_betweenness_attack(
    G_main, fraction=FRACTION_TO_REMOVE, step_size=STEP_SIZE
)
lcc_random, eff_random = random_attack(
    G_main, fraction=FRACTION_TO_REMOVE, step_size=STEP_SIZE, runs=5
)

# Resilience summary
def find_collapse_point(data, threshold=0.5):
    num_steps = len(data)
    for i, value in enumerate(data):
        if value < threshold:
            percent_removed = (i / (num_steps - 1)) * (FRACTION_TO_REMOVE * 100)
            return f"{percent_removed:.2f}%"
    return f"Not reached within {FRACTION_TO_REMOVE * 100:.0f}% removal"


print("\n--- Resilience Summary (50% Collapse Threshold) ---")
print(f"{'Attack Profile':<35} | {'LCC Collapse':<22} | {'Efficiency Collapse'}")
print("-" * 85)
summary_data = {
    "Random Failure": (find_collapse_point(lcc_random), find_collapse_point(eff_random)),
    "Static Degree Attack": (find_collapse_point(lcc_static_degree), find_collapse_point(eff_static_degree)),
    "Sequential Degree Attack": (find_collapse_point(lcc_seq_degree), find_collapse_point(eff_seq_degree)),
    "Sequential Betweenness Attack": (find_collapse_point(lcc_seq_betweenness), find_collapse_point(eff_seq_betweenness)),
}
for name, (lcc_pt, eff_pt) in summary_data.items():
    print(f"{name:<35} | {lcc_pt:<22} | {eff_pt}")

# Attack simulation plot
print("\n  Saving attack simulation plot...")
plt.style.use("dark_background")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), dpi=200)
x_axis = np.linspace(0, FRACTION_TO_REMOVE * 100, len(lcc_static_degree))

ax1.plot(x_axis, lcc_random, "c--", linewidth=2, label="Random Failure")
ax1.plot(x_axis, lcc_static_degree, "y-", linewidth=2, label="Static Degree Attack")
ax1.plot(x_axis, lcc_seq_degree, color="orange", linestyle="-", linewidth=2.5, label="Sequential Degree Attack")
ax1.plot(x_axis, lcc_seq_betweenness, "r-", linewidth=2.5, label="Sequential Betweenness Attack")
ax1.set_xlabel("Nodes Removed (%)", fontsize=13)
ax1.set_ylabel("LCC Size (% of Original)", fontsize=13)
ax1.set_title("Network Fragmentation under Attack", fontsize=15, fontweight="bold")
ax1.legend(fontsize=11)
ax1.grid(True, linestyle="--", alpha=0.3)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax1.set_xlim(0, FRACTION_TO_REMOVE * 100)

ax2.plot(x_axis, eff_random, "c--", linewidth=2, label="Random Failure")
ax2.plot(x_axis, eff_static_degree, "y-", linewidth=2, label="Static Degree Attack")
ax2.plot(x_axis, eff_seq_degree, color="orange", linestyle="-", linewidth=2.5, label="Sequential Degree Attack")
ax2.plot(x_axis, eff_seq_betweenness, "r-", linewidth=2.5, label="Sequential Betweenness Attack")
ax2.set_xlabel("Nodes Removed (%)", fontsize=13)
ax2.set_ylabel("Normalized Global Efficiency", fontsize=13)
ax2.set_title("Network Efficiency Degradation", fontsize=15, fontweight="bold")
ax2.legend(fontsize=11)
ax2.grid(True, linestyle="--", alpha=0.3)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax2.set_xlim(0, FRACTION_TO_REMOVE * 100)

fig.tight_layout(pad=3.0)
plt.savefig("kathmandu_attack_simulations.png")
plt.close()
print("  Saved: kathmandu_attack_simulations.png")

# ============================================================
# PHASE 3: Articulation Points and Bridges
# ============================================================
# These are the graph-theoretic core of the project:
# - Articulation points: intersections whose removal disconnects the graph
# - Bridges: roads with no alternate route (single path between endpoints)

print("\n[Phase 3] Finding articulation points and bridge edges...")

articulation_points = set(nx.articulation_points(G_simple))
bridges = list(nx.bridges(G_simple))

print(f"  Articulation points (critical intersections): {len(articulation_points)}")
print(f"  Bridge edges (no-alternate roads): {len(bridges)}")

# Betweenness centrality for all nodes (k=200 approximation)
print("  Computing node betweenness centrality (k=200)...")
betweenness_k200 = nx.betweenness_centrality(G_simple, k=200, normalized=True)

# Build articulation point DataFrame
print("  Computing component splits for each articulation point...")
articulation_rows = []
for node in articulation_points:
    node_data = G_main.nodes[node]
    degree = G_simple.degree(node)
    bet = betweenness_k200.get(node, 0.0)
    g_tmp = G_simple.copy()
    g_tmp.remove_node(node)
    comps_after = nx.number_connected_components(g_tmp)
    articulation_rows.append({
        "OSM_Node_ID": node,
        "Latitude": node_data.get("y", np.nan),
        "Longitude": node_data.get("x", np.nan),
        "Degree": degree,
        "Betweenness_Centrality": bet,
        "Components_After_Removal": comps_after,
    })

df_articulation = (
    pd.DataFrame(articulation_rows)
    .sort_values(["Components_After_Removal", "Betweenness_Centrality"], ascending=[False, False])
    .reset_index(drop=True)
)
df_articulation.to_csv("kathmandu_articulation_points.csv", index=False)

print("\n  Top 20 Articulation Points (Critical Intersections):")
print(df_articulation.head(20).to_markdown(index=False))

# Bridge edge betweenness
print("\n  Computing edge betweenness centrality (k=200)...")
edge_betweenness_k200 = nx.edge_betweenness_centrality(G_simple, k=200, normalized=True)

bridge_rows = []
for (u, v) in bridges:
    u_data = G_main.nodes[u]
    v_data = G_main.nodes[v]
    eb = edge_betweenness_k200.get((u, v), edge_betweenness_k200.get((v, u), 0.0))
    bridge_rows.append({
        "Node_U": u,
        "Node_V": v,
        "Edge_Betweenness": eb,
        "U_Latitude": u_data.get("y", np.nan),
        "U_Longitude": u_data.get("x", np.nan),
        "V_Latitude": v_data.get("y", np.nan),
        "V_Longitude": v_data.get("x", np.nan),
    })

df_bridges = pd.DataFrame(bridge_rows)
if not df_bridges.empty:
    df_bridges = df_bridges.sort_values("Edge_Betweenness", ascending=False).reset_index(drop=True)
df_bridges.to_csv("kathmandu_bridge_edges.csv", index=False)

print("\n  Top 20 Bridge Edges (Roads With No Alternate Route):")
print(df_bridges.head(20).to_markdown(index=False))

# ============================================================
# PHASE 7: Composite Vulnerability Score
# ============================================================

print("\n[Phase 7] Computing composite vulnerability scores...")


def minmax_norm(series):
    s_min, s_max = float(series.min()), float(series.max())
    if s_max == s_min:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - s_min) / (s_max - s_min)

df_final = df_articulation.copy()

df_final["Norm_Betweenness"]            = minmax_norm(df_final["Betweenness_Centrality"])
df_final["Norm_Components"]             = minmax_norm(df_final["Components_After_Removal"])
df_final["Norm_Degree"]                 = minmax_norm(df_final["Degree"])

# Weighted composite score
# Betweenness: 35% — structural centrality (how many paths go through this node)
# Components after removal: 35% — how badly the graph splits
# Degree: 30% — local connectivity loss
# Betweenness: 35% — structural centrality (how many paths go through this node)
df_final["Vulnerability_Score"] = (
    0.35 * df_final["Norm_Betweenness"]
    + 0.35 * df_final["Norm_Components"]
    + 0.30 * df_final["Norm_Degree"]
)

df_final_ranked = df_final.sort_values("Vulnerability_Score", ascending=False).reset_index(drop=True)

final_cols = [
    "OSM_Node_ID", "Latitude", "Longitude", "Degree",
    "Betweenness_Centrality", "Components_After_Removal",
    "Norm_Betweenness", "Norm_Components", "Norm_Degree",
    "Vulnerability_Score",
]
df_final_ranked[final_cols].to_csv("kathmandu_final_vulnerability_rankings.csv", index=False)

print("\n  Final Top 20 Critical Nodes by Composite Vulnerability Score:")
print(df_final_ranked[final_cols].head(20).to_markdown(index=False))

# Top 10 nodes for map overlays
top10_critical_ids = df_final_ranked["OSM_Node_ID"].head(10).tolist()

# ============================================================
# PHASE 8: Spatial Visualization (1x2 Topological Layout)
# ============================================================

print("\n[Phase 8] Generating spatial vulnerability maps...")


def node_xy(node_id):
    nd = G_main.nodes[node_id]
    return nd.get("x", np.nan), nd.get("y", np.nan)


def split_xy(nodes):
    xs, ys = [], []
    for n in nodes:
        if not G_main.has_node(n):
            continue
        x, y = node_xy(n)
        if not (np.isnan(x) or np.isnan(y)):
            xs.append(x)
            ys.append(y)
    return xs, ys


plt.style.use("dark_background")
fig, axes = plt.subplots(1, 2, figsize=(22, 10), dpi=200)
fig.patch.set_facecolor("#0d0d0d")

BASE_KWARGS = dict(show=False, close=False, bgcolor="#0d0d0d", edge_color="#2a2a2a", node_size=0)
for ax in axes.flat:
    ox.plot_graph(G_main, ax=ax, **BASE_KWARGS)

# Panel 1: Articulation points and top-ranked structural nodes.
ax1 = axes[0]
art_x, art_y, art_s = [], [], []
for n in articulation_points:
    if not G_main.has_node(n):
        continue
    x, y = node_xy(n)
    if np.isnan(x) or np.isnan(y):
        continue
    art_x.append(x)
    art_y.append(y)
    art_s.append(max(4.0, G_simple.degree(n) * 0.6))

ax1.scatter(
    art_x,
    art_y,
    c="red",
    s=art_s,
    alpha=0.65,
    zorder=3,
    label=f"Articulation Points ({len(articulation_points)})",
)

top10_x, top10_y = split_xy(top10_critical_ids)
ax1.scatter(
    top10_x,
    top10_y,
    c="yellow",
    s=150,
    marker="*",
    alpha=1.0,
    zorder=5,
    label="Top 10 Vulnerable Nodes",
)
ax1.set_title("Topological Vulnerability: Articulation Points", color="white", fontsize=13, pad=10)
ax1.legend(loc="upper right", fontsize=9, facecolor="#1a1a1a", edgecolor="#555", labelcolor="white")

# Panel 2: Bridge edges, with the strongest bridge edges emphasized.
ax2 = axes[1]
for u, v in bridges:
    if not (G_main.has_node(u) and G_main.has_node(v)):
        continue
    xu, yu = node_xy(u)
    xv, yv = node_xy(v)
    if any(np.isnan(c) for c in [xu, yu, xv, yv]):
        continue
    ax2.plot([xu, xv], [yu, yv], color="orange", linewidth=1.5, alpha=0.7, zorder=3)

for _, row in df_bridges.head(20).iterrows():
    u, v = int(row["Node_U"]), int(row["Node_V"])
    if not (G_main.has_node(u) and G_main.has_node(v)):
        continue
    xu, yu = node_xy(u)
    xv, yv = node_xy(v)
    if any(np.isnan(c) for c in [xu, yu, xv, yv]):
        continue
    ax2.plot([xu, xv], [yu, yv], color="red", linewidth=3, alpha=0.95, zorder=4)

ax2.set_title(f"Topological Vulnerability: Bridge Edges ({len(bridges)} total)", color="white", fontsize=13, pad=10)
legend_handles = [
    mpatches.Patch(color="orange", label=f"All bridge edges ({len(bridges)})"),
    mpatches.Patch(color="red", label="Top 20 by edge betweenness"),
]
ax2.legend(handles=legend_handles, loc="upper right", fontsize=9, facecolor="#1a1a1a", edgecolor="#555", labelcolor="white")


def simulate_and_plot_fracture(graph):
    graph_undirected = nx.Graph(graph.to_undirected())
    if df_bridges.empty:
        raise ValueError("Bridge edge table is empty; cannot simulate a structural fracture.")

    bridge_row = df_bridges.iloc[0]
    start_node = int(bridge_row["Node_U"])
    end_node = int(bridge_row["Node_V"])

    if not (graph_undirected.has_node(start_node) and graph_undirected.has_node(end_node)):
        raise ValueError(f"Top bridge edge nodes ({start_node}, {end_node}) are not present in the graph.")

    baseline_efficiency = nx.global_efficiency(graph_undirected)

    fractured_graph = graph_undirected.copy()
    if not fractured_graph.has_edge(start_node, end_node):
        raise nx.NetworkXError(
            f"The targeted bridge edge ({start_node}, {end_node}) is not present in the undirected graph."
        )
    fractured_graph.remove_edge(start_node, end_node)

    if not isinstance(fractured_graph, nx.MultiDiGraph):
        fractured_graph = nx.MultiDiGraph(fractured_graph)

    components = [set(component) for component in nx.weakly_connected_components(fractured_graph)]
    giant_component = max(components, key=len) if components else set()
    stranded_components = [component for component in components if component != giant_component]
    stranded_intersections = sum(len(component) for component in stranded_components)
    component_count = len(components)

    fractured_efficiency = nx.global_efficiency(fractured_graph)
    efficiency_loss_pct = 0.0 if baseline_efficiency <= 0 else ((baseline_efficiency - fractured_efficiency) / baseline_efficiency) * 100.0

    node_colors = ["#2C3E50" if node in giant_component else "#E74C3C" for node in fractured_graph.nodes()]
    edge_colors = ["#2C3E50" if u in giant_component and v in giant_component else "#E74C3C" for u, v in fractured_graph.edges()]
    edge_widths = [1.0 if (u in giant_component and v in giant_component) else 1.4 for u, v in fractured_graph.edges()]

    fig2, ax = ox.plot_graph(
        fractured_graph,
        node_color=node_colors,
        node_size=8,
        edge_color=edge_colors,
        edge_linewidth=edge_widths,
        bgcolor="#111111",
        show=False,
        close=False,
    )

    x_start, y_start = fractured_graph.nodes[start_node]["x"], fractured_graph.nodes[start_node]["y"]
    x_end, y_end = fractured_graph.nodes[end_node]["x"], fractured_graph.nodes[end_node]["y"]
    ax.plot(
        [x_start, x_end],
        [y_start, y_end],
        color="#F1C40F",
        linewidth=6,
        solid_capstyle="round",
        zorder=10,
    )

    ax.set_title("Fracture Simulation: Severing the Top Bridge Edge", color="white", fontsize=13, pad=12)
    legend_handles = [
        Line2D([0], [0], color="#2C3E50", linewidth=6, label="Largest Connected Component"),
        Line2D([0], [0], color="#E74C3C", linewidth=6, label="Isolated Components"),
        Line2D([0], [0], color="#F1C40F", linewidth=6, label="Severed Bridge Edge"),
    ]
    legend = ax.legend(
        handles=legend_handles,
        loc="lower left",
        frameon=True,
        facecolor="#1A1A1A",
        edgecolor="#555555",
        framealpha=0.95,
    )
    plt.setp(legend.get_texts(), color="white")

    ax.text(
        0.02,
        0.02,
        "\n".join(
            [
                f"Components Created: {component_count}",
                f"Stranded Intersections: {stranded_intersections}",
                f"Network Efficiency Loss: {efficiency_loss_pct:.2f}%",
            ]
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        color="white",
        bbox=dict(facecolor="#1A1A1A", edgecolor="#555555", boxstyle="round,pad=0.45", alpha=0.95),
    )
    ax.set_axis_off()
    fig2.savefig("kathmandu_network_fracture.png", dpi=300, bbox_inches="tight", facecolor="#111111")
    plt.close(fig2)

    return {
        "start_node": start_node,
        "end_node": end_node,
        "component_count": component_count,
        "giant_component_size": len(giant_component),
        "stranded_intersections": stranded_intersections,
        "isolated_component_sizes": [len(component) for component in stranded_components],
        "network_efficiency_loss_pct": efficiency_loss_pct,
    }


fig.suptitle(
    "Kathmandu Road Network Vulnerability Analysis: Topological Traits",
    color="white",
    fontsize=17,
    fontweight="bold",
    y=1.01,
)
fig.tight_layout(pad=2.0)
plt.savefig("kathmandu_vulnerability_maps.png", bbox_inches="tight", facecolor="#0d0d0d")
plt.close()
print("  Saved: kathmandu_vulnerability_maps.png")

fracture_summary = simulate_and_plot_fracture(G)
print("  Saved: kathmandu_network_fracture.png")
print(
    "  Fracture summary — "
    f"Components: {fracture_summary['component_count']}, "
    f"Stranded intersections: {fracture_summary['stranded_intersections']}, "
    f"Efficiency loss: {fracture_summary['network_efficiency_loss_pct']:.2f}%"
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE — Output Files")
print("=" * 60)
print("  kathmandu_attack_simulations.png       — LCC & efficiency curves")
print("  kathmandu_vulnerability_maps.png       — 2-panel spatial maps")
print("  kathmandu_articulation_points.csv      — all critical intersections")
print("  kathmandu_bridge_edges.csv             — all no-alternate roads")
print("  kathmandu_final_vulnerability_rankings.csv — composite ranked nodes")
print("  kathmandu_network_fracture.png         — severed bridge fracture simulation")
print()
print(f"  Articulation points found:             {len(articulation_points)}")
print(f"  Bridge edges (no-alternate roads):     {len(bridges)}")
print(f"  Top critical node (OSM ID):            {df_final_ranked['OSM_Node_ID'].iloc[0]}")
print(f"  Top node vulnerability score:          {df_final_ranked['Vulnerability_Score'].iloc[0]:.4f}")
print("=" * 60)