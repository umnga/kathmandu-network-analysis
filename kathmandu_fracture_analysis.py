from __future__ import annotations

from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd

try:
    import osmnx as ox
except ModuleNotFoundError:
    import sys

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
    venv_site = next((path for path in candidates if path.exists()), None)
    if venv_site is None:
        raise
    sys.path.insert(0, str(venv_site))
    import osmnx as ox


PROJECT_ROOT = Path(__file__).resolve().parent
PLACE_NAME = "Kathmandu, Nepal"
NETWORK_TYPE = "drive"

GRAPHML_CANDIDATES = [
    PROJECT_ROOT / "kathmandu.graphml",
    PROJECT_ROOT / "kathmandu_network.graphml",
    PROJECT_ROOT / "kathmandu_drive_network.graphml",
]

ARTICULATION_CSV = PROJECT_ROOT / "kathmandu_articulation_points.csv"
BRIDGE_CSV = PROJECT_ROOT / "kathmandu_bridge_edges.csv"
FRACTURE_FIGURE = PROJECT_ROOT / "kathmandu_severe_bottlenecks_fracture.png"

GIANT_COLOR = "#36454F"  # muted dark slate/gray
FRINGE_COLOR = "#DC143C"  # crimson red
SEVERED_EDGE_COLOR = "#FFD400"  # bright yellow
BACKGROUND_COLOR = "#0B0F14"
TEXT_COLOR = "#F4F6F8"
BOX_COLOR = "#111820"
BOX_EDGE_COLOR = "#9AA4AE"


def _coerce_osm_id(value) -> Optional[int]:
    """Convert a CSV value into an exact integer OSM ID.

    Scientific notation such as "1.95048e+09" is parsed with Decimal so the
    resulting node/edge identifiers remain exact integers rather than floats.
    """

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)

    text = str(value).strip().strip('"').strip("'")
    if not text:
        return None

    try:
        return int(Decimal(text))
    except (InvalidOperation, ValueError, OverflowError):
        try:
            return int(float(text))
        except (ValueError, OverflowError):
            raise ValueError(f"Unable to parse OSM identifier: {value!r}") from None


def load_graph() -> nx.Graph:
    """Load the Kathmandu road network from GraphML when available.

    If a saved GraphML file is not present, the graph is downloaded directly
    from OpenStreetMap and then cached to disk for future runs.
    """

    for graphml_path in GRAPHML_CANDIDATES:
        if graphml_path.exists():
            graph = ox.load_graphml(graphml_path)
            return graph

    graph = ox.graph_from_place(PLACE_NAME, network_type=NETWORK_TYPE, simplify=True)
    cache_path = GRAPHML_CANDIDATES[-1]
    try:
        ox.save_graphml(graph, cache_path)
    except Exception:
        pass
    return graph


def load_top_bridge_edge(csv_path: Path) -> tuple[int, int, pd.DataFrame]:
    """Read the bridge CSV and return the highest-ranked bridge edge."""

    if not csv_path.exists():
        raise FileNotFoundError(f"Bridge CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str)
    if df.empty:
        raise ValueError(f"Bridge CSV is empty: {csv_path}")

    if "Edge_Betweenness" in df.columns:
        df["Edge_Betweenness"] = pd.to_numeric(df["Edge_Betweenness"], errors="coerce")
        df = df.sort_values("Edge_Betweenness", ascending=False, na_position="last").reset_index(drop=True)

    df["Node_U"] = df["Node_U"].map(_coerce_osm_id)
    df["Node_V"] = df["Node_V"].map(_coerce_osm_id)

    top_row = df.iloc[0]
    edge_u = int(top_row["Node_U"])
    edge_v = int(top_row["Node_V"])
    return edge_u, edge_v, df


def load_top_articulation_node(csv_path: Path) -> tuple[int, pd.DataFrame]:
    """Read the articulation CSV and return the highest-ranked articulation point."""

    if not csv_path.exists():
        raise FileNotFoundError(f"Articulation CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str)
    if df.empty:
        raise ValueError(f"Articulation CSV is empty: {csv_path}")

    if "Components_After_Removal" in df.columns:
        df["Components_After_Removal"] = pd.to_numeric(
            df["Components_After_Removal"], errors="coerce"
        )
    if "Betweenness_Centrality" in df.columns:
        df["Betweenness_Centrality"] = pd.to_numeric(df["Betweenness_Centrality"], errors="coerce")

    sort_columns = [column for column in ["Components_After_Removal", "Betweenness_Centrality"] if column in df.columns]
    if sort_columns:
        ascending = [False] * len(sort_columns)
        df = df.sort_values(sort_columns, ascending=ascending, na_position="last").reset_index(drop=True)

    df["OSM_Node_ID"] = df["OSM_Node_ID"].map(_coerce_osm_id)
    top_node = int(df.iloc[0]["OSM_Node_ID"])
    return top_node, df


def _simple_undirected(graph: nx.Graph) -> nx.Graph:
    """Return a simple undirected graph with a stable component structure."""

    if graph.is_multigraph():
        undirected_graph = nx.Graph(graph.to_undirected())
    else:
        undirected_graph = graph.to_undirected(as_view=False)
        if not isinstance(undirected_graph, nx.Graph):
            undirected_graph = nx.Graph(undirected_graph)
    return undirected_graph


def _component_metrics(fractured_graph: nx.Graph) -> dict:
    """Compute the exact graph-theoretic impact of a bridge-edge removal."""

    components = [set(component) for component in nx.connected_components(fractured_graph)]
    if not components:
        return {
            "components": [],
            "component_count": 0,
            "giant_component": set(),
            "giant_size": 0,
            "fragment_sizes": [],
            "nodes_isolated": 0,
            "components_created": 0,
        }

    giant_component = max(components, key=len)
    fragment_components = [component for component in components if component is not giant_component]
    fragment_sizes = sorted((len(component) for component in fragment_components), reverse=True)

    return {
        "components": components,
        "component_count": len(components),
        "giant_component": giant_component,
        "giant_size": len(giant_component),
        "fragment_sizes": fragment_sizes,
        "nodes_isolated": sum(fragment_sizes),
        "components_created": max(len(components) - 1, 0),
    }


def visualize_network_fracture(
    graph: nx.Graph,
    edge_u: int,
    edge_v: int,
    output_path: Optional[Path] = FRACTURE_FIGURE,
) -> dict:
    """Remove a bridge edge, visualize the fracture, and return summary metrics."""

    simple_graph = _simple_undirected(graph)
    if not simple_graph.has_edge(edge_u, edge_v):
        if simple_graph.has_edge(edge_v, edge_u):
            edge_u, edge_v = edge_v, edge_u
        else:
            raise nx.NetworkXError(f"Edge ({edge_u}, {edge_v}) is not present in the graph")

    baseline_efficiency = nx.global_efficiency(simple_graph)

    fractured_graph = simple_graph.copy()
    fractured_graph.remove_edge(edge_u, edge_v)

    metrics = _component_metrics(fractured_graph)
    fractured_efficiency = nx.global_efficiency(fractured_graph)
    efficiency_loss_pct = (
        0.0
        if baseline_efficiency <= 0
        else ((baseline_efficiency - fractured_efficiency) / baseline_efficiency) * 100.0
    )

    metrics["baseline_efficiency"] = baseline_efficiency
    metrics["fractured_efficiency"] = fractured_efficiency
    metrics["network_efficiency_loss_pct"] = efficiency_loss_pct

    giant_nodes = metrics["giant_component"]
    node_colors = [GIANT_COLOR if node in giant_nodes else FRINGE_COLOR for node in fractured_graph.nodes()]
    edge_colors = [GIANT_COLOR if u in giant_nodes else FRINGE_COLOR for u, v in fractured_graph.edges()]
    edge_widths = [1.2 for _ in fractured_graph.edges()]

    fig, ax = ox.plot_graph(
        fractured_graph,
        node_color=node_colors,
        node_size=12,
        edge_color=edge_colors,
        edge_linewidth=edge_widths,
        bgcolor=BACKGROUND_COLOR,
        show=False,
        close=False,
    )

    if fractured_graph.has_node(edge_u) and fractured_graph.has_node(edge_v):
        x1, y1 = fractured_graph.nodes[edge_u]["x"], fractured_graph.nodes[edge_u]["y"]
        x2, y2 = fractured_graph.nodes[edge_v]["x"], fractured_graph.nodes[edge_v]["y"]
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=SEVERED_EDGE_COLOR,
            linewidth=5.0,
            solid_capstyle="round",
            zorder=10,
        )

    legend_handles = [
        Line2D([0], [0], color=GIANT_COLOR, linewidth=6, label="Giant Component"),
        Line2D([0], [0], color=FRINGE_COLOR, linewidth=6, label="Isolated Subgraphs"),
        Line2D([0], [0], color=SEVERED_EDGE_COLOR, linewidth=6, label="Severed Bridge Edge"),
    ]
    legend = ax.legend(
        handles=legend_handles,
        loc="lower left",
        frameon=True,
        facecolor=BOX_COLOR,
        edgecolor=BOX_EDGE_COLOR,
        framealpha=0.95,
    )
    plt.setp(legend.get_texts(), color=TEXT_COLOR)

    metric_lines = [
        f"Nodes Isolated: {metrics['nodes_isolated']}",
        f"Components Created: {metrics['components_created']}",
        f"Network Efficiency Loss: {efficiency_loss_pct:.2f}%",
        f"Component Sizes: {metrics['fragment_sizes'] if metrics['fragment_sizes'] else '[0]'}",
    ]
    ax.text(
        0.98,
        0.02,
        "\n".join(metric_lines),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        color=TEXT_COLOR,
        bbox={
            "facecolor": BOX_COLOR,
            "edgecolor": BOX_EDGE_COLOR,
            "boxstyle": "round,pad=0.45",
            "alpha": 0.95,
        },
    )

    ax.set_title(
        f"Kathmandu Road Network Fracture at Bridge Edge ({edge_u}, {edge_v})",
        color=TEXT_COLOR,
        fontsize=16,
        fontweight="bold",
        pad=18,
    )
    ax.set_axis_off()

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=BACKGROUND_COLOR)

    return metrics


def _feature_geometries_to_points(gdf: pd.DataFrame) -> list[tuple[float, float]]:
    """Convert OSM feature geometries into latitude/longitude point pairs."""

    if gdf is None or gdf.empty or "geometry" not in gdf.columns:
        return []

    points: list[tuple[float, float]] = []
    for geometry in gdf.geometry:
        if geometry is None or geometry.is_empty:
            continue
        point = geometry if geometry.geom_type == "Point" else geometry.centroid
        points.append((float(point.y), float(point.x)))
    return points


def _map_features_to_graph_nodes(graph: nx.Graph, place_name: str, tags: dict) -> list[int]:
    """Snap area-based OSM features to the nearest road-network nodes.

    OSMnx delegates nearest-node lookup to SciPy or scikit-learn backends when
    available. With scikit-learn installed, this uses a BallTree under the hood.
    """

    try:
        features = ox.features_from_place(place_name, tags=tags)
    except Exception:
        return []

    points = _feature_geometries_to_points(features)
    if not points:
        return []

    nearest_nodes = ox.distance.nearest_nodes(
        graph,
        X=[longitude for latitude, longitude in points],
        Y=[latitude for latitude, longitude in points],
    )

    if isinstance(nearest_nodes, Iterable) and not isinstance(nearest_nodes, (str, bytes)):
        unique_nodes = list(dict.fromkeys(int(node) for node in nearest_nodes))
    else:
        unique_nodes = [int(nearest_nodes)]
    return unique_nodes


def compute_residential_isolation_count(
    graph: nx.Graph,
    articulation_node: int,
    place_name: str = PLACE_NAME,
) -> dict:
    """Count residential road nodes that lose access to every hospital after deletion."""

    residential_nodes = _map_features_to_graph_nodes(graph, place_name, {"landuse": "residential"})
    hospital_nodes = _map_features_to_graph_nodes(graph, place_name, {"amenity": "hospital"})

    simple_graph = _simple_undirected(graph)
    fractured_graph = simple_graph.copy()
    if fractured_graph.has_node(articulation_node):
        fractured_graph.remove_node(articulation_node)

    hospital_reachable_nodes: set[int] = set()
    for component in nx.connected_components(fractured_graph):
        component_nodes = set(component)
        if component_nodes.intersection(hospital_nodes):
            hospital_reachable_nodes.update(component_nodes)

    isolated_residential_nodes = [
        node for node in residential_nodes if node in fractured_graph and node not in hospital_reachable_nodes
    ]

    return {
        "residential_nodes": residential_nodes,
        "hospital_nodes": hospital_nodes,
        "isolated_residential_nodes": isolated_residential_nodes,
        "isolated_residential_count": len(isolated_residential_nodes),
        "remaining_components": nx.number_connected_components(fractured_graph),
    }


def main() -> None:
    ox.settings.use_cache = True
    ox.settings.log_console = False

    graph = load_graph()
    edge_u, edge_v, _bridge_df = load_top_bridge_edge(BRIDGE_CSV)
    articulation_node, _articulation_df = load_top_articulation_node(ARTICULATION_CSV)

    fracture_metrics = visualize_network_fracture(graph, edge_u, edge_v, output_path=FRACTURE_FIGURE)
    isolation_metrics = compute_residential_isolation_count(graph, articulation_node)

    print("Kathmandu road network fracture metrics")
    print(f"Bridge edge: ({edge_u}, {edge_v})")
    print(f"Total resulting components: {fracture_metrics['component_count']}")
    print(f"Giant component size: {fracture_metrics['giant_size']}")
    print(f"Isolated fragment sizes: {fracture_metrics['fragment_sizes']}")
    print(f"Nodes isolated: {fracture_metrics['nodes_isolated']}")
    print(f"Network efficiency loss: {fracture_metrics['network_efficiency_loss_pct']:.2f}%")
    print(f"Saved fracture figure: {FRACTURE_FIGURE}")
    print()
    print("Residential accessibility after deleting top articulation point")
    print(f"Top articulation node: {articulation_node}")
    print(f"Residential nodes mapped: {len(isolation_metrics['residential_nodes'])}")
    print(f"Hospital nodes mapped: {len(isolation_metrics['hospital_nodes'])}")
    print(f"Residential nodes isolated from all hospitals: {isolation_metrics['isolated_residential_count']}")


if __name__ == "__main__":
    main()