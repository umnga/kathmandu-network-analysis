from __future__ import annotations

from pathlib import Path
from typing import Iterable

import folium
import pandas as pd
from folium.plugins import MarkerCluster

SOURCE_BRIDGES_CSV = Path(__file__).resolve().with_name("kathmandu_bridge_edges.csv")
SOURCE_ARTICULATION_CSV = Path(__file__).resolve().with_name("kathmandu_articulation_points.csv")
OUTPUT_HTML = Path(__file__).resolve().with_name("kathmandu_severe_bottlenecks.html")

KATHMANDU_CENTER = (27.7172, 85.3240)
MAP_ZOOM_START = 12

ROAD_BASE_COLOR = "#3A3A3A"
ROAD_BASE_WEIGHT = 1.2
ROAD_BASE_OPACITY = 0.18

BRIDGE_COLOR = "#E74C3C"
BRIDGE_WEIGHT = 5
BRIDGE_OPACITY = 0.95

ARTICULATION_COLOR = "#FF3B30"
ARTICULATION_RADIUS = 6
ARTICULATION_FILL_OPACITY = 0.92


def _safe_int(value) -> int | None:
    if pd.isna(value):
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        try:
            return int(float(str(value).strip()))
        except (TypeError, ValueError):
            return None


def _safe_float(value) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_bridge_edges(source_path: Path) -> pd.DataFrame:
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source file: {source_path}")

    df = pd.read_csv(source_path)
    required_columns = {
        "Node_U",
        "Node_V",
        "Edge_Betweenness",
        "U_Latitude",
        "U_Longitude",
        "V_Latitude",
        "V_Longitude",
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required bridge columns: {missing}")

    cleaned = df.copy()
    for column in ["Node_U", "Node_V"]:
        cleaned[column] = cleaned[column].map(_safe_int)
    for column in ["Edge_Betweenness", "U_Latitude", "U_Longitude", "V_Latitude", "V_Longitude"]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.dropna(subset=sorted(required_columns)).reset_index(drop=True)
    cleaned["Node_U"] = cleaned["Node_U"].astype(int)
    cleaned["Node_V"] = cleaned["Node_V"].astype(int)
    return cleaned


def load_articulation_points(source_path: Path) -> pd.DataFrame:
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source file: {source_path}")

    df = pd.read_csv(source_path)
    required_columns = {"OSM_Node_ID", "Latitude", "Longitude"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required articulation columns: {missing}")

    cleaned = df.copy()
    cleaned["OSM_Node_ID"] = cleaned["OSM_Node_ID"].map(_safe_int)
    cleaned["Latitude"] = pd.to_numeric(cleaned["Latitude"], errors="coerce")
    cleaned["Longitude"] = pd.to_numeric(cleaned["Longitude"], errors="coerce")
    cleaned = cleaned.dropna(subset=sorted(required_columns)).reset_index(drop=True)
    cleaned["OSM_Node_ID"] = cleaned["OSM_Node_ID"].astype(int)
    return cleaned


def _add_base_road_context(map_object: folium.Map, bridges: pd.DataFrame) -> None:
    for _, row in bridges.iterrows():
        folium.PolyLine(
            locations=[
                (float(row["U_Latitude"]), float(row["U_Longitude"])),
                (float(row["V_Latitude"]), float(row["V_Longitude"])),
            ],
            color=ROAD_BASE_COLOR,
            weight=ROAD_BASE_WEIGHT,
            opacity=ROAD_BASE_OPACITY,
        ).add_to(map_object)


def _add_bridge_layer(map_object: folium.Map, bridges: pd.DataFrame) -> None:
    bridge_group = folium.FeatureGroup(name="Bridge edges with no alternate route", show=True)
    for _, row in bridges.iterrows():
        node_u = int(row["Node_U"])
        node_v = int(row["Node_V"])
        betweenness = float(row["Edge_Betweenness"])
        folium.PolyLine(
            locations=[
                (float(row["U_Latitude"]), float(row["U_Longitude"])),
                (float(row["V_Latitude"]), float(row["V_Longitude"])),
            ],
            color=BRIDGE_COLOR,
            weight=BRIDGE_WEIGHT,
            opacity=BRIDGE_OPACITY,
            tooltip=(
                f"Bridge edge {node_u} → {node_v} | edge betweenness: {betweenness:.6f}"
            ),
        ).add_to(bridge_group)
    bridge_group.add_to(map_object)


def _add_articulation_layer(map_object: folium.Map, articulation_points: pd.DataFrame) -> None:
    cluster = MarkerCluster(name="Articulation points", disableClusteringAtZoom=15)
    cluster.add_to(map_object)

    for _, row in articulation_points.iterrows():
        node_id = int(row["OSM_Node_ID"])
        latitude = float(row["Latitude"])
        longitude = float(row["Longitude"])
        degree = row.get("Degree")
        betweenness = row.get("Betweenness_Centrality")
        components_after_removal = row.get("Components_After_Removal")

        popup_html = (
            f"<b>OSM Node ID:</b> {node_id}<br>"
            f"<b>Degree:</b> {degree if pd.notna(degree) else 'n/a'}<br>"
            f"<b>Betweenness:</b> {betweenness if pd.notna(betweenness) else 'n/a'}<br>"
            f"<b>Components After Removal:</b> {components_after_removal if pd.notna(components_after_removal) else 'n/a'}"
        )

        folium.CircleMarker(
            location=(latitude, longitude),
            radius=ARTICULATION_RADIUS,
            color=ARTICULATION_COLOR,
            fill=True,
            fill_color=ARTICULATION_COLOR,
            fill_opacity=ARTICULATION_FILL_OPACITY,
            weight=2,
            popup=folium.Popup(popup_html, max_width=340),
            tooltip=f"Articulation node {node_id}",
        ).add_to(cluster)


def build_map(bridges: pd.DataFrame, articulation_points: pd.DataFrame) -> folium.Map:
    map_object = folium.Map(location=KATHMANDU_CENTER, zoom_start=MAP_ZOOM_START, tiles="CartoDB dark_matter")
    _add_base_road_context(map_object, bridges)
    _add_bridge_layer(map_object, bridges)
    _add_articulation_layer(map_object, articulation_points)

    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        z-index: 9999;
        background: rgba(17, 17, 17, 0.92);
        color: white;
        padding: 14px 16px;
        border: 1px solid #444;
        border-radius: 10px;
        box-shadow: 0 6px 22px rgba(0, 0, 0, 0.35);
        font-size: 13px;
        line-height: 1.45;
        max-width: 300px;
    ">
        <div style="font-size: 14px; font-weight: 700; margin-bottom: 8px;">Kathmandu bottlenecks</div>
        <div><span style="display:inline-block;width:12px;height:12px;background:#E74C3C;margin-right:8px;border-radius:2px;"></span>Bridge edge with no alternate route</div>
        <div><span style="display:inline-block;width:12px;height:12px;background:#FF3B30;margin-right:8px;border-radius:50%;"></span>Articulation point</div>
    </div>
    """
    map_object.get_root().html.add_child(folium.Element(legend_html))
    return map_object


def main() -> None:
    bridges = load_bridge_edges(SOURCE_BRIDGES_CSV)
    articulation_points = load_articulation_points(SOURCE_ARTICULATION_CSV)

    print(f"Loaded {len(bridges)} bridge edges")
    print(f"Loaded {len(articulation_points)} articulation points")

    bottleneck_map = build_map(bridges, articulation_points)
    bottleneck_map.save(str(OUTPUT_HTML))
    print(f"Saved {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
