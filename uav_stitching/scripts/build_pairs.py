#!/usr/bin/env python
"""Phase 3: convert WGS84 to ENU and build the GPS candidate-pair graph."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.gps import metadata_rows_to_positions, write_positions_csv  # noqa: E402
from src.metadata import read_metadata_csv  # noqa: E402
from src.neighbor_graph import (  # noqa: E402
    build_knn_neighbor_graph,
    build_quadrant_neighbor_graph,
    graph_statistics,
)
from src.utils import configure_logging, write_json_atomic  # noqa: E402
from src.visualization import plot_gps_neighbor_graph  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--method", choices=("quadrant", "knn"), help="Override configured neighbor method")
    parser.add_argument("--k", type=int, help="K for the optional KNN baseline")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    config = load_config(args.config)
    coordinate_system = str(config.gps.get("coordinate_system", "enu")).casefold()
    if coordinate_system != "enu":
        raise ValueError(f"Phase 3 currently requires coordinate_system=enu, got {coordinate_system!r}")

    metadata_path = config.cache_dir / "image_metadata.csv"
    rows = read_metadata_csv(metadata_path)
    positions, origin = metadata_rows_to_positions(rows)
    positions_path = config.cache_dir / "image_positions.csv"
    write_positions_csv(positions_path, positions)

    method = (args.method or config.gps.get("neighbor_method", "quadrant")).casefold()
    max_distance = config.gps.get("max_neighbor_distance_m")
    max_distance = float(max_distance) if max_distance is not None else None
    if method == "quadrant":
        pairs = build_quadrant_neighbor_graph(positions, max_neighbor_distance_m=max_distance)
    elif method == "knn":
        k = args.k if args.k is not None else int(config.gps.get("knn_k", 4))
        pairs = build_knn_neighbor_graph(positions, k=k, max_neighbor_distance_m=max_distance)
    else:
        raise ValueError(f"Unknown GPS neighbor method: {method}")

    statistics = graph_statistics(positions, pairs)
    pairs_path = config.cache_dir / "image_pairs.json"
    payload = {
        "schema_version": 1,
        "coordinate_system": "local_enu_meters",
        "method": method,
        "max_neighbor_distance_m": max_distance,
        "origin": dataclasses.asdict(origin),
        "statistics": dataclasses.asdict(statistics),
        "pairs": [list(pair) for pair in pairs],
    }
    write_json_atomic(pairs_path, payload)
    figure_path = config.debug_dir / "gps_neighbor_graph.png"
    plot_gps_neighbor_graph(positions, pairs, figure_path, method=method)

    east_values = [position.east for position in positions]
    north_values = [position.north for position in positions]
    print("[GPS coordinates]")
    print(
        f"Origin: lat={origin.latitude:.9f}, lon={origin.longitude:.9f}, "
        f"ellh={origin.altitude:.3f} m"
    )
    print(f"East extent: {min(east_values):.3f} .. {max(east_values):.3f} m")
    print(f"North extent: {min(north_values):.3f} .. {max(north_values):.3f} m")
    print(f"Positions CSV: {positions_path}")
    print("\n[GPS graph]")
    print(f"Method: {method}")
    print(f"Images: {statistics.image_count}")
    print(f"Pairs: {statistics.pair_count}")
    print(f"Connected components: {statistics.connected_components}")
    print(f"Isolated images: {list(statistics.isolated_images)}")
    print(
        f"Degree min/mean/max: {statistics.min_degree}/"
        f"{statistics.mean_degree:.3f}/{statistics.max_degree}"
    )
    print(f"Pairs JSON: {pairs_path}")
    print(f"Debug graph: {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
