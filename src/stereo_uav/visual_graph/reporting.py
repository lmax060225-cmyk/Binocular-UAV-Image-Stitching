"""Reporting for the visual graph algorithm."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .features import (
    VisualVerifier,
)
from .geometry import (
    assert_edge_orientation,
    find_connected_components,
    image_order,
    key_text,
)
from .models import (
    Config,
    GraphBuildResult,
    ImageEdge,
    ImageKey,
)


def save_csv(path: Path, rows: list[dict], columns=None) -> None:
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False, encoding="utf-8-sig")


def save_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            default=lambda x: x.item() if isinstance(x, np.generic) else str(x),
        ),
        encoding="utf-8",
    )


def graph_statistics(pairs, nodes, graph: GraphBuildResult) -> dict:
    edges = list(graph.optimization_edges_by_pair.values())
    degree = Counter({key: 0 for key in nodes})
    for edge in edges:
        degree.update((edge.key_i, edge.key_j))
    values = np.array(list(degree.values()))
    stats = dict(
        num_stereo_pairs=len(pairs),
        num_images=len(nodes),
        intra_pair_edges=sum("intra_pair" in e.roles for e in edges),
        total_optimization_edges=len(edges),
        image_graph_connected_components=len(find_connected_components(nodes, edges)),
        degree_min=int(values.min()),
        degree_mean=float(values.mean()),
        degree_median=float(np.median(values)),
        degree_max=int(values.max()),
    )
    for role in ("tree", "strong", "loop"):
        stats[f"pair_{role}_relations"] = sum(role in r.roles for r in graph.relations.values())
        stats[f"image_{role}_edges"] = sum(role in e.roles for e in edges)
    print("\n" + "=" * 60 + "\nINCREMENTAL GRAPH CONSTRUCTION\n" + "=" * 60)
    print(f"Stereo pairs = {len(pairs)}\nImages = {len(nodes)}")
    print(f"Sequential pair candidates = {graph.candidate_counts['sequential']}")
    print(f"Retrieval pair candidates = {graph.candidate_counts['retrieval']}")
    print(f"Verified pair relations = {graph.candidate_counts['verified']}")
    for role in ("tree", "strong", "loop"):
        print(f"{role.title()} relations = {stats[f'pair_{role}_relations']}")
    print("\n" + "=" * 60 + "\nIMAGE OPTIMIZATION GRAPH\n" + "=" * 60)
    print(f"Nodes = {len(nodes)}\nIntra-pair ordinary edges = {stats['intra_pair_edges']}")
    for role in ("tree", "strong", "loop"):
        print(f"{role.title()} ordinary edges = {stats[f'image_{role}_edges']}")
    print(
        f"Total ordinary edges = {len(edges)}\nConnected components = "
        f"{stats['image_graph_connected_components']}"
    )
    print(
        f"Degree min/median/mean/max = {values.min()}/{np.median(values)}/"
        f"{values.mean():.3f}/{values.max()}"
    )
    print("IMPORTANT:\nAll optimization edges are ordinary = True\nStereo-special edges = 0")
    if values.mean() > 10:
        print("WARNING: optimization graph is unexpectedly dense")
    return stats


def save_graph_diagnostics(
    output: Path, pairs, paths, graph: GraphBuildResult, verifier: VisualVerifier, cfg: Config
) -> dict:
    data = output / "data"
    pair_columns = [
        "frame_i",
        "frame_j",
        "roles",
        "relation_score",
        "endpoint_coverage",
        "num_verified_LL_LR_RL_RR",
        "source",
    ]
    pair_rows = [
        dict(
            frame_i=r.frame_i,
            frame_j=r.frame_j,
            roles="|".join(sorted(r.roles)),
            relation_score=r.relation_score,
            endpoint_coverage=r.endpoint_coverage,
            num_verified_LL_LR_RL_RR=len(r.verified_image_edges),
            source=r.source,
        )
        for r in graph.relations.values()
    ]
    save_csv(data / "pair_topology_graph.csv", pair_rows, pair_columns)
    edge_rows = []
    arrays = {}
    for index, edge in enumerate(graph.optimization_edges_by_pair.values()):
        row = {
            name: getattr(edge, name)
            for name in (
                "raw_matches",
                "ratio_matches",
                "ransac_inliers",
                "inlier_ratio",
                "mean_reproj_error",
                "coverage_i",
                "coverage_j",
                "quality_score",
                "selected_matches",
            )
        }
        row.update(
            key_i=key_text(edge.key_i),
            key_j=key_text(edge.key_j),
            roles="|".join(sorted(edge.roles)),
            provenance=edge.provenance,
            observation_index=index,
        )
        edge_rows.append(row)
        for name in (
            "H_i_to_j",
            "inlier_pts_i",
            "inlier_pts_j",
            "selected_pts_i",
            "selected_pts_j",
        ):
            arrays[f"e{index}_{name}"] = getattr(edge, name)
    edge_columns = [
        "key_i",
        "key_j",
        "roles",
        "provenance",
        "raw_matches",
        "ratio_matches",
        "ransac_inliers",
        "inlier_ratio",
        "mean_reproj_error",
        "coverage_i",
        "coverage_j",
        "quality_score",
        "selected_matches",
        "observation_index",
    ]
    save_csv(data / "image_optimization_graph.csv", edge_rows, edge_columns)
    np.savez_compressed(data / "optimization_observations.npz", **arrays)
    save_csv(data / "matching_diagnostics.csv", verifier.diagnostics)
    save_csv(
        data / "retrieval_candidates.csv",
        graph.retrieval_rows,
        ["query_frame", "query_position", "frame", "position", "votes", "score", "weighted_score"],
    )
    save_csv(data / "pair_registration_status.csv", graph.pair_status_rows)
    manifest = []
    for key, path in paths.items():
        feature = verifier.feature_cache[key]
        manifest.append(
            dict(
                camera=key[0],
                frame=key[1],
                path=str(path.resolve()),
                bytes=path.stat().st_size,
                mtime_ns=path.stat().st_mtime_ns,
                height=feature.shape[0],
                width=feature.shape[1],
                sift_features=len(feature.descriptors),
                sift_extractions=verifier.extraction_counts[key],
            )
        )
    save_csv(data / "input_manifest.csv", manifest)
    stats = graph_statistics(pairs, list(paths), graph)
    stats.update(graph.candidate_counts)
    save_json(data / "graph_summary.json", stats)
    save_json(data / "graph_config.json", asdict(cfg))
    return stats


def load_saved_graph(output: Path, paths: dict[ImageKey, Path], cfg: Config):
    """Explicit replay of this implementation's final graph, with input validation."""
    data = output / "data"
    recorded_cfg = json.loads((data / "graph_config.json").read_text(encoding="utf-8"))
    current_cfg = json.loads(json.dumps(asdict(cfg)))
    rendering_options = {"MOSAIC_MAX_PIXELS", "MOSAIC_MAX_SIDE", "GRAPHCUT_MAX_PIXELS"}
    if {k: v for k, v in recorded_cfg.items() if k not in rendering_options} != {
        k: v for k, v in current_cfg.items() if k not in rendering_options
    }:
        raise ValueError("Saved graph config differs; use a new output directory")
    manifest = pd.read_csv(data / "input_manifest.csv")
    shapes = {}
    if len(manifest) != len(paths):
        raise ValueError("Saved graph image count differs")
    for row in manifest.itertuples(index=False):
        key = (row.camera, int(row.frame))
        path = paths.get(key)
        if (
            path is None
            or str(path.resolve()) != row.path
            or path.stat().st_size != row.bytes
            or path.stat().st_mtime_ns != row.mtime_ns
        ):
            raise ValueError(f"Input changed since graph construction: {key}")
        shapes[key] = (int(row.height), int(row.width))
    edges = []
    with np.load(data / "optimization_observations.npz", allow_pickle=False) as arrays:
        for row in pd.read_csv(data / "image_optimization_graph.csv").to_dict("records"):
            index = row.pop("observation_index")
            for name in ("key_i", "key_j"):
                camera, frame = row[name].split(":")
                row[name] = (camera, int(frame))
            row["roles"] = set(row["roles"].split("|"))
            for name in (
                "H_i_to_j",
                "inlier_pts_i",
                "inlier_pts_j",
                "selected_pts_i",
                "selected_pts_j",
            ):
                row[name] = arrays[f"e{index}_{name}"].copy()
            edge = ImageEdge(**row)
            assert_edge_orientation(edge)
            edges.append(edge)
    return edges, shapes, json.loads((data / "graph_summary.json").read_text(encoding="utf-8"))


def save_transforms(
    data: Path, nodes, initial, affine, projective, component_ids, references
) -> None:
    affine_rows, projective_rows, initial_rows = [], [], []
    names = ("a", "b", "c", "d", "e", "f", "g", "h")
    for key in sorted(nodes, key=image_order):
        common = dict(
            camera=key[0],
            frame=key[1],
            component=component_ids[key],
            is_reference=key == references[component_ids[key]],
        )
        affine_rows.append(dict(common, **dict(zip(names[:6], affine[key].ravel()[:6]))))
        row = dict(common, delta_norm=float(np.linalg.norm(projective[key] - affine[key])))
        row.update(
            {f"affine_init_{name}": value for name, value in zip(names, affine[key].ravel()[:8])}
        )
        row.update(
            {
                f"projective_final_{name}": value
                for name, value in zip(names, projective[key].ravel()[:8])
            }
        )
        projective_rows.append(row)
        initial_rows.append(dict(common, **dict(zip(names, initial[key].ravel()[:8]))))
    save_csv(data / "global_affine_transforms.csv", affine_rows)
    save_csv(data / "global_projective_transforms.csv", projective_rows)
    save_csv(data / "initial_traversal_transforms.csv", initial_rows)
