"""Pipeline for the visual graph algorithm."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .features import (
    point_policy,
)
from .geometry import (
    choose_reference,
    find_connected_components,
    key_text,
)
from .metrics import (
    compute_registration_metrics,
    edge_errors,
    error_statistics,
)
from .models import (
    Config,
)
from .optimization import (
    initial_graph_transforms,
    solve_global_affine_sparse,
    solve_global_projective,
)
from .rendering import (
    canvas_render_scale,
    mosaic_bounds,
    render_mosaic,
)
from .reporting import (
    save_csv,
    save_json,
    save_transforms,
)


def run_registration_and_mosaics(paths, edges, shapes, output: Path, cfg: Config):
    nodes, data = list(paths), output / "data"
    components = find_connected_components(nodes, edges)
    initial, affine, projective = {}, {}, {}
    component_ids, references, optimizer_rows = {}, {}, []
    for component_id, group in enumerate(components):
        members = set(group)
        component_edges = [edge for edge in edges if edge.key_i in members]
        assert all(edge.key_j in members for edge in component_edges)
        reference = choose_reference(group)
        references[component_id] = reference
        component_ids.update({key: component_id for key in group})
        initial.update(initial_graph_transforms(group, component_edges, reference))
        print("\n" + "=" * 60 + "\nGLOBAL AFFINE OPTIMIZATION\n" + "=" * 60, flush=True)
        print(
            f"Component={component_id}, N={len(group)}, reference={key_text(reference)}\n"
            f"M = {len(component_edges)}\nP = {point_policy(len(nodes))[0]}\n"
            "solver = sparse linear LSMR",
            flush=True,
        )
        affine_result, affine_info = solve_global_affine_sparse(
            group, component_edges, reference, cfg
        )
        affine.update(affine_result)
        _, affine_metrics = compute_registration_metrics(
            component_edges, dict(initial=initial, affine=affine)
        )
        print(
            f"RMSE before = {affine_metrics[0]['rmse']}\nRMSE after = {affine_metrics[2]['rmse']}",
            flush=True,
        )
        print("\n" + "=" * 60 + "\nGLOBAL PROJECTIVE OPTIMIZATION\n" + "=" * 60, flush=True)
        print(
            "Objective: Eproj = E_match/(M*P) + 800 * E_rigid/N\n"
            "sigma_tr = 5000\nNo GPS prior\nNo stereo residual\n"
            "No safety penalty\nNo robust loss",
            flush=True,
        )
        result, projective_info = solve_global_projective(
            group, component_edges, reference, affine_result, cfg
        )
        projective.update(result)
        projective_errors = [edge_errors(edge, result) for edge in component_edges]
        projective_rmse = error_statistics(
            np.concatenate(projective_errors) if projective_errors else np.array([])
        )["rmse"]
        print(
            f"RMSE before = {affine_metrics[2]['rmse']}\nRMSE after = {projective_rmse}", flush=True
        )
        for info in (affine_info, projective_info):
            optimizer_rows.append(dict(component=component_id, **info))
        print(json.dumps(projective_info, indent=2, default=str), flush=True)
        save_csv(data / "optimizer_summary.csv", optimizer_rows)
    save_transforms(data, nodes, initial, affine, projective, component_ids, references)
    edge_metrics, summary_metrics = compute_registration_metrics(
        edges, dict(initial=initial, affine=affine, projective=projective)
    )
    save_csv(data / "edge_registration_metrics.csv", edge_metrics)
    save_csv(data / "registration_metrics_summary.csv", summary_metrics)
    print(
        "\nAll-RANSAC paper-style and selected-P metrics:\n"
        + pd.DataFrame(summary_metrics).to_string(index=False),
        flush=True,
    )
    mosaic_rows = []
    for component_id, group in enumerate(components):
        scale = min(
            canvas_render_scale(mosaic_bounds(group, stage, shapes), cfg)
            for stage in (affine, projective)
        )
        for stage_name, transforms in (("affine", affine), ("projective", projective)):
            filename = f"global_{stage_name}_mosaic"
            if len(components) > 1:
                filename += f"_component_{component_id:03d}"
            path = output / "mosaics" / f"{filename}.jpg"
            info = render_mosaic(group, paths, shapes, transforms, path, cfg, scale)
            mosaic_rows.append(dict(stage=stage_name, component=component_id, **info))
            if len(components) > 1 and component_id == 0:
                for suffix in (".jpg", "_preview.jpg", "_mask.png"):
                    shutil.copy2(
                        path.with_name(filename + suffix),
                        path.with_name(f"global_{stage_name}_mosaic" + suffix),
                    )
            save_json(data / "mosaic_summary.json", mosaic_rows)
    return summary_metrics, optimizer_rows, mosaic_rows
