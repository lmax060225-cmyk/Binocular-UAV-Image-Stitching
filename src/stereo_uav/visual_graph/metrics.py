"""Metrics for the visual graph algorithm."""

from __future__ import annotations

import numpy as np

from .geometry import (
    key_text,
    warp_points,
)
from .models import (
    ImageEdge,
    ImageKey,
)


def edge_errors(
    edge: ImageEdge, transforms: dict[ImageKey, np.ndarray], selected: bool = False
) -> np.ndarray:
    pi = edge.selected_pts_i if selected else edge.inlier_pts_i
    pj = edge.selected_pts_j if selected else edge.inlier_pts_j
    return np.linalg.norm(
        warp_points(transforms[edge.key_i], pi) - warp_points(transforms[edge.key_j], pj), axis=1
    )


def error_statistics(errors: np.ndarray) -> dict:
    if not len(errors):
        return dict(count=0, rmse=None, mean=None, median=None, p90=None, p95=None, max=None)
    return dict(
        count=len(errors),
        rmse=float(np.sqrt(np.mean(errors**2))),
        mean=float(errors.mean()),
        median=float(np.median(errors)),
        p90=float(np.percentile(errors, 90)),
        p95=float(np.percentile(errors, 95)),
        max=float(errors.max()),
    )


def compute_registration_metrics(
    edges: list[ImageEdge], stages: dict[str, dict[ImageKey, np.ndarray]]
) -> tuple[list[dict], list[dict]]:
    rows, summaries = [], []
    for edge in edges:
        row = dict(
            key_i=key_text(edge.key_i),
            key_j=key_text(edge.key_j),
            roles="|".join(sorted(edge.roles)),
            ransac_inliers=edge.ransac_inliers,
            selected_matches=edge.selected_matches,
        )
        for stage, transforms in stages.items():
            stat = error_statistics(edge_errors(edge, transforms))
            row[f"rmse_{stage}"] = stat["rmse"]
            if stage == "projective":
                row.update(median_projective=stat["median"], p95_projective=stat["p95"])
        rows.append(row)
    for stage, transforms in stages.items():
        for selected in (False, True):
            arrays = [edge_errors(edge, transforms, selected) for edge in edges]
            errors = np.concatenate(arrays) if arrays else np.array([])
            summaries.append(
                dict(
                    stage=stage,
                    points="selected_P" if selected else "all_RANSAC_inliers",
                    **error_statistics(errors),
                )
            )
    return rows, summaries
