"""Reporting for the backbone algorithm."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import config
from .geometry import (
    projective_stereo_pair_frame,
    transform_points_affine,
    transform_points_projective,
    wrap_angle,
)
from .initialization import (
    affine_stereo_pair_frame,
)
from .io import (
    to_jsonable_matrix,
)
from .models import (
    ImageRecord,
    PairMatch_Edge,
)


# ==================== RIGID STEREO DEBUG OUTPUT ====================
def rigid_stereo_config_dict() -> Dict[str, object]:
    """Return the active rigid-stereo configuration for reproducible reporting."""

    return {
        "method": "rigid_stereo_common_similarity",
        "lambda_perp": config.LAMBDA_STEREO_PERP,
        "perp_sigma_px": config.STEREO_PERP_SIGMA_PX,
        "lambda_theta": config.LAMBDA_STEREO_THETA,
        "theta_sigma_rad": config.STEREO_THETA_SIGMA_RAD,
        "theta_sigma_deg": math.degrees(config.STEREO_THETA_SIGMA_RAD),
        "lambda_scale": config.LAMBDA_STEREO_SCALE,
        "log_scale_sigma": config.STEREO_LOG_SCALE_SIGMA,
        "affine_stereo_weight_schedule": list(config.AFFINE_STEREO_WEIGHT_SCHEDULE),
        "affine_stereo_energy": ("r_perp + relative_rotation + relative_log_scale"),
        "projective_stereo_energy": (
            "center-Jacobian local r_perp + relative_rotation + relative_log_scale"
        ),
        "parallel_disparity_constraint": "none",
        "projective_safety_energy": "denominator + singular_value_bounds",
        "lambda_projective_denom": config.LAMBDA_PROJECTIVE_DENOM,
        "projective_denom_min": config.PROJECTIVE_DENOM_MIN,
        "lambda_projective_sv": config.LAMBDA_PROJECTIVE_SV,
        "projective_sv_min": config.PROJECTIVE_SV_MIN,
        "projective_sv_max": config.PROJECTIVE_SV_MAX,
        "projective_safety_grid_rows": config.PROJECTIVE_SAFETY_GRID_ROWS,
        "projective_safety_grid_cols": config.PROJECTIVE_SAFETY_GRID_COLS,
    }


def save_rigid_stereo_initialization_debug(
    block_idx: int,
    initialization_debug: Dict[str, object],
    out_dir: Path,
) -> None:
    """Save rigid-stereo initialization and direction diagnostics as CSV files."""

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = rigid_stereo_config_dict()
    config["block_idx"] = block_idx
    config_row = {
        key: json.dumps(value, ensure_ascii=False)
        if isinstance(value, (dict, list, tuple))
        else value
        for key, value in config.items()
    }
    pd.DataFrame([config_row]).to_csv(
        out_dir / "rigid_stereo_config.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_row = {"block_idx": block_idx}
    summary_row.update(
        {
            key: json.dumps(value, ensure_ascii=False)
            if isinstance(value, (dict, list, tuple))
            else value
            for key, value in initialization_debug.items()
            if key != "events"
        }
    )
    pd.DataFrame([summary_row]).to_csv(
        out_dir / "initialization_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    events = initialization_debug.get(
        "events",
        [],
    )
    if events:
        pd.DataFrame(events).to_csv(
            out_dir / "initialization_events.csv",
            index=False,
            encoding="utf-8-sig",
        )


def collect_rigid_stereo_stage_debug(
    block_idx: int,
    stage_name: str,
    transforms: np.ndarray,
    edges: Sequence[PairMatch_Edge],
    images: Sequence[ImageRecord],
    transform_kind: str,
) -> Tuple[
    List[Dict[str, object]],
    List[Dict[str, object]],
]:
    """Collect stereo diagnostics. Similarity stages use global linear parts; projective stages use

    center Jacobians. Summaries record scale ratio, angle difference, and normal residuals. Per-
    point rows include parallel displacement for diagnosis only, not as an optimization
    constraint.
    """

    if transform_kind not in {
        "affine",
        "projective",
    }:
        raise ValueError(f"Unsupported transform_kind: {transform_kind}")

    stereo_edges = [
        edge for edge in edges if edge.edge_type == "stereo" and edge.selected_matches > 0
    ]

    summary_rows: List[Dict[str, object]] = []
    point_rows: List[Dict[str, object]] = []

    for edge in stereo_edges:
        if transform_kind == "affine":
            (
                scale_i,
                scale_j,
                angle_i,
                angle_j,
                log_scale_ratio,
                common_direction,
                normal_direction,
            ) = affine_stereo_pair_frame(
                transforms,
                edge.i,
                edge.j,
            )

            pi = transform_points_affine(
                transforms[edge.i],
                edge.selected_pts_i,
            )
            pj = transform_points_affine(
                transforms[edge.j],
                edge.selected_pts_j,
            )

        else:
            (
                scale_i,
                scale_j,
                angle_i,
                angle_j,
                log_scale_ratio,
                common_direction,
                normal_direction,
            ) = projective_stereo_pair_frame(
                transforms,
                images,
                edge.i,
                edge.j,
            )

            pi = transform_points_projective(
                transforms[edge.i],
                edge.selected_pts_i,
            )
            pj = transform_points_projective(
                transforms[edge.j],
                edge.selected_pts_j,
            )

        delta = pi - pj

        perpendicular = delta @ normal_direction
        parallel = delta @ common_direction

        angle_difference = wrap_angle(angle_i - angle_j)

        summary_rows.append(
            {
                "block_idx": block_idx,
                "stage": stage_name,
                "transform_kind": transform_kind,
                "edge_name": edge.edge_name,
                "i": edge.i,
                "j": edge.j,
                "image_i": images[edge.i].name,
                "image_j": images[edge.j].name,
                "selected_matches": edge.selected_matches,
                "scale_i": scale_i,
                "scale_j": scale_j,
                "scale_ratio_i_over_j": (
                    scale_i
                    / max(
                        scale_j,
                        config.PROJECTIVE_REG_EPS,
                    )
                ),
                "log_scale_ratio": log_scale_ratio,
                "angle_i_deg": math.degrees(angle_i),
                "angle_j_deg": math.degrees(angle_j),
                "angle_difference_deg": math.degrees(angle_difference),
                "common_direction_x": float(common_direction[0]),
                "common_direction_y": float(common_direction[1]),
                "common_angle_deg": math.degrees(
                    math.atan2(
                        common_direction[1],
                        common_direction[0],
                    )
                ),
                "normal_direction_x": float(normal_direction[0]),
                "normal_direction_y": float(normal_direction[1]),
                "perp_mean_px": float(np.mean(perpendicular)),
                "perp_median_px": float(np.median(perpendicular)),
                "perp_std_px": float(np.std(perpendicular)),
                "perp_rmse_px": float(math.sqrt(np.mean(perpendicular * perpendicular))),
                "perp_max_abs_px": float(np.max(np.abs(perpendicular))),
                "parallel_mean_px_debug_only": float(np.mean(parallel)),
                "parallel_median_px_debug_only": float(np.median(parallel)),
                "parallel_std_px_debug_only": float(np.std(parallel)),
            }
        )

        for match_idx in range(len(perpendicular)):
            point_rows.append(
                {
                    "block_idx": block_idx,
                    "stage": stage_name,
                    "transform_kind": transform_kind,
                    "edge_name": edge.edge_name,
                    "i": edge.i,
                    "j": edge.j,
                    "match_idx": match_idx,
                    "raw_i_x": float(
                        edge.selected_pts_i[
                            match_idx,
                            0,
                        ]
                    ),
                    "raw_i_y": float(
                        edge.selected_pts_i[
                            match_idx,
                            1,
                        ]
                    ),
                    "raw_j_x": float(
                        edge.selected_pts_j[
                            match_idx,
                            0,
                        ]
                    ),
                    "raw_j_y": float(
                        edge.selected_pts_j[
                            match_idx,
                            1,
                        ]
                    ),
                    "warped_i_x": float(pi[match_idx, 0]),
                    "warped_i_y": float(pi[match_idx, 1]),
                    "warped_j_x": float(pj[match_idx, 0]),
                    "warped_j_y": float(pj[match_idx, 1]),
                    "delta_x": float(delta[match_idx, 0]),
                    "delta_y": float(delta[match_idx, 1]),
                    "perp_residual_px": float(perpendicular[match_idx]),
                    "parallel_component_px_debug_only": float(parallel[match_idx]),
                }
            )

    return summary_rows, point_rows


def save_rigid_stereo_stage_debug(
    block_idx: int,
    stage_name: str,
    transforms: np.ndarray,
    edges: Sequence[PairMatch_Edge],
    images: Sequence[ImageRecord],
    transform_kind: str,
    out_dir: Path,
) -> None:
    """Save one stage's stereo summaries and per-correspondence diagnostics."""

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows, point_rows = collect_rigid_stereo_stage_debug(
        block_idx,
        stage_name,
        transforms,
        edges,
        images,
        transform_kind,
    )

    safe_stage = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        stage_name,
    )

    pd.DataFrame(summary_rows).to_csv(
        out_dir / f"{safe_stage}_stereo_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(point_rows).to_csv(
        out_dir / f"{safe_stage}_stereo_points.csv",
        index=False,
        encoding="utf-8-sig",
    )


# ==================== RIGID STEREO DEBUG OUTPUT END ====================


def block_image_role(local_idx: int) -> Tuple[str, int]:
    """Map a local block index to camera side and within-stream position."""

    if local_idx < 2:
        return "left", local_idx
    return "right", local_idx - 2


def make_block_image_rows(block_idx: int, images: Sequence[ImageRecord]) -> List[Dict[str, object]]:
    """Build per-block image metadata rows for CSV output."""

    rows: List[Dict[str, object]] = []
    for local_idx, image in enumerate(images):
        camera, camera_local_idx = block_image_role(local_idx)
        rows.append(
            {
                "block_idx": block_idx,
                "local_idx": local_idx,
                "camera": camera,
                "camera_local_idx": camera_local_idx,
                "original_index": image.index,
                "name": image.name,
                "path": str(image.path),
                "width": image.width,
                "height": image.height,
            }
        )
    return rows


def make_edge_report_rows(
    block_idx: int,
    images: Sequence[ImageRecord],
    pair_results: Sequence[PairMatch_Edge],
    valid_edges: Sequence[PairMatch_Edge],
) -> List[Dict[str, object]]:
    """Build per-block edge matching rows for CSV output."""

    valid_pairs = {(edge.i, edge.j) for edge in valid_edges}
    rows: List[Dict[str, object]] = []

    for edge in pair_results:
        rows.append(
            {
                "block_idx": block_idx,
                "i": edge.i,
                "j": edge.j,
                "edge_type": edge.edge_type,
                "edge_name": edge.edge_name,
                "raw_stereo_dx_median": (
                    float(np.median(edge.selected_pts_i[:, 0] - edge.selected_pts_j[:, 0]))
                    if edge.edge_type == "stereo" and edge.selected_matches > 0
                    else None
                ),
                "raw_stereo_dy_median": (
                    float(np.median(edge.selected_pts_i[:, 1] - edge.selected_pts_j[:, 1]))
                    if edge.edge_type == "stereo" and edge.selected_matches > 0
                    else None
                ),
                "image_i": images[edge.i].name,
                "image_j": images[edge.j].name,
                "is_valid": (edge.i, edge.j) in valid_pairs,
                "raw_matches": edge.raw_matches,
                "ratio_matches": edge.ratio_matches,
                "ransac_inliers": edge.ransac_inliers,
                "selected_matches": edge.selected_matches,
                "inlier_ratio": edge.inlier_ratio,
                "mean_reproj_error": edge.mean_reproj_error,
                "H_i_to_j_json": json.dumps(to_jsonable_matrix(edge.H_i_to_j), ensure_ascii=False),
            }
        )

    return rows


def save_block_transforms(
    block_idx: int,
    images: Sequence[ImageRecord],
    transforms: np.ndarray,
    out_dir: Path,
    transform_name: str,
    image_roles: Optional[Sequence[Tuple[str, int]]] = None,
) -> None:
    """Save each image's transform in the local block coordinate system."""

    rows: List[Dict[str, object]] = []

    for local_idx, (image, H) in enumerate(zip(images, transforms)):
        if image_roles is None:
            camera, camera_local_idx = block_image_role(local_idx)
        else:
            camera, camera_local_idx = image_roles[local_idx]
        matrix = H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H

        row = {
            "block_idx": block_idx,
            "local_idx": local_idx,
            "camera": camera,
            "camera_local_idx": camera_local_idx,
            "original_index": image.index,
            "name": image.name,
            "path": str(image.path),
        }

        for r in range(3):
            for c in range(3):
                row[f"H{r + 1}{c + 1}"] = float(matrix[r, c])

        rows.append(row)

    pd.DataFrame(rows).to_csv(
        out_dir / f"{transform_name}_transforms.csv", index=False, encoding="utf-8-sig"
    )


def make_block_summary_row(
    block_idx: int,
    images: Sequence[ImageRecord],
    status: str,
    candidate_edges: int = 0,
    valid_edges: int = 0,
    total_selected_matches: int = 0,
    affine_initial_cost: Optional[float] = None,
    affine_final_cost: Optional[float] = None,
    affine_success: Optional[bool] = None,
    projective_initial_cost: Optional[float] = None,
    projective_final_cost: Optional[float] = None,
    projective_success: Optional[bool] = None,
    persistent_applied: Optional[bool] = None,
    persistent_initial_cost: Optional[float] = None,
    persistent_final_cost: Optional[float] = None,
    persistent_success: Optional[bool] = None,
    error: str = "",
) -> Dict[str, object]:
    """Build the block summary row for CSV output."""

    row: Dict[str, object] = {
        "block_idx": block_idx,
        "status": status,
        "error": error,
        "candidate_edges": candidate_edges,
        "valid_edges": valid_edges,
        "total_selected_matches": total_selected_matches,
        "affine_initial_cost": affine_initial_cost,
        "affine_final_cost": affine_final_cost,
        "affine_success": affine_success,
        "projective_initial_cost": projective_initial_cost,
        "projective_final_cost": projective_final_cost,
        "projective_success": projective_success,
        "persistent_applied": persistent_applied,
        "persistent_initial_cost": persistent_initial_cost,
        "persistent_final_cost": persistent_final_cost,
        "persistent_success": persistent_success,
    }

    for local_idx, image in enumerate(images):
        camera, camera_local_idx = block_image_role(local_idx)
        prefix = f"image_{local_idx}_{camera}_{camera_local_idx}"
        row[f"{prefix}_name"] = image.name
        row[f"{prefix}_path"] = str(image.path)

    return row


def write_report_csvs(
    output_dirs: Dict[str, Path],
    block_rows: Sequence[Dict[str, object]],
    image_rows: Sequence[Dict[str, object]],
    edge_rows: Sequence[Dict[str, object]],
) -> None:
    """Write accumulated block summaries, image rows, and edge rows to CSV files."""

    logs_dir = output_dirs["logs"]
    logs_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(block_rows).to_csv(
        logs_dir / "block_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(image_rows).to_csv(
        logs_dir / "block_images.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(edge_rows).to_csv(logs_dir / "block_edges.csv", index=False, encoding="utf-8-sig")
