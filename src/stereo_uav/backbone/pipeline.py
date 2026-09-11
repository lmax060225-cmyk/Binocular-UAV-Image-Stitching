"""Pipeline for the backbone algorithm."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from . import config
from .geometry import (
    decompose_global_transforms,
    normalize_projective_homography,
    rebase_variable_global_states,
)
from .initialization import (
    initialize_affines,
)
from .io import (
    load_incremental_image_windows,
    make_block_output_dirs,
    make_output_dirs,
)
from .matching import (
    build_candidate_edges,
    extract_features,
    filter_valid_edges,
    match_pair,
)
from .models import (
    FeatureRecord,
    ImageRecord,
    PairMatch_Edge,
    PersistentBlockCache,
)
from .persistent import (
    build_persistent_active_window,
    initialize_persistent_active_state,
    optimize_persistent_window_projectives,
)
from .projective import (
    optimize_projectives,
)
from .rendering import (
    save_graphcut_mosaic_preview,
)
from .reporting import (
    block_image_role,
    make_block_image_rows,
    make_block_summary_row,
    make_edge_report_rows,
    save_block_transforms,
    save_rigid_stereo_initialization_debug,
    save_rigid_stereo_stage_debug,
    write_report_csvs,
)
from .similarity import (
    optimize_affines,
)


def run_stitching() -> None:

    np.random.seed(config.RANDOM_SEED)

    left_in_dir = Path(config.LEFT_IN)
    right_in_dir = Path(config.RIGHT_IN)
    output_dir = Path(config.OUTPUT_DIR)

    out_dirs = make_output_dirs(output_dir)

    print(f"LEFT_IN_DIR  = {left_in_dir}")
    print(f"RIGHT_IN_DIR = {right_in_dir}")
    print(f"OUTPUT_DIR   = {output_dir}")

    block_rows: List[Dict[str, object]] = []
    image_rows: List[Dict[str, object]] = []
    edge_rows: List[Dict[str, object]] = []

    # Reuse the SIFT object and shared-frame features across windows.
    # Trim the cache to the two images needed by the next window.
    sift = cv2.SIFT_create(nfeatures=config.SIFT_NFEATURES)
    feature_cache: Dict[Path, FeatureRecord] = {}

    # The current stereo_t1 observation becomes the next window's stereo_t.
    previous_stereo_cache: Optional[Tuple[Path, Path, PairMatch_Edge]] = None

    # Cache typed observations from the last successful global update.
    # Combine them with the next block in a two-block fixed-lag window.
    previous_block_cache: Optional[PersistentBlockCache] = None

    # Accumulated global state.
    global_images: List[ImageRecord] = []

    # key:
    #   ("left", frame_number)
    #   ("right", frame_number)
    #
    # G_i maps the current image into the global projective coordinate system.
    global_transforms_by_key: Dict[
        Tuple[str, int],
        np.ndarray,
    ] = {}

    # S_i is the accumulated similarity backbone used for cross-block initialization.
    # Exclude the projective correction in G_i from propagation.
    global_backbones_by_key: Dict[
        Tuple[str, int],
        np.ndarray,
    ] = {}

    # Keep image records, image keys, and transform ordering consistent.
    global_image_keys: List[Tuple[str, int]] = []
    global_image_key_set = set()

    # Iterate over adjacent-frame sliding windows:
    # (frame_0, frame_1)
    # (frame_1, frame_2)
    # (frame_2, frame_3)
    # ...
    for block_idx, (left_images, right_images) in enumerate(
        load_incremental_image_windows(
            left_in_dir,
            right_in_dir,
        )
    ):
        print(f"\n========== Processing block {block_idx:04d} ==========")

        # Every block contains:
        # left_t、left_t+1、right_t、right_t+1
        image_block = [
            left_images[0],
            left_images[1],
            right_images[0],
            right_images[1],
        ]

        block_dirs = make_block_output_dirs(
            out_dirs,
            block_idx,
        )

        print("Block images:")

        for local_idx, image in enumerate(image_block):
            camera, camera_local_idx = block_image_role(local_idx)

            print(
                f"  local_idx={local_idx}, camera={camera}[{camera_local_idx}], name={image.name}"
            )

        image_rows.extend(
            make_block_image_rows(
                block_idx,
                image_block,
            )
        )

        block_candidate_pairs: List[Tuple[int, int]] = []
        block_pair_ret: List[PairMatch_Edge] = []
        block_valid_edges: List[PairMatch_Edge] = []
        try:
            # 1. Extract features for the four block images.
            block_features = extract_features(
                image_block,
                cache=feature_cache,
                sift=sift,
            )

            # 2. Construct six candidate edges over the four images.
            block_candidate_specs = build_candidate_edges(len(image_block))
            block_candidate_pairs = [(spec.i, spec.j) for spec in block_candidate_specs]

            print("Block Matching candidate image pairs")

            for spec in tqdm(
                block_candidate_specs,
                desc=f"Block {block_idx:04d} matching",
            ):
                reused_pair = None
                if spec.edge_name == "stereo_t" and previous_stereo_cache is not None:
                    cached_left, cached_right, cached_pair = previous_stereo_cache
                    if (
                        cached_left == image_block[spec.i].path
                        and cached_right == image_block[spec.j].path
                    ):
                        reused_pair = replace(
                            cached_pair,
                            i=spec.i,
                            j=spec.j,
                            edge_type=spec.edge_type,
                            edge_name=spec.edge_name,
                        )

                if reused_pair is not None:
                    block_pair_ret.append(reused_pair)
                else:
                    block_pair_ret.append(
                        match_pair(
                            spec,
                            block_features,
                        )
                    )

            next_stereo = next(edge for edge in block_pair_ret if edge.edge_name == "stereo_t1")
            previous_stereo_cache = (
                image_block[1].path,
                image_block[3].path,
                next_stereo,
            )

            # 3. Filter valid matching edges.
            block_valid_edges = filter_valid_edges(
                block_pair_ret,
                len(image_block),
            )

            edge_rows.extend(
                make_edge_report_rows(
                    block_idx,
                    image_block,
                    block_pair_ret,
                    block_valid_edges,
                )
            )

            # 4. Initialize and optimize similarities with the rigid-stereo prior.
            (
                initial_affines,
                initialization_debug,
            ) = initialize_affines(
                len(image_block),
                block_valid_edges,
                image_block,
            )

            save_rigid_stereo_initialization_debug(
                block_idx,
                initialization_debug,
                block_dirs["rigid_stereo_debug"],
            )

            save_rigid_stereo_stage_debug(
                block_idx=block_idx,
                stage_name="affine_initial",
                transforms=initial_affines,
                edges=block_valid_edges,
                images=image_block,
                transform_kind="affine",
                out_dir=block_dirs["rigid_stereo_debug"],
            )

            (
                affine_transforms,
                affine_result,
                affine_initial_cost,
                affine_final_cost,
            ) = optimize_affines(
                initial_affines,
                block_valid_edges,
            )

            save_rigid_stereo_stage_debug(
                block_idx=block_idx,
                stage_name="affine_optimized",
                transforms=affine_transforms,
                edges=block_valid_edges,
                images=image_block,
                transform_kind="affine",
                out_dir=block_dirs["rigid_stereo_debug"],
            )

            if not affine_result.success:
                print(
                    "  Warning: affine optimization "
                    "did not report success: "
                    f"{affine_result.message}"
                )

            # 5. Refine projectively while retaining the local rigid-stereo prior.
            (
                projective_transforms,
                projective_result,
                projective_initial_cost,
                projective_final_cost,
            ) = optimize_projectives(
                affine_transforms,
                block_valid_edges,
                image_block,
            )

            save_rigid_stereo_stage_debug(
                block_idx=block_idx,
                stage_name="projective_optimized",
                transforms=projective_transforms,
                edges=block_valid_edges,
                images=image_block,
                transform_kind="projective",
                out_dir=block_dirs["rigid_stereo_debug"],
            )

            if not projective_result.success:
                print(
                    "  Warning: projective optimization "
                    "did not report success: "
                    f"{projective_result.message}"
                )

            # 6. Save block CSV data without rendering a local mosaic.
            save_block_transforms(
                block_idx,
                image_block,
                affine_transforms,
                block_dirs["transforms"],
                "affine",
            )

            # Save local projective transforms.
            save_block_transforms(
                block_idx,
                image_block,
                projective_transforms,
                block_dirs["transforms"],
                "projective",
            )

            # Assign globally unique keys to the block images.
            local_keys = [
                ("left", left_images[0].index),
                ("left", left_images[1].index),
                ("right", right_images[0].index),
                ("right", right_images[1].index),
            ]

            # 8-9. Persistent two-block similarity-backbone + local-correction accumulation
            #
            # Local projective transforms are solved independently:
            #   T_i : current image_i -> current block local reference
            #
            # Persistent transforms are jointly updated rather than overwritten using:
            #   block_to_global @ T_i
            # as a per-block composition.
            #
            # For subsequent blocks:
            #   previous block + current block -> 6 unique nodes
            # Fix the oldest stereo pair.
            # Optimize shared old and new transforms in one persistent window.
            # Ordinary edges use ordinary residuals.
            # Stereo edges use rigid-stereo residuals.
            persistent_applied = False
            persistent_initial_cost: Optional[float] = None
            persistent_final_cost: Optional[float] = None
            persistent_result = None

            start_new_segment = block_idx == 0 or previous_block_cache is None

            if not start_new_segment:
                previous_key_set = set(previous_block_cache.keys)
                current_key_set = set(local_keys)
                shared_keys = previous_key_set & current_key_set

                # Consecutive blocks must share both images of the preceding stereo frame.
                # Both shared transforms must already exist in the persistent state.
                if len(shared_keys) != 2 or any(
                    key not in global_transforms_by_key or key not in global_backbones_by_key
                    for key in shared_keys
                ):
                    print(
                        "  Warning: previous/current blocks do not have a complete "
                        "persistent shared stereo state; start a new global segment."
                    )
                    start_new_segment = True

            if start_new_segment:
                if block_idx > 0:
                    # Start a new segment if continuity with the previous global segment is lost.
                    global_images.clear()
                    global_transforms_by_key.clear()
                    global_backbones_by_key.clear()
                    global_image_keys.clear()
                    global_image_key_set.clear()

                current_global_transforms = np.stack(
                    [normalize_projective_homography(H) for H in projective_transforms],
                    axis=0,
                )
                (
                    current_global_backbones,
                    _,
                ) = decompose_global_transforms(
                    current_global_transforms,
                    image_block,
                )

                global_transforms_by_key.update(
                    {
                        key: current_global_transforms[local_idx].copy()
                        for local_idx, key in enumerate(local_keys)
                    }
                )
                global_backbones_by_key.update(
                    {
                        key: current_global_backbones[local_idx].copy()
                        for local_idx, key in enumerate(local_keys)
                    }
                )

                print(
                    "  Similarity-backbone state: initialize new global segment; "
                    "future blocks propagate S only, never the projective correction C."
                )

            else:
                assert previous_block_cache is not None

                (
                    active_keys,
                    active_images,
                    active_edges,
                ) = build_persistent_active_window(
                    previous_block_cache,
                    local_keys,
                    image_block,
                    block_valid_edges,
                )

                (
                    initial_active_backbones,
                    initial_active_transforms,
                ) = initialize_persistent_active_state(
                    active_keys=active_keys,
                    current_keys=local_keys,
                    current_images=image_block,
                    current_local_projective=projective_transforms,
                    global_transforms_by_key=global_transforms_by_key,
                    global_backbones_by_key=global_backbones_by_key,
                )

                current_key_set = set(local_keys)
                # The oldest pair has participated in two windows and leaves the active window.
                # Fix it while optimizing the shared and newly added nodes.
                fixed_keys = [
                    key for key in previous_block_cache.keys if key not in current_key_set
                ]
                if len(fixed_keys) != 2:
                    raise RuntimeError(
                        "Persistent two-block window should finalize exactly one "
                        f"old stereo pair, got fixed_keys={fixed_keys}."
                    )

                active_index = {key: idx for idx, key in enumerate(active_keys)}
                fixed_indices = {active_index[key] for key in fixed_keys}
                variable_indices = [
                    idx for idx in range(len(active_keys)) if idx not in fixed_indices
                ]

                (
                    optimized_active_transforms,
                    persistent_result,
                    persistent_initial_cost,
                    persistent_final_cost,
                    influential_window_edges,
                ) = optimize_persistent_window_projectives(
                    initial_global_backbones=initial_active_backbones,
                    initial_global_transforms=initial_active_transforms,
                    variable_indices=variable_indices,
                    window_edges=active_edges,
                    window_images=active_images,
                )

                # Absorb optimized similarity components into S without changing G; propagate only S next.
                (
                    optimized_active_transforms,
                    optimized_active_backbones,
                    optimized_active_corrections,
                ) = rebase_variable_global_states(
                    global_transforms=optimized_active_transforms,
                    old_backbones=initial_active_backbones,
                    images=active_images,
                    variable_indices=variable_indices,
                )
                persistent_applied = True

                print(
                    "  persistent cost: "
                    f"{persistent_initial_cost:.6f} -> "
                    f"{persistent_final_cost:.6f}; "
                    f"success={persistent_result.success}; "
                    f"active_factors={len(influential_window_edges)}"
                )

                # Commit G and S together so a projective correction cannot become the propagation backbone.
                global_transforms_by_key.update(
                    {
                        key: optimized_active_transforms[idx].copy()
                        for idx, key in enumerate(active_keys)
                    }
                )
                global_backbones_by_key.update(
                    {
                        key: optimized_active_backbones[idx].copy()
                        for idx, key in enumerate(active_keys)
                    }
                )

                save_block_transforms(
                    block_idx,
                    active_images,
                    optimized_active_transforms,
                    block_dirs["transforms"],
                    "persistent_window_global_projective",
                    active_keys,
                )
                save_block_transforms(
                    block_idx,
                    active_images,
                    optimized_active_backbones,
                    block_dirs["transforms"],
                    "persistent_window_similarity_backbone",
                    active_keys,
                )
                save_block_transforms(
                    block_idx,
                    active_images,
                    optimized_active_corrections,
                    block_dirs["transforms"],
                    "persistent_window_projective_correction",
                    active_keys,
                )

            # Read S, G, and C from the authoritative state for every branch.
            current_global_transforms = np.stack(
                [global_transforms_by_key[key] for key in local_keys],
                axis=0,
            )
            current_global_backbones = np.stack(
                [global_backbones_by_key[key] for key in local_keys],
                axis=0,
            )
            _, current_global_corrections = decompose_global_transforms(
                current_global_transforms,
                image_block,
                backbone_template=current_global_backbones,
            )

            # Save the current block's final persistent global transforms.
            save_block_transforms(
                block_idx,
                image_block,
                current_global_transforms,
                block_dirs["transforms"],
                "persistent_current_global_projective",
                local_keys,
            )
            save_block_transforms(
                block_idx,
                image_block,
                current_global_backbones,
                block_dirs["transforms"],
                "persistent_current_similarity_backbone",
                local_keys,
            )
            save_block_transforms(
                block_idx,
                image_block,
                current_global_corrections,
                block_dirs["transforms"],
                "persistent_current_projective_correction",
                local_keys,
            )

            # Append new image records without duplicating the shared frame.
            for local_idx, key in enumerate(local_keys):
                if key not in global_image_key_set:
                    global_image_key_set.add(key)
                    global_image_keys.append(key)
                    global_images.append(image_block[local_idx])

            # Assemble all accumulated global projective transforms.
            global_transforms = np.stack(
                [global_transforms_by_key[key] for key in global_image_keys],
                axis=0,
            )

            # Save all accumulated global projective transforms.
            save_block_transforms(
                block_idx,
                global_images,
                global_transforms,
                block_dirs["transforms"],
                "incremental_global_projective",
                global_image_keys,
            )

            global_backbones = np.stack(
                [global_backbones_by_key[key] for key in global_image_keys],
                axis=0,
            )
            _, global_corrections = decompose_global_transforms(
                global_transforms,
                global_images,
                backbone_template=global_backbones,
            )
            save_block_transforms(
                block_idx,
                global_images,
                global_backbones,
                block_dirs["transforms"],
                "incremental_global_similarity_backbone",
                global_image_keys,
            )
            save_block_transforms(
                block_idx,
                global_images,
                global_corrections,
                block_dirs["transforms"],
                "incremental_global_projective_correction",
                global_image_keys,
            )

            # Avoid rendering the accumulated mosaic inside the block loop.
            # This avoids repeatedly warping all previously processed images.
            #
            # Render the final accumulated mosaic once after the loop.

            block_rows.append(
                make_block_summary_row(
                    block_idx=block_idx,
                    images=image_block,
                    status="success",
                    candidate_edges=len(block_candidate_pairs),
                    valid_edges=len(block_valid_edges),
                    total_selected_matches=sum(edge.selected_matches for edge in block_valid_edges),
                    affine_initial_cost=(affine_initial_cost),
                    affine_final_cost=(affine_final_cost),
                    affine_success=bool(affine_result.success),
                    projective_initial_cost=(projective_initial_cost),
                    projective_final_cost=(projective_final_cost),
                    projective_success=bool(projective_result.success),
                    persistent_applied=persistent_applied,
                    persistent_initial_cost=(persistent_initial_cost),
                    persistent_final_cost=(persistent_final_cost),
                    persistent_success=(
                        bool(persistent_result.success) if persistent_result is not None else None
                    ),
                )
            )

            # Cache this successful block's original typed observations for the next window.
            # Cache block-local endpoints, not remapped active-window endpoints.
            previous_block_cache = PersistentBlockCache(
                keys=list(local_keys),
                images=list(image_block),
                valid_edges=list(block_valid_edges),
            )
        except Exception as exc:
            print(f"  Block {block_idx:04d} failed: {exc}")

            if block_pair_ret and not any(row.get("block_idx") == block_idx for row in edge_rows):
                edge_rows.extend(
                    make_edge_report_rows(
                        block_idx,
                        image_block,
                        block_pair_ret,
                        block_valid_edges,
                    )
                )

            # This block produced no trusted persistent update.
            # Clear the active-window cache to prevent joining nonconsecutive blocks.
            previous_block_cache = None

            block_rows.append(
                make_block_summary_row(
                    block_idx=block_idx,
                    images=image_block,
                    status="failed",
                    candidate_edges=len(block_candidate_pairs),
                    valid_edges=len(block_valid_edges),
                    total_selected_matches=sum(edge.selected_matches for edge in block_valid_edges),
                    error=str(exc),
                )
            )

        # The next window reuses only the second time step of this block.
        # Release inactive image pixels and discard descriptors that will not be reused.
        reusable_paths = {
            image_block[1].path,
            image_block[3].path,
        }
        feature_cache = {
            path: feature for path, feature in feature_cache.items() if path in reusable_paths
        }
        for stale_image in (image_block[0], image_block[2]):
            stale_image.image = None
            stale_image.gray = None

        # Save reports after each block.
        # Preserve completed statistics if a later block fails.
        write_report_csvs(
            out_dirs,
            block_rows,
            image_rows,
            edge_rows,
        )

    # Render one final accumulated mosaic after processing all blocks.
    if global_images and global_image_keys:
        final_global_transforms = np.stack(
            [global_transforms_by_key[key] for key in global_image_keys],
            axis=0,
        )

        final_mosaic_path = out_dirs["mosaics"] / "incremental_mosaic_final.jpg"

        print(f"\nGenerating final accumulated projective mosaic: {final_mosaic_path}")

        save_graphcut_mosaic_preview(
            global_images,
            final_global_transforms,
            final_mosaic_path,
        )

    else:
        print("Warning: no valid global images were accumulated; final mosaic was not generated.")

    print("Done. Block-level results are saved under OUTPUT_DIR.")
