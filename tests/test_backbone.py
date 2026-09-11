"""Regression checks for typed observations and similarity-only propagation."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from stereo_uav.backbone import config
from stereo_uav.backbone.geometry import (
    compose_similarity_corrections,
    decompose_global_transforms,
    pack_affine_params,
    pack_variable_projective_params,
    unpack_affine_params,
    unpack_variable_projective_params,
)
from stereo_uav.backbone.matching import build_candidate_edges
from stereo_uav.backbone.models import ImageRecord, PairMatch_Edge, PersistentBlockCache
from stereo_uav.backbone.persistent import build_persistent_active_window
from stereo_uav.backbone.projective import (
    projective_native_safety_residuals,
    projective_objective_expected_residual_size,
    projective_objective_residuals,
)
from stereo_uav.backbone.similarity import affine_ordinary_residuals, affine_stereo_residuals


def records(count):
    return [
        ImageRecord(i, Path(f"{i}.png"), f"{i}.png", None, None, 640, 480) for i in range(count)
    ]


def edge(i=0, j=1, kind="stereo"):
    points = np.array([[20.0, 30.0], [100.0, 200.0], [300.0, 100.0], [500.0, 350.0]])
    return PairMatch_Edge(
        i,
        j,
        kind,
        selected_matches=len(points),
        selected_pts_i=points,
        selected_pts_j=points.copy(),
    )


def test_six_edge_block_retains_stereo_types():
    specs = build_candidate_edges(4)
    assert len(specs) == 6
    assert {(s.i, s.j) for s in specs if s.edge_type == "stereo"} == {(0, 2), (1, 3)}
    assert sum(s.edge_type == "ordinary" for s in specs) == 4
    with pytest.raises(ValueError):
        build_candidate_edges(3)


def test_stereo_keeps_parallel_disparity_free_but_penalizes_normal_error():
    transforms = np.tile(np.eye(3), (2, 1, 1))
    transforms[1, 0, 2] = 80.0
    np.testing.assert_allclose(affine_stereo_residuals(transforms, [edge()], 1.0), 0.0, atol=1e-12)
    assert np.linalg.norm(affine_ordinary_residuals(transforms, [edge(kind="ordinary")])) > 0
    transforms[1, 1, 2] = 4.0
    assert np.linalg.norm(affine_stereo_residuals(transforms, [edge()], 1.0)) > 0


def test_similarity_and_projective_packing_preserve_fixed_gauge():
    transforms = np.tile(np.eye(3), (3, 1, 1))
    transforms[1] = [[1.1, 0.2, 2000.0], [-0.2, 1.1, -3000.0], [0, 0, 1]]
    np.testing.assert_allclose(unpack_affine_params(pack_affine_params(transforms), 3), transforms)
    transforms[2, 2, 0] = 1e-5
    packed = pack_variable_projective_params(transforms, [1, 2])
    actual = unpack_variable_projective_params(packed, transforms, [1, 2])
    np.testing.assert_allclose(actual, transforms)


def test_backbone_decomposition_preserves_global_warp():
    global_transforms = np.tile(np.eye(3), (3, 1, 1))
    global_transforms[1] = [[1.03, 0.03, 90], [-0.02, 0.98, 40], [2e-5, -3e-5, 1]]
    global_transforms[2] = [[0.98, -0.03, 180], [0.02, 1.02, 30], [-1e-5, 2e-5, 1]]
    images = records(3)
    backbones, corrections = decompose_global_transforms(global_transforms, images)
    np.testing.assert_allclose(
        compose_similarity_corrections(backbones, corrections), global_transforms, atol=1e-10
    )
    np.testing.assert_allclose(backbones[:, 2], np.tile([0, 0, 1], (3, 1)))
    np.testing.assert_allclose(backbones[:, 0, 0], backbones[:, 1, 1])
    np.testing.assert_allclose(backbones[:, 0, 1], -backbones[:, 1, 0])


def test_native_safety_penalizes_near_collapsed_correction():
    corrections = np.tile(np.eye(3), (2, 1, 1))
    safe = projective_native_safety_residuals(corrections, records(2), [1])
    corrections[1, 0, 0] = 1e-6
    collapsed = projective_native_safety_residuals(corrections, records(2), [1])
    assert safe.shape == collapsed.shape
    assert np.isfinite(collapsed).all()
    assert np.linalg.norm(collapsed) > np.linalg.norm(safe)


def test_projective_objective_has_typed_residual_dimensions():
    transforms = np.tile(np.eye(3), (2, 1, 1))
    edges = [edge(), edge(kind="ordinary")]
    residual = projective_objective_residuals(transforms, transforms, edges, records(2), [1])
    assert len(residual) == projective_objective_expected_residual_size(edges, 1)
    assert np.isfinite(residual).all()


def test_two_block_window_preserves_original_observation_multiplicity():
    old_keys = [("left", 0), ("left", 1), ("right", 0), ("right", 1)]
    new_keys = [("left", 1), ("left", 2), ("right", 1), ("right", 2)]
    edges = [
        replace(edge(s.i, s.j, s.edge_type), edge_name=s.edge_name) for s in config.BLOCK_EDGE_SPECS
    ]
    previous = PersistentBlockCache(old_keys, records(4), edges)
    keys, images, active = build_persistent_active_window(previous, new_keys, records(4), edges)
    assert len(keys) == len(images) == 6
    # Both original blocks contribute the shared stereo observation. This is intentional
    # compatibility with the original weighting, not a claim of factor deduplication.
    assert len(active) == 12
    assert len({(e.i, e.j) for e in active}) == 11
    assert {(e.i, e.j, e.edge_type) for e in edges} == {
        (s.i, s.j, s.edge_type) for s in config.BLOCK_EDGE_SPECS
    }
    with pytest.raises(ValueError):
        build_persistent_active_window(
            previous, [(s, t + 5) for s, t in new_keys], records(4), edges
        )
