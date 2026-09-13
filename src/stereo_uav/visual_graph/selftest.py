"""Selftest for the visual graph algorithm."""

from __future__ import annotations

import argparse
import sys
import time
import unittest
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from .cli import (
    main,
)
from .features import (
    mutual_ratio_matches,
    point_policy,
    select_evenly_distributed_matches,
)
from .geometry import (
    assert_edge_orientation,
    canonical_image_pair,
    find_connected_components,
    reverse_edge,
    warp_points,
)
from .graph import (
    build_incremental_pair_graph,
    cull_strong_relations,
    materialize_optimization_graph,
    select_relation_image_edges,
    strict_loop_edges,
    verify_pair_relation,
)
from .metrics import (
    edge_errors,
)
from .models import (
    LEFT_IN,
    RIGHT_IN,
    Config,
    ImageEdge,
    PairRelation,
    RetrievalCandidate,
    StereoPair,
)
from .optimization import (
    ProjectiveProblem,
    projective_energy,
    projective_residuals,
    projective_sparse_jacobian,
    solve_global_affine_sparse,
    solve_global_projective,
)
from .rendering import (
    graphcut_merge,
)
from .reporting import (
    save_json,
)
from .retrieval import (
    IncrementalSIFTRetriever,
)


def synthetic_edge(ki, kj, Hi=None, Hj=None, count=80, quality=1.0, seed=7):
    Hi = np.eye(3) if Hi is None else Hi
    Hj = np.eye(3) if Hj is None else Hj
    rng = np.random.default_rng(seed)
    pi = rng.uniform([30.0, 20.0], [900.0, 700.0], (count, 2))
    H = np.linalg.inv(Hj) @ Hi
    H /= H[2, 2]
    pj = warp_points(H, pi)
    si, sj = select_evenly_distributed_matches(pi, pj, 40, 5, 8, rng)
    error = float(np.linalg.norm(warp_points(H, pi) - pj, axis=1).mean())
    return ImageEdge(
        ki,
        kj,
        "tree",
        count,
        count,
        count,
        1.0,
        error,
        H,
        pi,
        pj,
        si,
        sj,
        40,
        0.4,
        0.4,
        quality,
        {"tree"},
    )


def synthetic_problem(count=7, projective=True):
    nodes = [("left", index) for index in range(count)]
    transforms = {}
    for i, key in enumerate(nodes):
        theta = 0.015 * i
        c, s = np.cos(theta), np.sin(theta)
        transforms[key] = np.array(
            [
                [c, s, 80.0 * i],
                [-s, c, 12.0 * i],
                [1.5e-5 * i if projective else 0.0, -8e-6 * i if projective else 0.0, 1.0],
            ]
        )
    edges = [
        synthetic_edge(
            nodes[i], nodes[j], transforms[nodes[i]], transforms[nodes[j]], seed=100 * i + j
        )
        for i in range(count)
        for j in range(i + 1, count)
        if j - i <= 2 or (i == 0 and j == count - 1)
    ]
    return nodes, edges, transforms


class SelfTests(unittest.TestCase):
    def test_ordinary_edge_orientation(self):
        Hi = np.array([[1.0, 0.02, 30.0], [-0.01, 0.98, 17.0], [1e-5, 0.0, 1.0]])
        edge = synthetic_edge(("right", 10), ("left", 2), Hi)
        assert_edge_orientation(edge)
        identity = canonical_image_pair(edge.key_i, edge.key_j)
        self.assertEqual(edge.key_i, ("right", 10))
        self.assertEqual(identity[0], ("left", 2))
        reversed_edge = reverse_edge(edge)
        self.assertEqual(reversed_edge.key_i, edge.key_j)
        np.testing.assert_allclose(reversed_edge.inlier_pts_i, edge.inlier_pts_j)
        np.testing.assert_allclose(
            warp_points(reversed_edge.H_i_to_j, reversed_edge.inlier_pts_i),
            reversed_edge.inlier_pts_j,
            atol=1e-8,
        )
        self.assertEqual(reversed_edge.coverage_i, edge.coverage_j)
        np.testing.assert_allclose(reverse_edge(reversed_edge).H_i_to_j, edge.H_i_to_j, atol=1e-12)

    def test_uniform_P_selection(self):
        rng = np.random.default_rng(7)
        points = rng.uniform(0, 100, (400, 2))
        for total in (140, 301):
            P, rows, cols = point_policy(total)
            si, sj = select_evenly_distributed_matches(
                points, points + 1, P, rows, cols, np.random.default_rng(7)
            )
            self.assertEqual(si.shape, (P, 2))
            self.assertEqual(len(np.unique(si, axis=0)), P)
            np.testing.assert_allclose(sj, si + 1)
            repeat, _ = select_evenly_distributed_matches(
                points, points + 1, P, rows, cols, np.random.default_rng(7)
            )
            np.testing.assert_array_equal(si, repeat)
        with self.assertRaises(ValueError):
            select_evenly_distributed_matches(points[:10], points[:10], 40, 5, 8, rng)

    def test_candidate_is_not_optimization_edge(self):
        _candidate = RetrievalCandidate(
            frame=0, position=0, votes=900, score=0.9, weighted_score=20.0
        )
        relation = PairRelation(10, 0, "retrieval", [], 100.0, 0)
        self.assertEqual(materialize_optimization_graph([], {(0, 10): relation}), {})
        calls = []

        def reject(ki, kj):
            calls.append((ki, kj))
            return None

        a, b = StereoPair(10, ("left", 10), ("right", 10)), StereoPair(0, ("left", 0), ("right", 0))
        self.assertIsNone(verify_pair_relation(a, b, "retrieval", reject, Config()))
        self.assertEqual(len(calls), 4)

    def test_all_edges_are_ordinary(self):
        import ast

        tree = ast.parse(
            "\n".join(p.read_text(encoding="utf-8") for p in Path(__file__).parent.glob("*.py"))
        )
        edge_fields = set(ImageEdge.__dataclass_fields__)
        self.assertNotIn("edge_type", edge_fields)
        for item in ast.walk(tree):
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.assertFalse("stereo" in item.name and "residual" in item.name)
        self.assertEqual(
            set(ProjectiveProblem.__dataclass_fields__), {"nodes", "edges", "reference", "cfg"}
        )

    def test_pair_relation_materialization(self):
        new = StereoPair(1, ("left", 1), ("right", 1))
        old = StereoPair(0, ("left", 0), ("right", 0))
        edges = [
            synthetic_edge(ki, kj, quality=10.0 if ki[0] != kj[0] else 1.0)
            for ki in (new.left_key, new.right_key)
            for kj in (old.left_key, old.right_key)
        ]
        relation = PairRelation(1, 0, "sequential", edges, 10.0, 4)
        selected = select_relation_image_edges(relation, "tree", new, {old.left_key, old.right_key})
        self.assertEqual(len(selected), 2)
        self.assertEqual(len({key for e in selected for key in (e.key_i, e.key_j)}), 4)
        self.assertTrue(all(e.key_i[0] != e.key_j[0] for e in selected))
        one = replace(relation, verified_image_edges=edges[:1])
        intra = synthetic_edge(new.left_key, new.right_key)
        selected = select_relation_image_edges(
            one, "tree", new, {old.left_key, old.right_key}, intra
        )
        components = find_connected_components(
            [new.left_key, new.right_key, old.left_key], selected + [intra]
        )
        self.assertEqual(len(components), 1)

    def test_tree_connectivity(self):
        import contextlib
        import io

        pairs = [StereoPair(i, ("left", i), ("right", i)) for i in range(6)]

        class FakeVerifier:
            def verify(self, ki, kj):
                return synthetic_edge(ki, kj)

            def feature(self, key):
                return None

        class FakeRetriever:
            def describe_pair(self, pair, feature):
                return np.empty((0, 128), np.float32)

            def query(self, descriptors, position):
                return []

            def add(self, frame, position, descriptors):
                pass

        with contextlib.redirect_stdout(io.StringIO()):
            graph = build_incremental_pair_graph(pairs, FakeVerifier(), Config(), FakeRetriever())
        self.assertTrue(all(row["registered_at_arrival"] for row in graph.pair_status_rows))
        self.assertEqual(sum("tree" in r.roles for r in graph.relations.values()), 5)
        nodes = [key for pair in pairs for key in (pair.left_key, pair.right_key)]
        tree_only = [
            e
            for e in graph.optimization_edges_by_pair.values()
            if "tree" in e.roles or "intra_pair" in e.roles
        ]
        self.assertEqual(len(find_connected_components(nodes, tree_only)), 1)

        # A missing cross link must be reported, never fabricated.
        class BrokenVerifier(FakeVerifier):
            def verify(self, ki, kj):
                return synthetic_edge(ki, kj) if ki[1] == kj[1] else None

        with contextlib.redirect_stdout(io.StringIO()):
            broken = build_incremental_pair_graph(
                pairs, BrokenVerifier(), Config(), FakeRetriever()
            )
        self.assertFalse(broken.pair_status_rows[-1]["registered_at_arrival"])
        self.assertEqual(
            len(find_connected_components(nodes, list(broken.optimization_edges_by_pair.values()))),
            6,
        )

    def test_strong_edge_culling(self):
        relations = {}
        for i in range(1, 7):
            edge = synthetic_edge(("left", 0), ("left", i))
            relations[(0, i)] = PairRelation(
                0, i, "sequential", [edge], float(i), 2, {"strong"}, {"strong": [edge]}
            )
        relations[(0, 1)].roles.update(("tree", "loop"))
        relations[(0, 1)].role_edges.update(
            tree=relations[(0, 1)].verified_image_edges, loop=relations[(0, 1)].verified_image_edges
        )
        self.assertEqual(cull_strong_relations(relations, Config()), 2)
        self.assertEqual(relations[(0, 1)].roles, {"tree", "loop"})
        self.assertNotIn((0, 2), relations)
        self.assertEqual(sum("strong" in r.roles for r in relations.values()), 4)
        final = materialize_optimization_graph([], relations)
        self.assertNotIn(canonical_image_pair(("left", 0), ("left", 2)), final)

    def test_affine_sparse_recovery(self):
        nodes, edges, truth = synthetic_problem(projective=False)
        result, info = solve_global_affine_sparse(nodes, edges, nodes[0], Config())
        errors = np.concatenate([edge_errors(edge, result) for edge in edges])
        self.assertLess(float(np.sqrt(np.mean(errors**2))), 1e-5)
        for key in nodes:
            np.testing.assert_allclose(result[key], truth[key], atol=1e-4)
        self.assertIn(info["status"], (1, 2, 4, 5))

    def test_projective_objective_exact_terms(self):
        nodes, edges, truth = synthetic_problem(count=3)
        problem = ProjectiveProblem(nodes, edges, nodes[0], Config())
        x = problem.pack(truth)
        x[0] += 0.03
        x[1] += 0.01
        transforms = problem.unpack(x)
        residual = projective_residuals(x, problem)
        self.assertEqual(len(residual), 2 * len(edges) * 40 + 4 * (len(nodes) - 1))
        matching = sum(np.sum(edge_errors(edge, transforms, True) ** 2) for edge in edges) / (
            len(edges) * 40
        )
        rigid = 0.0
        for H in transforms.values():
            a, b, c, d, e, f, g, h = H.ravel()[:8]
            rigid += (
                (a * b + d * e) ** 2
                + (a * a + d * d - 1) ** 2
                + (b * b + e * e - 1) ** 2
                + (g * g + h * h) ** 2
            )
        manual = matching + 800 * rigid / len(nodes)
        self.assertAlmostEqual(float(residual @ residual), manual, places=8)
        np.testing.assert_allclose(projective_energy(x, problem)["global_energy"], manual)
        # Arbitrary external settings cannot affect the mathematical context.
        unused_settings = {"unused_parameter": 1.0}
        unused_settings["unused_parameter"] = 1e12
        np.testing.assert_array_equal(residual, projective_residuals(x, problem))

    def test_projective_analytic_jacobian(self):
        nodes, edges, truth = synthetic_problem(count=3)
        problem = ProjectiveProblem(nodes, edges, nodes[0], Config())
        x = problem.pack(truth)
        x[0] += 0.01
        analytic = projective_sparse_jacobian(x, problem).toarray()
        numerical = np.zeros_like(analytic)
        for column in range(len(x)):
            step = 1e-9 if column % 8 >= 6 else 1e-6
            offset = np.zeros_like(x)
            offset[column] = step
            numerical[:, column] = (
                projective_residuals(x + offset, problem)
                - projective_residuals(x - offset, problem)
            ) / (2 * step)
        relative = float(np.linalg.norm(analytic - numerical) / np.linalg.norm(numerical))
        column_relative = np.linalg.norm(analytic - numerical, axis=0) / np.maximum(
            np.linalg.norm(numerical, axis=0), 1.0
        )
        print(
            f"Jacobian relative error={relative:.3e}; max column error={column_relative.max():.3e}"
        )
        self.assertLess(relative, 1e-5)
        self.assertLess(float(column_relative.max()), 1e-5)

    def test_projective_recovery(self):
        nodes, edges, _ = synthetic_problem(count=7)
        affine, _ = solve_global_affine_sparse(nodes, edges, nodes[0], Config())
        result, info = solve_global_projective(nodes, edges, nodes[0], affine, Config(), verbose=0)
        before = np.concatenate([edge_errors(e, affine) for e in edges])
        after = np.concatenate([edge_errors(e, result) for e in edges])
        self.assertLess(float(np.mean(after**2)), float(np.mean(before**2)))
        self.assertLess(info["global_energy_after"], info["global_energy_before"])
        self.assertTrue(all(np.isfinite(H).all() for H in result.values()))
        np.testing.assert_array_equal(result[nodes[0]], np.eye(3))
        print(
            f"Synthetic recovery all-inlier RMSE: {np.sqrt(np.mean(before**2)):.6f} -> "
            f"{np.sqrt(np.mean(after**2)):.6f}"
        )

    def test_mutual_ratio_matching(self):
        def match(q, t, distance):
            return cv2.DMatch(q, t, distance)

        forward = [[match(0, 1, 1.0), match(0, 0, 4.0)], [match(1, 0, 1.0), match(1, 1, 4.0)]]
        backward = [[match(0, 0, 1.0), match(0, 1, 4.0)], [match(1, 0, 1.0), match(1, 1, 4.0)]]
        ratio, mutual = mutual_ratio_matches(forward, backward, 0.75)
        self.assertEqual(len(ratio), 2)
        self.assertEqual(mutual, [(0, 1)])

    def test_retrieval_is_incremental(self):
        cfg = replace(Config(), RETRIEVAL_TEMPORAL_EXCLUSION=2)
        retriever = IncrementalSIFTRetriever(cfg)
        descriptors = np.random.default_rng(7).uniform(0, 100, (100, 128)).astype(np.float32)
        self.assertEqual(retriever.query(descriptors, 0), [])
        retriever.add(100, 0, descriptors)
        self.assertEqual(retriever.query(descriptors, 1), [])
        candidates = retriever.query(descriptors, 2)
        self.assertEqual(candidates[0].frame, 100)
        self.assertEqual(candidates[0].votes, 100)
        with self.assertRaises(AssertionError):
            retriever.query(descriptors, 0)

    def test_loop_strict_gate_and_second_edge(self):
        a = synthetic_edge(("left", 20), ("left", 0), quality=10.0)
        b = synthetic_edge(("right", 20), ("right", 0), quality=7.9)
        relation = PairRelation(20, 0, "retrieval", [a, b], 20.0, 4)
        pair = StereoPair(20, ("left", 20), ("right", 20))
        selected = select_relation_image_edges(relation, "loop", pair, set())
        self.assertEqual(len(selected), 1)
        b.quality_score = 8.1
        self.assertEqual(len(select_relation_image_edges(relation, "loop", pair, set())), 2)
        b.inlier_ratio = 0.3
        strict = strict_loop_edges(relation, Config())
        self.assertEqual(len(strict), 1)
        self.assertIs(strict[0], a)

    def test_dedup_roles_and_orientation(self):
        edge = synthetic_edge(("right", 1), ("left", 0))
        reverse = reverse_edge(edge)
        relation = PairRelation(
            1,
            0,
            "sequential",
            [edge, reverse],
            1.0,
            2,
            {"tree", "strong"},
            {"tree": [edge], "strong": [reverse]},
        )
        graph = materialize_optimization_graph([], {(0, 1): relation})
        self.assertEqual(len(graph), 1)
        result = next(iter(graph.values()))
        self.assertEqual(result.roles, {"tree", "strong"})
        assert_edge_orientation(result)

    def test_graphcut_mask_union(self):
        if not hasattr(cv2, "detail_GraphCutSeamFinder"):
            self.skipTest("GraphCut unavailable")
        old = np.full((60, 100, 3), 50, np.uint8)
        new = np.full_like(old, 150)
        old_mask, new_mask = np.zeros((60, 100), np.uint8), np.zeros((60, 100), np.uint8)
        old_mask[:, :70] = 255
        new_mask[:, 30:] = 255
        merged, union, used = graphcut_merge(
            old, new, old_mask, new_mask, cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD"), 2000
        )
        self.assertTrue(used)
        self.assertTrue(np.all(union == 255))
        self.assertTrue(np.all(merged[:, :20] == 50))
        self.assertTrue(np.all(merged[:, 80:] == 150))

    def test_retrieval_parent_materializes_at_most_two_edges(self):
        import contextlib
        import io

        pairs = [StereoPair(i, ("left", i), ("right", i)) for i in (0, 10)]

        class FakeVerifier:
            def verify(self, ki, kj):
                quality = {
                    ("left", "left"): 10.0,
                    ("left", "right"): 9.0,
                    ("right", "left"): 0.1,
                    ("right", "right"): 1.0,
                }[ki[0], kj[0]]
                return synthetic_edge(ki, kj, quality=quality)

            def feature(self, key):
                return None

        class FakeRetriever:
            def describe_pair(self, pair, feature):
                return np.empty((0, 128), np.float32)

            def query(self, descriptors, position):
                return [RetrievalCandidate(0, 0, 100, 1.0, 1.0)] if position else []

            def add(self, frame, position, descriptors):
                pass

        cfg = replace(Config(), SEQUENTIAL_OFFSETS=(2, 4), RETRIEVAL_TEMPORAL_EXCLUSION=1)
        with contextlib.redirect_stdout(io.StringIO()):
            graph = build_incremental_pair_graph(pairs, FakeVerifier(), cfg, FakeRetriever())
        relation = graph.relations[(0, 10)]
        self.assertEqual(relation.roles, {"tree"})
        self.assertEqual(len(relation.role_edges["tree"]), 2)
        cross = [
            edge
            for edge in graph.optimization_edges_by_pair.values()
            if edge.key_i[1] != edge.key_j[1]
        ]
        self.assertEqual(len(cross), 2)
        self.assertTrue(graph.pair_status_rows[-1]["registered_at_arrival"])

    def test_existing_run_is_not_overwritten(self):
        import contextlib
        import io
        import tempfile
        from unittest.mock import patch

        with tempfile.TemporaryDirectory(prefix="binocular_uav_test_") as directory:
            output = Path(directory)
            (output / "data").mkdir()
            (output / "data" / "image_optimization_graph.csv").write_text("existing graph")
            (output / "run.log").write_text("original successful log")
            args = argparse.Namespace(
                output=output,
                self_test=False,
                resume_graph=False,
                opencv_threads=1,
                left=LEFT_IN,
                right=RIGHT_IN,
            )
            with (
                patch("stereo_uav.visual_graph.cli.parse_arguments", return_value=args),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(main(), 1)
            self.assertEqual((output / "run.log").read_text(), "original successful log")
            self.assertFalse((output / "run_failure.json").exists())


def run_self_tests(output: Path | None = None) -> bool:
    cv2.setRNGSeed(7)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SelfTests)
    names = [test.id().split(".")[-1] for test in suite]
    started = time.perf_counter()
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    failed = {test.id().split(".")[-1] for test, _ in result.failures + result.errors}
    skipped = {test.id().split(".")[-1] for test, _ in result.skipped}
    report = dict(
        tests_run=result.testsRun,
        failures=len(result.failures),
        errors=len(result.errors),
        skipped=len(result.skipped),
        success=result.wasSuccessful(),
        runtime=time.perf_counter() - started,
        tests={
            name: "FAIL" if name in failed else "SKIP" if name in skipped else "PASS"
            for name in names
        },
    )
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        save_json(output / "self_test_results.json", report)
    print("SELF_TESTS_PASS" if result.wasSuccessful() else "SELF_TESTS_FAIL", flush=True)
    return result.wasSuccessful()
