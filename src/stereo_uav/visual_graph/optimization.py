"""Optimization for the visual graph algorithm."""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import lsmr

from .geometry import (
    signed_denominator,
)
from .models import (
    Config,
    ImageEdge,
    ImageKey,
)


def initial_graph_transforms(
    nodes: list[ImageKey], edges: list[ImageEdge], reference: ImageKey
) -> dict[ImageKey, np.ndarray]:
    """Measured-H traversal used ONLY for the pre-affine diagnostic baseline."""
    adjacency: defaultdict = defaultdict(list)
    for edge in edges:
        adjacency[edge.key_i].append((edge, True))
        adjacency[edge.key_j].append((edge, False))
    transforms = {reference: np.eye(3)}
    queue = deque([reference])
    while queue:
        key = queue.popleft()
        for edge, forward in sorted(adjacency[key], key=lambda pair: -pair[0].quality_score):
            other = edge.key_j if forward else edge.key_i
            if other in transforms:
                continue
            # p_j = H_ij p_i, so global H_j = global H_i inv(H_ij).
            step = np.linalg.inv(edge.H_i_to_j) if forward else edge.H_i_to_j
            H = transforms[key] @ step
            H /= H[2, 2]
            if not np.isfinite(H).all():
                raise ValueError("Non-finite initial graph traversal")
            transforms[other] = H
            queue.append(other)
    assert set(transforms) == set(nodes)
    return transforms


def solve_global_affine_sparse(
    nodes: list[ImageKey], edges: list[ImageEdge], reference: ImageKey, cfg: Config
) -> tuple[dict[ImageKey, np.ndarray], dict]:
    """4DoF shared a=e, b=-d. The fixed identity contribution moves to RHS."""
    started = time.perf_counter()
    variables = [key for key in nodes if key != reference]
    indices = {key: index for index, key in enumerate(variables)}
    if not variables:
        return {reference: np.eye(3)}, dict(
            stage="affine", status=0, message="singleton", runtime=0.0
        )
    assert edges
    P = edges[0].selected_matches
    assert all(edge.selected_matches == P for edge in edges)
    residual_count = 2 * len(edges) * P
    rhs = np.zeros(residual_count)
    row_parts, col_parts, value_parts = [], [], []
    for edge_index, edge in enumerate(edges):
        rows = np.arange(edge_index * 2 * P, (edge_index + 1) * 2 * P).reshape(P, 2)
        for key, points, sign in (
            (edge.key_i, edge.selected_pts_i, 1.0),
            (edge.key_j, edge.selected_pts_j, -1.0),
        ):
            if key == reference:
                rhs[rows.ravel()] -= sign * points.ravel()
                continue
            x, y = points.T
            coefficients = np.zeros((P, 2, 4))
            coefficients[:, 0, 0] = x
            coefficients[:, 0, 1] = y
            coefficients[:, 0, 2] = cfg.TRANSLATION_SCALE
            coefficients[:, 1, 0] = y
            coefficients[:, 1, 1] = -x
            coefficients[:, 1, 3] = cfg.TRANSLATION_SCALE
            row_parts.append(np.repeat(rows.ravel(), 4))
            col_parts.append(np.tile(indices[key] * 4 + np.arange(4), P * 2))
            value_parts.append(sign * coefficients.ravel())
    A = coo_matrix(
        (np.concatenate(value_parts), (np.concatenate(row_parts), np.concatenate(col_parts))),
        shape=(residual_count, 4 * len(variables)),
    ).tocsr()
    A.eliminate_zeros()
    solution = lsmr(
        A, rhs, atol=1e-11, btol=1e-11, conlim=1e12, maxiter=max(2000, 20 * len(variables))
    )
    if solution[1] not in (0, 1, 2, 4, 5):
        raise RuntimeError(f"Affine LSMR did not converge: istop={solution[1]}")
    transforms = {reference: np.eye(3)}
    for key, index in indices.items():
        a, b, c, f = solution[0][index * 4 : (index + 1) * 4]
        transforms[key] = np.array(
            [[a, b, c * cfg.TRANSLATION_SCALE], [-b, a, f * cfg.TRANSLATION_SCALE], [0.0, 0.0, 1.0]]
        )
    assert all(np.isfinite(H).all() for H in transforms.values())
    return transforms, dict(
        stage="affine",
        solver="sparse linear LSMR",
        status=solution[1],
        iterations=solution[2],
        residual_norm=solution[3],
        normal_residual_norm=solution[4],
        condition_estimate=solution[6],
        matrix_nnz=A.nnz,
        runtime=time.perf_counter() - started,
    )


@dataclass
class ProjectiveProblem:
    nodes: list[ImageKey]
    edges: list[ImageEdge]
    reference: ImageKey
    cfg: Config

    def __post_init__(self):
        self.variables = [key for key in self.nodes if key != self.reference]
        self.variable_index = {key: i for i, key in enumerate(self.variables)}
        self.node_index = {key: i for i, key in enumerate(self.nodes)}
        self.variable_node_indices = np.array([self.node_index[key] for key in self.variables], int)
        self.N, self.M = len(self.nodes), len(self.edges)
        self.P = self.edges[0].selected_matches if self.edges else 0
        assert all(e.selected_matches == self.P for e in self.edges)
        self.data_scale = 1.0 / math.sqrt(self.M * self.P) if self.M else 0.0
        self.rigid_scale = math.sqrt(self.cfg.OMEGA_RIGID / self.N)
        self.points = (
            np.stack(
                [
                    np.stack([e.selected_pts_i for e in self.edges]),
                    np.stack([e.selected_pts_j for e in self.edges]),
                ]
            )
            if self.M
            else np.empty((2, 0, 0, 2))
        )
        self.endpoint_node_indices = np.array(
            [
                [self.node_index[e.key_i] for e in self.edges],
                [self.node_index[e.key_j] for e in self.edges],
            ],
            int,
        )
        self.endpoint_variable_indices = np.array(
            [
                [self.variable_index.get(e.key_i, -1) for e in self.edges],
                [self.variable_index.get(e.key_j, -1) for e in self.edges],
            ],
            int,
        )
        self.data_rows = 2 * self.M * self.P
        self.residual_count = self.data_rows + 4 * len(self.variables)
        # Build sparsity once; only values change between nonlinear iterations.
        rows, cols = [], []
        for side in range(2):
            valid = self.endpoint_variable_indices[side] >= 0
            row = np.arange(self.data_rows).reshape(self.M, self.P, 2)[valid]
            col = self.endpoint_variable_indices[side, valid, None, None, None] * 8 + np.arange(8)
            rows.append(np.broadcast_to(row[..., None], (*row.shape, 8)).ravel())
            cols.append(np.broadcast_to(col, (*row.shape, 8)).ravel())
        rigid_rows = self.data_rows + np.arange(4 * len(self.variables)).reshape(-1, 4, 1)
        rigid_cols = np.arange(len(self.variables))[:, None, None] * 8 + np.arange(8)
        rows.append(np.broadcast_to(rigid_rows, (len(self.variables), 4, 8)).ravel())
        cols.append(np.broadcast_to(rigid_cols, (len(self.variables), 4, 8)).ravel())
        self.jac_rows, self.jac_cols = np.concatenate(rows), np.concatenate(cols)

    def pack(self, transforms: dict[ImageKey, np.ndarray]) -> np.ndarray:
        packed = np.array([transforms[key].ravel()[:8] for key in self.variables]).reshape(-1, 8)
        packed[:, [2, 5]] /= self.cfg.SIGMA_TR
        return packed.ravel()

    def matrices(self, x: np.ndarray) -> np.ndarray:
        matrices = np.tile(np.eye(3), (self.N, 1, 1))
        packed = x.reshape(-1, 8).copy()
        packed[:, [2, 5]] *= self.cfg.SIGMA_TR
        matrices[self.variable_node_indices, :2, :] = packed[:, :6].reshape(-1, 2, 3)
        matrices[self.variable_node_indices, 2, :2] = packed[:, 6:8]
        return matrices

    def unpack(self, x: np.ndarray) -> dict[ImageKey, np.ndarray]:
        return dict(zip(self.nodes, self.matrices(x)))


def projective_warps(x: np.ndarray, problem: ProjectiveProblem):
    H = problem.matrices(x)
    endpoints = H[problem.endpoint_node_indices]
    numerator = np.einsum("semp,seqp->seqm", endpoints[:, :, :2, :2], problem.points)
    numerator += endpoints[:, :, None, :2, 2]
    raw_w = np.einsum("sep,seqp->seq", endpoints[:, :, 2, :2], problem.points) + 1.0
    w = signed_denominator(raw_w)
    return H, numerator / w[..., None], w, raw_w


def rigid_q(matrices: np.ndarray) -> np.ndarray:
    a, b, d, e = (matrices[:, 0, 0], matrices[:, 0, 1], matrices[:, 1, 0], matrices[:, 1, 1])
    g, h = matrices[:, 2, 0], matrices[:, 2, 1]
    return np.stack(
        (a * b + d * e, a * a + d * d - 1.0, b * b + e * e - 1.0, g * g + h * h), axis=1
    )


def projective_residuals(x: np.ndarray, problem: ProjectiveProblem) -> np.ndarray:
    H, uv, _, _ = projective_warps(x, problem)
    data = (uv[0] - uv[1]).ravel() * problem.data_scale
    # Exactly four Eq.(16) residuals per non-reference image. Identity contributes zero.
    rigid = rigid_q(H[problem.variable_node_indices]).ravel() * problem.rigid_scale
    return np.concatenate((data, rigid))


def projective_sparse_jacobian(x: np.ndarray, problem: ProjectiveProblem) -> csr_matrix:
    H, uv, w, raw_w = projective_warps(x, problem)
    values = []
    for side, sign in ((0, 1.0), (1, -1.0)):
        valid = problem.endpoint_variable_indices[side] >= 0
        points, denominator, warped = problem.points[side, valid], w[side, valid], uv[side, valid]
        xy_over_w = points / denominator[..., None]
        blocks = np.zeros((int(valid.sum()), problem.P, 2, 8))
        blocks[:, :, 0, :2] = xy_over_w
        blocks[:, :, 0, 2] = problem.cfg.SIGMA_TR / denominator
        blocks[:, :, 1, 3:5] = xy_over_w
        blocks[:, :, 1, 5] = problem.cfg.SIGMA_TR / denominator
        blocks[:, :, :, 6:8] = -warped[..., None] * xy_over_w[:, :, None, :]
        # Inside the epsilon branch w is constant; its derivative is zero.
        blocks[:, :, :, 6:8] *= (np.abs(raw_w[side, valid]) >= 1e-10)[:, :, None, None]
        values.append((blocks * sign * problem.data_scale).ravel())
    variable_H = H[problem.variable_node_indices]
    a, b, d, e = (
        variable_H[:, 0, 0],
        variable_H[:, 0, 1],
        variable_H[:, 1, 0],
        variable_H[:, 1, 1],
    )
    g, h = variable_H[:, 2, 0], variable_H[:, 2, 1]
    rigid = np.zeros((len(problem.variables), 4, 8))
    rigid[:, 0, 0], rigid[:, 0, 1], rigid[:, 0, 3], rigid[:, 0, 4] = b, a, e, d
    rigid[:, 1, 0], rigid[:, 1, 3] = 2 * a, 2 * d
    rigid[:, 2, 1], rigid[:, 2, 4] = 2 * b, 2 * e
    rigid[:, 3, 6], rigid[:, 3, 7] = 2 * g, 2 * h
    values.append((rigid * problem.rigid_scale).ravel())
    return coo_matrix(
        (np.concatenate(values), (problem.jac_rows, problem.jac_cols)),
        shape=(problem.residual_count, 8 * len(problem.variables)),
    ).tocsr()


def projective_energy(x: np.ndarray, problem: ProjectiveProblem) -> dict[str, float]:
    residual = projective_residuals(x, problem)
    data_energy = float(residual[: problem.data_rows] @ residual[: problem.data_rows])
    raw_rigid = rigid_q(problem.matrices(x))
    rigid_energy = float(np.sum(raw_rigid**2) / problem.N)
    return dict(
        data_energy=data_energy,
        rigid_energy=rigid_energy,
        global_energy=data_energy + problem.cfg.OMEGA_RIGID * rigid_energy,
    )


def solve_global_projective(
    nodes: list[ImageKey],
    edges: list[ImageEdge],
    reference: ImageKey,
    affine: dict[ImageKey, np.ndarray],
    cfg: Config,
    verbose: int = 2,
) -> tuple[dict[ImageKey, np.ndarray], dict]:
    if len(nodes) == 1:
        return {reference: np.eye(3)}, dict(
            stage="projective", status=0, message="singleton", runtime=0.0
        )
    started = time.perf_counter()
    problem = ProjectiveProblem(nodes, edges, reference, cfg)
    x0 = problem.pack(affine)
    before = projective_energy(x0, problem)
    result = least_squares(
        projective_residuals,
        x0,
        jac=projective_sparse_jacobian,
        args=(problem,),
        method="trf",
        tr_solver="lsmr",
        loss="linear",
        x_scale="jac",
        max_nfev=cfg.MAX_NFEV,
        verbose=verbose,
    )
    after = projective_energy(result.x, problem)
    transforms = problem.unpack(result.x)
    finite = (
        all(np.isfinite(H).all() for H in transforms.values()) and np.isfinite(result.fun).all()
    )
    if not finite:
        raise RuntimeError("Projective optimization failed: non-finite result")
    if after["global_energy"] > before["global_energy"] + 1e-8 * max(1.0, before["global_energy"]):
        raise RuntimeError("Projective optimization failed: objective increased")
    if not result.success:
        print(f"WARNING: projective optimizer did not converge: {result.message}", flush=True)
    summary = dict(
        stage="projective",
        solver="TRF/LSMR analytic sparse Jacobian",
        nfev=result.nfev,
        njev=result.njev,
        cost=result.cost,
        optimality=result.optimality,
        status=result.status,
        message=result.message,
        converged=bool(result.success),
        finite=finite,
        runtime=time.perf_counter() - started,
        M=problem.M,
        P=problem.P,
        N=problem.N,
        omega_rigid=cfg.OMEGA_RIGID,
        sigma_tr=cfg.SIGMA_TR,
    )
    for name, energy in (("before", before), ("after", after)):
        summary.update({f"{term}_{name}": value for term, value in energy.items()})
    assert np.isclose(2 * result.cost, after["global_energy"], rtol=1e-10)
    return transforms, summary
