# Algorithm definitions and implementation scope

This document describes the executable implementations in this repository. Both
methods estimate image-to-mosaic transformations from image correspondences. Neither
method estimates calibrated 3D camera poses, reconstructs depth, or uses geolocation
as an observation. A synchronized left/right input layout alone does not imply a
stereo disparity or epipolar constraint; the two methods treat it differently.

For image `i`, let `p_ik` denote a matched source pixel and let `w(H_i, p_ik)`
denote its inhomogeneous projection under the image-to-global matrix `H_i`.
All registration coordinates and errors refer to input-image pixels. Rendering
may downsample the canvas without changing the saved registration matrices.

## 1. Incremental visual graph with global registration

Source: [`src/stereo_uav/visual_graph`](../src/stereo_uav/visual_graph).
Compatibility entry point:
[`uav_stereo_incremental_visual_graph_global_registration.py`](../uav_stereo_incremental_visual_graph_global_registration.py).

### Input and candidate discovery

The loader synchronizes left and right filenames by leading integer frame ID.
Duplicate IDs within one stream are rejected. Unpaired frames are reported and
excluded. Sequential candidates use offsets 1, 2, and 4 in the synchronized stream,
not numerical differences between filename IDs.

SIFT features are extracted once per image. Incremental FLANN retrieval uses up
to 500 high-response descriptors per image. Each stereo pair queries historical
descriptors before adding its own descriptors. Retrieval excludes pairs less than
10 stream positions away and proposes at most 20 historical candidates.

For a candidate pair-to-pair relation, all four LL, LR, RL, and RR image
combinations undergo the same verification. Verification uses mutual Lowe-ratio
matches, homography RANSAC, inlier count and ratio, mean reprojection error, and
convex-hull spatial coverage in both images. Retrieval scores nominate candidates;
they are not optimization measurements.

### Pair topology and image observations

The topology distinguishes `tree`, `strong`, and `loop` roles. Tree relations
provide the retained connection to previously accepted pairs. Strong relations
provide additional support and are pruned when their degree limit is exceeded.
Loop relations must satisfy stricter image-level verification gates. Tree and
loop roles persist. The relation-selection logic materializes at most two
verified image observations per selected relation.

The image optimization graph deduplicates endpoint pairs, retains the higher-quality
observation, and combines topology roles. **Every image observation uses the same
ordinary 2D residual**, including same-time left/right matches. There is no separate
stereo rotation, scale, or disparity prior in this algorithm.

Disconnected image components are solved independently with separate reference
gauges. Their relative placement is not recovered by global registration; rendering
them does not establish a common geometric coordinate system.

### Two-stage registration

The first stage is named `affine` in output files for historical compatibility, but
its matrix is a four-parameter similarity:

```text
H_i = [[ a_i,  b_i, c_i],
       [-b_i,  a_i, f_i],
       [   0,    0,   1]]
```

One image per connected component is fixed to identity. The remaining transforms
minimize the sum of squared 2D matched-point differences. Translation parameters
are scaled by 5000. The system is linear in these parameters and is solved using
sparse LSMR; it is not a general six-degree-of-freedom affine fit.

The second stage uses eight-parameter homographies initialized by the first stage:

```text
H_i = [[a_i, b_i, c_i],
       [d_i, e_i, f_i],
       [g_i, h_i,   1]]
```

For a component with `N` images, `M` edges, and `P` selected matches per edge,
the implemented squared-residual objective is:

```text
E = (1 / (M P)) sum_edges sum_k ||w(H_i,p_ik) - w(H_j,p_jk)||^2
    + (omega / N) sum_variable_images ||q(H_i)||^2

q(H) = [a*b + d*e,
        a*a + d*d - 1,
        b*b + e*e - 1,
        g*g + h*h]
```

The default `omega` is 800. In particular, the last penalty is the square of
`g*g + h*h`, not a linear penalty on projective coefficients. This regularizer acts
on the original matrix entries; it is not the correction-matrix safety penalty
used by the second algorithm. SciPy least squares uses an analytic sparse Jacobian.
The reference matrix remains identity.

The match budget is 40 points on a 5-by-8 grid for up to 300 images, and 20 points
on a 5-by-4 grid above that threshold. Both endpoint arrays use identical selected
indices. All retained RANSAC inliers and the selected optimization points have
separate error reports.

### Execution and limitations

Graph construction is incremental. The global numerical solve runs after graph
construction, or during explicit replay of a saved final graph. The implementation
does not update a converged full-history solution at each incoming pair and does
not provide a bounded-latency incremental solver.

Rendering uses sequential Graph-Cut composition with bounded seam-solver resolution.
Planar homographies cannot generally eliminate depth-dependent parallax, moving
objects, illumination changes, or incorrect correspondences. A low pooled inlier
RMSE does not establish metric map accuracy or seamless rendering everywhere.

## 2. Block-incremental registration with a rigid-stereo similarity backbone

Source: [`src/stereo_uav/backbone`](../src/stereo_uav/backbone).
Compatibility entry point:
[`UAV_Binocular_Camera_block_incremental_rigid_stereo_similarity_backbone_core.py`](../UAV_Binocular_Camera_block_incremental_rigid_stereo_similarity_backbone_core.py).

### Fixed block and typed observations

Each block contains `[left_t, left_t+1, right_t, right_t+1]`. Its six candidate
edges comprise four ordinary edges and two same-time stereo edges `(0,2)` and
`(1,3)`. Consecutive blocks share one stereo pair. SIFT features and the shared
stereo observation are reused across adjacent blocks.

This loader extracts the **last numeric group** from each filename stem, whereas
the visual-graph loader uses the leading integer. Matching purely numeric stems
are the safest shared input convention. Do not append camera numbers after a frame
ID and assume both loaders will interpret the filename identically.

Matching uses SIFT, L2 KNN, Lowe's ratio test, homography RANSAC, and spatially
distributed match selection. Unlike the first method's mutual verifier, this
matching path applies a forward ratio test. Failed edges are filtered and the
remaining block graph must connect all four images to the reference.

### Similarity initialization and stereo residuals

Stage one uses the same four-parameter similarity matrix convention as above.
Initialization propagates ordinary temporal edges first, supplements with other
ordinary edges, and permits a valid stereo edge only as an initialization bridge
when needed. A bridge does not change the edge's objective type.

Ordinary edges constrain both coordinates of matched points. Stereo edges use
a shared transformed image-x direction `u_ij` and its perpendicular `n_ij`:

```text
r_perp,k = n_ij dot (w(H_i,p_ik) - w(H_j,p_jk))
r_theta  = wrap(theta_i - theta_j)
r_scale  = log(s_i / s_j)
```

The prior discourages relative rotation, relative scale, and normal displacement.
It leaves displacement parallel to `u_ij` unconstrained. The direction rotates
with the current pair; it is not fixed to the mosaic x axis. This is an image-plane
rigidity assumption, not a calibrated epipolar model or a recovered stereo baseline.

Ordinary residuals are normalized by the square root of the total ordinary match
count. Stereo point residuals are normalized by the square root of the stereo match
count; rotation and scale residuals by the square root of the stereo edge count.
The default normal weight is 9 with a 2-pixel sigma, rotation weight is 1 with a
0.5-degree sigma, and log-scale weight is 1 with a sigma of **0.5**. The previous
comment mentioning 0.02 did not describe the configured value.

The solver first performs an ordinary-only warm start, then applies stereo residual
multipliers `sqrt(alpha)` with `alpha` equal to 0.1, 0.3, and 1.0 while retaining
ordinary residuals. Similarity least squares uses componentwise Huber loss with
`f_scale=4.0`. Therefore `alpha` is an energy multiplier only in the quadratic
region; outside that region it also changes the effective robust threshold.
Reported initial/final costs use the full, unrobustified squared-residual objective
at `alpha=1`. These comparable diagnostics are distinct from SciPy's robust cost.

### Projective correction and propagation

Global transformations are represented as:

```text
G_i = S_i @ C_i
```

`S_i` is a similarity backbone. `C_i` is a local projective correction. The local
second stage holds `S_i` fixed and optimizes selected eight-parameter corrections.
Typed data residuals use the composed `G_i`; denominator and local-Jacobian
singular-value safety penalties act directly on `C_i` in image-normalized coordinates.
These are soft penalties, not guaranteed hard bounds or proofs against folding.

In the projective stereo term, rotation and scale come from the Jacobian at each
image center. Normal matched-point residuals use the pair's shared local direction.
This center-based approximation does not enforce an identical local frame over
the entire image.

Two adjacent blocks form a six-image persistent window. The oldest stereo pair is
fixed, while the shared and new images' corrections can change. Both blocks' valid
typed observations are retained. **The shared stereo observation occurs twice when
both blocks contain it**: the implementation concatenates the original factors and
does not deduplicate this observation. The refactor preserves that effective weighting.

The persistent solver discards factors whose endpoints are both fixed. For two
fully matched blocks, this leaves 11 influential factor entries: the oldest stereo
factor is excluded and the shared stereo factor still occurs twice. Local and
persistent projective least squares use `soft_l1` loss with `f_scale=4.0`; reported squared-residual
costs are not the same quantity as the solver's robust cost.

After optimization, similarity components are absorbed back into `S_i` while
preserving `G_i`. New blocks propagate only the similarity backbone, not a product
of accumulated projective corrections. This limits one mechanism of projective
drift, but it does not guarantee globally correct geometry.

### Execution and limitations

This is a fixed-lag local optimization method. Once images leave the active window,
their states are not jointly readjusted by a later full-history solve. There is no
long-range retrieval or global loop-closure mechanism in this implementation.
Accumulated transforms and final rendering still grow with sequence length; a
bounded optimization window is not a constant-memory whole application.

If successful-block continuity is lost, the original pipeline may start a new
segment. Such segments do not acquire a verified relative registration simply by
appearing in the same output canvas. Inspect block status reports before treating a
final mosaic as a single connected map.

Graph-Cut is disabled by default: later valid pixels overwrite earlier ones. The
CLI's `--graphcut` option enables the existing seam-selection path. Preview scaling
and reduced seam ROIs do not alter saved global transformations.

## Comparison

| Property | Incremental visual graph | Similarity backbone |
|---|---|---|
| Candidate structure | Sequential and retrieved historical pair relations | Consecutive four-image blocks |
| Same-time left/right observations | Ordinary full 2D residual | Special normal, rotation, and scale residuals |
| Numerical optimization scope | Full connected component after graph construction | Local block, then two-block window |
| Long-range constraints | Verified retained tree/loop relations | No long-range retrieval or loop closure |
| Projective regularization | Four penalties on global matrix entries | Denominator and singular-value penalties on local corrections |
| Propagation state | Graph traversal initialization followed by global solve | Similarity backbone with local projective corrections |
| Default compositing | Sequential Graph-Cut | Ordered overwrite; optional Graph-Cut |
| Evidence in this repository | Archived 140-image experiment and regression validation | Regression validation and bounded real-image smoke comparison |

The repository does not provide a controlled full-dataset comparison proving that
either method is more accurate or faster. Such a comparison requires the same
images, resolution, correspondence budget, rendering settings, and failure criteria.
