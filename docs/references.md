# Research provenance and references

This document identifies the research basis of the implementations without claiming
that repository-specific engineering choices were proposed by the cited authors.

## Which paper is the primary source?

The primary source is **Wang, Fu, and Xu (2025)** [1]. Both implementations retain
its central two-stage pattern: similarity-constrained registration provides the
initialization for projective refinement. The incremental visual-graph method applies
that model to each complete connected component and additionally retains the paper's
normalized projective correspondence term, rigid matrix regularizer, translation
scale `sigma_tr = 5000`, default rigid weight `omega = 800`, spatially distributed
correspondence sampling, and sequential Graph-Cut composition. The backbone method
instead applies the two stages within bounded local windows.

Accordingly, the repository should be described as an **adaptation and extension of
Wang et al.**, not as an independent reproduction and not as an implementation of a
generic `Sim(2)` paper.

## What is the `Sim(2)` reference?

The first-stage matrix in both implementations is

```text
S_i = [[ a_i,  b_i, c_i],
       [-b_i,  a_i, f_i],
       [   0,    0,   1]].
```

It contains translation, rotation, and one uniform scale: four degrees of freedom.
It is therefore a planar similarity transformation, commonly denoted `Sim(2)`. The
code optimizes the four matrix coefficients directly; it does not use a Lie-algebra
solver or cite a separate `Sim(2)` estimation algorithm.

For this repository, **MegaStitch** [2] is the appropriate direct citation for the
similarity-constrained global formulation. Wang et al. use the shared constraints
`a_i = e_i` and `b_i = -d_i` in their first-stage objective and explicitly attribute
that construction to MegaStitch. MegaStitch also motivates a global linear solution
for transformations restricted to similarity or affine models and the use of that
solution to initialize a later homography optimization.

Chen and Chuang's **Natural Image Stitching with the Global Similarity Prior** [3]
is useful related work because it formalizes similarity preservation as a global
image-stitching prior. It should not be presented as the direct source of this code:
their method optimizes a mesh warp and selects desired per-image scales and rotations,
whereas this repository estimates one global matrix per image (plus local correction
matrices in the backbone method). None of their mesh objective, focal-length/3D
rotation estimation, or scale/rotation-selection procedure is implemented here.

## Feature-level attribution

| Repository element | Relationship to prior work | Citation |
|---|---|---|
| Similarity-constrained first stage followed by projective refinement | Core two-stage basis retained and adapted | Wang et al. [1] |
| Four-parameter shared-variable similarity matrix | Used by Wang et al.; direct formulation lineage is MegaStitch | Wang et al. [1]; Zarei et al. [2] |
| Spatially distributed optimization matches, projective rigid regularizer, `sigma_tr = 5000`, and `omega = 800` | Retained by the incremental visual-graph method | Wang et al. [1] |
| Sequential Graph-Cut composition | Same high-level blending strategy; implemented here with OpenCV rather than claimed as source-equivalent code | Wang et al. [1] |
| Global similarity as a distortion-control idea in image stitching | Related conceptual background only; its mesh method is not implemented | Chen and Chuang [3] |
| Stereo-pair synchronization, descriptor retrieval, tree/strong/loop topology, and component-wise visual-graph solution | Repository-specific modifications | This repository |
| Four-image blocks and ordinary-versus-stereo typed residuals | Repository-specific modifications | This repository |
| Stereo normal-displacement, relative-rotation, and relative-scale penalties | Repository-specific modifications; image-plane prior rather than calibrated stereo geometry | This repository |
| `G_i = S_i @ C_i`, correction safety barriers, and persistent two-block fixed-lag refinement | Repository-specific modifications | This repository |

“Repository-specific” states provenance only. It is not, by itself, a novelty claim;
novelty requires a separate prior-art review and controlled experimental evidence.

## Suggested description for papers or reports

> Our method is built on the two-stage global registration framework of Wang et al.
> [1], in which a similarity-constrained global solution initializes projective
> refinement. The similarity parameterization follows the shared-variable global
> formulation used by MegaStitch [2]. We replace GPS-based neighborhood construction
> with a synchronized binocular visual graph and introduce stereo-aware graph or
> fixed-lag constraints, depending on the implementation. These additions are
> image-plane constraints and do not constitute calibrated stereo reconstruction.

For the **incremental visual-graph** implementation, append:

> Candidate relations are built incrementally from sequential and retrieved visual
> evidence, while all verified image edges, including same-time left/right edges,
> retain ordinary two-dimensional correspondence residuals. The completed graph is
> optimized component-wise using the Wang-derived similarity-to-projective objective.

For the **rigid-stereo similarity-backbone** implementation, append:

> Consecutive synchronized pairs form four-image blocks. Ordinary edges retain full
> two-dimensional correspondence residuals, whereas stereo edges penalize normal
> displacement, relative rotation, and relative scale. Projective corrections are
> optimized over a persistent two-block window on top of a propagated similarity
> backbone.

## References

1. Z. Wang, Z. Fu, and J. Xu, “Large-scale UAV image stitching based on global
   registration optimization and graph-cut method,” *Journal of Visual Communication
   and Image Representation*, vol. 107, article 104354, 2025.
   <https://doi.org/10.1016/j.jvcir.2024.104354>

2. A. Zarei, E. Gonzalez, N. Merchant, D. Pauli, E. Lyons, and K. Barnard,
   “MegaStitch: Robust Large-Scale Image Stitching,” *IEEE Transactions on Geoscience
   and Remote Sensing*, vol. 60, article 4408309, pp. 1-9, 2022.
   <https://doi.org/10.1109/TGRS.2022.3141907>

3. Y.-S. Chen and Y.-Y. Chuang, “Natural Image Stitching with the Global Similarity
   Prior,” in *Computer Vision - ECCV 2016*, Part V, pp. 186-201, 2016.
   <https://doi.org/10.1007/978-3-319-46454-1_12>

Machine-readable entries are available in [`references.bib`](../references.bib).
