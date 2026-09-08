# Large-scale UAV stitching reproduction

Runnable engineering reproduction of **“Large-scale UAV image stitching based
on global registration optimization and graph-cut method”** (Wang, Fu, Xu,
JVCIR 107, 2025, 104354). The implementation operates on the supplied 70 DJI
M4E survey images and supports the same cached workflow for larger sets.

## Verified result on the supplied survey

- 70/70 JPG–MRK records uniquely associated by DJI exposure index.
- Survey-centered WGS84→ENU coordinates; paper-style four-quadrant GPS graph:
  132 unique edges, one connected component, no isolated images.
- 8,000 SIFT features per image, cached once. All 132 GPS-neighbor pairs pass
  Lowe ratio + RANSAC; every valid pair contributes at most 40 spatially
  distributed matches from an 8×5 overlap grid.
- Global shared-variable affine optimization, then global eight-parameter
  projective refinement with `sigma_tr=5000` and paper rigid weight `omega=800`.
  Reference image 35 is fixed to identity.
- Full-data all-RANSAC-inlier projection RMSE: **11.126 px**. This metric uses
  all inliers, not only the 40 optimization points. All 70 transformed
  footprints pass the finite/area/edge/denominator/perspective checks.
- Full-resolution geometry is 10200×13337. The delivered mosaic is rendered at
  25% scale (2550×3335), while Graph-Cut seams are solved at 10% scale to bound
  memory. Its support mask is one connected component; 81.203% of the
  rectangular canvas is covered. Black outside the irregular footprint is
  expected and is described by `final_mosaic_mask.png`.

The real 5-image, 20-image, and 70-image pipelines were all executed. All seam
steps used OpenCV Graph-Cut without falling back to the distance seam.

## Method boundary

The registration objective, translation parameter scaling, shared affine
variables, averaged projective matching energy, and averaged rigid energy
follow the paper directly.

The paper does not publish every engineering detail. These are explicitly
treated as implementation choices:

- strongest-RANSAC-edge propagation for affine initialization;
- center-reference maximum-overlap traversal for frame insertion order;
- standard WGS84 ECEF→ENU conversion and deterministic axis-boundary handling;
- Graph-Cut seams solved at a configurable lower resolution and transferred to
  the uint8 output canvas.

**Registration optimization follows the paper directly. The graph-cut blending
stage uses OpenCV `GraphCutSeamFinder` as an engineering-equivalent
implementation unless the cited reference implementation is later
incorporated.** Exposure compensation is off by default and was not mixed into
the reported run.

## Environment and tests

Use the supplied environment, not the system Python:

```powershell
$py = 'C:\Users\Lenovo\.conda\envs\uav_gtsam\python.exe'
& $py -m pytest -q
```

The project bootstraps the Conda `Library\bin` DLL directory process-locally so
direct invocation of `python.exe` supports NumPy BLAS and SciPy reliably. The
verified suite contains GPS quadrant, metadata/MRK, sampling coverage,
affine/projective recovery, rigid suppression, transform scaling, numeric
runtime, Graph-Cut order, and a synthetic sequential blend test.

## Commands

From `D:\UAV_Paper\uav_stitching`:

```powershell
& $py scripts/inspect_dataset.py --images ../image --mrk ../image/information.MRK
& $py scripts/build_pairs.py --config configs/my_uav.yaml
& $py scripts/match_features.py --config configs/my_uav.yaml
& $py scripts/optimize_affine.py --config configs/my_uav.yaml
& $py scripts/optimize_projective.py --config configs/my_uav.yaml
& $py scripts/stitch.py --config configs/my_uav.yaml
& $py scripts/evaluate.py --config configs/my_uav.yaml --render
```

One command, or one selected stage:

```powershell
& $py run_pipeline.py --config configs/my_uav.yaml --stage all
& $py run_pipeline.py --config configs/my_uav.yaml --stage blend
```

Supported stages are `metadata`, `pairs`, `matches`, `affine`, `projective`,
`warp`, `blend`, and `all`. Feature and match caches make unchanged reruns
incremental.

## Primary artifacts

```text
outputs/final_mosaic.jpg
outputs/final_mosaic_mask.png
outputs/projection_rmse.csv
outputs/evaluation_summary.json
outputs/stitch_summary.json
cache/affine_transforms.npy
cache/projective_transforms.npy
debug/projective_preview.jpg
debug/projective_footprints.png
debug/seams/step_*.jpg
```

Matrices consistently map **image pixel coordinates → global mosaic
coordinates**. OpenCV `warpPerspective` receives the translated and scaled
forward matrix `S @ T_canvas @ H_i`.
