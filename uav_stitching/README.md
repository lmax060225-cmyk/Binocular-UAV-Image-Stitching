# StereoUAV-Graph-Stitching

## GPS-guided image stitching

This subproject provides an engineering implementation of *Large-scale UAV image stitching based on global registration optimization and graph-cut method* by Zhongxing Wang, Zhizhong Fu, and Jin Xu (*Journal of Visual Communication and Image Representation*, 107, 104354, 2025).

[Paper DOI](https://doi.org/10.1016/j.jvcir.2024.104354) · [Publisher page](https://www.sciencedirect.com/science/article/pii/S1047320324003109) · [Repository](https://github.com/lmax060225-cmyk/StereoUAV-Graph-Stitching)

The pipeline uses GPS metadata to select neighboring image pairs, estimates a global affine registration, refines it with projective transformations and rigid regularization, and blends the registered images using sequential graph-cut seams. Feature and match caches can be reused when their inputs and configuration remain valid.

## Installation

Use Python 3.10 or later in a dedicated environment. For example, with Conda:

```sh
git clone https://github.com/lmax060225-cmyk/StereoUAV-Graph-Stitching.git
cd StereoUAV-Graph-Stitching/uav_stitching
conda create -n uav-stitching python=3.11 -y
conda activate uav-stitching
python -m pip install -r requirements.txt
```

The main dependencies are NumPy, SciPy, OpenCV with contrib modules, Pillow, PyYAML, and tqdm. On Windows, the project adds the active Conda environment's `Library/bin` directory to the process DLL search path when available.

## Dataset and configuration

Raw survey images and DJI positioning files are not distributed with this repository. Supply your own data and update `configs/my_uav.yaml`. Relative paths in this file are resolved from the `uav_stitching/` directory, not from the shell's working directory.

The default configuration expects an `image/` directory at the repository root. See [data placement instructions](data/README.md) for the directory structure. Image and MRK records are associated by DJI exposure index rather than by file or row order.

Key configuration values are:

| Setting | Default | Meaning |
| --- | --- | --- |
| `features.nfeatures` | `8000` | Requested SIFT feature count per image; the detected count may vary |
| `features.ratio_test` | `0.75` | Lowe's descriptor-distance ratio threshold |
| `features.ransac_threshold_px` | `4.0` | RANSAC reprojection threshold in pixels |
| `sampling.selected_points` | `40` | Maximum number of correspondences selected per valid pair |
| `sampling.grid_rows`, `sampling.grid_cols` | `5`, `8` | Spatial sampling grid over the overlap region |
| `optimization.sigma_tr` | `5000.0` | Translation parameter scale |
| `optimization.rigid_weight` | `800.0` | Rigid regularization weight (`omega`) |
| `warp.final_scale` | `0.25` | Output scale relative to full-resolution geometry |
| `blend.seam_scale` | `0.10` | Seam estimation scale relative to full-resolution geometry |

Exposure compensation is disabled by default. To retain the published results when running a new experiment, set `output_dir`, `cache_dir`, and `debug_dir` to separate experiment directories.

## Running the pipeline

Run these commands from `StereoUAV-Graph-Stitching/uav_stitching/` after activating your environment and configuring the dataset.

Run all stages:

```sh
python run_pipeline.py --config configs/my_uav.yaml --stage all
```

Run only blending after the required registration artifacts have been generated:

```sh
python run_pipeline.py --config configs/my_uav.yaml --stage blend
```

Available stages are `metadata`, `pairs`, `matches`, `affine`, `projective`, `warp`, `blend`, and `all`. Individual stages require the outputs of earlier stages. The `warp` stage generates registration previews; `blend` generates the final mosaic. The runner also evaluates the final result after `blend` or `all`.

For step-by-step execution with the default data layout:

```sh
python scripts/inspect_dataset.py --images ../image --mrk ../image/information.MRK
python scripts/build_pairs.py --config configs/my_uav.yaml
python scripts/match_features.py --config configs/my_uav.yaml
python scripts/optimize_affine.py --config configs/my_uav.yaml
python scripts/optimize_projective.py --config configs/my_uav.yaml
python scripts/evaluate.py --config configs/my_uav.yaml --render
python scripts/stitch.py --config configs/my_uav.yaml
python scripts/evaluate.py --config configs/my_uav.yaml
```

If you change the dataset paths, update the arguments to `inspect_dataset.py` as well, or use the configuration-driven runner.

## Recorded survey results

The published artifacts document a run on 70 DJI M4E survey images, along with 5-image and 20-image subset runs. These are results from the saved experiments, not guarantees for other datasets or dependency versions.

For the 70-image run, [evaluation_summary.json](outputs/evaluation_summary.json) and [stitch_summary.json](outputs/stitch_summary.json) report:

| Metric | Recorded value |
| --- | --- |
| Registered images | 70 |
| Valid image-pair edges | 132 |
| Reference image ID | 35 (zero-based; transform fixed to identity) |
| Projection RMSE over all RANSAC inliers | 11.126 px |
| Transform sanity warnings | 0 |
| Full-resolution canvas, width × height | 10200 × 13337 px |
| Rendered mosaic, width × height | 2550 × 3335 px |
| Canvas coverage | 81.203% |
| Connected components in the final mask | 1 |

The RMSE measures correspondence alignment in the global mosaic coordinate system using all RANSAC inliers, rather than only the sampled optimization points. It is not an absolute geolocation accuracy measurement or a held-out evaluation. Black regions outside the image coverage are identified by `final_mosaic_mask.png`.

The saved summaries for the 5-, 20-, and 70-image runs record OpenCV graph-cut for every seam step after the initial image, with no distance-seam fallback. The current repository does not include an automated test suite for this subproject; the saved artifacts provide the evidence for the results reported above.

## Implementation scope

The implementation is based on the paper's two-stage affine-to-projective registration approach, spatially distributed correspondences, and rigid regularization. It is an engineering reproduction, not the authors' reference code.

Implementation choices include initialization through the strongest RANSAC-supported edges, maximum-overlap image insertion starting from a central reference, WGS84-to-local-ENU coordinate conversion, and seam estimation at reduced resolution. OpenCV's `GraphCutSeamFinder` provides the seam solver; masks are transferred to the output canvas for blending. Exposure compensation was disabled in the recorded runs.

## Output files

Published results:

```text
outputs/final_mosaic.jpg
outputs/final_mosaic_mask.png
outputs/projection_rmse.csv
outputs/evaluation_summary.json
outputs/stitch_summary.json
```

The same directory contains files with `_small5` and `_small20` suffixes for the subset runs.

Intermediate artifacts generated locally include:

```text
cache/affine_transforms.npy
cache/projective_transforms.npy
debug/projective_preview.jpg
debug/projective_footprints.png
debug/seams/step_*.jpg
```

Each transformation `H_i` maps image pixel coordinates to global mosaic coordinates. Rendering uses the forward transformation `S @ T_canvas @ H_i`, where `T_canvas` translates the mosaic bounds onto the output canvas and `S` applies the rendering scale. This matrix is passed to OpenCV's `warpPerspective`.
