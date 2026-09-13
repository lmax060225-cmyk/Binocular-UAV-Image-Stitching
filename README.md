# Binocular-UAV-Image-Stitching

Image stitching for large-scale synchronized binocular UAV image collections.

This repository provides two visual-only registration and mosaicking algorithms.

| Algorithm | Registration strategy | Stereo treatment |
|---|---|---|
| **Incremental visual graph** | Build a visual graph incrementally, then solve all images in each connected component | Every verified image edge uses an ordinary 2D correspondence residual |
| **Rigid-stereo similarity backbone** | Register four-image blocks and refine a persistent two-block window | Separate normal-displacement, relative-rotation, and relative-scale constraints |

Read the [algorithm definitions and comparison](docs/algorithms.md) for the exact
objectives, propagation rules, assumptions, and limitations. The first stage is
called `affine` in legacy outputs, but both implementations use a four-parameter
similarity model in that stage.

## Research basis and citation

The primary methodological basis of this repository is Wang, Fu, and Xu's
two-stage global-registration method for large-scale UAV mosaicking. The visual-graph
implementation retains its similarity-constrained first stage, projective refinement,
rigid regularizer, translation scaling, and sequential Graph-Cut composition, while
replacing GPS-neighbor construction with a stereo-aware incremental visual graph.
The similarity-backbone implementation is a further structural extension that uses
typed stereo residuals and a persistent two-block optimization window.

The four-parameter transform used here is a member of the planar similarity group
`Sim(2)`. Its direct methodological lineage is the constrained global linear model
used by MegaStitch and then adopted by Wang et al. Chen and Chuang's global similarity
prior is included as related image-stitching literature, but this repository does not
implement their mesh-warp objective or their scale/rotation-selection procedure.

See [research provenance and references](docs/references.md) for the precise
implemented-versus-adapted boundary and [references.bib](references.bib) for reusable
BibTeX entries. If this repository supports academic work, cite the Wang et al. paper
as the primary source and MegaStitch for the similarity-constrained global formulation.

## Installation

Python 3.10 or newer is required. From the repository root:

```sh
python -m pip install -e ".[test]"
```

Runtime dependencies are NumPy, SciPy, OpenCV, pandas, and tqdm. The OpenCV build
must expose SIFT and the Graph-Cut seam finder for Graph-Cut rendering. Raw input
images are not distributed. Put synchronized images in separate left/right folders.
For both methods, use matching purely numeric stems such as `000001.png` and
`000002.png`. The visual-graph loader reads the leading integer; the backbone loader
reads the last numeric group. Filenames with additional camera or timestamp numbers
can therefore synchronize differently between the two loaders.

## Run the visual-graph algorithm

```sh
python uav_stereo_incremental_visual_graph_global_registration.py --left data/left --right data/right --output outputs/visual_graph
```

After installation, `uav-visual-graph` provides the same CLI. Use `--self-test` for
the numerical/topology self-tests. Use `--resume-graph` with the same output and
unchanged input files to replay a saved final graph; provenance and configuration
are checked before reuse. A fresh run refuses to overwrite an existing saved graph.

## Run the similarity-backbone algorithm

```sh
python UAV_Binocular_Camera_block_incremental_rigid_stereo_similarity_backbone_core.py --left data/left --right data/right --output outputs/backbone
```

After installation, the equivalent command is `uav-stereo-backbone`. Add `--graphcut`
to enable seam selection; the original ordered-overwrite compositing remains the
default. The CLI requires an empty output directory. Review block status reports:
a final image alone is not evidence that every block registered successfully.

## Results

### Visual graph: archived 140-image experiment

The preserved experiment contains 70 stereo pairs, 478 image edges, and one
connected component. Pooled all-RANSAC-inlier RMSE was **215.149 → 16.828 → 4.522 px**
for initialization, similarity, and projective registration, respectively. These
are historical measurements, not a new full-dataset benchmark after refactoring.

![Archived projective mosaic](results/visual_graph/mosaics/global_projective_mosaic_preview.jpg)

[Experiment report](results/visual_graph/implementation_report.md) ·
[Metrics and transformations](results/visual_graph/data) ·
[Local visual diagnostics](results/visual_graph/visual_checks)

The image retains visible local seam/brightness differences and an irregular
coverage boundary. Pooled registration error does not measure geographic accuracy
or establish that every overlap is seamless.

### Similarity backbone: refactor validation

The refactor was checked on three consecutive real stereo pairs, resized to an
800-pixel longest side. The original and reorganized implementations produced
matching numerical CSV outputs and a pixel-identical final mosaic. This bounded
integration check is not a full-sequence accuracy evaluation or a comparison
against the visual-graph method. See [validation evidence](docs/validation.md).

## Source organization

- [`src/stereo_uav/visual_graph`](src/stereo_uav/visual_graph): feature verification, retrieval, topology, global optimization, reporting, and rendering.
- [`src/stereo_uav/backbone`](src/stereo_uav/backbone): typed blocks, similarity initialization, projective corrections, persistent windows, and rendering.
- [`tests`](tests): numerical, graph, stereo-residual, and propagation regression checks.
- [`docs`](docs): algorithm definitions, migration guidance, and validation scope.
- [`docs/references.md`](docs/references.md): research provenance, citation guidance, and references.
- [`references.bib`](references.bib): verified BibTeX metadata for the principal references.
- [`results`](results): reviewed experiment artifacts for these two algorithms.

The original root script names remain as compatibility entry points. The repository
contains only these two stereo methods and their supporting artifacts. See the
[layout and migration guide](docs/repository_layout.md) for configuration and result paths.

## Validation

```sh
python -m pytest -q
python uav_stereo_incremental_visual_graph_global_registration.py --self-test --output outputs/selftest
```

See [validation evidence and limits](docs/validation.md) before interpreting the
archived experiment or the smoke run as an accuracy claim.
