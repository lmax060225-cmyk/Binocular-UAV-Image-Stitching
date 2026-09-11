# Repository layout and migration

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── uav_stereo_incremental_visual_graph_global_registration.py
├── UAV_Binocular_Camera_block_incremental_rigid_stereo_similarity_backbone_core.py
├── src/stereo_uav/
│   ├── visual_graph/
│   │   ├── models.py, geometry.py, features.py, retrieval.py, graph.py
│   │   ├── optimization.py, metrics.py, rendering.py, reporting.py
│   │   └── pipeline.py, cli.py, selftest.py
│   └── backbone/
│       ├── config.py, models.py, io.py, matching.py, geometry.py
│       ├── initialization.py, similarity.py, projective.py, persistent.py
│       └── rendering.py, reporting.py, pipeline.py, cli.py
├── tests/
├── docs/
└── results/
    └── visual_graph/
        ├── data/
        ├── mosaics/
        ├── visual_checks/
        └── implementation_report.md
```

The root scripts are compatibility entry points and re-export the corresponding
package APIs. Import new code from `stereo_uav.visual_graph` or
`stereo_uav.backbone`. The implementations remain separate: their edge semantics,
state representations, objectives, and matching policies are not interchangeable.

The complete former `uav_stitching/` project, including its code, configuration,
documentation, and generated results, is removed from the repository. No monocular
algorithm is published. Raw data, local experiments, editor settings, and unrelated
files are excluded by the root allowlist in `.gitignore`.

The archived visual-graph experiment moved from `out_new_visual_graph/` to
`results/visual_graph/`. Numerical artifacts and images retain their original contents.
Recorded paths inside provenance files describe that historical run. They are not
rewritten to imply a newly executed experiment. Saved-graph replay checks input
paths, sizes, timestamps, and algorithm configuration; a fresh clone or relocated
dataset may not satisfy those checks. Run a new experiment in a new output directory
when the original provenance is unavailable.

For new work, use `data/left/` and `data/right/` for local inputs and `outputs/` for
generated artifacts. These directories are ignored. Publish only reviewed,
documented results under `results/`. The visual-graph root script's existing
`data_set_2` and `out_new_visual_graph_test` defaults were preserved; explicit CLI
paths are recommended for reproducibility. The backbone CLI replaces the former
machine-specific `D:` drive paths with portable input/output defaults.

Backbone configuration lives in `src/stereo_uav/backbone/config.py`; runtime modules
read that module directly. Assigning a copied constant on the old compatibility
module is not the supported configuration interface. Use CLI paths or configure
`stereo_uav.backbone.config` before calling `pipeline.run_stitching()`.

Run validation from the repository root:

```sh
python -m pip install -e ".[test]"
python -m pytest -q
python uav_stereo_incremental_visual_graph_global_registration.py --self-test --output outputs/selftest
```

All source comments, diagnostics, and maintained documentation use English.
Historical metrics are kept separate from refactor validation in
[`validation.md`](validation.md).
