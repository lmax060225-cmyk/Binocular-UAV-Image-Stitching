# StereoUAV Graph Stitching

Incremental visual-graph registration and binocular UAV image stitching with stereo constraints and global optimization.

This repository collects three related implementations for large-scale UAV image registration and mosaicking:

- `uav_stereo_incremental_visual_graph_global_registration.py`: incremental stereo visual-graph registration with ordinary, tree, strong, and loop edges, followed by global optimization and mosaic rendering.
- `UAV_Binocular_Camera_block_incremental_rigid_stereo_similarity_backbone_core.py`: block-incremental binocular stitching with a rigid stereo similarity backbone.
- `uav_stitching/`: a modular, reproducible stitching pipeline covering metadata association, GPS-neighbor graph construction, feature matching, affine and projective optimization, warping, Graph-Cut seams, blending, and evaluation.

## Repository scope

Source code, configuration, documentation, and lightweight data descriptors are versioned. Runtime caches, debug imagery, rendered mosaics, and other generated outputs are intentionally excluded so the repository remains compact and reproducible.

See [`uav_stitching/README.md`](uav_stitching/README.md) for environment setup, pipeline commands, method boundaries, verified metrics, and artifact descriptions.

## Reference paper

Zhongxing Wang, Zhizhong Fu, and Jin Xu, “Large-scale UAV image stitching based on global registration optimization and graph-cut method,” *Journal of Visual Communication and Image Representation*, vol. 107, article 104354, 2025.

- DOI: [10.1016/j.jvcir.2024.104354](https://doi.org/10.1016/j.jvcir.2024.104354)
- Publisher page: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1047320324003109)
