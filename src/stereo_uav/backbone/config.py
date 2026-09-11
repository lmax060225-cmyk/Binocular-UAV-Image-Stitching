"""Config for the backbone algorithm."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

from .models import (
    EdgeSpec,
)

LEFT_IN = Path("data/left")


RIGHT_IN = Path("data/right")


OUTPUT_DIR = Path("outputs/backbone")


# Each four-image block has two stereo and four ordinary edges, followed by two-stage optimization.

# Maximum SIFT feature count per image.
SIFT_NFEATURES = 10000


# Lowe ratio criterion: d1 < RATIO_TEST * d2.
RATIO_TEST = 0.75


# Homography RANSAC reprojection threshold in pixels.
RANSAC_REPROJ_THRESH = 4.0


# Reject edges with fewer than this many RANSAC inliers.
MIN_INLIERS = 30


# Maximum spatially selected correspondences per valid edge.
SELECTED_MATCHES_PER_PAIR = 40


# Partition the source inlier bounding box into GRID_ROWS by GRID_COLS cells.
GRID_ROWS = 5


GRID_COLS = 8


# Pixel translations can be much larger than other matrix entries.
# Use c_scaled = c / TRANSLATION_SCALE to improve numerical conditioning.
TRANSLATION_SCALE = 5000.0


# ==================== RIGID STEREO PAIR CONFIG ====================
# Ordinary edges constrain full 2D correspondence alignment across images.
#
# Stereo edges leave parallel disparity free and do not bind the normal to global y.
# Use a shared similarity prior for each rigid stereo pair:
#
# 1. Relative rotation:
#          r_theta = wrap(theta_i - theta_j)
#
# 2. Relative scale:
#          r_scale = log(s_i / s_j)
#
# 3. Residual normal to the current shared stereo direction:
#          r_perp,k = n_ij^T (p_i,k' - p_j,k')
#
# u_ij is the normalized mean of the transformed image-x directions.
#      n_ij = [-u_y, u_x]^T.
#
# The pair can rotate, translate, and scale jointly.
# Penalize relative rotation, relative scale, and displacement normal to the shared direction.
LAMBDA_STEREO_PERP = 9.0


STEREO_PERP_SIGMA_PX = 2.0


LAMBDA_STEREO_THETA = 1.0


STEREO_THETA_SIGMA_RAD = math.radians(0.5)


LAMBDA_STEREO_SCALE = 1.0


# Normalization of log(s_i / s_j); the configured value is 0.5.
STEREO_LOG_SCALE_SIGMA = 0.5


# Stage-one continuation:
# Warm start with ordinary edges, then increase the rigid-stereo residual scale.
# Scale stereo residuals by sqrt(alpha) before the Huber loss.
# This equals an energy multiplier alpha only in the quadratic loss region.
AFFINE_STEREO_WEIGHT_SCHEDULE: Tuple[float, ...] = (0.1, 0.3, 1.0)


# ==================== PROJECTIVE NATIVE SAFETY CONFIG ====================
# Soft safety penalties become significant near homography degeneration.
# Local and persistent objectives share denominator and singular-value barriers.
LAMBDA_PROJECTIVE_DENOM = 2000.0


LAMBDA_PROJECTIVE_SV = 200.0


# Lower bound on corner denominators normalized relative to the image center.
PROJECTIVE_DENOM_MIN = 0.20


PROJECTIVE_DENOM_SOFTNESS = 0.02


# Permitted local Jacobian singular-value range in normalized coordinates.
# Safety barriers discourage collapse and extreme stretching; they cannot repair incorrect matches or models.
PROJECTIVE_SV_MIN = 0.05


PROJECTIVE_SV_MAX = 20.0


PROJECTIVE_SV_SOFTNESS = 0.10


PROJECTIVE_SAFETY_GRID_ROWS = 5


PROJECTIVE_SAFETY_GRID_COLS = 5


# Numerical floor for normalization, SVD, logarithms, and denominators.
PROJECTIVE_REG_EPS = 1.0e-12


# Maximum least-squares function evaluations for the two stages.
MAX_OPT_NFEV_AFFINE = 300


MAX_OPT_NFEV_PROJECTIVE = 500


# ==================== PERSISTENT TWO-BLOCK WINDOW CONFIG ====================
# Retain two-stage local registration to estimate each block's geometry.
# Decompose accumulated state as:
#   G_i = S_i @ C_i
# S_i contains rotation, uniform scale, and translation for propagation across blocks.
# C_i is a locally penalized projective correction excluded from the propagation chain.
# The two-block window updates shared and new global transforms through their C_i variables.
MAX_OPT_NFEV_PERSISTENT_PROJECTIVE = 500


PERSISTENT_PROJECTIVE_F_SCALE = 4.0


PERSISTENT_INVALID_RESIDUAL = 1.0e6


# Limit preview dimensions only; saved transforms remain in original pixels.
MAX_CANVAS_SIZE = 12000


# Find seams on a reduced ROI and restore the seam masks to the rendering resolution.
MAX_GRAPHCUT_ROI_SIZE = 1200


# False: later images overwrite earlier pixels in input order.
# True: select seams with Graph-Cut in overlapping regions.
USE_GRAPHCUT = False


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


RANDOM_SEED = 7


# Fixed graph for [left_t, left_t+1, right_t, right_t+1].
# Exactly two special stereo edges and four ordinary edges.
BLOCK_EDGE_SPECS: Tuple[EdgeSpec, ...] = (
    EdgeSpec(0, 1, "ordinary", "left_temporal"),
    EdgeSpec(0, 2, "stereo", "stereo_t"),
    EdgeSpec(0, 3, "ordinary", "left_t_to_right_t1"),
    EdgeSpec(1, 2, "ordinary", "left_t1_to_right_t"),
    EdgeSpec(1, 3, "stereo", "stereo_t1"),
    EdgeSpec(2, 3, "ordinary", "right_temporal"),
)
