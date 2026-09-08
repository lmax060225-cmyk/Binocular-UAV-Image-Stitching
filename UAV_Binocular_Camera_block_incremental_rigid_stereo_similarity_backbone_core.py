import json
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Iterator

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from tqdm import tqdm

LEFT_IN = r"D:\UAV_Strip_Test\left"
RIGHT_IN = r"D:\UAV_Strip_Test\right"
OUTPUT_DIR = r"D:\UAV_Strip_Test\out_no_strip_1"


# 对每个 4 图 block 建立固定六边图：2 条 stereo 特殊边 + 4 条 ordinary 普通边，再做两阶段全局优化。

# SIFT 每张图最多保留的特征数。图像分辨率高时可以适当调大。
SIFT_NFEATURES = 10000

# Lowe ratio test 阈值。判据是 d1 < RATIO_TEST * d2。
RATIO_TEST = 0.75

# RANSAC 单应性估计的重投影误差阈值，单位是像素。
RANSAC_REPROJ_THRESH = 4.0

# RANSAC 内点数少于该值时，认为该图片对不可靠，丢弃这条边。
MIN_INLIERS = 30

# 每条有效边最终送入全局优化的均匀匹配点数量上限。
SELECTED_MATCHES_PER_PAIR = 40

# 均匀选点时，把 RANSAC 内点覆盖区域切成 GRID_ROWS x GRID_COLS 网格。
GRID_ROWS = 5
GRID_COLS = 8

# 平移参数 c/f 的数值通常是几千像素，远大于 a/b/d/e/g/h。
# 优化变量里用 c_scaled = c / TRANSLATION_SCALE，改善数值尺度。
TRANSLATION_SCALE = 5000.0

# ==================== RIGID STEREO PAIR CONFIG ====================
# ordinary 边：完整二维匹配约束，用于建立跨时刻/跨视图的全局拼接运动。
#
# stereo 边：不再固定原始水平视差 d_ij^0，也不再把“竖直方向”绑定到全局 y 轴。
# 对刚性双目 pair 使用共同 similarity 先验：
#
#   1) 相对旋转约束：
#          r_theta = wrap(theta_i - theta_j)
#
#   2) 相对尺度约束：
#          r_scale = log(s_i / s_j)
#
#   3) 当前共同 stereo 方向的法向残差：
#          r_perp,k = n_ij^T (p_i,k' - p_j,k')
#
#      其中 u_ij 是左右图当前 x 轴方向的单位向量平均，
#      n_ij = [-u_y, u_x]^T。
#
# 这套约束允许整个双目 pair 一起旋转、平移、缩放，
# 但抑制左右图之间的相对旋转、相对缩放和垂直于当前极线方向的错位。
LAMBDA_STEREO_PERP = 9.0
STEREO_PERP_SIGMA_PX = 2.0

LAMBDA_STEREO_THETA = 1.0
STEREO_THETA_SIGMA_RAD = math.radians(0.5)

LAMBDA_STEREO_SCALE = 1.0
# log(s_i / s_j) 的尺度。0.02 大约对应 2% 的相对尺度变化。
STEREO_LOG_SCALE_SIGMA = 0.5

# 第一阶段采用 continuation：
#   ordinary-only 预优化 -> 逐步增加整个 rigid-stereo 能量权重。
# 系数 alpha 乘在整个 stereo 能量上，因此残差乘 sqrt(alpha)。
AFFINE_STEREO_WEIGHT_SCHEDULE: Tuple[float, ...] = (0.1, 0.3, 1.0)
# ==================== RIGID STEREO PAIR CONFIG END ====================

# ==================== PROJECTIVE NATIVE SAFETY CONFIG ====================
# 仅在 Homography 接近数值退化时激活。local block 与 persistent window
# 共用 denominator + singular-value 两组宽松安全 barrier。
LAMBDA_PROJECTIVE_DENOM = 2000.0
LAMBDA_PROJECTIVE_SV = 200.0

# 角点相对图像中心的归一化齐次分母下界。
PROJECTIVE_DENOM_MIN = 0.20
PROJECTIVE_DENOM_SOFTNESS = 0.02

# 归一化坐标系中局部 Jacobian 两个主方向的宽松尺度范围。
# safety 只阻止任一方向接近塌缩或无限拉伸，不负责修正错误匹配或错误模型。
PROJECTIVE_SV_MIN = 0.05
PROJECTIVE_SV_MAX = 20.0
PROJECTIVE_SV_SOFTNESS = 0.10

PROJECTIVE_SAFETY_GRID_ROWS = 5
PROJECTIVE_SAFETY_GRID_COLS = 5

# 防止坐标归一化、SVD、log 和 denominator normalization 中出现除零。
PROJECTIVE_REG_EPS = 1.0e-12
# ==================== PROJECTIVE NATIVE SAFETY CONFIG END ====================

# 两阶段 least_squares 的最大函数评估次数。
MAX_OPT_NFEV_AFFINE = 300
MAX_OPT_NFEV_PROJECTIVE = 500

# ==================== PERSISTENT TWO-BLOCK WINDOW CONFIG ====================
# block-local affine/projective 两阶段仍保留，用于估计当前 block 的独立局部几何。
# 长期累计状态拆成：
#   G_i = S_i @ C_i
# 其中 S_i 是只含旋转/统一尺度/平移的 similarity backbone，负责跨 block 传播；
# C_i 是受 native safety barrier 限制的局部 projective correction，永不进入下一次累计链。
# 两 block persistent window 仍联合优化 shared H_old 与当前新 H，但变量实际是 C_i。
MAX_OPT_NFEV_PERSISTENT_PROJECTIVE = 500
PERSISTENT_PROJECTIVE_F_SCALE = 4.0
PERSISTENT_INVALID_RESIDUAL = 1.0e6
# ==================== PERSISTENT TWO-BLOCK WINDOW CONFIG END ====================

# 只限制预览图最大边长，保存的 H 矩阵仍是原始像素尺度。
MAX_CANVAS_SIZE = 12000

# GraphCut 只在缩小后的 ROI 上求接缝，再把接缝 mask 放回原尺寸；最终拼接图分辨率不变。
MAX_GRAPHCUT_ROI_SIZE = 1200

# False：关闭 GraphCut，按输入顺序直接硬覆盖；后加入的图像覆盖前面的图像。
# True ：在重叠区域使用 GraphCut 寻找接缝。
USE_GRAPHCUT = False


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

RANDOM_SEED = 7


@dataclass
class ImageRecord:
    """一张输入图片及其基础信息。"""
    index: int #ImageRecord的记录
    path: Path #图片路径
    name: str #图片名称
    image: Optional[np.ndarray]  # OpenCV BGR 原图；退出活动窗口后释放。
    gray: Optional[np.ndarray]  # 灰度图；SIFT 完成且退出活动窗口后释放。
    width: int #宽度
    height: int #高度

@dataclass
class FeatureRecord:
    """一张图片的 SIFT 特征结果。"""

    keypoints: List[cv2.KeyPoint] #第i张照片的关键点集合 List[cv2.KeyPoint]
    # shape=(K_i , 128)，无特征时为 None。K_i是总的像素点数,i是图片标号,descriptors是按行堆叠起来
    descriptors: Optional[np.ndarray]

@dataclass(frozen=True)
class EdgeSpec:
    i: int
    j: int
    edge_type: str
    edge_name: str

@dataclass
class PairMatch_Edge:
    """一条候选图片边 (i, j) 的匹配、RANSAC、均匀选点结果。"""

    i: int #第i张照片
    j: int #第j张照片
    edge_type: str = "ordinary"  # ordinary / stereo
    edge_name: str = ""          # 便于日志和调试定位具体边
    raw_matches: int = 0  # BFMatcher KNN 返回的初始匹配数量。
    ratio_matches: int = 0  # Lowe ratio test 后保留的匹配数量。
    ransac_inliers: int = 0  # RANSAC 单应性筛选后的内点数量。
    selected_matches: int = 0  # 均匀选点后真正送入全局优化的匹配数量。
    inlier_ratio: float = 0.0  # ransac_inliers / ratio_matches。
    mean_reproj_error: float = math.inf  # RANSAC 内点平均重投影误差。
    H_i_to_j: Optional[np.ndarray] = None  # 局部单应矩阵，满足 p_j ~ H_i_to_j p_i。
    inlier_matches: List[cv2.DMatch] = field(default_factory=list)  # RANSAC 内点匹配。
    selected_dmatches: List[cv2.DMatch] = field(default_factory=list)  # 用于 debug 绘图的均匀匹配。

    #第i张图中边(i,j)的均匀选中匹配点
    selected_pts_i: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.float64))

    #第j张照片中边(i,j)对应的匹配点
    selected_pts_j: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.float64))


@dataclass
class PersistentBlockCache:
    """保存上一成功 block 的原始 typed-edge 因子，供下一次两 block 窗口继续优化。"""

    keys: List[Tuple[str, int]]
    images: List[ImageRecord]
    valid_edges: List[PairMatch_Edge]


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


def imread_unicode(path: Path) -> Optional[np.ndarray]:
    """支持 Windows 中文路径的 OpenCV 读图。"""

    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    """支持 Windows 中文路径的 OpenCV 写图。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix if path.suffix else ".jpg"
    ok, data = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    data.tofile(str(path))



def to_jsonable_matrix(matrix: Optional[np.ndarray]) -> Optional[List[List[float]]]:
    """把 numpy 矩阵转成 json.dumps 可序列化的 Python list。"""
    if matrix is None:
        return None
    return [[float(value) for value in row] for row in matrix]


def make_output_dirs(output_dir: Path) -> Dict[str, Path]:
    """创建结果目录；最终拼接图与中间数据分开存放。"""
    dirs = {
        "root": output_dir,
        "mosaics": output_dir / "mosaics",
        "data": output_dir / "data",
        "logs": output_dir / "data" / "logs",
        "blocks": output_dir / "data" / "blocks",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def make_block_output_dirs(output_dirs: Dict[str, Path], block_idx: int) -> Dict[str, Path]:
    """为每个 block 创建只包含 CSV 数据的目录。"""

    block_root = output_dirs["blocks"] / f"block_{block_idx:04d}"
    dirs = {
        "rigid_stereo_debug": block_root / "rigid_stereo_debug",
        "transforms": block_root / "transforms",
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def natural_sort_key(path: Path) -> Tuple[object, ...]:
    """按文件名中的数字自然排序"""
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    )


def extract_frame_number(path: Path) -> int:
    #path.stem是文件去掉后缀的部分
    #r"\d+"是正则表达式，匹配的连续的数字
    numbers = re.findall(r"\d+", path.stem) #r"\d+"匹配的正则表达式

    """
        re.findall(正则,字符串)
        在字符串里找出所有满足正则的字串,返回一个字符串列表
    """

    if not numbers:
        #如果文件名里面没有数字，就没法按帧号对齐，属于不可恢复的逻辑错误，
        #所以直接抛出异常中断程序，而不是静默跳过
        raise ValueError(f"Cannot extract frame number from image name: {path.name}")
    
    return int(numbers[-1])


def collect_image_paths(image_dir: Path) -> List[Path]:
    """收集并按自然顺序排列某一路图片路径。"""

    root = Path(image_dir) #打开图像目录
    if not root.exists():
        raise FileNotFoundError(f"IMAGE_DIR does not exist: {root}")
    paths = [p for p in root.iterdir()if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(paths, key=natural_sort_key)#f返回的是以路径为元素的List


def read_one_image(path:Path , idx: int) -> ImageRecord:
    image = imread_unicode(path)
    if image is None:
        raise RuntimeError(f"Unreadable image: {path}")
    #将图片转成灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height,width = image.shape[:2]
    return ImageRecord(
        idx,
        path,
        path.name,
        image,
        gray,
        width,
        height
    )


def image_pixels(image: ImageRecord) -> np.ndarray:
    """返回 BGR 像素；已释放的历史帧在最终渲染时按需重新读取。"""

    if image.image is not None:
        return image.image

    pixels = imread_unicode(image.path)
    if pixels is None:
        raise RuntimeError(f"Unreadable image: {image.path}")
    return pixels


def load_image_batches(
        left_in: Path,
        right_in: Path,
        batch_size_per_side: int = 2,
) -> Iterator[Tuple[List[ImageRecord], List[ImageRecord]]]:
    """兼容旧调用：按共同帧号生成非重叠双帧 batch。主流程不使用。"""

    left_paths = collect_image_paths(left_in)
    right_paths = collect_image_paths(right_in)
    left_by_frame = {extract_frame_number(path): path for path in left_paths}
    right_by_frame = {extract_frame_number(path): path for path in right_paths}
    if len(left_by_frame) != len(left_paths) or len(right_by_frame) != len(right_paths):
        raise RuntimeError("Duplicate frame numbers found in stereo input directories.")

    common_frames = sorted(set(left_by_frame) & set(right_by_frame))
    usable_count = len(common_frames) // batch_size_per_side * batch_size_per_side
    if usable_count < batch_size_per_side:
        raise RuntimeError("Need at least one frame-aligned stereo batch.")

    for start in range(0, usable_count, batch_size_per_side):
        frame_numbers = common_frames[start:start + batch_size_per_side]
        yield (
            [read_one_image(left_by_frame[number], number) for number in frame_numbers],
            [read_one_image(right_by_frame[number], number) for number in frame_numbers],
        )


def load_incremental_image_windows(left_in: Path,right_in: Path,) -> Iterator[Tuple[List[ImageRecord], List[ImageRecord]]]:

    left_paths = collect_image_paths(left_in)
    right_paths = collect_image_paths(right_in)

    print(f"Found {len(left_paths)} left images in {left_in}")
    print(f"Found {len(right_paths)} right images in {right_in}")

    left_by_frame = {extract_frame_number(path): path for path in left_paths}
    right_by_frame = {extract_frame_number(path): path for path in right_paths}

    if len(left_by_frame) != len(left_paths):
        raise RuntimeError("Duplicate frame numbers found in left_in.")
    
    if len(right_by_frame) != len(right_paths):
        raise RuntimeError("Duplicate frame numbers found in right_in.")

    common_frames = sorted(set(left_by_frame) & set(right_by_frame))
    left_only = sorted(set(left_by_frame) - set(right_by_frame))
    right_only = sorted(set(right_by_frame) - set(left_by_frame))

    if left_only:
        print(f"Warning: skip {len(left_only)} left-only frames: {left_only}")

    if right_only:
        print(f"Warning: skip {len(right_only)} right-only frames: {right_only}")

    if len(common_frames) < 2:
        raise RuntimeError("Need at least 2 frame-aligned stereo pairs.")

    #窗口每次只滑动一个位置。相邻block共享1帧重叠
    total_windows = len(common_frames) - 1

    previous_frame: Optional[int] = None
    previous_left: Optional[ImageRecord] = None
    previous_right: Optional[ImageRecord] = None

    for window_idx in tqdm(range(total_windows), desc="Loading incremental image windows"):

        frame_numbers = common_frames[window_idx: window_idx + 2]

        first_frame, second_frame = frame_numbers

        # 相邻窗口共享第一帧，直接复用上一窗口的第二帧，避免重复解码。
        if (
            previous_frame == first_frame
            and previous_left is not None
            and previous_right is not None
        ):
            left_first = previous_left
            right_first = previous_right
        else:
            left_first = read_one_image(left_by_frame[first_frame], first_frame)
            right_first = read_one_image(right_by_frame[first_frame], first_frame)

        left_second = read_one_image(left_by_frame[second_frame], second_frame)
        right_second = read_one_image(right_by_frame[second_frame], second_frame)

        left_batch = [left_first, left_second]
        right_batch = [right_first, right_second]

        previous_frame = second_frame
        previous_left = left_second
        previous_right = right_second

        yield left_batch, right_batch
# ==================== MODIFIED CODE END: incremental sliding-window input ====================



def extract_features(
        images: Sequence[ImageRecord],
        cache: Optional[Dict[Path, FeatureRecord]] = None,
        sift=None,
) -> List[FeatureRecord]:
    """对每张灰度图提取 SIFT keypoints 和 descriptors。"""

    print("Extracting SIFT features")

    if sift is None:
        sift = cv2.SIFT_create(nfeatures=SIFT_NFEATURES)

    features: List[FeatureRecord] = [] #特征记录表

    for image in tqdm(images, desc="SIFT"):

        if cache is not None and image.path in cache:
            features.append(cache[image.path])
            continue

        if image.gray is None:
            pixels = image_pixels(image)
            gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.gray

        keypoints, descriptors = sift.detectAndCompute(gray, None)

        if descriptors is None or len(keypoints) == 0:
            print(f"Warning: no SIFT features in image {image.index}: {image.name}")
            keypoints = []
            descriptors = None

        feature = FeatureRecord(keypoints, descriptors)
        features.append(feature)
        if cache is not None:
            cache[image.path] = feature
    return features


# 构建候选匹配边集合
def build_candidate_edges(num_images: int) -> List[EdgeSpec]:
    """
    为一个固定的 4 图 block 返回带类型的六条边。

    图像顺序必须是：
        0 = left_t
        1 = left_t+1
        2 = right_t
        3 = right_t+1

    其中：
        - (0, 2)、(1, 3) 是同步双目 stereo 特殊边；
        - 其余四条是 ordinary 普通边。

    不再先构造无类型完全图再在后面猜测边的语义，
    边的几何类型从生成阶段开始就随 EdgeSpec 一起传递。
    """
    if num_images != 4:
        raise ValueError(
            "Special-edge block requires exactly 4 images ordered as "
            "[left_t, left_t+1, right_t, right_t+1], "
            f"but got num_images={num_images}."
        )

    edges = list(BLOCK_EDGE_SPECS)
    print(
        "Built 6 typed candidate edges: "
        "2 stereo special edges + 4 ordinary edges"
    )
    return edges


def build_candidate_pairs(num_images: int) -> List[Tuple[int, int]]:
    """兼容只需要端点编号的调用；正式流程应保留 typed edge。"""
    return [(spec.i, spec.j) for spec in build_candidate_edges(num_images)]


"""
计算局部单应矩阵 H_(i,j) 的平均重投影误差
"""
def compute_reprojection_error(H_i_to_j: np.ndarray, pts_i: np.ndarray, pts_j: np.ndarray) -> float:
    """计算 H_i_to_j 把 pts_i 投影到图 j 后，与 pts_j 的平均像素误差。"""

    if len(pts_i) == 0:
        return math.inf
    # pts_i.shape == (3,2)也就是N个点,2个坐标

    projected = cv2.perspectiveTransform(pts_i.reshape(-1, 1, 2).astype(np.float64), H_i_to_j)
    # cv2.perspectiveTransform 要求输入点集的形状通常是 (N,1,2),而不是(N,2)
    # -1:让NumPy自动推断这一维的大小,因为总共有2N个数，后两维固定为 1 * 2, 所以第一维自动推断为N

    # 又把projective转成(N,2)的张量
    projected = projected.reshape(-1, 2)

    # 求每一行的二范数,返回平均重投影误差
    errors = np.linalg.norm(projected - pts_j, axis=1)
    return float(np.mean(errors))  # 返回errors均值


# 在RANSAC内点中按空间网络选点,并使用fathest_point规则不点,返回P_i和P_j

def select_uniform_matches(
        inlier_matches: Sequence[cv2.DMatch],
        keypoints_i: Sequence[cv2.KeyPoint],
        keypoints_j: Sequence[cv2.KeyPoint],
        target_count: int = SELECTED_MATCHES_PER_PAIR,
) -> Tuple[List[cv2.DMatch], np.ndarray, np.ndarray]:
    """
    从 RANSAC 内点中选空间分布尽量均匀的匹配点。

    不能简单取 descriptor distance 最小的 Top-P，因为这些点容易集中在纹理丰富区域。
    这里先在图 i 的内点覆盖 bbox 上划分网格，每个有点网格选一个最优点；
    如果数量不足，再用 farthest-point 策略补点，让点尽量覆盖整个重叠区域。
    """

    if not inlier_matches:
        return [], np.empty((0, 2), np.float64), np.empty((0, 2), np.float64)

    pts_i = np.array([keypoints_i[m.queryIdx].pt for m in inlier_matches], dtype=np.float64)
    pts_j = np.array([keypoints_j[m.trainIdx].pt for m in inlier_matches], dtype=np.float64)
    distances = np.array([m.distance for m in inlier_matches], dtype=np.float64)

    x0, y0 = np.min(pts_i, axis=0)
    x1, y1 = np.max(pts_i, axis=0)
    if x1 <= x0 or y1 <= y0:
        order = np.argsort(distances)[:target_count]
        selected = [inlier_matches[int(idx)] for idx in order]
        return selected, pts_i[order], pts_j[order]

    selected_indices: List[int] = []
    used = set()
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            cx0 = x0 + (x1 - x0) * col / GRID_COLS
            cx1 = x0 + (x1 - x0) * (col + 1) / GRID_COLS
            cy0 = y0 + (y1 - y0) * row / GRID_ROWS
            cy1 = y0 + (y1 - y0) * (row + 1) / GRID_ROWS
            in_x = (pts_i[:, 0] >= cx0) & (pts_i[:, 0] <= cx1 if col == GRID_COLS - 1 else pts_i[:, 0] < cx1)
            in_y = (pts_i[:, 1] >= cy0) & (pts_i[:, 1] <= cy1 if row == GRID_ROWS - 1 else pts_i[:, 1] < cy1)
            cell_indices = np.where(in_x & in_y)[0]
            if len(cell_indices) == 0:
                continue
            best = int(cell_indices[np.argmin(distances[cell_indices])])
            if best not in used:
                selected_indices.append(best)
                used.add(best)

    remaining = [idx for idx in range(len(inlier_matches)) if idx not in used]
    if len(selected_indices) < target_count and remaining:
        norm = pts_i.copy()
        norm[:, 0] = (norm[:, 0] - x0) / max(x1 - x0, 1e-9)
        norm[:, 1] = (norm[:, 1] - y0) / max(y1 - y0, 1e-9)

        if not selected_indices:
            first = int(remaining[int(np.argmin(distances[remaining]))])
            selected_indices.append(first)
            used.add(first)
            remaining = [idx for idx in remaining if idx != first]

        while len(selected_indices) < target_count and remaining:
            selected_pts = norm[np.array(selected_indices)]
            rem_pts = norm[np.array(remaining)]
            min_d2 = np.min(np.sum((rem_pts[:, None, :] - selected_pts[None, :, :]) ** 2, axis=2), axis=1)
            candidate_order = np.lexsort((distances[remaining], -min_d2))
            best_remaining_pos = int(candidate_order[0])
            best_idx = int(remaining[best_remaining_pos])
            selected_indices.append(best_idx)
            used.add(best_idx)
            remaining.pop(best_remaining_pos)

    selected_indices = selected_indices[: min(target_count, len(selected_indices))]
    selected_indices_np = np.array(selected_indices, dtype=np.int64)
    selected = [inlier_matches[int(idx)] for idx in selected_indices_np]
    return selected, pts_i[selected_indices_np], pts_j[selected_indices_np]


"""
对每一条边(i,j)执行descriptor匹配、low_ratio_test、RANSAC单应性估计、内点筛选、均匀选点
"""


def match_pair(spec: EdgeSpec, features: Sequence[FeatureRecord]) -> PairMatch_Edge:
    """
    对一对图片做局部匹配和 RANSAC 筛选。

    流程：
    1. SIFT descriptor 使用 BFMatcher L2 距离做 KNN(k=2)。
    2. Lowe ratio test 删除二义性强的匹配。
    3. cv2.findHomography + RANSAC 删除外点。
    4. RANSAC 内点足够时，再做空间均匀选点。
    """

    i, j = spec.i, spec.j
    pair = PairMatch_Edge(
        i=i,
        j=j,
        edge_type=spec.edge_type,
        edge_name=spec.edge_name,
    )

    fi = features[i]
    fj = features[j]

    if fi.descriptors is None or fj.descriptors is None:
        return pair

    if len(fi.descriptors) < 2 or len(fj.descriptors) < 2:
        return pair

    # KNN匹配：对候选边(i,j)用BMatcher的L2距离进行k = 2最邻近搜索。
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn = matcher.knnMatch(fi.descriptors, fj.descriptors, k=2)

    pair.raw_matches = len(knn)

    # 使用Low_Ratio_Test，筛选稳定的特征点

    good: List[cv2.DMatch] = []
    for candidates in knn:
        if len(candidates) != 2:
            continue
        m, n = candidates
        if m.distance < RATIO_TEST * n.distance:
            good.append(m)
    pair.ratio_matches = len(good)
    if len(good) < 4:
        return pair

    # 取出通过 low_ratio_test 的图片i中的特征点
    pts_i = np.float64([fi.keypoints[m.queryIdx].pt for m in good])

    # 取出通过 low_ratio_test 的图片j中的特征点
    pts_j = np.float64([fj.keypoints[m.trainIdx].pt for m in good])

    # 用RANSAC估计出单应估计
    H, mask = cv2.findHomography(pts_i, pts_j, cv2.RANSAC, RANSAC_REPROJ_THRESH)

    if H is None or mask is None:
        return pair

    mask = mask.reshape(-1).astype(bool)

    # 计算RANSAC内点
    inlier_matches = [m for m, keep in zip(good, mask) if keep]

    # 将齐次化估计H
    pair.H_i_to_j = H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H

    pair.ransac_inliers = len(inlier_matches)

    pair.inlier_ratio = float(pair.ransac_inliers / max(pair.ratio_matches, 1))

    if pair.ransac_inliers > 0:
        inlier_pts_i = np.float64([fi.keypoints[m.queryIdx].pt for m in inlier_matches])
        inlier_pts_j = np.float64([fj.keypoints[m.trainIdx].pt for m in inlier_matches])
        pair.mean_reproj_error = compute_reprojection_error(pair.H_i_to_j, inlier_pts_i, inlier_pts_j)

    if pair.ransac_inliers < MIN_INLIERS:
        return pair

    # 在RANSAC内点中按空间网络选点,并使用fathest_point规则不点,返回P_i和P_j

    selected, selected_pts_i, selected_pts_j = select_uniform_matches(
        inlier_matches,
        fi.keypoints,
        fj.keypoints,
        SELECTED_MATCHES_PER_PAIR,
    )

    min_selected = min(MIN_INLIERS, SELECTED_MATCHES_PER_PAIR)

    if len(selected) < min_selected:
        pair.selected_matches = len(selected)
        return pair

    pair.inlier_matches = inlier_matches
    pair.selected_dmatches = selected
    pair.selected_pts_i = selected_pts_i
    pair.selected_pts_j = selected_pts_j
    pair.selected_matches = len(selected)

    return pair


def match_block_edges(features: Sequence[FeatureRecord]) -> List[PairMatch_Edge]:
    """按固定 typed-edge 图批量匹配；主循环因缓存复用而逐边调用。"""
    return [
        match_pair(spec, features)
        for spec in build_candidate_edges(len(features))
    ]


"""
    建立局部similarity估计和affine初始化
"""


def estimate_local_similarity(edge: Optional[PairMatch_Edge]) -> Optional[np.ndarray]:
    """用一条相邻边的均匀匹配点估计局部 similarity/partial affine。"""

    if edge is None or len(edge.selected_pts_i) < 2:
        return None

    """
        使用用 cv2.estimateAffinePartial2D从一条边的选中点估计局部similarity/partial affine
    """

    A, _ = cv2.estimateAffinePartial2D(
        edge.selected_pts_i.astype(np.float64),
        edge.selected_pts_j.astype(np.float64),
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_REPROJ_THRESH,
    )

    if A is None:
        return None

    # 先得到一个3 * 3的矩阵
    H = np.eye(3, dtype=np.float64)
    # 用A替换H矩阵的1,2行
    H[:2, :] = A
    return H


# ==================== RIGID STEREO INITIALIZATION ====================
def project_to_similarity(H: np.ndarray) -> np.ndarray:
    """把任意 3x3 近似变换投影为当前代码约定的 2D similarity。"""

    H = np.asarray(H, dtype=np.float64)
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]

    a = 0.5 * (H[0, 0] + H[1, 1])
    b = 0.5 * (H[0, 1] - H[1, 0])
    c = H[0, 2]
    f = H[1, 2]

    return np.array(
        [
            [a, b, c],
            [-b, a, f],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def similarity_components(
        H: np.ndarray,
) -> Tuple[float, float, np.ndarray]:
    """
    从当前代码约定的 similarity 中提取：
        scale     : s > 0
        angle     : 图像 x 轴在全局坐标中的角度
        direction : 对应单位方向 [cos(theta), sin(theta)]
    """

    a = float(H[0, 0])
    b = float(H[0, 1])

    scale = max(math.hypot(a, b), PROJECTIVE_REG_EPS)

    # 当前矩阵形式：
    #   [[a,  b],
    #    [-b, a]]
    # 第一列就是原始 x 轴经过变换后的方向 [a, -b]^T。
    direction = np.array(
        [a / scale, -b / scale],
        dtype=np.float64,
    )
    angle = float(math.atan2(direction[1], direction[0]))
    return scale, angle, direction


def normalize_direction(direction: np.ndarray) -> np.ndarray:
    """把二维方向向量归一化；退化时回退到全局 x 轴。"""

    direction = np.asarray(direction, dtype=np.float64).reshape(2)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        return np.array([1.0, 0.0], dtype=np.float64)
    return direction / norm


def average_directions(
        direction_i: np.ndarray,
        direction_j: np.ndarray,
) -> np.ndarray:
    """
    求两个单位方向的圆周平均。

    rigid stereo 初值和残差期望左右图方向接近，因此直接对单位向量求和后归一化。
    若两方向恰好近似相反导致退化，则回退到 direction_i。
    """

    direction_i = normalize_direction(direction_i)
    direction_j = normalize_direction(direction_j)

    merged = direction_i + direction_j
    if float(np.linalg.norm(merged)) < 1e-12:
        return direction_i.copy()
    return normalize_direction(merged)


def similarity_linear_from_scale_direction(
        scale: float,
        direction: np.ndarray,
) -> np.ndarray:
    """由共同尺度和图像 x 轴方向构造 2x2 similarity 线性部分。"""

    direction = normalize_direction(direction)
    ux, uy = float(direction[0]), float(direction[1])

    # 标准旋转形式：
    #   s [[ cos(theta), -sin(theta)],
    #      [ sin(theta),  cos(theta)]]
    return float(scale) * np.array(
        [
            [ux, -uy],
            [uy, ux],
        ],
        dtype=np.float64,
    )


def image_center_point(image: ImageRecord) -> np.ndarray:
    """返回图像中心的二维像素坐标。"""

    return np.array(
        [
            0.5 * (image.width - 1.0),
            0.5 * (image.height - 1.0),
        ],
        dtype=np.float64,
    )


def replace_similarity_linear_preserve_center(
        H: np.ndarray,
        image: ImageRecord,
        new_linear: np.ndarray,
) -> np.ndarray:
    """
    替换 similarity 的 2x2 线性部分，同时保持图像中心的全局位置不变。

    这样共同化左右图的旋转/尺度时，不会因为默认绕原点变化而造成大幅位置跳动。
    """

    H = project_to_similarity(H)
    center = image_center_point(image)

    old_global_center = (
        H[:2, :2] @ center
        + H[:2, 2]
    )

    new_H = H.copy()
    new_H[:2, :2] = np.asarray(new_linear, dtype=np.float64)
    new_H[:2, 2] = (
        old_global_center
        - new_H[:2, :2] @ center
    )
    return new_H


def get_edge_by_name(
        edges: Sequence[PairMatch_Edge],
        edge_name: str,
) -> Optional[PairMatch_Edge]:
    """按固定六边图中的语义名称查找边。"""
    return next((edge for edge in edges if edge.edge_name == edge_name), None)


def build_local_similarity_cache(
        edges: Sequence[PairMatch_Edge],
) -> Dict[str, np.ndarray]:
    """为有效边预先估计局部 similarity，供初始化传播重复使用。"""

    cache: Dict[str, np.ndarray] = {}

    for edge in edges:
        local = estimate_local_similarity(edge)

        if local is None and edge.H_i_to_j is not None:
            local = edge.H_i_to_j

        if local is None:
            continue

        cache[edge.edge_name] = project_to_similarity(local)

    return cache


def propagate_similarity_across_edge(
        known_transform: np.ndarray,
        edge: PairMatch_Edge,
        local_i_to_j: np.ndarray,
        known_idx: int,
        unknown_idx: int,
) -> np.ndarray:
    """
    沿一条边把已知全局变换传播到未知节点。

    局部关系：
        p_j ~= H_i_to_j p_i

    因而：
        若 T_i 已知，则 T_j ~= T_i inv(H_i_to_j)
        若 T_j 已知，则 T_i ~= T_j H_i_to_j
    """

    if known_idx == edge.i and unknown_idx == edge.j:
        try:
            propagated = (
                known_transform
                @ np.linalg.inv(local_i_to_j)
            )
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                f"Cannot invert local similarity for edge {edge.edge_name}"
            ) from exc

    elif known_idx == edge.j and unknown_idx == edge.i:
        propagated = (
            known_transform
            @ local_i_to_j
        )

    else:
        raise ValueError(
            "known_idx / unknown_idx do not match the supplied edge."
        )

    return project_to_similarity(propagated)


def commonize_stereo_pair_similarity(
        transforms: List[np.ndarray],
        images: Sequence[ImageRecord],
        i: int,
        j: int,
        lock_i: bool,
) -> Dict[str, object]:
    """
    把一对 rigid stereo 图像的线性部分共同化。

    - lock_i=True：
        保持第 i 张图完整不动，第 j 张图采用 i 的尺度和方向。
        用于参考 pair (0, 2)，因为第 0 张图固定为全局 gauge。
    - lock_i=False：
        左右图共同采用：
            scale = sqrt(s_i * s_j)
            direction = normalize(u_i + u_j)
        同时分别保持两张图自己的全局图像中心位置。
    """

    H_i_before = project_to_similarity(transforms[i])
    H_j_before = project_to_similarity(transforms[j])

    scale_i, angle_i, direction_i = similarity_components(H_i_before)
    scale_j, angle_j, direction_j = similarity_components(H_j_before)

    if lock_i:
        common_scale = scale_i
        common_direction = direction_i.copy()
    else:
        common_scale = math.sqrt(
            max(scale_i * scale_j, PROJECTIVE_REG_EPS)
        )
        common_direction = average_directions(
            direction_i,
            direction_j,
        )

    common_linear = similarity_linear_from_scale_direction(
        common_scale,
        common_direction,
    )

    if lock_i:
        transforms[i] = H_i_before
        transforms[j] = replace_similarity_linear_preserve_center(
            H_j_before,
            images[j],
            common_linear,
        )
    else:
        transforms[i] = replace_similarity_linear_preserve_center(
            H_i_before,
            images[i],
            common_linear,
        )
        transforms[j] = replace_similarity_linear_preserve_center(
            H_j_before,
            images[j],
            common_linear,
        )

    scale_i_after, angle_i_after, _ = similarity_components(transforms[i])
    scale_j_after, angle_j_after, _ = similarity_components(transforms[j])

    return {
        "action": "commonize_pair_similarity",
        "i": i,
        "j": j,
        "lock_i": lock_i,
        "scale_i_before": scale_i,
        "scale_j_before": scale_j,
        "scale_ratio_before": scale_i / max(scale_j, PROJECTIVE_REG_EPS),
        "angle_i_before_deg": math.degrees(angle_i),
        "angle_j_before_deg": math.degrees(angle_j),
        "angle_difference_before_deg": math.degrees(
            wrap_angle(angle_i - angle_j)
        ),
        "common_scale": common_scale,
        "common_direction_x": float(common_direction[0]),
        "common_direction_y": float(common_direction[1]),
        "common_angle_deg": math.degrees(
            math.atan2(common_direction[1], common_direction[0])
        ),
        "scale_i_after": scale_i_after,
        "scale_j_after": scale_j_after,
        "scale_ratio_after": scale_i_after / max(
            scale_j_after,
            PROJECTIVE_REG_EPS,
        ),
        "angle_i_after_deg": math.degrees(angle_i_after),
        "angle_j_after_deg": math.degrees(angle_j_after),
        "angle_difference_after_deg": math.degrees(
            wrap_angle(angle_i_after - angle_j_after)
        ),
    }


def affine_stereo_pair_frame(
        transforms: Sequence[np.ndarray],
        i: int,
        j: int,
) -> Tuple[
    float,
    float,
    float,
    float,
    float,
    np.ndarray,
    np.ndarray,
]:
    """
    返回 affine/similarity rigid-stereo pair 的当前局部量：
        scale_i, scale_j, angle_i, angle_j, log_scale_ratio, u, n
    """

    scale_i, angle_i, direction_i = similarity_components(
        transforms[i]
    )
    scale_j, angle_j, direction_j = similarity_components(
        transforms[j]
    )

    common_direction = average_directions(
        direction_i,
        direction_j,
    )
    normal_direction = np.array(
        [
            -common_direction[1],
            common_direction[0],
        ],
        dtype=np.float64,
    )

    log_scale_ratio = math.log(
        max(scale_i, PROJECTIVE_REG_EPS)
        / max(scale_j, PROJECTIVE_REG_EPS)
    )

    return (
        scale_i,
        scale_j,
        angle_i,
        angle_j,
        log_scale_ratio,
        common_direction,
        normal_direction,
    )


def correct_stereo_perpendicular_translation(
        transforms: List[np.ndarray],
        edge: Optional[PairMatch_Edge],
        lock_i: bool,
) -> Dict[str, object]:
    """
    只沿当前 stereo 共同方向的法向 n 修正平移。

    不修改沿 u 方向的平移，因此不主动压缩/固定水平视差。
    """

    if edge is None or edge.selected_matches <= 0:
        return {
            "action": "perpendicular_translation_correction",
            "status": "skipped_no_valid_stereo_edge",
        }

    i, j = edge.i, edge.j

    (
        _,
        _,
        angle_i,
        angle_j,
        _,
        common_direction,
        normal_direction,
    ) = affine_stereo_pair_frame(
        transforms,
        i,
        j,
    )

    pi = transform_points_affine(
        transforms[i],
        edge.selected_pts_i,
    )
    pj = transform_points_affine(
        transforms[j],
        edge.selected_pts_j,
    )

    perp_before = (
        (pi - pj)
        @ normal_direction
    )

    correction = float(np.median(perp_before))

    if lock_i:
        # r_perp = n^T (p_i - p_j)
        # t_j += correction * n  =>  r_perp_new = r_perp - correction
        transforms[j][:2, 2] += (
            correction
            * normal_direction
        )
    else:
        # 对非参考 pair 对称修正，减少单侧图像跳动。
        transforms[i][:2, 2] -= (
            0.5
            * correction
            * normal_direction
        )
        transforms[j][:2, 2] += (
            0.5
            * correction
            * normal_direction
        )

    pi_after = transform_points_affine(
        transforms[i],
        edge.selected_pts_i,
    )
    pj_after = transform_points_affine(
        transforms[j],
        edge.selected_pts_j,
    )
    perp_after = (
        (pi_after - pj_after)
        @ normal_direction
    )

    return {
        "action": "perpendicular_translation_correction",
        "status": "applied",
        "edge_name": edge.edge_name,
        "i": i,
        "j": j,
        "lock_i": lock_i,
        "angle_i_deg": math.degrees(angle_i),
        "angle_j_deg": math.degrees(angle_j),
        "common_direction_x": float(common_direction[0]),
        "common_direction_y": float(common_direction[1]),
        "normal_direction_x": float(normal_direction[0]),
        "normal_direction_y": float(normal_direction[1]),
        "median_perp_before_px": float(np.median(perp_before)),
        "mean_perp_before_px": float(np.mean(perp_before)),
        "std_perp_before_px": float(np.std(perp_before)),
        "applied_correction_px": correction,
        "median_perp_after_px": float(np.median(perp_after)),
        "mean_perp_after_px": float(np.mean(perp_after)),
        "std_perp_after_px": float(np.std(perp_after)),
    }


def initialize_affines(
        num_images: int,
        edges: Sequence[PairMatch_Edge],
        images: Sequence[ImageRecord],
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    按 rigid stereo pair 语义初始化四图 block。

    固定顺序：
        0 = left_t
        1 = left_t+1
        2 = right_t
        3 = right_t+1

    初始化职责：
        1. reference stereo pair (0, 2) 使用共同 identity similarity；
        2. stereo_t 只沿当前法向修正右图平移；
        3. ordinary temporal edges 优先传播到下一时刻；
        4. 缺失时再使用其他 ordinary edges 补充传播；
        5. 只有 ordinary 图仍无法连通时，才允许 stereo edge 作初始化桥接回退；
        6. pair (1, 3) 的尺度/旋转共同化，并保持各自图像中心全局位置；
        7. stereo_t1 只沿当前法向做对称平移修正。

    返回：
        initial_transforms, initialization_debug
    """

    if num_images != 4 or len(images) != 4:
        raise ValueError(
            "Rigid-stereo initialization requires exactly four images "
            "[left_t, left_t+1, right_t, right_t+1]."
        )

    print(
        "Initializing global similarity transforms with rigid stereo-pair prior"
    )

    transforms: List[np.ndarray] = [
        np.eye(3, dtype=np.float64)
        for _ in range(num_images)
    ]
    initialized = [
        False
        for _ in range(num_images)
    ]

    debug: Dict[str, object] = {
        "method": "rigid_stereo_common_similarity_initialization",
        "image_order": [
            "left_t",
            "left_t+1",
            "right_t",
            "right_t+1",
        ],
        "events": [],
    }
    events = debug["events"]

    local_cache = build_local_similarity_cache(edges)
    edge_by_name = {
        edge.edge_name: edge
        for edge in edges
    }

    # ------------------------------------------------------------
    # Step 1: reference pair (0, 2) 共享 identity similarity。
    # ------------------------------------------------------------
    transforms[0] = np.eye(3, dtype=np.float64)
    transforms[2] = np.eye(3, dtype=np.float64)
    initialized[0] = True
    initialized[2] = True

    events.append(
        {
            "action": "seed_reference_pair",
            "pair": [0, 2],
            "description": (
                "T0=I; T2 starts with the same rotation/scale. "
                "Stereo translation remains free."
            ),
        }
    )

    # reference pair 只沿法向修正 T2 平移；T0 作为 gauge 固定不动。
    stereo_t = edge_by_name.get("stereo_t")
    events.append(
        correct_stereo_perpendicular_translation(
            transforms,
            stereo_t,
            lock_i=True,
        )
    )

    # ------------------------------------------------------------
    # Step 2: temporal ordinary edges 优先传播。
    # ------------------------------------------------------------
    def try_direct_propagation(
            edge_name: str,
            known_idx: int,
            unknown_idx: int,
            reason: str,
    ) -> bool:

        edge = edge_by_name.get(edge_name)
        local = local_cache.get(edge_name)

        if edge is None or local is None:
            events.append(
                {
                    "action": "propagate",
                    "edge_name": edge_name,
                    "status": "skipped_missing_edge_or_local_similarity",
                    "reason": reason,
                }
            )
            return False

        if not initialized[known_idx] or initialized[unknown_idx]:
            return False

        transforms[unknown_idx] = propagate_similarity_across_edge(
            transforms[known_idx],
            edge,
            local,
            known_idx,
            unknown_idx,
        )
        initialized[unknown_idx] = True

        events.append(
            {
                "action": "propagate",
                "edge_name": edge_name,
                "status": "applied",
                "known_idx": known_idx,
                "unknown_idx": unknown_idx,
                "reason": reason,
                "result_transform": to_jsonable_matrix(
                    transforms[unknown_idx]
                ),
            }
        )
        return True

    try_direct_propagation(
        "left_temporal",
        0,
        1,
        "preferred_left_temporal_motion",
    )
    try_direct_propagation(
        "right_temporal",
        2,
        3,
        "preferred_right_temporal_motion",
    )

    # ------------------------------------------------------------
    # Step 3: 其他 ordinary 边只用于补齐尚未初始化的节点。
    # ------------------------------------------------------------
    ordinary_edges = [
        edge
        for edge in edges
        if edge.edge_type == "ordinary"
        and edge.edge_name in local_cache
    ]

    progress = True
    while progress and not all(initialized):
        progress = False

        for edge in ordinary_edges:
            local = local_cache[edge.edge_name]

            if initialized[edge.i] and not initialized[edge.j]:
                transforms[edge.j] = propagate_similarity_across_edge(
                    transforms[edge.i],
                    edge,
                    local,
                    edge.i,
                    edge.j,
                )
                initialized[edge.j] = True
                progress = True

                events.append(
                    {
                        "action": "propagate",
                        "edge_name": edge.edge_name,
                        "status": "applied",
                        "known_idx": edge.i,
                        "unknown_idx": edge.j,
                        "reason": "ordinary_fallback_propagation",
                        "result_transform": to_jsonable_matrix(
                            transforms[edge.j]
                        ),
                    }
                )

            elif initialized[edge.j] and not initialized[edge.i]:
                transforms[edge.i] = propagate_similarity_across_edge(
                    transforms[edge.j],
                    edge,
                    local,
                    edge.j,
                    edge.i,
                )
                initialized[edge.i] = True
                progress = True

                events.append(
                    {
                        "action": "propagate",
                        "edge_name": edge.edge_name,
                        "status": "applied",
                        "known_idx": edge.j,
                        "unknown_idx": edge.i,
                        "reason": "ordinary_fallback_propagation",
                        "result_transform": to_jsonable_matrix(
                            transforms[edge.i]
                        ),
                    }
                )

    # ------------------------------------------------------------
    # Step 4: 仅在 ordinary 无法连通时，允许任意有效边作初始化桥接。
    # 该回退只用于产生数值初值；正式目标函数仍按 typed edge 处理。
    # ------------------------------------------------------------
    if not all(initialized):
        all_cached_edges = [
            edge
            for edge in edges
            if edge.edge_name in local_cache
        ]

        progress = True
        while progress and not all(initialized):
            progress = False

            for edge in all_cached_edges:
                local = local_cache[edge.edge_name]

                if initialized[edge.i] and not initialized[edge.j]:
                    transforms[edge.j] = propagate_similarity_across_edge(
                        transforms[edge.i],
                        edge,
                        local,
                        edge.i,
                        edge.j,
                    )
                    initialized[edge.j] = True
                    progress = True

                    events.append(
                        {
                            "action": "propagate",
                            "edge_name": edge.edge_name,
                            "status": "applied",
                            "known_idx": edge.i,
                            "unknown_idx": edge.j,
                            "reason": "last_resort_any_edge_bridge",
                            "result_transform": to_jsonable_matrix(
                                transforms[edge.j]
                            ),
                        }
                    )

                elif initialized[edge.j] and not initialized[edge.i]:
                    transforms[edge.i] = propagate_similarity_across_edge(
                        transforms[edge.j],
                        edge,
                        local,
                        edge.j,
                        edge.i,
                    )
                    initialized[edge.i] = True
                    progress = True

                    events.append(
                        {
                            "action": "propagate",
                            "edge_name": edge.edge_name,
                            "status": "applied",
                            "known_idx": edge.j,
                            "unknown_idx": edge.i,
                            "reason": "last_resort_any_edge_bridge",
                            "result_transform": to_jsonable_matrix(
                                transforms[edge.i]
                            ),
                        }
                    )

    for idx, ok in enumerate(initialized):
        if not ok:
            print(
                f"  Warning: image {idx} could not be initialized; "
                "use identity as a numerical fallback."
            )
            transforms[idx] = np.eye(3, dtype=np.float64)

            events.append(
                {
                    "action": "identity_fallback",
                    "image_idx": idx,
                }
            )

    # ------------------------------------------------------------
    # Step 5: 下一时刻 pair (1, 3) 的 rotation / scale 共同化。
    # ------------------------------------------------------------
    events.append(
        commonize_stereo_pair_similarity(
            transforms,
            images,
            1,
            3,
            lock_i=False,
        )
    )

    # ------------------------------------------------------------
    # Step 6: stereo_t1 仅沿当前共同方向的法向做对称平移修正。
    # ------------------------------------------------------------
    stereo_t1 = edge_by_name.get("stereo_t1")
    events.append(
        correct_stereo_perpendicular_translation(
            transforms,
            stereo_t1,
            lock_i=False,
        )
    )

    # 第 0 张图始终固定为 gauge。
    transforms[0] = np.eye(3, dtype=np.float64)

    initial_transforms = np.stack(
        [
            project_to_similarity(H)
            for H in transforms
        ],
        axis=0,
    )
    initial_transforms[0] = np.eye(3, dtype=np.float64)

    debug["initialized_flags"] = [
        bool(value)
        for value in initialized
    ]
    debug["final_initial_transforms"] = [
        to_jsonable_matrix(H)
        for H in initial_transforms
    ]

    return initial_transforms, debug
# ==================== RIGID STEREO INITIALIZATION END ====================


def pack_affine_params(transforms: np.ndarray) -> np.ndarray:
    """把第 1..N-1 张图的 similarity 参数打包成 least_squares 的一维变量。"""

    params = []

    """
        优化向量中不直接使用c_i和f_i,而是使用c'_i = c_i / sigma f'_i = f_i / sigma
    """
    for H in transforms[1:]:
        params.extend([H[0, 0], H[0, 1], H[0, 2] / TRANSLATION_SCALE, H[1, 2] / TRANSLATION_SCALE])
    return np.array(params, dtype=np.float64)


"""
    unpack的过程；把一维参数向量转成3*3矩阵
"""


def unpack_affine_params(params: np.ndarray, num_images: int) -> np.ndarray:
    """把一维变量解包成每张图的全局 similarity 矩阵，第 0 张固定为单位阵。"""

    transforms = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], num_images, axis=0)
    for idx in range(1, num_images):
        k = (idx - 1) * 4
        a, b, c_scaled, f_scaled = params[k: k + 4]
        # Similarity constraint from the paper: a=e and b=-d. This preserves
        # rotation + uniform scale + translation before projective refinement.
        transforms[idx] = np.array(
            [[a, b, c_scaled * TRANSLATION_SCALE], [-b, a, f_scaled * TRANSLATION_SCALE], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    return transforms


"""用 affine/similarity 齐次矩阵变换二维点。"""
def transform_points_affine(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
        对于P个二维点组成的矩阵 P.shape(N,2)
        转成 P' = P || [1 * N]^T
    """
    hom = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
    return (H @ hom.T).T[:, :2]
"""
    affine 阶段残差。
    对每条边 (i,j) 的每个 selected match，希望：
    H_aff_i p_i == H_aff_j p_j
    每个匹配点贡献 rx, ry 两个残差。
"""


def wrap_angle(angle: float) -> float:
    """把角度差规范到 [-pi, pi]。"""
    return float(math.atan2(math.sin(angle), math.cos(angle)))


def homography_jacobian(H: np.ndarray, x: float, y: float) -> np.ndarray:
    """计算 projective homography 在指定像素位置的一阶 2x2 Jacobian。"""
    a, b, c = H[0]
    d, e, f = H[1]
    g, h, _ = H[2]

    denominator = g * x + h * y + 1.0
    if abs(denominator) < 1e-9:
        denominator = -1e-9 if denominator < 0.0 else 1e-9

    numerator_u = a * x + b * y + c
    numerator_v = d * x + e * y + f
    denominator2 = denominator * denominator

    return np.array(
        [
            [
                (a * denominator - g * numerator_u) / denominator2,
                (b * denominator - h * numerator_u) / denominator2,
            ],
            [
                (d * denominator - g * numerator_v) / denominator2,
                (e * denominator - h * numerator_v) / denominator2,
            ],
        ],
        dtype=np.float64,
    )


def local_rotation_angle(H: np.ndarray, image: ImageRecord) -> float:
    """用图像中心处 Jacobian 的最近旋转分量表示局部旋转角。"""
    J = homography_jacobian(
        H,
        0.5 * (image.width - 1.0),
        0.5 * (image.height - 1.0),
    )
    return float(math.atan2(J[1, 0] - J[0, 1], J[0, 0] + J[1, 1]))


def similarity_rotation_angle(H: np.ndarray) -> float:
    """Similarity 变换的旋转角；该角度与像素位置无关。"""
    return float(math.atan2(H[1, 0] - H[0, 1], H[0, 0] + H[1, 1]))


def count_matches_by_type(edges: Sequence[PairMatch_Edge]) -> Tuple[int, int]:
    """返回 ordinary 与 stereo selected-match 数。"""
    return (
        sum(edge.selected_matches for edge in edges if edge.edge_type == "ordinary"),
        sum(edge.selected_matches for edge in edges if edge.edge_type == "stereo"),
    )


def split_edges_by_type(
        edges: Sequence[PairMatch_Edge],
) -> Tuple[List[PairMatch_Edge], List[PairMatch_Edge]]:
    """把有效边显式拆成 ordinary / stereo 两组。"""

    ordinary_edges = [
        edge for edge in edges
        if edge.edge_type == "ordinary"
    ]
    stereo_edges = [
        edge for edge in edges
        if edge.edge_type == "stereo"
    ]
    return ordinary_edges, stereo_edges


def affine_ordinary_residuals(
        transforms: np.ndarray,
        ordinary_edges: Sequence[PairMatch_Edge],
) -> np.ndarray:
    """第一阶段 ordinary 边：完整二维点一致性。"""

    ordinary_count = sum(edge.selected_matches for edge in ordinary_edges)
    if ordinary_count <= 0:
        return np.empty(0, dtype=np.float64)

    ordinary_scale = math.sqrt(float(ordinary_count))
    residual_chunks: List[np.ndarray] = []

    for edge in ordinary_edges:
        pi = transform_points_affine(
            transforms[edge.i],
            edge.selected_pts_i,
        )
        pj = transform_points_affine(
            transforms[edge.j],
            edge.selected_pts_j,
        )
        residual_chunks.append(
            (pi - pj).reshape(-1) / ordinary_scale
        )

    return np.concatenate(residual_chunks)


def affine_stereo_residuals(
        transforms: np.ndarray,
        stereo_edges: Sequence[PairMatch_Edge],
        stereo_weight_scale: float,
) -> np.ndarray:
    """
    第一阶段 rigid-stereo residual。

    对每条同步双目边 (i, j)：

        r_theta = wrap(theta_i - theta_j)
        r_scale = log(s_i / s_j)
        r_perp,k = n_ij^T (p_i,k' - p_j,k')

    其中：
        - u_ij 是左右图当前 x 轴方向的单位向量平均；
        - n_ij = [-u_y, u_x]^T；
        - 不约束沿 u_ij 方向的位移，因此不固定水平视差。

    整个 stereo 能量再乘 continuation 系数 alpha。
    """

    if stereo_weight_scale <= 0.0 or not stereo_edges:
        return np.empty(0, dtype=np.float64)

    stereo_count = sum(
        edge.selected_matches
        for edge in stereo_edges
    )
    if stereo_count <= 0:
        return np.empty(0, dtype=np.float64)

    point_normalization = math.sqrt(float(stereo_count))
    edge_normalization = math.sqrt(
        float(max(len(stereo_edges), 1))
    )
    alpha_sqrt = math.sqrt(
        float(stereo_weight_scale)
    )

    residual_chunks: List[np.ndarray] = []

    for edge in stereo_edges:
        (
            scale_i,
            scale_j,
            angle_i,
            angle_j,
            log_scale_ratio,
            _,
            normal_direction,
        ) = affine_stereo_pair_frame(
            transforms,
            edge.i,
            edge.j,
        )

        pi = transform_points_affine(
            transforms[edge.i],
            edge.selected_pts_i,
        )
        pj = transform_points_affine(
            transforms[edge.j],
            edge.selected_pts_j,
        )

        perpendicular = (
            (pi - pj)
            @ normal_direction
        )

        # 每个 stereo 匹配点贡献一个法向残差。
        residual_chunks.append(
            alpha_sqrt
            * math.sqrt(LAMBDA_STEREO_PERP)
            * perpendicular
            / (
                STEREO_PERP_SIGMA_PX
                * point_normalization
            )
        )

        # 每条 stereo edge 只贡献一个相对旋转残差。
        angle_difference = wrap_angle(
            angle_i
            - angle_j
        )
        residual_chunks.append(
            np.array(
                [
                    alpha_sqrt
                    * math.sqrt(LAMBDA_STEREO_THETA)
                    * angle_difference
                    / (
                        STEREO_THETA_SIGMA_RAD
                        * edge_normalization
                    )
                ],
                dtype=np.float64,
            )
        )

        # 每条 stereo edge 只贡献一个相对尺度残差。
        residual_chunks.append(
            np.array(
                [
                    alpha_sqrt
                    * math.sqrt(LAMBDA_STEREO_SCALE)
                    * log_scale_ratio
                    / (
                        STEREO_LOG_SCALE_SIGMA
                        * edge_normalization
                    )
                ],
                dtype=np.float64,
            )
        )

        # 防止静态检查误判未使用；这些量也用于调试函数中的同一定义。
        _ = scale_i, scale_j

    return np.concatenate(residual_chunks)


def affine_residuals(
        params: np.ndarray,
        num_images: int,
        edges: Sequence[PairMatch_Edge],
        stereo_weight_scale: float = 1.0,
) -> np.ndarray:
    """
    第一阶段总残差。

    ordinary / stereo 在代码上分开计算，但共同作用于同一组 similarity 参数。
    stereo_weight_scale=0 时就是 ordinary-only 预优化。
    """

    transforms = unpack_affine_params(params, num_images)
    ordinary_edges, stereo_edges = split_edges_by_type(edges)

    residual_chunks: List[np.ndarray] = []

    ordinary_residual = affine_ordinary_residuals(
        transforms,
        ordinary_edges,
    )
    if ordinary_residual.size > 0:
        residual_chunks.append(ordinary_residual)

    stereo_residual = affine_stereo_residuals(
        transforms,
        stereo_edges,
        stereo_weight_scale,
    )
    if stereo_residual.size > 0:
        residual_chunks.append(stereo_residual)

    if not residual_chunks:
        return np.empty(0, dtype=np.float64)

    return np.concatenate(residual_chunks)


"""
    第一阶段优化器，优化的是(a_i,b_i,c_i,f_i)。

    稳定化流程：
        Step A: ordinary-only 预优化；
        Step B+: ordinary 始终保留，同时按 0.1 -> 0.3 -> 1.0
                 逐步增加 stereo 能量权重。

    不会出现“先 ordinary，再只用 stereo 覆盖优化”的情况。
"""
def optimize_affines(
        initial_transforms: np.ndarray,
        edges: Sequence[PairMatch_Edge],
):
    """分阶段 continuation 的全局 similarity 优化。"""

    print("Optimizing global affine/similarity transforms")

    num_images = len(initial_transforms)
    ordinary_edges, stereo_edges = split_edges_by_type(edges)

    ordinary_count = sum(edge.selected_matches for edge in ordinary_edges)
    stereo_count = sum(edge.selected_matches for edge in stereo_edges)

    print(
        f"  valid ordinary edges={len(ordinary_edges)}, matches={ordinary_count}; "
        f"stereo edges={len(stereo_edges)}, matches={stereo_count}"
    )

    x_initial = pack_affine_params(initial_transforms)

    # 对外报告的 initial/final cost 始终使用完整最终目标 alpha=1，
    # 这样不会因为 continuation 阶段目标不同而失去可比性。
    r0_full = affine_residuals(
        x_initial,
        num_images,
        edges,
        1.0,
    )
    initial_cost = float(0.5 * np.sum(r0_full * r0_full))

    current_x = x_initial.copy()
    last_result = None

    # ------------------------------------------------------------
    # Step A: ordinary-only 预优化
    # ------------------------------------------------------------
    if ordinary_count > 0:
        print("  [Affine stage A] ordinary-only")
        stage_r0 = affine_residuals(
            current_x,
            num_images,
            edges,
            0.0,
        )
        stage_initial_cost = float(0.5 * np.sum(stage_r0 * stage_r0))

        last_result = least_squares(
            affine_residuals,
            current_x,
            args=(num_images, edges, 0.0),
            loss="huber",
            f_scale=4.0,
            max_nfev=MAX_OPT_NFEV_AFFINE,
            verbose=0,
        )
        current_x = last_result.x

        stage_r1 = affine_residuals(
            current_x,
            num_images,
            edges,
            0.0,
        )
        stage_final_cost = float(0.5 * np.sum(stage_r1 * stage_r1))
        print(
            f"    cost: {stage_initial_cost:.6f} -> {stage_final_cost:.6f}; "
            f"success={last_result.success}"
        )
    else:
        print("  [Affine stage A] skipped: no valid ordinary matches")

    # ------------------------------------------------------------
    # Step B+: ordinary 始终保留，逐步增强 stereo
    # ------------------------------------------------------------
    if stereo_count > 0:
        for stage_idx, alpha in enumerate(
            AFFINE_STEREO_WEIGHT_SCHEDULE,
            start=1,
        ):
            print(
                f"  [Affine stereo stage {stage_idx}] "
                f"ordinary + stereo alpha={alpha:.3f}"
            )

            stage_r0 = affine_residuals(
                current_x,
                num_images,
                edges,
                alpha,
            )
            stage_initial_cost = float(0.5 * np.sum(stage_r0 * stage_r0))

            last_result = least_squares(
                affine_residuals,
                current_x,
                args=(num_images, edges, alpha),
                loss="huber",
                f_scale=4.0,
                max_nfev=MAX_OPT_NFEV_AFFINE,
                verbose=0,
            )
            current_x = last_result.x

            stage_r1 = affine_residuals(
                current_x,
                num_images,
                edges,
                alpha,
            )
            stage_final_cost = float(0.5 * np.sum(stage_r1 * stage_r1))
            print(
                f"    cost: {stage_initial_cost:.6f} -> {stage_final_cost:.6f}; "
                f"success={last_result.success}"
            )

    if last_result is None:
        raise RuntimeError("Affine optimization has no valid residuals to optimize.")

    final_transforms = unpack_affine_params(
        current_x,
        num_images,
    )

    r1_full = affine_residuals(
        current_x,
        num_images,
        edges,
        1.0,
    )
    final_cost = float(0.5 * np.sum(r1_full * r1_full))

    return (
        final_transforms,
        last_result,
        initial_cost,
        final_cost,
    )


"""
    Projective参数向量化,和affine阶段一样，代码对平移项c_i和f_i使用了尺度归一化

    第i张图的projective参数为:
    [a_i' , b_i' , c_i , d_i , e_i , f'_i , g_i, h_i]
"""


def normalize_projective_homography(H: np.ndarray) -> np.ndarray:
    """检查并把 Homography 归一化到 H[2, 2] == 1。"""

    matrix = np.asarray(H, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(
            "Homography must have shape (3, 3), "
            f"got {matrix.shape}."
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Homography contains NaN or Inf.")

    scale = float(matrix[2, 2])
    if abs(scale) < PROJECTIVE_REG_EPS:
        raise ValueError(
            "Homography cannot be normalized because H[2, 2] is too small."
        )

    normalized = matrix.copy() / scale
    if not np.all(np.isfinite(normalized)):
        raise ValueError("Normalized Homography contains NaN or Inf.")
    return normalized


def _pack_one_projective_homography(H: np.ndarray) -> List[float]:
    """统一把一个 H33=1 Homography 编码为 8-DOF 优化参数。"""

    H = normalize_projective_homography(H)
    return [
        float(H[0, 0]),
        float(H[0, 1]),
        float(H[0, 2]) / TRANSLATION_SCALE,
        float(H[1, 0]),
        float(H[1, 1]),
        float(H[1, 2]) / TRANSLATION_SCALE,
        float(H[2, 0]),
        float(H[2, 1]),
    ]


def _unpack_one_projective_homography(values: Sequence[float]) -> np.ndarray:
    """统一把 8-DOF 优化参数解码为 H33=1 的 3x3 Homography。"""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size != 8:
        raise ValueError(f"Expected 8 projective parameters, got {values.size}.")

    a, b, c_scaled, d, e, f_scaled, g, h = values
    return np.array(
        [
            [a, b, c_scaled * TRANSLATION_SCALE],
            [d, e, f_scaled * TRANSLATION_SCALE],
            [g, h, 1.0],
        ],
        dtype=np.float64,
    )


def pack_variable_projective_params(
        transforms: np.ndarray,
        variable_indices: Sequence[int],
) -> np.ndarray:
    """Persistent 参数化：只打包 variable_indices 指定的 global Homography。"""

    transforms = np.asarray(transforms, dtype=np.float64)
    if transforms.ndim != 3 or transforms.shape[1:] != (3, 3):
        raise ValueError("transforms must have shape (N, 3, 3).")

    indices = [int(idx) for idx in variable_indices]
    if len(indices) != len(set(indices)):
        raise ValueError("variable_indices contains duplicate indices.")

    params: List[float] = []
    for idx in indices:
        if idx < 0 or idx >= len(transforms):
            raise IndexError(f"Variable index {idx} is outside the transform array.")
        params.extend(_pack_one_projective_homography(transforms[idx]))
    return np.asarray(params, dtype=np.float64)


def unpack_variable_projective_params(
        params: np.ndarray,
        base_transforms: np.ndarray,
        variable_indices: Sequence[int],
) -> np.ndarray:
    """把 variable 参数写回 base_transforms；未列入 variable_indices 的 H 保持固定。"""

    base = np.asarray(base_transforms, dtype=np.float64)
    if base.ndim != 3 or base.shape[1:] != (3, 3):
        raise ValueError("base_transforms must have shape (N, 3, 3).")

    indices = [int(idx) for idx in variable_indices]
    if len(indices) != len(set(indices)):
        raise ValueError("variable_indices contains duplicate indices.")

    values = np.asarray(params, dtype=np.float64).reshape(-1)
    expected = 8 * len(indices)
    if values.size != expected:
        raise ValueError(
            f"Expected {expected} projective variables, got {values.size}."
        )

    transforms = base.copy()
    for slot, idx in enumerate(indices):
        if idx < 0 or idx >= len(transforms):
            raise IndexError(f"Variable index {idx} is outside the transform array.")
        k = 8 * slot
        transforms[idx] = _unpack_one_projective_homography(values[k:k + 8])
    return transforms



def compose_similarity_corrections(
        backbones: np.ndarray,
        corrections: np.ndarray,
) -> np.ndarray:
    """逐节点组合 G_i = S_i @ C_i，并统一归一化 H33。"""

    backbones = np.asarray(backbones, dtype=np.float64)
    corrections = np.asarray(corrections, dtype=np.float64)
    if backbones.shape != corrections.shape or backbones.ndim != 3 or backbones.shape[1:] != (3, 3):
        raise ValueError("backbones and corrections must both have shape (N, 3, 3).")

    return np.stack(
        [
            normalize_projective_homography(S @ C)
            for S, C in zip(backbones, corrections)
        ],
        axis=0,
    )


def project_homography_to_similarity_backbone(
        H: np.ndarray,
        image: ImageRecord,
) -> np.ndarray:
    """
    在图像中心处把一般 Homography 投影为 closest similarity，并保持中心映射位置。

    输出严格满足：
        [[a,  b, tx],
         [-b, a, ty],
         [0,  0,  1 ]]
    因而长期累计时不会携带 projective bottom row。
    """

    H = normalize_projective_homography(H)
    center = image_center_point(image)
    mapped_center = transform_points_projective(
        H,
        center.reshape(1, 2),
    )[0]

    J = homography_jacobian_general(
        H,
        float(center[0]),
        float(center[1]),
    )
    if J is None:
        raise ValueError("Cannot project a locally singular Homography to a similarity backbone.")

    # Frobenius 最近的当前代码约定 similarity：[[a,b],[-b,a]]。
    a = 0.5 * float(J[0, 0] + J[1, 1])
    b = 0.5 * float(J[0, 1] - J[1, 0])
    scale = math.hypot(a, b)
    if not math.isfinite(scale) or scale <= PROJECTIVE_REG_EPS:
        raise ValueError("Homography center Jacobian has a degenerate similarity component.")

    linear = np.array(
        [
            [a, b],
            [-b, a],
        ],
        dtype=np.float64,
    )
    translation = mapped_center - linear @ center
    backbone = np.array(
        [
            [a, b, float(translation[0])],
            [-b, a, float(translation[1])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return normalize_projective_homography(backbone)


def decompose_global_transforms(
        transforms: np.ndarray,
        images: Sequence[ImageRecord],
        backbone_template: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    把每个 G_i 分解为 G_i = S_i @ C_i。

    backbone_template=None 时，从 G_i 的中心 Jacobian提取新的 similarity backbone；
    提供 backbone_template 时保持给定 S_i，只重新计算 C_i = inv(S_i) @ G_i。
    """

    transforms = np.asarray(transforms, dtype=np.float64)
    if transforms.ndim != 3 or transforms.shape[1:] != (3, 3):
        raise ValueError("transforms must have shape (N, 3, 3).")
    if len(transforms) != len(images):
        raise ValueError("images and transforms must have equal length.")

    if backbone_template is None:
        backbones = np.stack(
            [
                project_homography_to_similarity_backbone(H, image)
                for H, image in zip(transforms, images)
            ],
            axis=0,
        )
    else:
        backbones = np.asarray(backbone_template, dtype=np.float64).copy()
        if backbones.shape != transforms.shape:
            raise ValueError("backbone_template must match transforms shape.")
        backbones = np.stack(
            [normalize_projective_homography(S) for S in backbones],
            axis=0,
        )

    corrections: List[np.ndarray] = []
    for S, G in zip(backbones, transforms):
        inverse_backbone = np.linalg.inv(S)
        correction = normalize_projective_homography(
            inverse_backbone @ normalize_projective_homography(G)
        )
        corrections.append(correction)

    corrections_array = np.stack(corrections, axis=0)
    # 组合后必须精确恢复原 G；这里只做数值一致性检查。
    reconstructed = compose_similarity_corrections(backbones, corrections_array)
    normalized_input = np.stack(
        [normalize_projective_homography(H) for H in transforms],
        axis=0,
    )
    if not np.allclose(reconstructed, normalized_input, rtol=1e-8, atol=1e-8):
        raise RuntimeError("Similarity-backbone decomposition failed to reconstruct G.")

    return backbones, corrections_array


def rebase_variable_global_states(
        global_transforms: np.ndarray,
        old_backbones: np.ndarray,
        images: Sequence[ImageRecord],
        variable_indices: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    优化结束后只对 variable 节点重新提取 similarity backbone。

    G_i 保持不变；只是把其中的 similarity 成分转移进 S_i，剩余 projective
    成分留在 C_i。这样下一 block 传播时只使用更新后的 S_i，而不会传播 C_i。
    """

    global_transforms = np.asarray(global_transforms, dtype=np.float64)
    backbones = np.asarray(old_backbones, dtype=np.float64).copy()
    if global_transforms.shape != backbones.shape:
        raise ValueError("global_transforms and old_backbones must have equal shape.")
    if len(images) != len(global_transforms):
        raise ValueError("images and transforms must have equal length.")

    variable_set = {int(idx) for idx in variable_indices}
    for idx in variable_set:
        if idx < 0 or idx >= len(global_transforms):
            raise IndexError(f"Variable index {idx} is outside global_transforms.")
        backbones[idx] = project_homography_to_similarity_backbone(
            global_transforms[idx],
            images[idx],
        )

    backbones, corrections = decompose_global_transforms(
        global_transforms,
        images,
        backbone_template=backbones,
    )
    reconstructed = compose_similarity_corrections(backbones, corrections)
    return reconstructed, backbones, corrections

def transform_points_projective(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    用 projective homography 变换二维点。

    齐次坐标：
        q = H [x, y, 1]^T
        x' = q0 / q2
        y' = q1 / q2
    """

    hom = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
    warped = (H @ hom.T).T
    w = warped[:, 2]
    safe_w = np.where(np.abs(w) < 1e-9, np.where(w < 0.0, -1e-9, 1e-9), w)
    return warped[:, :2] / safe_w[:, None]


# ==================== MODIFIED CODE START: typed projective optimization ====================
def projective_ordinary_residuals(
        transforms: np.ndarray,
        ordinary_edges: Sequence[PairMatch_Edge],
) -> np.ndarray:
    """第二阶段 ordinary 边：完整二维点一致性。"""

    ordinary_count = sum(edge.selected_matches for edge in ordinary_edges)
    if ordinary_count <= 0:
        return np.empty(0, dtype=np.float64)

    ordinary_scale = math.sqrt(float(ordinary_count))
    residual_chunks: List[np.ndarray] = []

    for edge in ordinary_edges:
        pi = transform_points_projective(
            transforms[edge.i],
            edge.selected_pts_i,
        )
        pj = transform_points_projective(
            transforms[edge.j],
            edge.selected_pts_j,
        )
        residual_chunks.append(
            (pi - pj).reshape(-1) / ordinary_scale
        )

    return np.concatenate(residual_chunks)


def projective_local_similarity_components(
        H: np.ndarray,
        image: ImageRecord,
) -> Tuple[float, float, np.ndarray]:
    """
    在图像中心处，用 projective Jacobian 提取局部 closest-similarity 信息。

    返回：
        scale     : 两列 Jacobian 范数的 RMS 尺度
        angle     : 最近旋转分量的角度
        direction : 局部 x 轴对应的单位方向
    """

    center = image_center_point(image)
    J = homography_jacobian(
        H,
        float(center[0]),
        float(center[1]),
    )

    u_norm2 = float(
        J[0, 0] * J[0, 0]
        + J[1, 0] * J[1, 0]
    )
    v_norm2 = float(
        J[0, 1] * J[0, 1]
        + J[1, 1] * J[1, 1]
    )

    scale = math.sqrt(
        max(
            0.5 * (u_norm2 + v_norm2),
            PROJECTIVE_REG_EPS,
        )
    )

    angle = float(
        math.atan2(
            J[1, 0] - J[0, 1],
            J[0, 0] + J[1, 1],
        )
    )

    direction = np.array(
        [
            math.cos(angle),
            math.sin(angle),
        ],
        dtype=np.float64,
    )

    return scale, angle, direction


def projective_stereo_pair_frame(
        transforms: Sequence[np.ndarray],
        images: Sequence[ImageRecord],
        i: int,
        j: int,
) -> Tuple[
    float,
    float,
    float,
    float,
    float,
    np.ndarray,
    np.ndarray,
]:
    """
    返回 projective 阶段 rigid-stereo pair 在图像中心处的局部量：
        scale_i, scale_j, angle_i, angle_j, log_scale_ratio, u, n
    """

    scale_i, angle_i, direction_i = (
        projective_local_similarity_components(
            transforms[i],
            images[i],
        )
    )
    scale_j, angle_j, direction_j = (
        projective_local_similarity_components(
            transforms[j],
            images[j],
        )
    )

    common_direction = average_directions(
        direction_i,
        direction_j,
    )
    normal_direction = np.array(
        [
            -common_direction[1],
            common_direction[0],
        ],
        dtype=np.float64,
    )

    log_scale_ratio = math.log(
        max(scale_i, PROJECTIVE_REG_EPS)
        / max(scale_j, PROJECTIVE_REG_EPS)
    )

    return (
        scale_i,
        scale_j,
        angle_i,
        angle_j,
        log_scale_ratio,
        common_direction,
        normal_direction,
    )


def projective_stereo_residuals(
        transforms: np.ndarray,
        stereo_edges: Sequence[PairMatch_Edge],
        images: Sequence[ImageRecord],
) -> np.ndarray:
    """
    第二阶段 rigid-stereo residual。

    projective H 可以产生位置相关的局部尺度和局部旋转，因此：
        - 在每张图中心处由 Jacobian 提取局部 closest-similarity；
        - 约束左右图中心处的局部旋转一致；
        - 约束左右图中心处的局部尺度一致；
        - 用两者共同局部 x 轴方向定义 n_ij；
        - stereo 匹配点只约束 n_ij 方向的残差。

    不固定沿共同 u_ij 方向的位移，不使用固定 d_ij^0。
    """

    if not stereo_edges:
        return np.empty(0, dtype=np.float64)

    stereo_count = sum(
        edge.selected_matches
        for edge in stereo_edges
    )
    if stereo_count <= 0:
        return np.empty(0, dtype=np.float64)

    point_normalization = math.sqrt(float(stereo_count))
    edge_normalization = math.sqrt(
        float(max(len(stereo_edges), 1))
    )

    residual_chunks: List[np.ndarray] = []

    for edge in stereo_edges:
        (
            scale_i,
            scale_j,
            angle_i,
            angle_j,
            log_scale_ratio,
            _,
            normal_direction,
        ) = projective_stereo_pair_frame(
            transforms,
            images,
            edge.i,
            edge.j,
        )

        pi = transform_points_projective(
            transforms[edge.i],
            edge.selected_pts_i,
        )
        pj = transform_points_projective(
            transforms[edge.j],
            edge.selected_pts_j,
        )

        perpendicular = (
            (pi - pj)
            @ normal_direction
        )

        residual_chunks.append(
            math.sqrt(LAMBDA_STEREO_PERP)
            * perpendicular
            / (
                STEREO_PERP_SIGMA_PX
                * point_normalization
            )
        )

        angle_difference = wrap_angle(
            angle_i
            - angle_j
        )
        residual_chunks.append(
            np.array(
                [
                    math.sqrt(LAMBDA_STEREO_THETA)
                    * angle_difference
                    / (
                        STEREO_THETA_SIGMA_RAD
                        * edge_normalization
                    )
                ],
                dtype=np.float64,
            )
        )

        residual_chunks.append(
            np.array(
                [
                    math.sqrt(LAMBDA_STEREO_SCALE)
                    * log_scale_ratio
                    / (
                        STEREO_LOG_SCALE_SIGMA
                        * edge_normalization
                    )
                ],
                dtype=np.float64,
            )
        )

        _ = scale_i, scale_j

    return np.concatenate(residual_chunks)


def projective_normalized_corner_denominators(
        H: np.ndarray,
        width: float,
        height: float,
) -> np.ndarray:
    """返回四个源图角点相对图像中心的同号归一化齐次分母。"""

    matrix = np.asarray(H, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"H must have shape (3, 3), got {matrix.shape}")

    x_max = float(width) - 1.0
    y_max = float(height) - 1.0
    cx = 0.5 * x_max
    cy = 0.5 * y_max
    h31, h32, h33 = matrix[2]

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        center_denominator = h31 * cx + h32 * cy + h33
        corner_denominators = np.array(
            [
                h33,
                h31 * x_max + h33,
                h32 * y_max + h33,
                h31 * x_max + h32 * y_max + h33,
            ],
            dtype=np.float64,
        )
        normalized = (
            corner_denominators
            * center_denominator
            / (center_denominator * center_denominator + PROJECTIVE_REG_EPS)
        )

    return np.nan_to_num(
        np.asarray(normalized, dtype=np.float64),
        nan=-1.0e6,
        posinf=-1.0e6,
        neginf=-1.0e6,
    )


def make_image_normalization_matrix(width: float, height: float) -> np.ndarray:
    """把图像像素坐标线性映射到固定归一化方形 [-1, 1]^2。"""

    width = float(width)
    height = float(height)
    if (
            not math.isfinite(width)
            or not math.isfinite(height)
            or width <= 1.0
            or height <= 1.0
    ):
        raise ValueError(
            "Image dimensions must be finite and greater than one, got "
            f"width={width}, height={height}"
        )

    return np.array(
        [
            [2.0 / (width - 1.0), 0.0, -1.0],
            [0.0, 2.0 / (height - 1.0), -1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def normalize_homography_for_safety(
        H: np.ndarray,
        source_image: ImageRecord,
        reference_image: ImageRecord,
) -> np.ndarray:
    """把 source->reference Homography 表达到分辨率无关的归一化坐标中。"""

    matrix = np.asarray(H, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"H must have shape (3, 3), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Homography contains NaN or Inf.")

    N_src = make_image_normalization_matrix(
        source_image.width,
        source_image.height,
    )
    N_ref = make_image_normalization_matrix(
        reference_image.width,
        reference_image.height,
    )
    H_bar = N_ref @ matrix @ np.linalg.inv(N_src)

    # Homography 的整体尺度无意义；只有 h33 足够安全时才做归一化。
    h33 = float(H_bar[2, 2])
    if math.isfinite(h33) and abs(h33) > PROJECTIVE_REG_EPS:
        H_bar = H_bar / h33
    return np.asarray(H_bar, dtype=np.float64)


def homography_jacobian_general(
        H: np.ndarray,
        x: float,
        y: float,
) -> Optional[np.ndarray]:
    """计算一般 3x3 Homography 的精确 2x2 Jacobian；退化时返回 None。"""

    matrix = np.asarray(H, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        return None

    h11, h12, h13 = matrix[0]
    h21, h22, h23 = matrix[1]
    h31, h32, h33 = matrix[2]
    x = float(x)
    y = float(y)

    w = h31 * x + h32 * y + h33
    if not math.isfinite(w) or abs(w) <= PROJECTIVE_REG_EPS:
        return None

    nu = h11 * x + h12 * y + h13
    nv = h21 * x + h22 * y + h23
    if not math.isfinite(nu) or not math.isfinite(nv):
        return None

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        inv_w2 = 1.0 / (w * w)
        J = np.array(
            [
                [
                    (h11 * w - h31 * nu) * inv_w2,
                    (h12 * w - h32 * nu) * inv_w2,
                ],
                [
                    (h21 * w - h31 * nv) * inv_w2,
                    (h22 * w - h32 * nv) * inv_w2,
                ],
            ],
            dtype=np.float64,
        )

    return J if np.all(np.isfinite(J)) else None


def make_projective_safety_grid() -> np.ndarray:
    """在归一化源图坐标 [-1, 1]^2 上生成固定 row-major 采样网格。"""

    rows = int(PROJECTIVE_SAFETY_GRID_ROWS)
    cols = int(PROJECTIVE_SAFETY_GRID_COLS)
    if rows <= 0 or cols <= 0:
        raise ValueError(
            "PROJECTIVE_SAFETY_GRID_ROWS/COLS must be positive, "
            f"got {rows}, {cols}."
        )

    u_values = np.linspace(-1.0, 1.0, cols, dtype=np.float64)
    v_values = np.linspace(-1.0, 1.0, rows, dtype=np.float64)
    return np.asarray(
        [(u, v) for v in v_values for u in u_values],
        dtype=np.float64,
    )


def smooth_positive_barrier(violation: float, softness: float) -> float:
    """稳定计算 tau * softplus(violation / tau)。"""

    tau = float(softness)
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError(f"softness must be finite and positive, got {softness}")

    violation = float(violation)
    if math.isnan(violation) or violation == math.inf:
        return 1.0e6
    if violation == -math.inf:
        return 0.0

    residual = tau * float(np.logaddexp(0.0, violation / tau))
    return residual if math.isfinite(residual) else 1.0e6


def projective_native_safety_residual_size(num_regularized: int) -> int:
    """返回 denominator + singular-value barrier 的固定 residual 维数。"""

    num_regularized = int(num_regularized)
    if num_regularized < 0:
        raise ValueError("num_regularized must be non-negative.")
    grid_size = int(PROJECTIVE_SAFETY_GRID_ROWS) * int(PROJECTIVE_SAFETY_GRID_COLS)
    return num_regularized * (4 + 2 * grid_size)


def _projective_local_singular_values(
        H_bar: np.ndarray,
        sample_points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """返回 safety 网格上 Jacobian 的最小和最大奇异值。"""

    dangerous_min = PROJECTIVE_REG_EPS
    dangerous_max = 1.0 / PROJECTIVE_REG_EPS
    sigma_mins: List[float] = []
    sigma_maxs: List[float] = []

    for u, v in np.asarray(sample_points, dtype=np.float64):
        J = homography_jacobian_general(H_bar, float(u), float(v))
        if J is None:
            sigma_mins.append(dangerous_min)
            sigma_maxs.append(dangerous_max)
            continue

        try:
            singular_values = np.linalg.svd(J, compute_uv=False)
        except np.linalg.LinAlgError:
            singular_values = np.empty(0, dtype=np.float64)

        if singular_values.shape != (2,) or not np.all(np.isfinite(singular_values)):
            sigma_min = dangerous_min
            sigma_max = dangerous_max
        else:
            sigma_min = max(float(np.min(singular_values)), PROJECTIVE_REG_EPS)
            sigma_max = max(float(np.max(singular_values)), sigma_min)

        sigma_mins.append(sigma_min)
        sigma_maxs.append(sigma_max)

    return (
        np.asarray(sigma_mins, dtype=np.float64),
        np.asarray(sigma_maxs, dtype=np.float64),
    )



def projective_native_safety_residuals(
        corrections: np.ndarray,
        images: Sequence[ImageRecord],
        regularized_indices: Sequence[int],
) -> np.ndarray:
    """
    对局部 projective correction C_i 施加 denominator + singular-value barrier。

    最终全局变换写成 G_i = S_i @ C_i。S_i 是稳定 similarity backbone；这里直接
    约束 C_i，而不是 inv(H_reference) @ H_i，因此公共 global projective 漂移不会
    被参考变换抵消。每个 C_i 都在对应源图自身的归一化坐标系中评估。
    """

    corrections = np.asarray(corrections, dtype=np.float64)
    if corrections.ndim != 3 or corrections.shape[1:] != (3, 3):
        raise ValueError("corrections must have shape (N, 3, 3).")
    if len(images) != len(corrections):
        raise ValueError("images and corrections must have equal length.")

    indices = [int(idx) for idx in regularized_indices]
    if len(indices) != len(set(indices)):
        raise ValueError("regularized_indices contains duplicate indices.")
    for idx in indices:
        if idx < 0 or idx >= len(corrections):
            raise IndexError(f"Regularized index {idx} is outside corrections.")
    if not indices:
        return np.empty(0, dtype=np.float64)

    lambda_denom = float(LAMBDA_PROJECTIVE_DENOM)
    lambda_sv = float(LAMBDA_PROJECTIVE_SV)
    for name, value in (
        ("LAMBDA_PROJECTIVE_DENOM", lambda_denom),
        ("LAMBDA_PROJECTIVE_SV", lambda_sv),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative, got {value}.")

    sv_min = float(PROJECTIVE_SV_MIN)
    sv_max = float(PROJECTIVE_SV_MAX)
    if (
            not math.isfinite(sv_min)
            or not math.isfinite(sv_max)
            or sv_min <= 0.0
            or sv_min >= sv_max
    ):
        raise ValueError("Projective singular-value limits must satisfy 0 < min < max.")

    sample_points = make_projective_safety_grid()
    num_regularized = len(indices)
    grid_size = len(sample_points)

    denom_scale = math.sqrt(lambda_denom / float(max(4 * num_regularized, 1)))
    sv_scale = math.sqrt(
        lambda_sv / float(max(2 * grid_size * num_regularized, 1))
    )

    log_sv_min = math.log(sv_min)
    log_sv_max = math.log(sv_max)

    denominator_residuals: List[float] = []
    singular_value_residuals: List[float] = []

    for idx in indices:
        correction = normalize_projective_homography(corrections[idx])

        normalized_denominators = projective_normalized_corner_denominators(
            correction,
            images[idx].width,
            images[idx].height,
        )
        denominator_residuals.extend(
            denom_scale * smooth_positive_barrier(
                PROJECTIVE_DENOM_MIN - float(value),
                PROJECTIVE_DENOM_SOFTNESS,
            )
            for value in normalized_denominators
        )

        # C_i 的输入和输出都使用该图像自己的像素尺度。
        correction_bar = normalize_homography_for_safety(
            correction,
            images[idx],
            images[idx],
        )
        sigma_mins, sigma_maxs = _projective_local_singular_values(
            correction_bar,
            sample_points,
        )

        for sigma_min, sigma_max in zip(sigma_mins, sigma_maxs):
            log_sigma_min = math.log(max(float(sigma_min), PROJECTIVE_REG_EPS))
            log_sigma_max = math.log(max(float(sigma_max), PROJECTIVE_REG_EPS))
            singular_value_residuals.extend(
                [
                    sv_scale * smooth_positive_barrier(
                        log_sv_min - log_sigma_min,
                        PROJECTIVE_SV_SOFTNESS,
                    ),
                    sv_scale * smooth_positive_barrier(
                        log_sigma_max - log_sv_max,
                        PROJECTIVE_SV_SOFTNESS,
                    ),
                ]
            )

    result = np.asarray(
        denominator_residuals + singular_value_residuals,
        dtype=np.float64,
    )
    expected_size = projective_native_safety_residual_size(num_regularized)
    if result.size != expected_size:
        raise RuntimeError(
            "Native projective safety residual dimension changed unexpectedly: "
            f"{result.size} != {expected_size}."
        )
    return np.nan_to_num(
        result,
        nan=1.0e6,
        posinf=1.0e6,
        neginf=1.0e6,
    )

def projective_data_residuals(
        transforms: np.ndarray,
        edges: Sequence[PairMatch_Edge],
        images: Sequence[ImageRecord],
) -> np.ndarray:
    """统一构造 typed-edge 数据项：ordinary 与 stereo 各走自己的残差。"""

    ordinary_edges, stereo_edges = split_edges_by_type(edges)
    chunks: List[np.ndarray] = []

    ordinary_residual = projective_ordinary_residuals(transforms, ordinary_edges)
    if ordinary_residual.size > 0:
        chunks.append(ordinary_residual)

    stereo_residual = projective_stereo_residuals(transforms, stereo_edges, images)
    if stereo_residual.size > 0:
        chunks.append(stereo_residual)

    return (
        np.concatenate(chunks).astype(np.float64, copy=False)
        if chunks
        else np.empty(0, dtype=np.float64)
    )



def projective_objective_residuals(
        global_transforms: np.ndarray,
        corrections: np.ndarray,
        edges: Sequence[PairMatch_Edge],
        images: Sequence[ImageRecord],
        regularized_indices: Sequence[int],
) -> np.ndarray:
    """统一目标：typed-edge 数据项作用于 G_i，native safety 只作用于 C_i。"""

    chunks: List[np.ndarray] = []

    data_residual = projective_data_residuals(global_transforms, edges, images)
    if data_residual.size > 0:
        chunks.append(data_residual)

    safety_residual = projective_native_safety_residuals(
        corrections,
        images,
        regularized_indices,
    )
    if safety_residual.size > 0:
        chunks.append(safety_residual)

    return (
        np.concatenate(chunks).astype(np.float64, copy=False)
        if chunks
        else np.empty(0, dtype=np.float64)
    )

def projective_objective_expected_residual_size(
        edges: Sequence[PairMatch_Edge],
        num_regularized: int,
) -> int:
    """返回 typed-edge 数据项与 native safety barrier 的固定 residual 总维数。"""

    data_size = 0
    for edge in edges:
        if edge.edge_type == "ordinary":
            data_size += 2 * int(edge.selected_matches)
        elif edge.edge_type == "stereo":
            data_size += int(edge.selected_matches) + 2
        else:
            raise ValueError(f"Unknown edge_type={edge.edge_type!r}")

    return data_size + projective_native_safety_residual_size(num_regularized)



def backbone_correction_projective_residuals(
        params: np.ndarray,
        backbones: np.ndarray,
        base_corrections: np.ndarray,
        variable_indices: Sequence[int],
        edges: Sequence[PairMatch_Edge],
        images: Sequence[ImageRecord],
) -> np.ndarray:
    """
    Local 与 persistent projective 优化共用的唯一残差入口。

    优化变量是指定节点的 C_i；先组成 G_i=S_i@C_i，再让 ordinary/stereo 数据项
    作用于 G_i，native safety 仅作用于 C_i。
    """

    expected_size = projective_objective_expected_residual_size(
        edges,
        len(variable_indices),
    )
    if expected_size <= 0:
        return np.empty(0, dtype=np.float64)

    try:
        corrections = unpack_variable_projective_params(
            params,
            base_corrections,
            variable_indices,
        )
        global_transforms = compose_similarity_corrections(
            backbones,
            corrections,
        )
        residual = projective_objective_residuals(
            global_transforms=global_transforms,
            corrections=corrections,
            edges=edges,
            images=images,
            regularized_indices=variable_indices,
        )
        if residual.size != expected_size:
            raise RuntimeError(
                "Projective residual dimension changed unexpectedly: "
                f"{residual.size} != {expected_size}."
            )
        if not np.all(np.isfinite(residual)):
            raise FloatingPointError("Projective residual contains NaN or Inf.")
        return residual

    except (
        ValueError,
        IndexError,
        RuntimeError,
        FloatingPointError,
        np.linalg.LinAlgError,
        OverflowError,
    ):
        return np.full(
            expected_size,
            PERSISTENT_INVALID_RESIDUAL,
            dtype=np.float64,
        )


def optimize_projectives(
        initial_affines: np.ndarray,
        edges: Sequence[PairMatch_Edge],
        images: Sequence[ImageRecord],
):
    """
    第二阶段 block-local projective 优化。

    第一阶段 similarity 结果直接作为固定 backbone S_i；第二阶段只优化 C_i，最终
    H_i=S_i@C_i。ordinary/stereo 保持 typed-edge 分流，native safety 直接限制 C_i。
    """

    print("Optimizing block-local projective corrections on similarity backbones")
    backbones = np.asarray(initial_affines, dtype=np.float64).copy()
    num_images = len(backbones)
    if len(images) != num_images:
        raise ValueError(
            "images 数量必须与 initial_affines 数量一致："
            f"{len(images)} != {num_images}"
        )

    backbones = np.stack(
        [normalize_projective_homography(S) for S in backbones],
        axis=0,
    )
    # Affine 阶段理论上已是 similarity；强制去掉数值 bottom-row 残留。
    backbones[:, 2, :] = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    ordinary_edges, stereo_edges = split_edges_by_type(edges)
    print(
        f"  projective objective: ordinary_edges={len(ordinary_edges)}, "
        f"stereo_edges={len(stereo_edges)}, correction_safety=denominator+singular_values"
    )

    base_corrections = np.repeat(
        np.eye(3, dtype=np.float64)[None, :, :],
        num_images,
        axis=0,
    )
    variable_indices = list(range(1, num_images))
    x0 = pack_variable_projective_params(base_corrections, variable_indices)
    residual_args = (
        backbones,
        base_corrections,
        variable_indices,
        edges,
        images,
    )

    r0 = backbone_correction_projective_residuals(x0, *residual_args)
    initial_cost = float(0.5 * np.sum(r0 * r0))

    result = least_squares(
        backbone_correction_projective_residuals,
        x0,
        args=residual_args,
        loss="soft_l1",
        f_scale=4.0,
        max_nfev=MAX_OPT_NFEV_PROJECTIVE,
        verbose=1,
    )

    final_corrections = unpack_variable_projective_params(
        result.x,
        base_corrections,
        variable_indices,
    )
    final_transforms = compose_similarity_corrections(
        backbones,
        final_corrections,
    )
    r1 = backbone_correction_projective_residuals(result.x, *residual_args)
    final_cost = float(0.5 * np.sum(r1 * r1))
    return final_transforms, result, initial_cost, final_cost

# ==================== MODIFIED CODE END: typed projective optimization ====================


# ==================== PERSISTENT TWO-BLOCK BACKBONE/CORRECTION OPTIMIZATION ====================
def build_persistent_active_window(
        previous_block: PersistentBlockCache,
        current_keys: Sequence[Tuple[str, int]],
        current_images: Sequence[ImageRecord],
        current_edges: Sequence[PairMatch_Edge],
) -> Tuple[
    List[Tuple[str, int]],
    List[ImageRecord],
    List[PairMatch_Edge],
]:
    """合并相邻两个 block，重映射并保留双方的全部合法 typed-edge factors。"""

    previous_keys = list(previous_block.keys)
    current_keys = list(current_keys)
    if len(previous_keys) != 4 or len(current_keys) != 4:
        raise ValueError("Persistent two-block window requires two 4-image blocks.")

    previous_key_set = set(previous_keys)
    current_key_set = set(current_keys)
    shared_keys = previous_key_set & current_key_set
    if len(shared_keys) != 2:
        raise ValueError(
            "Adjacent blocks must share exactly one stereo pair (2 images), "
            f"but shared {len(shared_keys)} keys: {sorted(shared_keys)}"
        )

    # 先保留 previous block 顺序，再追加 current block 的新节点。
    active_keys = previous_keys.copy()
    for key in current_keys:
        if key not in active_keys:
            active_keys.append(key)

    if len(active_keys) != 6:
        raise ValueError(
            "Two adjacent 4-image blocks should form exactly 6 unique images, "
            f"got {len(active_keys)}."
        )

    image_by_key: Dict[Tuple[str, int], ImageRecord] = {}
    for key, image in zip(previous_keys, previous_block.images):
        image_by_key[key] = image
    for key, image in zip(current_keys, current_images):
        image_by_key[key] = image

    active_images = [image_by_key[key] for key in active_keys]
    active_index = {key: idx for idx, key in enumerate(active_keys)}

    active_edges: List[PairMatch_Edge] = []

    def add_block_edges(
            block_label: str,
            block_keys: Sequence[Tuple[str, int]],
            edges: Sequence[PairMatch_Edge],
    ) -> None:
        for edge in edges:
            if edge.i < 0 or edge.i >= len(block_keys):
                raise IndexError(f"Invalid edge.i={edge.i} for {block_label} block.")
            if edge.j < 0 or edge.j >= len(block_keys):
                raise IndexError(f"Invalid edge.j={edge.j} for {block_label} block.")

            key_i = block_keys[edge.i]
            key_j = block_keys[edge.j]
            remapped = replace(
                edge,
                i=active_index[key_i],
                j=active_index[key_j],
                edge_name=f"{block_label}:{edge.edge_name}",
            )
            active_edges.append(remapped)

    add_block_edges(
        "previous",
        previous_keys,
        previous_block.valid_edges,
    )
    add_block_edges(
        "current",
        current_keys,
        current_edges,
    )

    return active_keys, active_images, active_edges



def initialize_persistent_active_state(
        active_keys: Sequence[Tuple[str, int]],
        current_keys: Sequence[Tuple[str, int]],
        current_images: Sequence[ImageRecord],
        current_local_projective: np.ndarray,
        global_transforms_by_key: Dict[Tuple[str, int], np.ndarray],
        global_backbones_by_key: Dict[Tuple[str, int], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    为两 block 窗口生成 (S_i, G_i) 初值。

    已存在节点直接复用 persistent S_old/G_old。新节点只通过 similarity backbone
    传播：Q_sim = S_shared_old @ inv(S_shared_local)，S_new=Q_sim@S_new_local；
    当前 block 的局部 correction C_local 只用于构造 G_new 初值，绝不进入 Q_sim。
    """

    current_keys = list(current_keys)
    current_local = np.asarray(current_local_projective, dtype=np.float64)
    if current_local.shape != (len(current_keys), 3, 3):
        raise ValueError(
            "current_local_projective must match current_keys and have shape (N,3,3)."
        )
    if len(current_images) != len(current_keys):
        raise ValueError("current_images and current_keys must have equal length.")

    current_local_backbones, current_local_corrections = decompose_global_transforms(
        current_local,
        current_images,
    )
    current_key_to_local = {key: idx for idx, key in enumerate(current_keys)}

    available_shared_local_indices = [
        idx
        for idx, key in enumerate(current_keys)
        if key in global_transforms_by_key and key in global_backbones_by_key
    ]
    if not available_shared_local_indices:
        raise ValueError(
            "Cannot initialize a connected persistent window without shared S_old/G_old."
        )

    anchor_local_idx = next(
        (
            idx
            for idx in (0, 2)
            if idx in available_shared_local_indices
        ),
        available_shared_local_indices[0],
    )
    anchor_key = current_keys[anchor_local_idx]
    S_anchor_global = normalize_projective_homography(
        global_backbones_by_key[anchor_key]
    )
    S_anchor_local = normalize_projective_homography(
        current_local_backbones[anchor_local_idx]
    )
    Q_similarity = normalize_projective_homography(
        S_anchor_global @ np.linalg.inv(S_anchor_local)
    )
    # 数值上再次投影，确保传播 gauge 没有 projective bottom row。
    Q_similarity = project_homography_to_similarity_backbone(
        Q_similarity,
        current_images[anchor_local_idx],
    )

    active_backbones: List[np.ndarray] = []
    active_transforms: List[np.ndarray] = []
    for key in active_keys:
        if key in global_transforms_by_key and key in global_backbones_by_key:
            S = normalize_projective_homography(global_backbones_by_key[key])
            G = normalize_projective_homography(global_transforms_by_key[key])
        elif key in current_key_to_local:
            local_idx = current_key_to_local[key]
            S = normalize_projective_homography(
                Q_similarity @ current_local_backbones[local_idx]
            )
            C_local = current_local_corrections[local_idx]
            G = normalize_projective_homography(S @ C_local)
        else:
            raise KeyError(
                f"Active key {key} has neither persistent state nor current local state."
            )
        active_backbones.append(S)
        active_transforms.append(G)

    return (
        np.stack(active_backbones, axis=0),
        np.stack(active_transforms, axis=0),
    )


def optimize_persistent_window_projectives(
        initial_global_backbones: np.ndarray,
        initial_global_transforms: np.ndarray,
        variable_indices: Sequence[int],
        window_edges: Sequence[PairMatch_Edge],
        window_images: Sequence[ImageRecord],
):
    """
    对 previous + current 两个 block 做 fixed-lag persistent 优化。

    S_i 在本次 least_squares 中固定，变量是 C_i；数据项使用 G_i=S_i@C_i，
    native safety 直接限制 C_i。最老 pair 的 C 固定以保留历史 gauge。
    """

    backbones = np.asarray(initial_global_backbones, dtype=np.float64)
    transforms = np.asarray(initial_global_transforms, dtype=np.float64)
    if backbones.shape != transforms.shape or backbones.ndim != 3 or backbones.shape[1:] != (3, 3):
        raise ValueError("initial backbones/transforms must both have shape (N,3,3).")
    if len(transforms) != len(window_images):
        raise ValueError("window_images and transforms must have equal length.")

    backbones = np.stack(
        [normalize_projective_homography(S) for S in backbones],
        axis=0,
    )
    transforms = np.stack(
        [normalize_projective_homography(G) for G in transforms],
        axis=0,
    )
    _, base_corrections = decompose_global_transforms(
        transforms,
        window_images,
        backbone_template=backbones,
    )

    variable_indices = [int(idx) for idx in variable_indices]
    variable_set = set(variable_indices)
    if not variable_indices:
        raise ValueError("Persistent window has no variable corrections.")
    if len(variable_indices) != len(variable_set):
        raise ValueError("variable_indices contains duplicate indices.")

    fixed_indices = [idx for idx in range(len(transforms)) if idx not in variable_set]
    if not fixed_indices:
        raise ValueError("Persistent optimization must keep at least one historical state fixed.")

    influential_edges = [
        edge
        for edge in window_edges
        if edge.i in variable_set or edge.j in variable_set
    ]
    if not influential_edges:
        raise ValueError("Persistent window has no factor touching a variable correction.")

    ordinary_edges, stereo_edges = split_edges_by_type(influential_edges)
    print("Optimizing persistent two-block corrections on similarity backbones")
    print(
        f"  nodes={len(transforms)}, fixed={fixed_indices}, variable={variable_indices}"
    )
    print(
        f"  typed factors: ordinary={len(ordinary_edges)}, "
        f"stereo={len(stereo_edges)}, total={len(influential_edges)}; "
        "correction_safety=denominator+singular_values"
    )

    x0 = pack_variable_projective_params(base_corrections, variable_indices)
    residual_args = (
        backbones,
        base_corrections,
        variable_indices,
        influential_edges,
        window_images,
    )

    r0 = backbone_correction_projective_residuals(x0, *residual_args)
    if r0.size == 0:
        raise RuntimeError("Persistent projective objective has no residuals.")
    initial_cost = float(0.5 * np.sum(r0 * r0))

    result = least_squares(
        backbone_correction_projective_residuals,
        x0,
        args=residual_args,
        loss="soft_l1",
        f_scale=PERSISTENT_PROJECTIVE_F_SCALE,
        max_nfev=MAX_OPT_NFEV_PERSISTENT_PROJECTIVE,
        verbose=1,
    )

    final_corrections = unpack_variable_projective_params(
        result.x,
        base_corrections,
        variable_indices,
    )
    final_transforms = compose_similarity_corrections(
        backbones,
        final_corrections,
    )
    r1 = backbone_correction_projective_residuals(result.x, *residual_args)
    final_cost = float(0.5 * np.sum(r1 * r1))

    return (
        final_transforms,
        result,
        initial_cost,
        final_cost,
        influential_edges,
    )

# ==================== PERSISTENT TWO-BLOCK BACKBONE/CORRECTION OPTIMIZATION END ====================


def compute_canvas(
    images: Sequence[ImageRecord],
    transforms: np.ndarray,
    max_canvas_size: Optional[int] = MAX_CANVAS_SIZE,
    verbose: bool = True,
):
    """
    根据最终 H_proj_i 计算全局画布。

    做法：
    1. 把每张原图四个角点通过 H_proj_i 投影到全局坐标。
    2. 统计所有角点的 min_x/min_y/max_x/max_y。
    3. 构造 T_canvas，把负坐标平移到正坐标区域。
    4. 若画布过大，只缩小 debug 预览图，不改变保存的 H 矩阵。


    image:所有图片的信息，每个ImageRecord里至少包含宽、高、图像索引等
    transforms:每张图对应的全局变换矩阵,也就是每张图的H_proj_i。这里的H是3*3单应矩阵.迎来把当前图像投影到全局坐标
    """
    if not images:
        raise ValueError("Cannot compute a canvas without images.")
    if len(images) != len(transforms):
        raise ValueError(
            "images 数量必须与 transforms 数量一致："
            f"{len(images)} != {len(transforms)}"
        )
    if not np.all(np.isfinite(transforms)):
        raise RuntimeError("Projective transforms contain NaN or Inf.")

    all_corners = [] #创建一个空列表,用来保存所有图像经过单应矩阵变换的4个角点

    """
        同时遍历图片和对应的单应矩阵
        每张图都有自己的H,表示这张图应该如何被投到全局拼接坐标系里
    """
    for image, H in zip(images, transforms):
        corners = (np.array
        (
            [[0.0, 0.0], #左上角
             [image.width - 1.0, 0.0], #右上角
             [image.width - 1.0, image.height - 1.0], #右下角
             [0.0, image.height - 1.0]], #左下角
            dtype=np.float64,
        ))

        #把当前图片的四个角点通过单应矩阵H投影到全局坐标系,然后加入all_corners
        projected_corners = transform_points_projective(H, corners)
        if not np.all(np.isfinite(projected_corners)):
            raise RuntimeError(
                f"Non-finite projected corners for image: {image.name}"
            )
        all_corners.append(projected_corners)

    #把所有图片变换后的角点拼成一个大数组
    """
        假设有5张图,每张图4个角点,那么最后corners的形状是:20 * 2
    """
    corners = np.vstack(all_corners)

    #计算所有角点里的最小x和最小y
    min_xy = np.floor(np.min(corners, axis=0)).astype(np.float64)
    max_xy = np.ceil(np.max(corners, axis=0)).astype(np.float64)


    #得到图片warp后完整容纳所需的画布的高度和宽度
    width = int(max_xy[0] - min_xy[0] + 1)
    height = int(max_xy[1] - min_xy[1] + 1)

    if width <= 0 or height <= 0:
        raise RuntimeError("Invalid canvas size computed from projective transforms.")

    """
        前面算出的全局坐标可能有负数,例如某张图warp后落在x = -300的区域。但图像数组不能用负坐标索引,
        所以需要整体平移
    """
    T_canvas = np.array(
        [[1.0, 0.0, -min_xy[0]],
         [0.0, 1.0, -min_xy[1]],
         [0.0, 0.0, 1.0]], dtype=np.float64)

    #默认预览图不缩放
    scale = 1.0

    #取画布宽高中较大的那个边,用来判断画布是否过大
    max_side = max(width, height)

    # 如果画布太大，就缩小预览图。max_canvas_size=None 表示不限制。
    if max_canvas_size is not None and max_side > max_canvas_size:
        scale = max_canvas_size / float(max_side)
        if verbose:
            print(
                f"  Canvas {width}x{height} exceeds preview limit="
                f"{max_canvas_size}; preview scale={scale:.4f}"
            )

    preview_width = max(1, int(math.ceil(width * scale)))
    preview_height = max(1, int(math.ceil(height * scale)))

    S = np.array(
        [[scale, 0.0, 0.0],
         [0.0, scale, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64)

    return {
        "T_canvas": T_canvas,
        "scale": scale,
        "width": width,
        "height": height,
        "preview_width": preview_width,
        "preview_height": preview_height,
        "preview_transform": S @ T_canvas,#先用T_canvas把负坐标平移到正坐标区域
        #再用S缩放到debug预览尺寸
    }

#创建一个OpenCV的Graph_cut接缝查找器(seam finder)对象
"""
    Graph_Cut Seam Finder会把重叠区域看成一个图优化问题，在其中寻找一条"切割路线",是路径尽量经过两张图差异较小的位置。
    接缝两侧分别保留不同图像的像素

"""
def make_graphcut_seam_finder():
    """
    创建 OpenCV Graph-Cut seam finder，
    并兼容不同 OpenCV Python 接口。
    """
    candidates = [] #创建一个空列表candidate,这个列表用于保存所有可能可用的构造方法
    #每一个元素都是一个二元组(接口名称, 构造函数)

    if hasattr(cv2, "detail_GraphCutSeamFinder"):
        candidates.append(
            ("cv2.detail_GraphCutSeamFinder",
             lambda: cv2.detail_GraphCutSeamFinder("COST_COLOR"))
        )

    if hasattr(cv2, "detail") and hasattr(cv2.detail, "GraphCutSeamFinder"):
        candidates.append(
            ("cv2.detail.GraphCutSeamFinder",
             lambda: cv2.detail.GraphCutSeamFinder("COST_COLOR"))
        )

    if hasattr(cv2, "detail") and hasattr(
        cv2.detail, "GraphCutSeamFinder_create"
    ):
        candidates.append(
            ("cv2.detail.GraphCutSeamFinder_create",
             lambda: cv2.detail.GraphCutSeamFinder_create("COST_COLOR"))
        )

    errors = []

    for name, ctor in candidates:
        try:
            return ctor()
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    raise RuntimeError(
        "无法创建 OpenCV GraphCutSeamFinder。\n"
        + "\n".join(errors)
    )


def get_mask_bounding_box(mask: np.ndarray):
    """计算二值 mask 中所有非零像素的最小外接矩形。"""

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x_0 = int(xs.min())
    x_1 = int(xs.max()) + 1
    y_0 = int(ys.min())
    y_1 = int(ys.max()) + 1
    return x_0, y_0, x_1, y_1

def compose_graphcut_mosaic_preview(
    images: Sequence[ImageRecord],
    transforms: np.ndarray,
    max_canvas_size: Optional[int] = MAX_CANVAS_SIZE,
    use_graphcut: Optional[bool] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
        使用 frame-to-frame graph-cut 生成拼接后的图。
        每次将当前 mosaic 和下一张 warped image 做 graph-cut，
        根据 seam mask 选择像素来源。
    """

    graphcut_enabled = USE_GRAPHCUT if use_graphcut is None else bool(use_graphcut)
    blend_mode = "graph-cut" if graphcut_enabled else "hard-overlay"
    if verbose:
        print(f"Composing {blend_mode} mosaic preview")

    #计算全局画布
    """
        1、用每张图的变换矩阵计算其变换后的四个角点
        2、找到所有像素覆盖范围的最小、最大坐标
        3、确定最终全局画布的宽度和高度
        4、生成一个平移或缩放矩阵，将可能出现的负坐标移动到画布内部
        5、如果这是preview，可能还会缩小画布尺寸
    """

    canvas_info = compute_canvas(
        images,
        transforms,
        max_canvas_size=max_canvas_size,
        verbose=verbose,
    )

    preview_T = canvas_info["preview_transform"]

    size = (
        canvas_info["preview_width"],
        canvas_info["preview_height"],
    )

    #创建空的全局mosaic，创建一张全黑RGB/BRG图像，作为全局拼接的画布,OpenCV中通常是BGR顺序，而不是RGB的顺序
    mosaic = np.zeros((size[1], size[0], 3), dtype=np.uint8)

    #创建与全局画布同样大小的掩码
    mosaic_mask = np.zeros((size[1], size[0]), dtype=np.uint8)

    #创建Graph_cut接缝查找器
    """
        这是选择是cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD")，
        寻找一条代条代价最小的接缝 
        两张图片颜色差异较小的位置,梯度差异较小的位置,不行明显穿过物体的位置
    """
    seam_finder = make_graphcut_seam_finder() if graphcut_enabled else None


    for image, H in tqdm(
        zip(images, transforms),
        total=len(images),
        desc=f"{blend_mode} preview",
        disable=not verbose,
    ):

        H_canvas = preview_T @ H
        
        #把当前图片warp到全局画布上
        warped = cv2.warpPerspective(
            image_pixels(image),
            H_canvas,
            size,
            flags=cv2.INTER_LINEAR #是双线性插值，用于图像内容变换，能够减小锯齿
        )
        #变换后：当前图像覆盖的位置有真实像素；当前图像未覆盖的位置通常为黑色0

        #创建当前图像的原始有效掩码
        mask_src = np.full((image.height, image.width), 255, dtype=np.uint8)

        #将掩码warp到全局画布上去
        warped_mask = cv2.warpPerspective(
            mask_src,
            H_canvas,
            size,
            flags=cv2.INTER_NEAREST
        )

        #强制将mask二值化
        warped_mask = ((warped_mask > 0).astype(np.uint8)) * 255



        """
        检查当前图像是否落在画布内：
            如果整个warped_mask都是0，说明:
                图像完全被变换到画布外;
                单应矩阵可能有问题;
                preview画布范围可能计算不正确
        """
        if not np.any(warped_mask):
            continue

        # 如果mosaic_mask里没有任何非零值，说明当前mosaic是空的，是第一张有效图
        if not np.any(mosaic_mask):
            valid = warped_mask > 0
            mosaic[valid] = warped[valid]
            mosaic_mask[valid] = 255
            continue

        # 关闭 GraphCut 时直接硬拼：后加入的图像覆盖此前内容。
        if not graphcut_enabled:
            valid = warped_mask > 0
            mosaic[valid] = warped[valid]
            mosaic_mask[valid] = 255
            continue

        overlap = (mosaic_mask > 0) & (warped_mask > 0)

        # 没有重叠区域，不需要 graph-cut，直接贴上
        if not np.any(overlap):
            valid = warped_mask > 0
            mosaic[valid] = warped[valid]
            mosaic_mask[valid] = 255
            continue
        
        # 当前 mosaic 和新 warped 图像只会在新图覆盖范围内发生变化。
        # 只裁剪当前 warped 图像的有效区域，避免累计 mosaic 变大后 GraphCut 的内存和耗时不断增长。
        current_mask = warped_mask > 0

        # 计算当前 warped 图像有效区域的最小外接矩形
        bbox = get_mask_bounding_box(current_mask)

        if bbox is None:
            continue

        x0, y0 , x1 , y1 = bbox

        mosaic_roi = mosaic[y0:y1, x0:x1]
        warped_roi = warped[y0:y1, x0:x1]

        mosaic_mask_roi = mosaic_mask[y0:y1, x0:x1]
        warped_mask_roi = warped_mask[y0:y1, x0:x1]

        
        # GraphCutSeamFinder 会原地修改 masks
        roi_height, roi_width = mosaic_roi.shape[:2]
        seam_scale = min(1.0, MAX_GRAPHCUT_ROI_SIZE / float(max(roi_width, roi_height)))

        if seam_scale < 1.0:
            seam_size = (
                max(1, int(round(roi_width * seam_scale))),
                max(1, int(round(roi_height * seam_scale))),
            )
            imgs_roi = [
                cv2.resize(mosaic_roi, seam_size, interpolation=cv2.INTER_AREA).astype(np.float32),
                cv2.resize(warped_roi, seam_size, interpolation=cv2.INTER_AREA).astype(np.float32),
            ]
            masks_roi = [
                cv2.resize(mosaic_mask_roi, seam_size, interpolation=cv2.INTER_NEAREST),
                cv2.resize(warped_mask_roi, seam_size, interpolation=cv2.INTER_NEAREST),
            ]
        else:
            imgs_roi = [
                mosaic_roi.astype(np.float32),
                warped_roi.astype(np.float32),
            ]
            masks_roi = [
                mosaic_mask_roi.copy(),
                warped_mask_roi.copy(),
            ]

        corners_roi = [
            (0, 0),
            (0, 0),
        ]

        assert seam_finder is not None
        seam_finder.find(
            imgs_roi,
            corners_roi,
            masks_roi,
        )

        # GraphCut 后局部区域中各自应该保留的位置
        if seam_scale < 1.0:
            old_keep_roi = cv2.resize(
                masks_roi[0], (roi_width, roi_height), interpolation=cv2.INTER_NEAREST
            ) > 0
            new_keep_roi = cv2.resize(
                masks_roi[1], (roi_width, roi_height), interpolation=cv2.INTER_NEAREST
            ) > 0
        else:
            old_keep_roi = masks_roi[0] > 0
            new_keep_roi = masks_roi[1] > 0

        # 先复制原来的局部 mosaic
        updated_roi = mosaic_roi.copy()

        # GraphCut 认为属于新图的像素，使用 warped_roi 覆盖
        updated_roi[new_keep_roi] = warped_roi[new_keep_roi]

        # 将局部结果写回全局画布
        mosaic[y0:y1, x0:x1] = updated_roi

        # 更新局部有效区域 mask
        updated_mask_roi = (
           (old_keep_roi | new_keep_roi).astype(np.uint8)
        ) * 255

        # 将局部 mask 写回全局 mask
        mosaic_mask[y0:y1, x0:x1] = updated_mask_roi

    return mosaic, canvas_info


def save_graphcut_mosaic_preview(
    images: Sequence[ImageRecord],
    transforms: np.ndarray,
    output_path: Path,
) -> Dict[str, object]:
    """生成并保存拼接预览，保持原有调用接口不变。"""

    mosaic, canvas_info = compose_graphcut_mosaic_preview(
        images,
        transforms,
        max_canvas_size=MAX_CANVAS_SIZE,
        use_graphcut=USE_GRAPHCUT,
        verbose=True,
    )
    imwrite_unicode(output_path, mosaic)
    return canvas_info


def filter_valid_edges(
    pair_results: Sequence[PairMatch_Edge],
    num_images: int,
) -> List[PairMatch_Edge]:
    """筛选有效边，并确认所有图片都与参考图连通。"""

    valid = [edge for edge in pair_results if edge.selected_matches >= min(MIN_INLIERS, SELECTED_MATCHES_PER_PAIR)]
    print(f"  Valid registration edges: {len(valid)} / {len(pair_results)}")
    if not valid:
        raise RuntimeError("No valid image pair survived matching/RANSAC/uniform selection.")
    if sum(edge.selected_matches for edge in valid) < 4 * max(1, len(valid)):
        raise RuntimeError("Too few selected matches for stable optimization.")

    adjacency = {idx: set() for idx in range(num_images)}
    for edge in valid:
        adjacency[edge.i].add(edge.j)
        adjacency[edge.j].add(edge.i)

    visited = {0}
    pending = [0]
    while pending:
        current = pending.pop()
        for neighbor in adjacency[current] - visited:
            visited.add(neighbor)
            pending.append(neighbor)

    if len(visited) != num_images:
        missing = sorted(set(range(num_images)) - visited)
        raise RuntimeError(
            f"Registration graph is disconnected; images {missing} cannot be aligned to reference image 0."
        )
    return valid



# ==================== RIGID STEREO DEBUG OUTPUT ====================
def rigid_stereo_config_dict() -> Dict[str, object]:
    """返回当前 rigid-stereo 方法的关键超参数，便于复现实验。"""

    return {
        "method": "rigid_stereo_common_similarity",
        "lambda_perp": LAMBDA_STEREO_PERP,
        "perp_sigma_px": STEREO_PERP_SIGMA_PX,
        "lambda_theta": LAMBDA_STEREO_THETA,
        "theta_sigma_rad": STEREO_THETA_SIGMA_RAD,
        "theta_sigma_deg": math.degrees(
            STEREO_THETA_SIGMA_RAD
        ),
        "lambda_scale": LAMBDA_STEREO_SCALE,
        "log_scale_sigma": STEREO_LOG_SCALE_SIGMA,
        "affine_stereo_weight_schedule": list(
            AFFINE_STEREO_WEIGHT_SCHEDULE
        ),
        "affine_stereo_energy": (
            "r_perp + relative_rotation + relative_log_scale"
        ),
        "projective_stereo_energy": (
            "center-Jacobian local r_perp + relative_rotation "
            "+ relative_log_scale"
        ),
        "parallel_disparity_constraint": "none",
        "projective_safety_energy": "denominator + singular_value_bounds",
        "lambda_projective_denom": LAMBDA_PROJECTIVE_DENOM,
        "projective_denom_min": PROJECTIVE_DENOM_MIN,
        "lambda_projective_sv": LAMBDA_PROJECTIVE_SV,
        "projective_sv_min": PROJECTIVE_SV_MIN,
        "projective_sv_max": PROJECTIVE_SV_MAX,
        "projective_safety_grid_rows": PROJECTIVE_SAFETY_GRID_ROWS,
        "projective_safety_grid_cols": PROJECTIVE_SAFETY_GRID_COLS,
    }


def save_rigid_stereo_initialization_debug(
        block_idx: int,
        initialization_debug: Dict[str, object],
        out_dir: Path,
) -> None:
    """把 rigid-stereo 初始化过程和方法配置保存为 CSV。"""

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = rigid_stereo_config_dict()
    config["block_idx"] = block_idx
    config_row = {
        key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else value
        for key, value in config.items()
    }
    pd.DataFrame([config_row]).to_csv(
        out_dir / "rigid_stereo_config.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_row = {"block_idx": block_idx}
    summary_row.update(
        {
            key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else value
            for key, value in initialization_debug.items()
            if key != "events"
        }
    )
    pd.DataFrame([summary_row]).to_csv(
        out_dir / "initialization_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    events = initialization_debug.get(
        "events",
        [],
    )
    if events:
        pd.DataFrame(events).to_csv(
            out_dir / "initialization_events.csv",
            index=False,
            encoding="utf-8-sig",
        )


def collect_rigid_stereo_stage_debug(
        block_idx: int,
        stage_name: str,
        transforms: np.ndarray,
        edges: Sequence[PairMatch_Edge],
        images: Sequence[ImageRecord],
        transform_kind: str,
) -> Tuple[
    List[Dict[str, object]],
    List[Dict[str, object]],
]:
    """
    收集某一阶段的 rigid-stereo 调试信息。

    transform_kind:
        "affine"     -> 使用全局 similarity 参数；
        "projective" -> 使用图像中心处 Jacobian 的局部 closest-similarity。

    summary_rows:
        每条 stereo edge 一行，记录尺度比、角度差和 r_perp 统计。

    point_rows:
        每个 stereo 匹配点一行，记录变换前后坐标及在共同 (u, n) 基底上的分量。
        parallel_component 只作为调试量保存，不参与当前优化约束。
    """

    if transform_kind not in {
        "affine",
        "projective",
    }:
        raise ValueError(
            f"Unsupported transform_kind: {transform_kind}"
        )

    stereo_edges = [
        edge
        for edge in edges
        if edge.edge_type == "stereo"
        and edge.selected_matches > 0
    ]

    summary_rows: List[Dict[str, object]] = []
    point_rows: List[Dict[str, object]] = []

    for edge in stereo_edges:

        if transform_kind == "affine":
            (
                scale_i,
                scale_j,
                angle_i,
                angle_j,
                log_scale_ratio,
                common_direction,
                normal_direction,
            ) = affine_stereo_pair_frame(
                transforms,
                edge.i,
                edge.j,
            )

            pi = transform_points_affine(
                transforms[edge.i],
                edge.selected_pts_i,
            )
            pj = transform_points_affine(
                transforms[edge.j],
                edge.selected_pts_j,
            )

        else:
            (
                scale_i,
                scale_j,
                angle_i,
                angle_j,
                log_scale_ratio,
                common_direction,
                normal_direction,
            ) = projective_stereo_pair_frame(
                transforms,
                images,
                edge.i,
                edge.j,
            )

            pi = transform_points_projective(
                transforms[edge.i],
                edge.selected_pts_i,
            )
            pj = transform_points_projective(
                transforms[edge.j],
                edge.selected_pts_j,
            )

        delta = pi - pj

        perpendicular = (
            delta
            @ normal_direction
        )
        parallel = (
            delta
            @ common_direction
        )

        angle_difference = wrap_angle(
            angle_i
            - angle_j
        )

        summary_rows.append(
            {
                "block_idx": block_idx,
                "stage": stage_name,
                "transform_kind": transform_kind,
                "edge_name": edge.edge_name,
                "i": edge.i,
                "j": edge.j,
                "image_i": images[edge.i].name,
                "image_j": images[edge.j].name,
                "selected_matches": edge.selected_matches,
                "scale_i": scale_i,
                "scale_j": scale_j,
                "scale_ratio_i_over_j": (
                    scale_i
                    / max(
                        scale_j,
                        PROJECTIVE_REG_EPS,
                    )
                ),
                "log_scale_ratio": log_scale_ratio,
                "angle_i_deg": math.degrees(
                    angle_i
                ),
                "angle_j_deg": math.degrees(
                    angle_j
                ),
                "angle_difference_deg": math.degrees(
                    angle_difference
                ),
                "common_direction_x": float(
                    common_direction[0]
                ),
                "common_direction_y": float(
                    common_direction[1]
                ),
                "common_angle_deg": math.degrees(
                    math.atan2(
                        common_direction[1],
                        common_direction[0],
                    )
                ),
                "normal_direction_x": float(
                    normal_direction[0]
                ),
                "normal_direction_y": float(
                    normal_direction[1]
                ),
                "perp_mean_px": float(
                    np.mean(perpendicular)
                ),
                "perp_median_px": float(
                    np.median(perpendicular)
                ),
                "perp_std_px": float(
                    np.std(perpendicular)
                ),
                "perp_rmse_px": float(
                    math.sqrt(
                        np.mean(
                            perpendicular
                            * perpendicular
                        )
                    )
                ),
                "perp_max_abs_px": float(
                    np.max(
                        np.abs(perpendicular)
                    )
                ),
                "parallel_mean_px_debug_only": float(
                    np.mean(parallel)
                ),
                "parallel_median_px_debug_only": float(
                    np.median(parallel)
                ),
                "parallel_std_px_debug_only": float(
                    np.std(parallel)
                ),
            }
        )

        for match_idx in range(
            len(perpendicular)
        ):
            point_rows.append(
                {
                    "block_idx": block_idx,
                    "stage": stage_name,
                    "transform_kind": transform_kind,
                    "edge_name": edge.edge_name,
                    "i": edge.i,
                    "j": edge.j,
                    "match_idx": match_idx,
                    "raw_i_x": float(
                        edge.selected_pts_i[
                            match_idx,
                            0,
                        ]
                    ),
                    "raw_i_y": float(
                        edge.selected_pts_i[
                            match_idx,
                            1,
                        ]
                    ),
                    "raw_j_x": float(
                        edge.selected_pts_j[
                            match_idx,
                            0,
                        ]
                    ),
                    "raw_j_y": float(
                        edge.selected_pts_j[
                            match_idx,
                            1,
                        ]
                    ),
                    "warped_i_x": float(
                        pi[match_idx, 0]
                    ),
                    "warped_i_y": float(
                        pi[match_idx, 1]
                    ),
                    "warped_j_x": float(
                        pj[match_idx, 0]
                    ),
                    "warped_j_y": float(
                        pj[match_idx, 1]
                    ),
                    "delta_x": float(
                        delta[match_idx, 0]
                    ),
                    "delta_y": float(
                        delta[match_idx, 1]
                    ),
                    "perp_residual_px": float(
                        perpendicular[match_idx]
                    ),
                    "parallel_component_px_debug_only": float(
                        parallel[match_idx]
                    ),
                }
            )

    return summary_rows, point_rows


def save_rigid_stereo_stage_debug(
        block_idx: int,
        stage_name: str,
        transforms: np.ndarray,
        edges: Sequence[PairMatch_Edge],
        images: Sequence[ImageRecord],
        transform_kind: str,
        out_dir: Path,
) -> None:
    """保存一个优化阶段的 rigid-stereo 汇总和逐匹配点调试表。"""

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows, point_rows = (
        collect_rigid_stereo_stage_debug(
            block_idx,
            stage_name,
            transforms,
            edges,
            images,
            transform_kind,
        )
    )

    safe_stage = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        stage_name,
    )

    pd.DataFrame(summary_rows).to_csv(
        out_dir
        / f"{safe_stage}_stereo_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(point_rows).to_csv(
        out_dir
        / f"{safe_stage}_stereo_points.csv",
        index=False,
        encoding="utf-8-sig",
    )
# ==================== RIGID STEREO DEBUG OUTPUT END ====================


def block_image_role(local_idx: int) -> Tuple[str, int]:
    """把 block 内局部编号映射为 left/right 及对应路内编号。"""

    if local_idx < 2:
        return "left", local_idx
    return "right", local_idx - 2


def make_block_image_rows(block_idx: int, images: Sequence[ImageRecord]) -> List[Dict[str, object]]:
    """生成当前 block 的图片信息记录，用于保存到 CSV。"""

    rows: List[Dict[str, object]] = []
    for local_idx, image in enumerate(images):
        camera, camera_local_idx = block_image_role(local_idx)
        rows.append(
            {
                "block_idx": block_idx,
                "local_idx": local_idx,
                "camera": camera,
                "camera_local_idx": camera_local_idx,
                "original_index": image.index,
                "name": image.name,
                "path": str(image.path),
                "width": image.width,
                "height": image.height,
            }
        )
    return rows


def make_edge_report_rows(
    block_idx: int,
    images: Sequence[ImageRecord],
    pair_results: Sequence[PairMatch_Edge],
    valid_edges: Sequence[PairMatch_Edge],
) -> List[Dict[str, object]]:
    """生成当前 block 的边匹配信息记录，用于保存到 CSV。"""

    valid_pairs = {(edge.i, edge.j) for edge in valid_edges}
    rows: List[Dict[str, object]] = []

    for edge in pair_results:
        rows.append(
            {
                "block_idx": block_idx,
                "i": edge.i,
                "j": edge.j,
                "edge_type": edge.edge_type,
                "edge_name": edge.edge_name,
                "raw_stereo_dx_median": (
                    float(np.median(edge.selected_pts_i[:, 0] - edge.selected_pts_j[:, 0]))
                    if edge.edge_type == "stereo" and edge.selected_matches > 0
                    else None
                ),
                "raw_stereo_dy_median": (
                    float(np.median(edge.selected_pts_i[:, 1] - edge.selected_pts_j[:, 1]))
                    if edge.edge_type == "stereo" and edge.selected_matches > 0
                    else None
                ),
                "image_i": images[edge.i].name,
                "image_j": images[edge.j].name,
                "is_valid": (edge.i, edge.j) in valid_pairs,
                "raw_matches": edge.raw_matches,
                "ratio_matches": edge.ratio_matches,
                "ransac_inliers": edge.ransac_inliers,
                "selected_matches": edge.selected_matches,
                "inlier_ratio": edge.inlier_ratio,
                "mean_reproj_error": edge.mean_reproj_error,
                "H_i_to_j_json": json.dumps(to_jsonable_matrix(edge.H_i_to_j), ensure_ascii=False),
            }
        )

    return rows


def save_block_transforms(
    block_idx: int,
    images: Sequence[ImageRecord],
    transforms: np.ndarray,
    out_dir: Path,
    transform_name: str,
    image_roles: Optional[Sequence[Tuple[str, int]]] = None,
) -> None:
    """保存当前 block 的每张图到 block 坐标系的变换矩阵。"""

    rows: List[Dict[str, object]] = []

    for local_idx, (image, H) in enumerate(zip(images, transforms)):
        if image_roles is None:
            camera, camera_local_idx = block_image_role(local_idx)
        else:
            camera, camera_local_idx = image_roles[local_idx]
        matrix = H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H

        row = {
            "block_idx": block_idx,
            "local_idx": local_idx,
            "camera": camera,
            "camera_local_idx": camera_local_idx,
            "original_index": image.index,
            "name": image.name,
            "path": str(image.path),
        }

        for r in range(3):
            for c in range(3):
                row[f"H{r + 1}{c + 1}"] = float(matrix[r, c])

        rows.append(row)

    pd.DataFrame(rows).to_csv(out_dir / f"{transform_name}_transforms.csv", index=False, encoding="utf-8-sig")


def make_block_summary_row(
    block_idx: int,
    images: Sequence[ImageRecord],
    status: str,
    candidate_edges: int = 0,
    valid_edges: int = 0,
    total_selected_matches: int = 0,
    affine_initial_cost: Optional[float] = None,
    affine_final_cost: Optional[float] = None,
    affine_success: Optional[bool] = None,
    projective_initial_cost: Optional[float] = None,
    projective_final_cost: Optional[float] = None,
    projective_success: Optional[bool] = None,
    persistent_applied: Optional[bool] = None,
    persistent_initial_cost: Optional[float] = None,
    persistent_final_cost: Optional[float] = None,
    persistent_success: Optional[bool] = None,
    error: str = "",
) -> Dict[str, object]:
    """生成当前 block 的汇总信息记录，用于保存到 CSV。"""

    row: Dict[str, object] = {
        "block_idx": block_idx,
        "status": status,
        "error": error,
        "candidate_edges": candidate_edges,
        "valid_edges": valid_edges,
        "total_selected_matches": total_selected_matches,
        "affine_initial_cost": affine_initial_cost,
        "affine_final_cost": affine_final_cost,
        "affine_success": affine_success,
        "projective_initial_cost": projective_initial_cost,
        "projective_final_cost": projective_final_cost,
        "projective_success": projective_success,
        "persistent_applied": persistent_applied,
        "persistent_initial_cost": persistent_initial_cost,
        "persistent_final_cost": persistent_final_cost,
        "persistent_success": persistent_success,
    }

    for local_idx, image in enumerate(images):
        camera, camera_local_idx = block_image_role(local_idx)
        prefix = f"image_{local_idx}_{camera}_{camera_local_idx}"
        row[f"{prefix}_name"] = image.name
        row[f"{prefix}_path"] = str(image.path)

    return row


def write_report_csvs(
    output_dirs: Dict[str, Path],
    block_rows: Sequence[Dict[str, object]],
    image_rows: Sequence[Dict[str, object]],
    edge_rows: Sequence[Dict[str, object]],
) -> None:
    """把所有 block 的汇总、图片、边匹配信息写入 CSV。"""

    logs_dir = output_dirs["logs"]
    logs_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(block_rows).to_csv(logs_dir / "block_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(image_rows).to_csv(logs_dir / "block_images.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(edge_rows).to_csv(logs_dir / "block_edges.csv", index=False, encoding="utf-8-sig")


def run_stitching() -> None:

    np.random.seed(RANDOM_SEED)

    left_in_dir = Path(LEFT_IN)
    right_in_dir = Path(RIGHT_IN)
    output_dir = Path(OUTPUT_DIR)

    out_dirs = make_output_dirs(output_dir)

    print(f"LEFT_IN_DIR  = {left_in_dir}")
    print(f"RIGHT_IN_DIR = {right_in_dir}")
    print(f"OUTPUT_DIR   = {output_dir}")

    block_rows: List[Dict[str, object]] = []
    image_rows: List[Dict[str, object]] = []
    edge_rows: List[Dict[str, object]] = []

    # SIFT 对象和共享帧特征跨窗口复用。cache 在每个 block 结束后会裁剪，
    # 始终只保留下一窗口仍然需要的左右两张图。
    sift = cv2.SIFT_create(nfeatures=SIFT_NFEATURES)
    feature_cache: Dict[Path, FeatureRecord] = {}

    # 上一窗口的 stereo_t1 就是下一窗口的 stereo_t。
    previous_stereo_cache: Optional[
        Tuple[Path, Path, PairMatch_Edge]
    ] = None

    # 上一个“成功完成 global 更新”的 block 因子缓存。
    # 下一 block 会与它组成 two-block fixed-lag window。
    previous_block_cache: Optional[PersistentBlockCache] = None

    # 增量全局状态
    global_images: List[ImageRecord] = []

    # key:
    #   ("left", frame_number)
    #   ("right", frame_number)
    #
    # G_i：当前图片到全局坐标系的最终 projective H。
    global_transforms_by_key: Dict[
        Tuple[str, int],
        np.ndarray,
    ] = {}

    # S_i：只含 similarity 的长期累计 backbone。跨 block 初始化只允许使用它，
    # 绝不使用 G_i 中的 projective correction。
    global_backbones_by_key: Dict[
        Tuple[str, int],
        np.ndarray,
    ] = {}

    # 保证 global_images、global_image_keys 和最终变换矩阵顺序一致
    global_image_keys: List[Tuple[str, int]] = []
    global_image_key_set = set()

    # 使用相邻帧滑动窗口：
    # (frame_0, frame_1)
    # (frame_1, frame_2)
    # (frame_2, frame_3)
    # ...
    for block_idx, (left_images, right_images) in enumerate(
        load_incremental_image_windows(
            left_in_dir,
            right_in_dir,
        )
    ):
        print(
            f"\n========== Processing block "
            f"{block_idx:04d} =========="
        )

        # 每个 block 固定包含：
        # left_t、left_t+1、right_t、right_t+1
        image_block = [
            left_images[0],
            left_images[1],
            right_images[0],
            right_images[1],
        ]

        block_dirs = make_block_output_dirs(
            out_dirs,
            block_idx,
        )

        print("Block images:")

        for local_idx, image in enumerate(image_block):
            camera, camera_local_idx = block_image_role(
                local_idx
            )

            print(
                f"  local_idx={local_idx}, "
                f"camera={camera}[{camera_local_idx}], "
                f"name={image.name}"
            )

        image_rows.extend(
            make_block_image_rows(
                block_idx,
                image_block,
            )
        )

        block_candidate_pairs: List[Tuple[int, int]] = []
        block_pair_ret: List[PairMatch_Edge] = []
        block_valid_edges: List[PairMatch_Edge] = []
        try:
            # 1. 提取当前四张图片的特征
            block_features = extract_features(
                image_block,
                cache=feature_cache,
                sift=sift,
            )

            # 2. 当前四张图建立完全图，共 6 条候选边
            block_candidate_specs = build_candidate_edges(
                len(image_block)
            )
            block_candidate_pairs = [
                (spec.i, spec.j)
                for spec in block_candidate_specs
            ]

            print("Block Matching candidate image pairs")

            for spec in tqdm(
                block_candidate_specs,
                desc=f"Block {block_idx:04d} matching",
            ):
                reused_pair = None
                if spec.edge_name == "stereo_t" and previous_stereo_cache is not None:
                    cached_left, cached_right, cached_pair = previous_stereo_cache
                    if (
                        cached_left == image_block[spec.i].path
                        and cached_right == image_block[spec.j].path
                    ):
                        reused_pair = replace(
                            cached_pair,
                            i=spec.i,
                            j=spec.j,
                            edge_type=spec.edge_type,
                            edge_name=spec.edge_name,
                        )

                if reused_pair is not None:
                    block_pair_ret.append(reused_pair)
                else:
                    block_pair_ret.append(
                        match_pair(
                            spec,
                            block_features,
                        )
                    )

            next_stereo = next(
                edge for edge in block_pair_ret
                if edge.edge_name == "stereo_t1"
            )
            previous_stereo_cache = (
                image_block[1].path,
                image_block[3].path,
                next_stereo,
            )

            # 3. 筛选有效匹配边
            block_valid_edges = filter_valid_edges(
                block_pair_ret,
                len(image_block),
            )

            edge_rows.extend(
                make_edge_report_rows(
                    block_idx,
                    image_block,
                    block_pair_ret,
                    block_valid_edges,
                )
            )

            # 4. 第一阶段：rigid-stereo aware similarity 初始化与优化
            (
                initial_affines,
                initialization_debug,
            ) = initialize_affines(
                len(image_block),
                block_valid_edges,
                image_block,
            )

            save_rigid_stereo_initialization_debug(
                block_idx,
                initialization_debug,
                block_dirs["rigid_stereo_debug"],
            )

            save_rigid_stereo_stage_debug(
                block_idx=block_idx,
                stage_name="affine_initial",
                transforms=initial_affines,
                edges=block_valid_edges,
                images=image_block,
                transform_kind="affine",
                out_dir=block_dirs["rigid_stereo_debug"],
            )

            (
                affine_transforms,
                affine_result,
                affine_initial_cost,
                affine_final_cost,
            ) = optimize_affines(
                initial_affines,
                block_valid_edges,
            )

            save_rigid_stereo_stage_debug(
                block_idx=block_idx,
                stage_name="affine_optimized",
                transforms=affine_transforms,
                edges=block_valid_edges,
                images=image_block,
                transform_kind="affine",
                out_dir=block_dirs["rigid_stereo_debug"],
            )

            if not affine_result.success:
                print(
                    "  Warning: affine optimization "
                    "did not report success: "
                    f"{affine_result.message}"
                )

            # 5. 第二阶段：projective 优化，并继续保持局部 rigid-stereo 先验
            (
                projective_transforms,
                projective_result,
                projective_initial_cost,
                projective_final_cost,
            ) = optimize_projectives(
                affine_transforms,
                block_valid_edges,
                image_block,
            )

            save_rigid_stereo_stage_debug(
                block_idx=block_idx,
                stage_name="projective_optimized",
                transforms=projective_transforms,
                edges=block_valid_edges,
                images=image_block,
                transform_kind="projective",
                out_dir=block_dirs["rigid_stereo_debug"],
            )

            if not projective_result.success:
                print(
                    "  Warning: projective optimization "
                    "did not report success: "
                    f"{projective_result.message}"
                )

            # 6. 只保存当前 block 的 CSV 数据，不生成调试图片或局部 mosaic。
            save_block_transforms(
                block_idx,
                image_block,
                affine_transforms,
                block_dirs["transforms"],
                "affine",
            )

            # 保存当前 block 的 projective H
            save_block_transforms(
                block_idx,
                image_block,
                projective_transforms,
                block_dirs["transforms"],
                "projective",
            )

            # 当前 block 中四张图的全局唯一标识
            local_keys = [
                ("left", left_images[0].index),
                ("left", left_images[1].index),
                ("right", right_images[0].index),
                ("right", right_images[1].index),
            ]

            # 8-9. Persistent two-block similarity-backbone + local-correction accumulation
            #
            # block-local projective_transforms 仍然是独立求解的：
            #   T_i : current image_i -> current block local reference
            #
            # 但真正写入 global_transforms_by_key 的 H 不再采用：
            #   block_to_global @ T_i
            # 的逐 block 覆盖。
            #
            # 对 block_idx > 0：
            #   previous block + current block -> 6 unique nodes
            #   最老 stereo pair 固定
            #   shared H_old + current new H 作为同一组 persistent global variables
            #   ordinary edge 只走 ordinary residual
            #   stereo edge 只走 rigid-stereo special residual
            persistent_applied = False
            persistent_initial_cost: Optional[float] = None
            persistent_final_cost: Optional[float] = None
            persistent_result = None

            start_new_segment = (
                block_idx == 0
                or previous_block_cache is None
            )

            if not start_new_segment:
                previous_key_set = set(previous_block_cache.keys)
                current_key_set = set(local_keys)
                shared_keys = previous_key_set & current_key_set

                # 正常相邻 block 必须共享 L_(t-1), R_(t-1) 两张图，
                # 且这两个 shared H_old 都必须已经存在于 global state。
                if (
                    len(shared_keys) != 2
                    or any(
                        key not in global_transforms_by_key
                        or key not in global_backbones_by_key
                        for key in shared_keys
                    )
                ):
                    print(
                        "  Warning: previous/current blocks do not have a complete "
                        "persistent shared stereo state; start a new global segment."
                    )
                    start_new_segment = True

            if start_new_segment:
                if block_idx > 0:
                    # 与已有 global 段失去连续连接时，保持旧代码“重新开始一段”的语义。
                    global_images.clear()
                    global_transforms_by_key.clear()
                    global_backbones_by_key.clear()
                    global_image_keys.clear()
                    global_image_key_set.clear()

                current_global_transforms = np.stack(
                    [
                        normalize_projective_homography(H)
                        for H in projective_transforms
                    ],
                    axis=0,
                )
                (
                    current_global_backbones,
                    _,
                ) = decompose_global_transforms(
                    current_global_transforms,
                    image_block,
                )

                global_transforms_by_key.update(
                    {
                        key: current_global_transforms[local_idx].copy()
                        for local_idx, key in enumerate(local_keys)
                    }
                )
                global_backbones_by_key.update(
                    {
                        key: current_global_backbones[local_idx].copy()
                        for local_idx, key in enumerate(local_keys)
                    }
                )

                print(
                    "  Similarity-backbone state: initialize new global segment; "
                    "future blocks propagate S only, never the projective correction C."
                )

            else:
                assert previous_block_cache is not None

                (
                    active_keys,
                    active_images,
                    active_edges,
                ) = build_persistent_active_window(
                    previous_block_cache,
                    local_keys,
                    image_block,
                    block_valid_edges,
                )

                (
                    initial_active_backbones,
                    initial_active_transforms,
                ) = initialize_persistent_active_state(
                    active_keys=active_keys,
                    current_keys=local_keys,
                    current_images=image_block,
                    current_local_projective=projective_transforms,
                    global_transforms_by_key=global_transforms_by_key,
                    global_backbones_by_key=global_backbones_by_key,
                )

                current_key_set = set(local_keys)
                # previous block 中不再出现在 current block 的最老 stereo pair 已完成
                # 两次窗口参与，现固定下来；其余 shared + new 节点继续优化。
                fixed_keys = [
                    key
                    for key in previous_block_cache.keys
                    if key not in current_key_set
                ]
                if len(fixed_keys) != 2:
                    raise RuntimeError(
                        "Persistent two-block window should finalize exactly one "
                        f"old stereo pair, got fixed_keys={fixed_keys}."
                    )

                active_index = {
                    key: idx
                    for idx, key in enumerate(active_keys)
                }
                fixed_indices = {
                    active_index[key]
                    for key in fixed_keys
                }
                variable_indices = [
                    idx
                    for idx in range(len(active_keys))
                    if idx not in fixed_indices
                ]

                (
                    optimized_active_transforms,
                    persistent_result,
                    persistent_initial_cost,
                    persistent_final_cost,
                    influential_window_edges,
                ) = optimize_persistent_window_projectives(
                    initial_global_backbones=initial_active_backbones,
                    initial_global_transforms=initial_active_transforms,
                    variable_indices=variable_indices,
                    window_edges=active_edges,
                    window_images=active_images,
                )

                # 把优化后的 similarity 成分吸收到 S，G 保持不变；下一 block 只传播 S。
                (
                    optimized_active_transforms,
                    optimized_active_backbones,
                    optimized_active_corrections,
                ) = rebase_variable_global_states(
                    global_transforms=optimized_active_transforms,
                    old_backbones=initial_active_backbones,
                    images=active_images,
                    variable_indices=variable_indices,
                )
                persistent_applied = True

                print(
                    "  persistent cost: "
                    f"{persistent_initial_cost:.6f} -> "
                    f"{persistent_final_cost:.6f}; "
                    f"success={persistent_result.success}; "
                    f"active_factors={len(influential_window_edges)}"
                )

                # G 与 S 原子写回，避免 projective correction 被误当作下一次传播骨架。
                global_transforms_by_key.update(
                    {
                        key: optimized_active_transforms[idx].copy()
                        for idx, key in enumerate(active_keys)
                    }
                )
                global_backbones_by_key.update(
                    {
                        key: optimized_active_backbones[idx].copy()
                        for idx, key in enumerate(active_keys)
                    }
                )

                save_block_transforms(
                    block_idx,
                    active_images,
                    optimized_active_transforms,
                    block_dirs["transforms"],
                    "persistent_window_global_projective",
                    active_keys,
                )
                save_block_transforms(
                    block_idx,
                    active_images,
                    optimized_active_backbones,
                    block_dirs["transforms"],
                    "persistent_window_similarity_backbone",
                    active_keys,
                )
                save_block_transforms(
                    block_idx,
                    active_images,
                    optimized_active_corrections,
                    block_dirs["transforms"],
                    "persistent_window_projective_correction",
                    active_keys,
                )


            # 从权威状态统一读取当前 block 的 S/G/C，避免分支内重复逻辑。
            current_global_transforms = np.stack(
                [global_transforms_by_key[key] for key in local_keys],
                axis=0,
            )
            current_global_backbones = np.stack(
                [global_backbones_by_key[key] for key in local_keys],
                axis=0,
            )
            _, current_global_corrections = decompose_global_transforms(
                current_global_transforms,
                image_block,
                backbone_template=current_global_backbones,
            )

            # 保存当前 4 图 block 在 persistent global state 中的最终 H。
            save_block_transforms(
                block_idx,
                image_block,
                current_global_transforms,
                block_dirs["transforms"],
                "persistent_current_global_projective",
                local_keys,
            )
            save_block_transforms(
                block_idx,
                image_block,
                current_global_backbones,
                block_dirs["transforms"],
                "persistent_current_similarity_backbone",
                local_keys,
            )
            save_block_transforms(
                block_idx,
                image_block,
                current_global_corrections,
                block_dirs["transforms"],
                "persistent_current_projective_correction",
                local_keys,
            )

            # 当前 block 中的新图片加入全局图片列表；shared frame 不重复添加。
            for local_idx, key in enumerate(local_keys):
                if key not in global_image_key_set:
                    global_image_key_set.add(key)
                    global_image_keys.append(key)
                    global_images.append(image_block[local_idx])

            # 10. 组装截至当前 block 的全部全局 projective H
            global_transforms = np.stack(
                [
                    global_transforms_by_key[key]
                    for key in global_image_keys
                ],
                axis=0,
            )

            # 保存截至当前 block 的全部全局 projective H
            save_block_transforms(
                block_idx,
                global_images,
                global_transforms,
                block_dirs["transforms"],
                "incremental_global_projective",
                global_image_keys,
            )

            global_backbones = np.stack(
                [global_backbones_by_key[key] for key in global_image_keys],
                axis=0,
            )
            _, global_corrections = decompose_global_transforms(
                global_transforms,
                global_images,
                backbone_template=global_backbones,
            )
            save_block_transforms(
                block_idx,
                global_images,
                global_backbones,
                block_dirs["transforms"],
                "incremental_global_similarity_backbone",
                global_image_keys,
            )
            save_block_transforms(
                block_idx,
                global_images,
                global_corrections,
                block_dirs["transforms"],
                "incremental_global_projective_correction",
                global_image_keys,
            )

            # 循环中不生成累计大图，避免每个 block 都重新 warp
            # 前面已经处理过的全部图片。
            #
            # 最终累计 mosaic 只在主循环结束后生成一次。

            block_rows.append(
                make_block_summary_row(
                    block_idx=block_idx,
                    images=image_block,
                    status="success",
                    candidate_edges=len(
                        block_candidate_pairs
                    ),
                    valid_edges=len(
                        block_valid_edges
                    ),
                    total_selected_matches=sum(
                        edge.selected_matches
                        for edge in block_valid_edges
                    ),
                    affine_initial_cost=(
                        affine_initial_cost
                    ),
                    affine_final_cost=(
                        affine_final_cost
                    ),
                    affine_success=bool(
                        affine_result.success
                    ),
                    projective_initial_cost=(
                        projective_initial_cost
                    ),
                    projective_final_cost=(
                        projective_final_cost
                    ),
                    projective_success=bool(
                        projective_result.success
                    ),
                    persistent_applied=persistent_applied,
                    persistent_initial_cost=(
                        persistent_initial_cost
                    ),
                    persistent_final_cost=(
                        persistent_final_cost
                    ),
                    persistent_success=(
                        bool(persistent_result.success)
                        if persistent_result is not None
                        else None
                    ),
                )
            )

            # 当前成功 block 的原始 typed-edge factors 留给下一次窗口。
            # 注意缓存的是 block-local edge observation，不是 active-window remap 后的 edge。
            previous_block_cache = PersistentBlockCache(
                keys=list(local_keys),
                images=list(image_block),
                valid_edges=list(block_valid_edges),
            )
        except Exception as exc:
            print(
                f"  Block {block_idx:04d} failed: "
                f"{exc}"
            )

            if (
                block_pair_ret
                and not any(
                    row.get("block_idx") == block_idx
                    for row in edge_rows
                )
            ):
                edge_rows.extend(
                    make_edge_report_rows(
                        block_idx,
                        image_block,
                        block_pair_ret,
                        block_valid_edges,
                    )
                )

            # 当前 block 没有产生可信的 persistent global state。下一 block 不能
            # 再与更早的非相邻 block 拼成 two-block window，因此断开 active-window cache。
            previous_block_cache = None

            block_rows.append(
                make_block_summary_row(
                    block_idx=block_idx,
                    images=image_block,
                    status="failed",
                    candidate_edges=len(
                        block_candidate_pairs
                    ),
                    valid_edges=len(
                        block_valid_edges
                    ),
                    total_selected_matches=sum(
                        edge.selected_matches
                        for edge in block_valid_edges
                    ),
                    error=str(exc),
                )
            )

        # 下一窗口只会复用当前 block 的第二个时间点。释放已经退出
        # 活动窗口的像素，并丢弃不会再次使用的 descriptor。
        reusable_paths = {
            image_block[1].path,
            image_block[3].path,
        }
        feature_cache = {
            path: feature
            for path, feature in feature_cache.items()
            if path in reusable_paths
        }
        for stale_image in (image_block[0], image_block[2]):
            stale_image.image = None
            stale_image.gray = None

        # 每处理完一个 block 就保存一次报告，
        # 避免程序中途异常导致已有统计结果丢失。
        write_report_csvs(
            out_dirs,
            block_rows,
            image_rows,
            edge_rows,
        )

    # 11. 所有 block 处理完成后，只生成一次最终累计 mosaic
    if global_images and global_image_keys:

        final_global_transforms = np.stack(
            [
                global_transforms_by_key[key]
                for key in global_image_keys
            ],
            axis=0,
        )

        final_mosaic_path = (
            out_dirs["mosaics"]
            / "incremental_mosaic_final.jpg"
        )

        print(
            "\nGenerating final accumulated "
            "projective mosaic: "
            f"{final_mosaic_path}"
        )

        save_graphcut_mosaic_preview(
            global_images,
            final_global_transforms,
            final_mosaic_path,
        )

    else:
        print(
            "Warning: no valid global images were "
            "accumulated; final mosaic was not generated."
        )

    print("Done. Block-level results are saved ""under OUTPUT_DIR.")


def main() -> None:
    """程序入口：纯命令行执行核心拼接算法。"""
    run_stitching()

if __name__ == "__main__":
    main()
