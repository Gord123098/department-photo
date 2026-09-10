#!/usr/bin/env python3
"""Build a registered, layered Deep Zoom viewer from one synchronized instant.

The color cameras form the panorama base. Monochrome cameras are geometrically
registered into the same coordinate system and receive chroma from the color base
while retaining their own luminance detail. Every raw and pan-sharpened layer is
written as an independent DZI pyramid.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
    from lightglue import ALIKED, LightGlue
    from lightglue.utils import rbd
except ImportError:
    torch = None

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import find_time_overlaps as overlap  # noqa: E402


OUTPUT = Path(__file__).resolve().parent
TILES = OUTPUT / "tiles"
REFERENCE_FRAME = 300.0
TEMPORAL_ALIGNMENT = PROJECT_ROOT / "temporal_overlap_results" / "verified-temporal-alignment.json"
FEATURE_SCALE = 0.5
TILE_SIZE = 256
COLOR_CAMERAS = {"642F", "8FF9"}
CAMERAS = [
    "447",
    "458",
    "642F",
    "643D_12mm_right",
    "6445",
    "644F",
    "6450_12mm_left",
    "6453",
    "645C",
    "8FF9",
]

# Each child is registered into its parent. Wide cameras are anchored directly
# to a color camera whenever possible so errors do not accumulate through a long
# chain. The close views use the strongest broad-coverage bridge available.
PARENTS = {
    "642F": "8FF9",
    "643D_12mm_right": "8FF9",
    "6450_12mm_left": "8FF9",
    "6453": "6450_12mm_left",
    "447": "6453",
    "6445": "6453",
    "645C": "6445",
    "458": "645C",
    "644F": "458",
}

DETAIL_TIERS = [
    {
        "id": "tier-12mm",
        "label": "12 mm stitched",
        "shortLabel": "12 mm",
        "cameras": ["6450_12mm_left", "643D_12mm_right"],
        "scale": 1.55,
        "minZoom": 2.2,
    },
    {
        "id": "tier-telephoto",
        "label": "Close detail (A + B stitched)",
        "shortLabel": "Close A + B",
        "cameras": ["447", "6453", "6445", "645C", "458", "644F"],
        "scale": 4.8,
        "minZoom": 3.6,
    },
    {
        "id": "tier-mono-array",
        "label": "All-monochrome array stitched",
        "shortLabel": "All Mono Array",
        "cameras": [
            "6450_12mm_left",
            "643D_12mm_right",
            "447",
            "6453",
            "6445",
            "645C",
            "458",
            "644F",
        ],
        "scale": 4.8,
        "minZoom": 2.0,
    },
]


@dataclass
class Edge:
    child: str
    parent: str
    homography: np.ndarray
    matches: int
    inliers: int
    median_error: float
    source_coverage: float


def nearest_frames() -> dict[str, overlap.Frame]:
    frames, _ = overlap.discover(PROJECT_ROOT)
    alignment = json.loads(TEMPORAL_ALIGNMENT.read_text(encoding="utf-8"))
    selected = {}
    for camera in CAMERAS:
        mapping = alignment["cameraMaps"][camera]
        scale = mapping["referenceFramePerCameraFrame"]
        intercept = mapping["referenceIntercept"]
        index = int(round((REFERENCE_FRAME - intercept) / scale))
        if not 0 <= index < len(frames[camera]):
            raise RuntimeError(f"{camera} was not recording at reference frame {REFERENCE_FRAME}")
        selected[camera] = frames[camera][index]
        residual = scale * index + intercept - REFERENCE_FRAME
        print(
            f"time {camera:20s} index={index:4d} residual={residual:+.3f} reference frames "
            f"file={frames[camera][index].path.name}", flush=True,
        )
    return selected


def feather(height: int, width: int, edge_width: int = 100) -> np.ndarray:
    y, x = np.mgrid[:height, :width]
    distance = np.minimum.reduce([x + 1, width - x, y + 1, height - y]).astype(np.float32)
    return np.clip(distance / edge_width, 0, 1)


def estimate_edge(
    child: str,
    parent: str,
    gray: dict[str, np.ndarray],
    features,
    matcher,
) -> Edge:
    with torch.inference_mode():
        match_result = matcher(
            {"image0": features[child], "image1": features[parent]}
        )
    child_features, parent_features, match_result = [
        rbd(item) for item in (features[child], features[parent], match_result)
    ]
    matches = match_result["matches"]
    if len(matches) < 8:
        raise RuntimeError(f"Not enough feature matches for {child} -> {parent}")
    source = child_features["keypoints"][matches[..., 0]].cpu().numpy().astype(np.float32)
    destination = (
        parent_features["keypoints"][matches[..., 1]].cpu().numpy().astype(np.float32)
    )
    homography_small, mask = cv2.findHomography(
        source, destination, cv2.USAC_MAGSAC, 3.0
    )
    if homography_small is None or mask is None:
        raise RuntimeError(f"Homography failed for {child} -> {parent}")
    inlier_mask = mask.ravel().astype(bool)
    prediction = cv2.perspectiveTransform(source[inlier_mask, None, :], homography_small)[
        :, 0, :
    ]
    errors = np.linalg.norm(prediction - destination[inlier_mask], axis=1)
    hull_area = (
        cv2.contourArea(cv2.convexHull(source[inlier_mask]))
        if int(inlier_mask.sum()) >= 3
        else 0.0
    )
    source_coverage = hull_area / float(gray[child].shape[0] * gray[child].shape[1])
    scale = np.diag([FEATURE_SCALE, FEATURE_SCALE, 1.0])
    homography_full = np.linalg.inv(scale) @ homography_small @ scale
    return Edge(
        child,
        parent,
        homography_full,
        len(matches),
        int(mask.sum()),
        float(np.median(errors)),
        float(source_coverage),
    )


def write_dzi(name: str, image: np.ndarray, extension: str) -> dict:
    descriptor = TILES / f"{name}.dzi"
    tile_root = TILES / f"{name}_files"
    if tile_root.exists():
        shutil.rmtree(tile_root)
    tile_root.mkdir(parents=True)
    height, width = image.shape[:2]
    max_level = math.ceil(math.log2(max(width, height)))
    current = image
    for level in range(max_level, -1, -1):
        level_path = tile_root / str(level)
        level_path.mkdir()
        rows = math.ceil(current.shape[0] / TILE_SIZE)
        columns = math.ceil(current.shape[1] / TILE_SIZE)
        for row in range(rows):
            for column in range(columns):
                tile = current[
                    row * TILE_SIZE : min((row + 1) * TILE_SIZE, current.shape[0]),
                    column * TILE_SIZE : min((column + 1) * TILE_SIZE, current.shape[1]),
                ]
                path = level_path / f"{column}_{row}.{extension}"
                if extension == "jpg":
                    cv2.imwrite(str(path), tile, [cv2.IMWRITE_JPEG_QUALITY, 90])
                else:
                    cv2.imwrite(str(path), tile, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if level:
            current = cv2.resize(
                current,
                (math.ceil(current.shape[1] / 2), math.ceil(current.shape[0] / 2)),
                interpolation=cv2.INTER_AREA,
            )
    descriptor.write_text(
        f'<Image TileSize="{TILE_SIZE}" Overlap="0" Format="{extension}" '
        'xmlns="http://schemas.microsoft.com/deepzoom/2008">\n'
        f'  <Size Width="{width}" Height="{height}"/>\n'
        '</Image>\n',
        encoding="utf-8",
    )
    return {"tileSource": f"tiles/{name}.dzi", "pixelWidth": width, "pixelHeight": height}


def confidence(inliers: int) -> str:
    if inliers >= 100:
        return "high"
    if inliers >= 40:
        return "medium"
    return "low"


CAMERA_GAINS = {
    "447": 0.99,
    "6453": 1.15,
    "6445": 1.20,
    "645C": 1.37,
    "458": 2.83,
    "644F": 2.83,
    "6450_12mm_left": 1.08,
    "643D_12mm_right": 1.01,
}


def apply_tone_curve(x_u8: np.ndarray, gain: float, knee: float = 180.0) -> np.ndarray:
    """Apply exposure gain with linear midtones and smooth exponential shoulder above knee."""
    x_f = x_u8.astype(np.float32)
    scaled = x_f * gain
    res = np.where(
        scaled <= knee,
        scaled,
        knee + (255.0 - knee) * (1.0 - np.exp(-(scaled - knee) / max(1.0, 255.0 - knee))),
    )
    return np.clip(res, 0, 255).astype(np.uint8)


def border_distance_mask(shape: tuple[int, int], edge_fade: int = 250) -> np.ndarray:
    """Distance-from-border weighting for smooth optimal-seam blending."""
    height, width = shape[:2]
    y, x = np.mgrid[:height, :width]
    distance = np.minimum.reduce([x, width - 1 - x, y, height - 1 - y]).astype(np.float32)
    return np.clip(distance / float(max(1, edge_fade)), 0.0, 1.0)


def source_feather_mask(shape: tuple[int, int], edge_width: int = 40) -> np.ndarray:
    """Feather only the outer rectangular frame edges to blend adjacent cameras smoothly."""
    height, width = shape[:2]
    feather_x = np.minimum(np.arange(width), width - 1 - np.arange(width)).astype(np.float32)
    feather_y = np.minimum(np.arange(height), height - 1 - np.arange(height)).astype(np.float32)
    mask_x = np.clip(feather_x / float(max(1, edge_width)), 0.0, 1.0)
    mask_y = np.clip(feather_y / float(max(1, edge_width)), 0.0, 1.0)
    return np.outer(mask_y, mask_x).astype(np.float32)


def soft_source_mask(image: np.ndarray, edge_width: int = 40) -> np.ndarray:
    return source_feather_mask(image.shape[:2], edge_width)


def align_color_to_mono(
    mono_u8: np.ndarray, color_u8: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Refine parallax with DIS optical flow and return smoothly aligned color + confidence."""
    height, width = mono_u8.shape[:2]
    scale = min(1.0, 1200.0 / max(height, width))
    small_size = (max(32, round(width * scale)), max(32, round(height * scale)))
    mono_small = cv2.resize(mono_u8, small_size, interpolation=cv2.INTER_AREA)
    color_gray = cv2.cvtColor(color_u8, cv2.COLOR_BGR2GRAY)
    color_small = cv2.resize(color_gray, small_size, interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    mono_features = clahe.apply(mono_small)
    color_features = clahe.apply(color_small)
    flow_solver = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow_solver.setFinestScale(0)
    forward = flow_solver.calc(mono_features, color_features, None)
    backward = flow_solver.calc(color_features, mono_features, None)

    small_y, small_x = np.mgrid[: small_size[1], : small_size[0]].astype(np.float32)
    backward_at_target = cv2.remap(
        backward,
        small_x + forward[:, :, 0],
        small_y + forward[:, :, 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    cycle_error = np.linalg.norm(forward + backward_at_target, axis=2)
    confidence_small = np.exp(-np.square(cycle_error / 2.5)).astype(np.float32)
    inside = (
        (small_x + forward[:, :, 0] >= 1)
        & (small_x + forward[:, :, 0] < small_size[0] - 2)
        & (small_y + forward[:, :, 1] >= 1)
        & (small_y + forward[:, :, 1] < small_size[1] - 2)
    )
    confidence_small *= inside

    # Smooth damping of large displacements to prevent tearing/shearing
    disp = np.linalg.norm(forward, axis=2)
    max_disp = 40.0
    damp = np.where(disp > max_disp, max_disp / np.maximum(disp, 1e-4), 1.0)
    forward[:, :, 0] *= damp
    forward[:, :, 1] *= damp

    flow = cv2.resize(forward, (width, height), interpolation=cv2.INTER_LINEAR)
    flow[:, :, 0] /= scale
    flow[:, :, 1] /= scale

    confidence = cv2.resize(confidence_small, (width, height), interpolation=cv2.INTER_LINEAR)
    confidence = cv2.GaussianBlur(confidence, (0, 0), 1.2)
    y, x = np.mgrid[:height, :width].astype(np.float32)
    aligned = cv2.remap(
        color_u8,
        x + flow[:, :, 0],
        y + flow[:, :, 1],
        cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT,
    )
    return aligned, np.clip(confidence, 0, 1)


def dense_flow_ycrcb_fusion(
    mono_u8: np.ndarray, aligned_color_u8: np.ndarray, confidence: np.ndarray
) -> np.ndarray:
    """Pan-sharpen by exact mono Y substitution with bilateral-smoothed, saturation-boosted chroma."""
    ycrcb = cv2.cvtColor(aligned_color_u8, cv2.COLOR_BGR2YCrCb)
    _, chroma_cr, chroma_cb = cv2.split(ycrcb)
    # Edge-preserving bilateral filter on chroma channels (r=7, sigma=25)
    cr_smooth = cv2.bilateralFilter(chroma_cr, 7, 25, 25)
    cb_smooth = cv2.bilateralFilter(chroma_cb, 7, 25, 25)
    
    # 1.10x gentle saturation expansion to maintain vibrant color contrast against high-res mono luminance
    cr_dev = (cr_smooth.astype(np.float32) - 128.0) * 1.10
    cb_dev = (cb_smooth.astype(np.float32) - 128.0) * 1.10
    cr_final = np.clip(128.0 + cr_dev, 0, 255).astype(np.uint8)
    cb_final = np.clip(128.0 + cb_dev, 0, 255).astype(np.uint8)
    return cv2.cvtColor(
        cv2.merge([mono_u8, cr_final, cb_final]),
        cv2.COLOR_YCrCb2BGR,
    )


def match_mono_tone(
    mono: np.ndarray, color: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    """Gently balance mono exposure to local color without clipping highlights."""
    color_luma = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    if int(valid.sum()) < 1000:
        return mono
    mono_med = float(np.median(mono[valid]))
    color_med = float(np.median(color_luma[valid]))
    if mono_med < 5:
        return mono
    gain = np.clip(color_med / mono_med, 0.85, 1.25)
    return np.clip(mono.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def compose_tier(
    tier: dict,
    products: dict[str, dict],
    canvas_width: int,
    canvas_height: int,
    channel: str = "color",
) -> tuple[np.ndarray, tuple[int, int, int, int], float]:
    """Blend already-warped camera products into one focal-length layer (color or mono)."""
    members = [products[camera] for camera in tier["cameras"] if camera in products]
    x0 = max(0, min(item["bbox"][0] for item in members))
    y0 = max(0, min(item["bbox"][1] for item in members))
    x1 = min(canvas_width, max(item["bbox"][2] for item in members))
    y1 = min(canvas_height, max(item["bbox"][3] for item in members))
    requested_scale = float(tier["scale"])
    base_pixels = max(1, (x1 - x0) * (y1 - y0))
    max_tier_pixels = 35_000_000
    scale = min(requested_scale, math.sqrt(max_tier_pixels / base_pixels))
    width = max(1, math.ceil((x1 - x0) * scale))
    height = max(1, math.ceil((y1 - y0) * scale))

    accum = np.zeros((height, width, 3), np.float32)
    weight_sum = np.zeros((height, width), np.float32)
    image_key = "pan" if channel == "color" else "mono"

    for item in members:
        ix0, iy0, ix1, iy1 = item["bbox"]
        dx0 = round((ix0 - x0) * scale)
        dy0 = round((iy0 - y0) * scale)
        dx1 = round((ix1 - x0) * scale)
        dy1 = round((iy1 - y0) * scale)
        if dx1 <= dx0 or dy1 <= dy0:
            continue
        tw = dx1 - dx0
        th = dy1 - dy0
        src_img = cv2.resize(item[image_key][:, :, :3], (tw, th), interpolation=cv2.INTER_AREA)
        weight = cv2.resize(item["weight"], (tw, th), interpolation=cv2.INTER_LINEAR)
        res_boost = float(item.get("resolutionScale", 1.0)) / 1.55
        w_p = np.power(weight * res_boost, 2.0)
        accum[dy0:dy1, dx0:dx1] += src_img.astype(np.float32) * w_p[:, :, None]
        weight_sum[dy0:dy1, dx0:dx1] += w_p

    valid = weight_sum > 1e-4
    output_rgb = np.zeros((height, width, 3), np.uint8)
    output_rgb[valid] = np.clip(accum[valid] / weight_sum[valid, None], 0, 255).astype(np.uint8)
    alpha = np.zeros((height, width), np.uint8)
    alpha[valid] = 255
    edge_feather = np.clip(weight_sum / 0.25, 0.0, 1.0)
    alpha = (alpha.astype(np.float32) * edge_feather).astype(np.uint8)
    result = np.dstack([output_rgb, alpha])
    return result, (x0, y0, x1, y1), scale


def main() -> None:
    if torch is None:
        raise RuntimeError(
            "ALIKED + LightGlue are required. Install the GPU requirements before building."
        )
    if TILES.exists():
        shutil.rmtree(TILES)
    TILES.mkdir(parents=True, exist_ok=True)
    selected = nearest_frames()
    color_images = {
        camera: cv2.imread(str(frame.path), cv2.IMREAD_COLOR)
        for camera, frame in selected.items()
    }
    gray_small = {
        camera: cv2.resize(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            None,
            fx=FEATURE_SCALE,
            fy=FEATURE_SCALE,
            interpolation=cv2.INTER_AREA,
        )
        for camera, image in color_images.items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = ALIKED(max_num_keypoints=8192).eval().to(device)
    matcher = LightGlue(features="aliked").eval().to(device)
    features = {}
    with torch.inference_mode():
        for camera, image in gray_small.items():
            tensor = torch.from_numpy(image.astype(np.float32) / 255.0)[None, None].to(device)
            features[camera] = extractor.extract(tensor)
    edges = {
        child: estimate_edge(child, parent, gray_small, features, matcher)
        for child, parent in PARENTS.items()
    }
    for child, edge in edges.items():
        print(
            f"{child:20s} -> {edge.parent:20s} "
            f"{edge.inliers:4d}/{edge.matches:4d} inliers · "
            f"{edge.median_error:.2f}px median · {edge.source_coverage:.1%} coverage",
            flush=True,
        )

    global_h = {"8FF9": np.eye(3)}

    def transform(camera: str) -> np.ndarray:
        if camera not in global_h:
            edge = edges[camera]
            global_h[camera] = transform(edge.parent) @ edge.homography
        return global_h[camera]

    def corners_for(camera: str) -> np.ndarray:
        height, width = color_images[camera].shape[:2]
        return np.float32([[[0, 0], [width, 0], [width, height], [0, height]]])

    unshifted = {
        camera: cv2.perspectiveTransform(corners_for(camera), transform(camera))[0]
        for camera in CAMERAS
    }
    # The wide color cameras define the world. Tight cameras are clipped to this
    # stable panorama instead of being allowed to stretch the canvas with a weak
    # or badly conditioned homography.
    wide_points = unshifted["8FF9"]
    padding = 72
    minimum = np.floor(wide_points.min(axis=0)) - padding
    maximum = np.ceil(wide_points.max(axis=0)) + padding
    translation = np.array(
        [[1, 0, -minimum[0]], [0, 1, -minimum[1]], [0, 0, 1]], dtype=np.float64
    )
    canvas_width = int(maximum[0] - minimum[0])
    canvas_height = int(maximum[1] - minimum[1])
    canvas_h = {camera: translation @ transform(camera) for camera in CAMERAS}
    projected = {
        camera: cv2.perspectiveTransform(corners_for(camera), canvas_h[camera])[0]
        for camera in CAMERAS
    }

    # Use the cleanest wide color camera as the base. The second wide camera is
    # still independently available, but blending the two exposures creates a
    # conspicuous rectangular vignette seam in the sky.
    accumulator = np.zeros((canvas_height, canvas_width, 3), np.float32)
    weights = np.zeros((canvas_height, canvas_width), np.float32)
    for camera in ("8FF9",):
        image = color_images[camera].astype(np.float32)
        warped_image = cv2.warpPerspective(image, canvas_h[camera], (canvas_width, canvas_height))
        source_weight = soft_source_mask(color_images[camera])
        warped_weight = cv2.warpPerspective(
            source_weight, canvas_h[camera], (canvas_width, canvas_height)
        )
        overlap_mask = (warped_weight > 0.25) & (weights > 0.25)
        if overlap_mask.sum() > 10000:
            current = accumulator[overlap_mask] / weights[overlap_mask, None]
            incoming = warped_image[overlap_mask]
            current_median = np.median(current, axis=0)
            incoming_median = np.maximum(np.median(incoming, axis=0), 1)
            gain = np.clip(current_median / incoming_median, 0.55, 1.8)
            warped_image *= gain
        accumulator += warped_image * warped_weight[:, :, None]
        weights += warped_weight
    base_color = np.clip(accumulator / np.maximum(weights[:, :, None], 1e-5), 0, 255).astype(
        np.uint8
    )
    base_color[weights == 0] = (10, 12, 15)
    base_mask = (weights > 0.02).astype(np.uint8) * 255
    cv2.imwrite(str(OUTPUT / "stitched-color-preview.jpg"), base_color, [cv2.IMWRITE_JPEG_QUALITY, 92])

    layers = []
    base_metadata = write_dzi("composite-color-base", base_color, "jpg")
    layers.append(
        {
            "id": "composite-color-base",
            "label": "Wide color panorama",
            "shortLabel": "Wide",
            "kind": "base",
            "channel": "color",
            "camera": "8FF9 color reference",
            "confidence": "high",
            "x": 0,
            "y": 0,
            "width": 1,
            "defaultOpacity": 1,
            **base_metadata,
        }
    )

    # Compute radiometric color match gain for 642F relative to 8FF9
    H_642f_to_8ff9 = transform("642F")
    warped_642f_in_8ff9 = cv2.warpPerspective(
        color_images["642F"],
        H_642f_to_8ff9,
        (color_images["8FF9"].shape[1], color_images["8FF9"].shape[0]),
    )
    mask_642f_in_8ff9 = cv2.warpPerspective(
        np.full(color_images["642F"].shape[:2], 255, dtype=np.uint8),
        H_642f_to_8ff9,
        (color_images["8FF9"].shape[1], color_images["8FF9"].shape[0]),
    )
    overlap_642f = (
        (mask_642f_in_8ff9 > 200)
        & (warped_642f_in_8ff9.sum(axis=2) > 30)
        & (color_images["8FF9"].sum(axis=2) > 30)
    )
    gain_642f = np.median(color_images["8FF9"][overlap_642f], axis=0) / np.maximum(
        np.median(warped_642f_in_8ff9[overlap_642f], axis=0), 1e-3
    )
    gain_642f = np.clip(gain_642f, 0.7, 1.5).astype(np.float32)
    print(f"Radiometric color match gain 642F -> 8FF9 (BGR): {gain_642f}", flush=True)
    color_642f_matched = np.clip(
        color_images["642F"].astype(np.float32) * gain_642f, 0, 255
    ).astype(np.uint8)

    raw_layers = []
    pan_layers = []
    products: dict[str, dict] = {}
    for camera in CAMERAS:
        source_height, source_width = color_images[camera].shape[:2]
        corners = projected[camera]
        lower = np.floor(corners.min(axis=0)).astype(int)
        upper = np.ceil(corners.max(axis=0)).astype(int)
        x0, y0 = np.maximum(lower, 0)
        x1 = min(upper[0], canvas_width)
        y1 = min(upper[1], canvas_height)
        bbox_width = x1 - x0
        bbox_height = y1 - y0
        if bbox_width < 16 or bbox_height < 16:
            raise RuntimeError(f"{camera} projects outside the stable wide panorama")
        polygon_area = max(abs(cv2.contourArea(corners.astype(np.float32))), 1)
        native_scale = float(
            np.clip(math.sqrt((source_width * source_height) / polygon_area), 0.55, 5.0)
        )
        local = np.array(
            [[native_scale, 0, -x0 * native_scale], [0, native_scale, -y0 * native_scale], [0, 0, 1]],
            dtype=np.float64,
        ) @ canvas_h[camera]
        output_width = max(1, math.ceil(bbox_width * native_scale))
        output_height = max(1, math.ceil(bbox_height * native_scale))
        image = color_images[camera]
        warped = cv2.warpPerspective(
            image, local, (output_width, output_height), flags=cv2.INTER_LANCZOS4
        )
        source_alpha = (source_feather_mask(image.shape[:2], edge_width=36) * 255).astype(np.uint8)
        alpha = cv2.warpPerspective(
            source_alpha, local, (output_width, output_height), flags=cv2.INTER_LINEAR
        )

        edge = edges.get(camera)
        layer_confidence = "high" if edge is None else confidence(edge.inliers)
        is_color = camera in COLOR_CAMERAS
        common = {
            "camera": camera,
            "sourceFile": str(selected[camera].path.relative_to(PROJECT_ROOT)),
            "confidence": layer_confidence,
            "registrationParent": edge.parent if edge else None,
            "registrationInliers": edge.inliers if edge else None,
            "registrationError": round(edge.median_error, 3) if edge else 0,
            "registrationCoverage": round(edge.source_coverage, 4) if edge else 1,
            "x": x0 / canvas_width,
            "y": y0 / canvas_width,
            "width": bbox_width / canvas_width,
            "resolutionScale": round(native_scale, 3),
            "footprint": [
                [round(float(x) / canvas_width, 7), round(float(y) / canvas_width, 7)]
                for x, y in corners
            ],
        }

        if is_color:
            # Color cameras (8FF9, 642F) are strictly kept native color
            raw_bgra = np.dstack([warped, alpha])
            raw_id = f"raw-{camera.lower()}"
            raw_metadata = write_dzi(raw_id, raw_bgra, "png")
            raw_layers.append(
                {
                    "id": raw_id,
                    "label": f"{camera} (Color)",
                    "shortLabel": f"{camera} Color",
                    "kind": "camera-color",
                    "channel": "color",
                    "fused": False,
                    "defaultOpacity": 1,
                    **common,
                    **raw_metadata,
                }
            )
        else:
            # Monochrome cameras: prepare tone-calibrated monochrome image
            mono = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            gain = CAMERA_GAINS.get(camera, 1.0)
            mono_corr = apply_tone_curve(mono, gain)
            mono_3ch = cv2.cvtColor(mono_corr, cv2.COLOR_GRAY2BGR)

            d_mask_src = border_distance_mask(image.shape[:2], edge_fade=250)
            warped_weight = cv2.warpPerspective(
                d_mask_src, local, (output_width, output_height), flags=cv2.INTER_LINEAR
            )
            mono_bgra = np.dstack([mono_3ch, alpha])

            # 1. Native Mono layer
            raw_id = f"raw-{camera.lower()}"
            raw_metadata = write_dzi(raw_id, mono_bgra, "png")
            raw_layers.append(
                {
                    "id": raw_id,
                    "label": f"{camera} (Native Mono)",
                    "shortLabel": f"{camera} Mono",
                    "kind": "camera-mono",
                    "channel": "mono",
                    "fused": False,
                    "defaultOpacity": 1,
                    **common,
                    **raw_metadata,
                }
            )

            # 2. Color Fused (pan-sharpened) layer - warp from highest resolution color source
            H_8ff9_to_local = local @ np.linalg.inv(transform(camera))
            H_642f_to_local = H_8ff9_to_local @ transform("642F")

            # Check coverage of high-res 642F camera (1.664x optical resolution scale)
            h_642f, w_642f = color_images["642F"].shape[:2]
            source_mask_642f = np.full((h_642f, w_642f), 255, dtype=np.uint8)
            mask_642f = cv2.warpPerspective(
                source_mask_642f,
                H_642f_to_local,
                (output_width, output_height),
                flags=cv2.INTER_NEAREST,
            )
            coverage_642f = float((mask_642f > 0).mean())

            h_8ff9, w_8ff9 = color_images["8FF9"].shape[:2]
            source_mask_8ff9 = np.full((h_8ff9, w_8ff9), 255, dtype=np.uint8)
            mask_8ff9 = cv2.warpPerspective(
                source_mask_8ff9,
                H_8ff9_to_local,
                (output_width, output_height),
                flags=cv2.INTER_NEAREST,
            )

            if coverage_642f > 0.98:
                # 100% inside high-resolution 642F camera (1.664x optical resolution scale)
                color_direct = cv2.warpPerspective(
                    color_642f_matched,
                    H_642f_to_local,
                    (output_width, output_height),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_REFLECT,
                )
                color_mask = mask_642f
                color_source_desc = "642F high-resolution (1.66x)"
            elif coverage_642f < 0.02:
                # Outside 642F FOV, use 8FF9 wide color reference
                color_direct = cv2.warpPerspective(
                    color_images["8FF9"],
                    H_8ff9_to_local,
                    (output_width, output_height),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_REFLECT,
                )
                color_mask = mask_8ff9
                color_source_desc = "8FF9 wide reference (1.00x)"
            else:
                # Hybrid overlap (e.g. 6450): blend high-res 642F with 8FF9 seamlessly using distance feathering
                color_8ff9 = cv2.warpPerspective(
                    color_images["8FF9"],
                    H_8ff9_to_local,
                    (output_width, output_height),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_REFLECT,
                )
                color_642f = cv2.warpPerspective(
                    color_642f_matched,
                    H_642f_to_local,
                    (output_width, output_height),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_REFLECT,
                )
                eroded_mask_642f = cv2.erode(mask_642f, np.ones((15, 15), np.uint8))
                dist_642f = cv2.distanceTransform((eroded_mask_642f > 0).astype(np.uint8), cv2.DIST_L2, 5)
                weight_642f = np.clip(dist_642f / 60.0, 0.0, 1.0)[:, :, None]
                color_direct = (
                    color_642f.astype(np.float32) * weight_642f
                    + color_8ff9.astype(np.float32) * (1.0 - weight_642f)
                ).astype(np.uint8)
                color_mask = np.maximum(mask_8ff9, mask_642f)
                color_source_desc = f"642F (1.66x, {coverage_642f:.0%}) + 8FF9 hybrid"

            valid = (warped_weight > 0.05) & (color_mask > 0)
            if valid.sum() > 100:
                aligned_color, flow_confidence = align_color_to_mono(mono_corr, color_direct)
                sharpened = dense_flow_ycrcb_fusion(
                    mono_corr, aligned_color, flow_confidence
                )
                sharpened[color_mask == 0] = mono_3ch[color_mask == 0]
            else:
                sharpened = mono_3ch.copy()

            pan_bgra = np.dstack([sharpened, alpha])
            pan_id = f"pan-{camera.lower()}"
            pan_metadata = write_dzi(pan_id, pan_bgra, "png")
            pan_layers.append(
                {
                    "id": pan_id,
                    "label": f"{camera} (Color Fused)",
                    "shortLabel": f"{camera} Fused",
                    "kind": "camera-fused",
                    "channel": "color",
                    "fused": True,
                    "fusionMethod": f"bilateral-smoothed YCrCb pan-sharpening from {color_source_desc}",
                    "colorSource": color_source_desc,
                    "defaultOpacity": 1,
                    **common,
                    **pan_metadata,
                }
            )

            products[camera] = {
                "mono": mono_bgra,
                "pan": pan_bgra,
                "weight": warped_weight,
                "bbox": (x0, y0, x1, y1),
                "resolutionScale": native_scale,
            }

    tier_layers = []
    for tier_index, tier in enumerate(DETAIL_TIERS, start=1):
        # 1. Color Fused tier
        tier_color_image, (x0, y0, x1, y1), tier_scale = compose_tier(
            tier, products, canvas_width, canvas_height, channel="color"
        )
        print(
            f"{tier['label']} (Color Fused): {tier_color_image.shape[1]} x {tier_color_image.shape[0]} "
            f"({tier_scale:.2f}x)",
            flush=True,
        )
        tile_color_metadata = write_dzi(tier["id"], tier_color_image, "png")
        tier_layers.append(
            {
                "id": tier["id"],
                "label": f"{tier['label']} (Color Fused)",
                "shortLabel": f"{tier['shortLabel']} Fused",
                "kind": "detail-tier",
                "channel": "color",
                "fused": True,
                "camera": " + ".join(tier["cameras"]),
                "cameras": tier["cameras"],
                "confidence": min(
                    (confidence(edges[camera].inliers) for camera in tier["cameras"]),
                    key={"low": 0, "medium": 1, "high": 2}.get,
                ),
                "tierIndex": tier_index,
                "minZoom": tier["minZoom"],
                "resolutionScale": round(tier_scale, 3),
                "defaultOpacity": 1,
                "x": x0 / canvas_width,
                "y": y0 / canvas_width,
                "width": (x1 - x0) / canvas_width,
                "footprints": [
                    [
                        [round(float(x) / canvas_width, 7), round(float(y) / canvas_width, 7)]
                        for x, y in projected[camera]
                    ]
                    for camera in tier["cameras"]
                ],
                **tile_color_metadata,
            }
        )

        # 2. Native Mono tier
        mono_id = f"{tier['id']}-mono"
        tier_mono_image, _, _ = compose_tier(
            tier, products, canvas_width, canvas_height, channel="mono"
        )
        print(
            f"{tier['label']} (Native Mono): {tier_mono_image.shape[1]} x {tier_mono_image.shape[0]} "
            f"({tier_scale:.2f}x)",
            flush=True,
        )
        tile_mono_metadata = write_dzi(mono_id, tier_mono_image, "png")
        tier_layers.append(
            {
                "id": mono_id,
                "label": f"{tier['label']} (Native Mono)",
                "shortLabel": f"{tier['shortLabel']} Mono",
                "kind": "detail-tier-mono",
                "channel": "mono",
                "fused": False,
                "camera": " + ".join(tier["cameras"]),
                "cameras": tier["cameras"],
                "confidence": min(
                    (confidence(edges[camera].inliers) for camera in tier["cameras"]),
                    key={"low": 0, "medium": 1, "high": 2}.get,
                ),
                "tierIndex": tier_index,
                "minZoom": tier["minZoom"],
                "resolutionScale": round(tier_scale, 3),
                "defaultOpacity": 1,
                "x": x0 / canvas_width,
                "y": y0 / canvas_width,
                "width": (x1 - x0) / canvas_width,
                "footprints": [
                    [
                        [round(float(x) / canvas_width, 7), round(float(y) / canvas_width, 7)]
                        for x, y in projected[camera]
                    ]
                    for camera in tier["cameras"]
                ],
                **tile_mono_metadata,
            }
        )

    # Order layers logically: base, stitched tiers, fused cameras, mono cameras, color cameras
    layers.extend(tier_layers)
    layers.extend(pan_layers)
    layers.extend(raw_layers)
    metadata = {
        "title": "Gigapixel Multi-Scale Camera Array",
        "visualReferenceTime": f"8FF9 reference frame {REFERENCE_FRAME:.0f}",
        "temporalAlignmentMethod": "all-frame visual motion search + affine clock fit",
        "temporalAlignmentManifest": "../temporal_overlap_results/verified-temporal-alignment.json",
        "canvasPixelWidth": canvas_width,
        "canvasPixelHeight": canvas_height,
        "registrationMethod": "ALIKED + LightGlue + USAC_MAGSAC",
        "excludedCameras": ["6434"],
        "inactiveAtReference": ["645C_2"],
        "colorCameras": ["8FF9", "642F"],
        "monoCameras": [
            "6450_12mm_left",
            "643D_12mm_right",
            "447",
            "6453",
            "6445",
            "645C",
            "458",
            "644F",
        ],
        "detailTiers": [
            {
                "id": tier["id"],
                "monoId": f"{tier['id']}-mono",
                "label": tier["label"],
                "shortLabel": tier["shortLabel"],
                "minZoom": tier["minZoom"],
                "cameras": tier["cameras"],
            }
            for tier in DETAIL_TIERS
        ],
        "layers": layers,
    }
    (OUTPUT / "layers.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (OUTPUT / "registration.json").write_text(
        json.dumps(
            {
                camera: {
                    "parent": edge.parent,
                    "matches": edge.matches,
                    "inliers": edge.inliers,
                    "confidence": confidence(edge.inliers),
                    "medianReprojectionError": edge.median_error,
                    "sourceCoverage": edge.source_coverage,
                    "homography": edge.homography.tolist(),
                }
                for camera, edge in edges.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Built {len(layers)} Deep Zoom layers")
    print(f"Panorama canvas: {canvas_width} x {canvas_height}")
    print(f"Output: {OUTPUT}")
    print("Generating QA inspection composites...")
    for t_info in tier_layers:
        t_id = t_info["id"]
        t_w, t_h = t_info["pixelWidth"], t_info["pixelHeight"]
        max_lev = math.ceil(math.log2(max(t_w, t_h)))
        t_dir = TILES / f"{t_id}_files" / str(max_lev)
        if t_dir.exists():
            canvas_qa = np.zeros((t_h, t_w, 3), np.uint8)
            for p in t_dir.glob("*.png"):
                c, r = map(int, p.stem.split("_"))
                im = cv2.imread(str(p), -1)
                if im is not None:
                    h, w = im.shape[:2]
                    rgb = im[:, :, :3]
                    canvas_qa[r * 256 : r * 256 + h, c * 256 : c * 256 + w] = rgb
            s = min(1.0, 1600.0 / t_w)
            qa_prev = cv2.resize(canvas_qa, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(OUTPUT / f"qa-{t_id}.jpg"), qa_prev, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"Saved qa-{t_id}.jpg")


if __name__ == "__main__":
    main()
