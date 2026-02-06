"""Math mesh image filter shared by the API and CLI layers."""
from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import Delaunay

DEFAULT_NUM_POINTS = 2500
MIN_NUM_POINTS = 500
MAX_NUM_POINTS = 5000
MAX_DIMENSION = 1000


class MeshFilterError(RuntimeError):
    """Raised when the mesh filter cannot produce an output image."""


@dataclass
class MeshStats:
    width: int
    height: int
    vertex_count: int
    triangle_count: int
    detail: int
    processing_seconds: float
    scale_factor: float

    def to_headers(self) -> Tuple[Tuple[str, str], ...]:
        return (
            ("X-Mesh-Width", str(self.width)),
            ("X-Mesh-Height", str(self.height)),
            ("X-Mesh-Vertices", str(self.vertex_count)),
            ("X-Mesh-Triangles", str(self.triangle_count)),
            ("X-Mesh-Detail", str(self.detail)),
            ("X-Mesh-Seconds", f"{self.processing_seconds:.2f}"),
            ("X-Mesh-Scale", f"{self.scale_factor:.3f}"),
        )


def _load_image(data: bytes) -> Image.Image:
    if not data:
        raise MeshFilterError("No image data supplied.")
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise MeshFilterError("Unable to read the uploaded image.") from exc


def _resize_for_processing(image: Image.Image, max_dimension: int) -> Tuple[Image.Image, float]:
    w, h = image.size
    max_dim = max(w, h)
    if max_dim <= max_dimension:
        return image.copy(), 1.0
    scale = max_dimension / float(max_dim)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS), scale


def generate_mesh_points(img_gray: np.ndarray, num_points: int) -> np.ndarray:
    corners = cv2.goodFeaturesToTrack(img_gray, num_points, 0.01, 10)
    if corners is not None:
        corners = np.uint16(corners).reshape(-1, 2)
    else:
        corners = np.zeros((0, 2), dtype=np.uint16)

    remaining = max(0, num_points - len(corners))
    if remaining > 0:
        h, w = img_gray.shape
        rand_pts = np.random.randint(0, [w, h], size=(remaining, 2), dtype=np.uint16)
        points = np.vstack((corners, rand_pts))
    else:
        points = corners[:num_points]

    bounds = np.array([[0, 0], [img_gray.shape[1] - 1, 0], [0, img_gray.shape[0] - 1], [img_gray.shape[1] - 1, img_gray.shape[0] - 1]], dtype=np.uint16)
    points = np.vstack((points, bounds))
    return points


def render_math_mesh(image_bytes: bytes, *, num_points: int = DEFAULT_NUM_POINTS, max_dimension: int = MAX_DIMENSION) -> Tuple[Image.Image, MeshStats]:
    detail = int(np.clip(num_points, MIN_NUM_POINTS, MAX_NUM_POINTS))
    start = time.perf_counter()

    source_img = _load_image(image_bytes)
    processed_img, scale = _resize_for_processing(source_img, max_dimension)
    img_np = np.array(processed_img)
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    points = generate_mesh_points(img_gray, detail)
    if len(points) < 3:
        raise MeshFilterError("Not enough vertices were generated for triangulation.")

    tri = Delaunay(points)
    simplices = tri.simplices

    triangle_colors = []
    width, height = processed_img.size
    for indices in simplices:
        pts = points[indices]
        cx = int(np.clip(np.mean(pts[:, 0]), 0, width - 1))
        cy = int(np.clip(np.mean(pts[:, 1]), 0, height - 1))
        triangle_colors.append(tuple(int(v) for v in img_np[cy, cx]))

    canvas = Image.new("RGB", processed_img.size, (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for color, indices in zip(triangle_colors, simplices):
        poly = [tuple(int(v) for v in points[i]) for i in indices]
        draw.polygon(poly, fill=color, outline=color)

    if scale < 1.0:
        canvas = canvas.resize(source_img.size, Image.Resampling.LANCZOS)

    elapsed = time.perf_counter() - start
    stats = MeshStats(
        width=canvas.width,
        height=canvas.height,
        vertex_count=len(points),
        triangle_count=len(simplices),
        detail=detail,
        processing_seconds=elapsed,
        scale_factor=scale,
    )
    return canvas, stats
