"""Flask server that exposes the math mesh filter as a tiny web API."""
from __future__ import annotations

import io
from typing import Dict

from flask import Flask, jsonify, render_template, request, send_file

from mesh_filter import (
    DEFAULT_NUM_POINTS,
    MAX_DIMENSION,
    MAX_NUM_POINTS,
    MIN_NUM_POINTS,
    MeshFilterError,
    render_math_mesh,
)

app = Flask(__name__, static_folder="static", template_folder="templates")

ALLOWED_EXPORTS: Dict[str, Dict[str, str]] = {
    "png": {"mime": "image/png", "pil": "PNG"},
    "jpeg": {"mime": "image/jpeg", "pil": "JPEG"},
    "jpg": {"mime": "image/jpeg", "pil": "JPEG"},
    "webp": {"mime": "image/webp", "pil": "WEBP"},
    "bmp": {"mime": "image/bmp", "pil": "BMP"},
}


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


def _coerce_int(value: str | None, default: int, min_value: int, max_value: int) -> int:
    try:
        coerced = int(value) if value is not None else default
    except (TypeError, ValueError):
        coerced = default
    return max(min_value, min(max_value, coerced))


@app.post("/api/filter")
def api_filter():  # type: ignore[override]
    file = request.files.get("image")
    if not file or file.filename == "":
        return jsonify({"error": "Please choose an image file."}), 400

    export_fmt = request.form.get("format", "png").lower()
    if export_fmt not in ALLOWED_EXPORTS:
        return jsonify({"error": "Unsupported export format."}), 400

    detail = _coerce_int(request.form.get("detail"), DEFAULT_NUM_POINTS, MIN_NUM_POINTS, MAX_NUM_POINTS)
    max_dimension = _coerce_int(request.form.get("maxDimension"), MAX_DIMENSION, 400, 1600)

    payload = file.read()
    if not payload:
        return jsonify({"error": "The uploaded file is empty."}), 400

    try:
        output_image, stats = render_math_mesh(payload, num_points=detail, max_dimension=max_dimension)
    except MeshFilterError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:  # noqa: BLE001
        return jsonify({"error": "Failed to create the math mesh image."}), 500

    buffer = io.BytesIO()
    pil_format = ALLOWED_EXPORTS[export_fmt]["pil"]
    output_image.save(buffer, format=pil_format)
    buffer.seek(0)

    response = send_file(
        buffer,
        mimetype=ALLOWED_EXPORTS[export_fmt]["mime"],
        download_name=f"math-mesh.{export_fmt if export_fmt != 'jpg' else 'jpeg'}",
    )
    for key, value in stats.to_headers():
        response.headers[key] = value
    return response


if __name__ == "__main__":
    app.run(debug=True)
