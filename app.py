"""
app.py
======
Servidor Flask – Vigilante de Inocuidad BPM.

Rutas:
  GET /              → UI web (index.html)
  GET /video_feed    → MJPEG stream anotado
  GET /api/anomalies → Últimas anomalías validadas (JSON)
  GET /api/status    → Estado del engine y modelo (JSON)
  POST /api/roi      → Actualizar polígono ROI en caliente (JSON)
"""

from __future__ import annotations

import os
import logging
import threading
from typing import Optional

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS

from vision_engine import BiosecurityVisionEngine

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("app")

# ---------------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Engine global (singleton)
# ---------------------------------------------------------------------------
def _parse_engine_source(raw: str) -> str | int:
    raw = (raw or "").strip()
    if not raw or raw.lower() == "auto":
        return "auto"
    if raw.isdigit():
        return int(raw)
    return raw


ENGINE_SOURCE: str | int = _parse_engine_source(
    os.getenv("ENGINE_SOURCE", "auto")
)  # webcam local; usar "auto", 0, 1 o "rtsp://..." para IP cam
MODEL_PATH:    str       = "best.pt"  # ruta al modelo YOLOv8 customizado

# ROI de ejemplo: rectángulo central (se puede actualizar vía /api/roi)
DEFAULT_ROI = None  # None = todo el frame es zona limpia

_engine: Optional[BiosecurityVisionEngine] = None
_engine_lock = threading.Lock()


def get_engine() -> BiosecurityVisionEngine:
    """Devuelve el engine singleton, creándolo si no existe."""
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = BiosecurityVisionEngine(
                source=ENGINE_SOURCE,
                model_path=MODEL_PATH,
                roi_polygon=DEFAULT_ROI,
                frame_skip=2,
                debounce_frames=5,
                reconnect_delay=3.0,
                jpeg_quality=85,
            )
            logger.info("Engine inicializado. Modelo: %s", _engine.model_status)
    return _engine


# ---------------------------------------------------------------------------
# MJPEG generator
# ---------------------------------------------------------------------------
def _mjpeg_generator():
    """Adapta el generador del engine al formato multipart/x-mixed-replace."""
    engine = get_engine()
    for jpeg_bytes, _ in engine.stream():
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg_bytes
            + b"\r\n"
        )


# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/anomalies")
def api_anomalies():
    """Devuelve la lista de anomalías actualmente validadas."""
    engine = get_engine()
    return jsonify({
        "anomalies": engine.last_anomalies,
        "count":     len(engine.last_anomalies),
    })


@app.route("/api/detections")
def api_detections():
    """Devuelve TODAS las detecciones del último frame procesado (raw)."""
    engine = get_engine()
    return jsonify({
        "detections": engine.last_detections,
        "count":      len(engine.last_detections),
    })


@app.route("/api/status")
def api_status():
    """Estado del sistema."""
    engine = get_engine()
    return jsonify({
        "engine_running":  engine.is_running,
        "model_status":    engine.model_status,
        "source":          str(engine.source),
        "capture_status":  engine.capture_status,
        "frame_skip":      engine.frame_skip,
        "debounce_frames": engine.debounce_frames,
        "roi_active":      engine.roi_polygon is not None,
    })


@app.route("/api/roi", methods=["POST"])
def api_set_roi():
    """
    Actualiza el polígono ROI en caliente.

    Body JSON esperado:
        {"polygon": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}
    Enviar {"polygon": null} para desactivar el ROI.
    """
    data = request.get_json(force=True, silent=True) or {}
    polygon = data.get("polygon")

    engine = get_engine()
    import numpy as np
    if polygon is None:
        engine.roi_polygon = None
        logger.info("ROI desactivado.")
    else:
        engine.roi_polygon = np.array(polygon, dtype=np.int32)
        logger.info("ROI actualizado con %d vertices.", len(polygon))

    return jsonify({"status": "ok", "roi_active": engine.roi_polygon is not None})


# ---------------------------------------------------------------------------
# Arranque
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 56)
    print("  Vigilante de Inocuidad BPM  –  Vision Engine")
    print("  http://localhost:5000")
    print("  API: /api/status  /api/anomalies  /api/roi")
    print("=" * 56)

    # Pre-calentar engine antes de recibir conexiones
    engine = get_engine()

    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        logger.info("Interrupcion de usuario. Cerrando...")
    finally:
        if _engine:
            _engine.stop()
