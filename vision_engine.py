"""
vision_engine.py
================
Motor de Vision Artificial – Vigilante de Inocuidad BPM.

Modos (seleccion automatica en orden de prioridad):
  1. custom  – best.pt con taxonomia BPM propia.
  2. ppe     – keremberke/yolov8n-hard-hat-detection (HuggingFace).
               Detecta: Persona, Cubrebocas, Sin Cubrebocas,
               Casco, Sin Casco, Chaleco, Sin Chaleco.
  3. coco    – YOLOv8n preentrenado (80 clases genericas).
  4. demo    – Sin modelo; solo stream con overlay informativo.

Python : 3.10+
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("vision_engine")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    ))
    logger.addHandler(_h)

# ---------------------------------------------------------------------------
# Taxonomia modelo CUSTOM (best.pt)
# ---------------------------------------------------------------------------
CUSTOM_NAMES: dict[int, str] = {
    0: "persona",
    1: "cubrebocas_ok",
    2: "sin_cubrebocas",
    3: "cofia_ok",
    4: "sin_cofia",
    5: "basura_suelo",
    6: "derrame_suciedad",
}
CUSTOM_PERSON_ID    = 0
CUSTOM_EPP_VIOL     = frozenset({2, 4})
CUSTOM_EPP_OK       = frozenset({1, 3})
CUSTOM_HYGIENE      = frozenset({5, 6})
CONF_EPP            = 0.80
CONF_HYGIENE        = 0.65
CONF_PERSON         = 0.65

# ---------------------------------------------------------------------------
# Taxonomia modelo PPE (keremberke/yolov8m-protective-equipment-detection)
# Clases: {0:'glove', 1:'goggles', 2:'helmet', 3:'mask', 4:'no_glove', 
#          5:'no_goggles', 6:'no_helmet', 7:'no_mask', 8:'no_shoes', 9:'shoes'}
# ---------------------------------------------------------------------------
PPE_PERSON_ID   = -1 # Modelo no detecta personas
PPE_VIOL_IDS    = frozenset({4, 5, 6, 7})   # no_glove, no_goggles, no_helmet, no_mask
PPE_OK_IDS      = frozenset({0, 1, 2, 3})   # glove, goggles, helmet, mask
PPE_KEEP_IDS    = frozenset({0, 1, 2, 3, 4, 5, 6, 7})
PPE_CONF_ITEM   = 0.40

# Mensajes en espanol para la UI
PPE_VIOL_ES: dict[str, str] = {
    "no_helmet":  "Sin casco/cofia",
    "no_mask":    "Sin cubrebocas",
    "no_goggles": "Sin gafas de seguridad",
    "no_glove":   "Sin guantes",
}
PPE_OK_ES: dict[str, str] = {
    "helmet":  "Casco/Cofia OK",
    "mask":    "Cubrebocas OK",
    "goggles": "Gafas OK",
    "glove":   "Guantes OK",
}
# EPP que el modelo PPE no puede detectar (informativo para la UI)
PPE_NOT_DETECTABLE: list[str] = [
    "Chaleco de seguridad",
]

# Colores BGR
C_OK     = (34, 211, 101)
C_DANGER = (54,  54, 247)
C_WARN   = (0,  180, 255)
C_PERSON = (247, 142, 79)
C_ROI    = (255, 200,  0)
C_WHITE  = (255, 255, 255)
C_GRAY   = (150, 150, 150)

DEBOUNCE_FRAMES = 5


# ---------------------------------------------------------------------------
# Estructuras de datos
# ---------------------------------------------------------------------------
@dataclass
class BBox:
    x1: int; y1: int; x2: int; y2: int
    confidence: float
    class_id: int
    class_name: str

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    @property
    def centroid(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


@dataclass
class Anomaly:
    type:       str
    label:      str
    message_es: str
    confidence: float
    centroid:   tuple[int, int]
    person_id:  Optional[int]
    in_roi:     Optional[bool]


# ---------------------------------------------------------------------------
# Motor principal
# ---------------------------------------------------------------------------
class BiosecurityVisionEngine:
    """
    Motor de vision artificial headless para deteccion de EPP en BPM.

    Modos de operacion:
      'custom'  – best.pt propio
      'ppe'     – modelo PPE pre-entrenado (HuggingFace)
      'coco'    – YOLOv8n generico
      'demo'    – sin modelo
    """

    MODE_CUSTOM = "custom"
    MODE_PPE    = "ppe"
    MODE_COCO   = "coco"
    MODE_DEMO   = "demo"

    def __init__(
        self,
        source: str | int = 0,
        model_path: str = "best.pt",
        roi_polygon: Optional[list[tuple[int, int]]] = None,
        frame_skip: int = 2,
        debounce_frames: int = DEBOUNCE_FRAMES,
        reconnect_delay: float = 3.0,
        jpeg_quality: int = 85,
    ) -> None:
        self.source          = source
        self.model_path      = model_path
        self.roi_polygon: Optional[np.ndarray] = (
            np.array(roi_polygon, dtype=np.int32) if roi_polygon else None
        )
        self._frame_skip     = max(1, frame_skip)
        self.debounce_frames = debounce_frames
        self.reconnect_delay = reconnect_delay
        self.jpeg_quality    = jpeg_quality

        self._model   = None
        self._person_model = None
        self._mode    = self.MODE_DEMO
        self._running = False
        self._fc      = 0                            # frame counter
        self._cap: Optional[cv2.VideoCapture] = None
        self._active_source: str = str(source)
        self._capture_status: str = "idle"

        self._deb: dict[str, deque[bool]] = {}       # debounce buffers

        self.last_anomalies:  list[dict] = []
        self.last_detections: list[dict] = []

        self._try_load_model()

    # ── Propiedades publicas ───────────────────────────────────────────────
    @property
    def frame_skip(self) -> int:
        return self._frame_skip

    @frame_skip.setter
    def frame_skip(self, v: int) -> None:
        self._frame_skip = max(1, v)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def model_status(self) -> str:
        return self._mode

    @property
    def capture_status(self) -> str:
        return self._capture_status

    @property
    def ppe_not_detectable(self) -> list[str]:
        return PPE_NOT_DETECTABLE if self._mode == self.MODE_PPE else []

    # ── Carga de modelo ────────────────────────────────────────────────────
    def _try_load_model(self) -> None:
        """
        Estrategia de carga en cascada:
          1. best.pt custom BPM
          2. keremberke/yolov8n-hard-hat-detection (HuggingFace)
          3. yolov8n.pt COCO generico
          4. modo DEMO
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("ultralytics no instalado. pip install ultralytics")
            return

        # Paso 1: modelo custom
        if Path(self.model_path).exists():
            try:
                self._model = YOLO(self.model_path)
                self._mode  = self.MODE_CUSTOM
                logger.info("Modelo custom BPM cargado: %s", self.model_path)
                return
            except Exception as e:
                logger.error("Error cargando modelo custom: %s", e)

        # Paso 2: modelo PPE pre-entrenado (HuggingFace)
        try:
            from huggingface_hub import hf_hub_download  # type: ignore
            logger.info("Descargando modelo PPE desde HuggingFace...")
            ppe_path = hf_hub_download(
                repo_id="keremberke/yolov8m-protective-equipment-detection",
                filename="best.pt",
            )
            self._model = YOLO(ppe_path)
            self._mode  = self.MODE_PPE
            try:
                # En modo PPE usamos un detector de personas local para no
                # interpretar como "OK" una escena donde hay gente pero no
                # hay evidencia de EPP detectada.
                self._person_model = YOLO("yolov8n.pt")
                logger.info("Detector de personas COCO listo para modo PPE.")
            except Exception as e:
                self._person_model = None
                logger.warning("No se pudo cargar detector de personas: %s", e)
            logger.info("Modelo PPE (keremberke) listo. Deteccion de EPP activa.")
            return
        except Exception as e:
            logger.warning("No se pudo cargar modelo PPE desde HuggingFace: %s", e)

        # Paso 3: YOLOv8n COCO generico
        try:
            self._model = YOLO("yolov8n.pt")
            self._mode  = self.MODE_COCO
            logger.info("YOLOv8n COCO cargado como fallback generico.")
        except Exception as e:
            logger.error("No se pudo cargar ningun modelo: %s", e)
            self._mode = self.MODE_DEMO

    # ── Gestion de camara ──────────────────────────────────────────────────
    def _open_capture(self) -> cv2.VideoCapture:
        candidates: list[str | int]
        if isinstance(self.source, str) and self.source.lower() == "auto":
            candidates = list(range(6))
        else:
            candidates = [self.source]

        last_cap: Optional[cv2.VideoCapture] = None
        for candidate in candidates:
            cap = cv2.VideoCapture(candidate)
            if isinstance(candidate, int):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS,          30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok = cap.isOpened()
            logger.info("VideoCapture(%s): %s", candidate, "OK" if ok else "FALLO")
            if ok:
                self._active_source = str(candidate)
                self._capture_status = f"opened:{candidate}"
                return cap
            last_cap = cap

        self._active_source = str(self.source)
        self._capture_status = "no_camera"
        return last_cap if last_cap is not None else cv2.VideoCapture()

    # ── Geometria ──────────────────────────────────────────────────────────
    @staticmethod
    def _inter(a: BBox, b: BBox) -> int:
        return (max(0, min(a.x2, b.x2) - max(a.x1, b.x1)) *
                max(0, min(a.y2, b.y2) - max(a.y1, b.y1)))

    @classmethod
    def _contain(cls, inner: BBox, outer: BBox) -> float:
        """Proporcion del area de inner dentro de outer."""
        return cls._inter(inner, outer) / inner.area if inner.area > 0 else 0.0

    @classmethod
    def _iou(cls, a: BBox, b: BBox) -> float:
        inter = cls._inter(a, b)
        union = a.area + b.area - inter
        return inter / union if union > 0 else 0.0

    def _in_roi(self, cx: int, cy: int, shape: tuple) -> bool:
        if self.roi_polygon is None:
            return True
        h, w = shape[:2]
        pt = (float(max(0, min(cx, w - 1))), float(max(0, min(cy, h - 1))))
        return cv2.pointPolygonTest(self.roi_polygon, pt, False) >= 0

    # ── Debounce ──────────────────────────────────────────────────────────
    def _debounce(self, key: str, detected: bool) -> bool:
        if key not in self._deb:
            self._deb[key] = deque([False] * self.debounce_frames,
                                   maxlen=self.debounce_frames)
        self._deb[key].append(detected)
        return all(self._deb[key])

    # ── Parseo de detecciones ─────────────────────────────────────────────
    def _parse(self, results) -> list[BBox]:
        boxes: list[BBox] = []
        if not results or results[0].boxes is None:
            return boxes

        res   = results[0]
        names = res.names or {}

        for box in res.boxes:
            cid  = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy().astype(int)

            if self._mode == self.MODE_CUSTOM:
                name = CUSTOM_NAMES.get(cid, f"cls_{cid}")
                if cid == CUSTOM_PERSON_ID and conf < CONF_PERSON:            continue
                if cid in CUSTOM_EPP_VIOL | CUSTOM_EPP_OK and conf < CONF_EPP: continue
                if cid in CUSTOM_HYGIENE  and conf < CONF_HYGIENE:            continue

            elif self._mode == self.MODE_PPE:
                if cid not in PPE_KEEP_IDS: continue
                name = {0:"glove", 1:"goggles", 2:"helmet", 3:"mask",
                        4:"no_glove", 5:"no_goggles", 6:"no_helmet", 7:"no_mask"}.get(cid, f"cls_{cid}")
                if conf < PPE_CONF_ITEM: continue

            else:  # COCO
                if conf < 0.40:                                               continue
                raw  = names.get(cid, f"obj_{cid}")
                name = "persona" if raw == "person" else raw

            boxes.append(BBox(
                x1=int(xyxy[0]), y1=int(xyxy[1]),
                x2=int(xyxy[2]), y2=int(xyxy[3]),
                confidence=conf, class_id=cid, class_name=name,
            ))
        return boxes

    def _parse_persons(self, results) -> list[BBox]:
        """Extrae solo personas del modelo COCO local."""
        persons: list[BBox] = []
        if not results or results[0].boxes is None:
            return persons

        res = results[0]
        names = res.names or {}

        for box in res.boxes:
            cid = int(box.cls[0].item())
            if cid != 0:
                continue
            conf = float(box.conf[0].item())
            if conf < 0.35:
                continue
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            persons.append(BBox(
                x1=int(xyxy[0]), y1=int(xyxy[1]),
                x2=int(xyxy[2]), y2=int(xyxy[3]),
                confidence=conf, class_id=PPE_PERSON_ID, class_name=names.get(cid, "person") if names.get(cid, "person") != "person" else "persona",
            ))
        return persons

    def _detect_persons(self, frame: np.ndarray) -> list[BBox]:
        """Detecta personas usando el modelo COCO auxiliar."""
        if self._person_model is None:
            return []
        try:
            results = self._person_model.predict(frame, verbose=False, conf=0.35, classes=[0])
            return self._parse_persons(results)
        except Exception as e:
            logger.error("Error detectando personas: %s", e)
            return []

    @staticmethod
    def _offline_frame(message: str, size: tuple[int, int] = (720, 1280)) -> np.ndarray:
        h, w = size
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(frame, "Vigilante de Inocuidad BPM", (40, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_OK, 2, cv2.LINE_AA)
        cv2.putText(frame, message, (40, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_WARN, 2, cv2.LINE_AA)
        cv2.putText(frame, "Verifica la camara o cambia ENGINE_SOURCE", (40, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_WHITE, 1, cv2.LINE_AA)
        return frame

    def _handle_anomaly(self, key: str, is_active: bool, type_: str, label: str, msg: str, conf: float, centroid: tuple, p_idx: Optional[int], in_roi: Optional[bool], anomalies: list[Anomaly]):
        if self._debounce(key, is_active):
            anomalies.append(Anomaly(
                type=type_, label=label, message_es=msg,
                confidence=conf, centroid=centroid, person_id=p_idx, in_roi=in_roi
            ))

    # ── Analisis EPP modelo PPE ────────────────────────────────────────────
    def _analyze_ppe(self, detections: list[BBox], persons: Optional[list[BBox]] = None) -> list[Anomaly]:
        anomalies: list[Anomaly] = []
        ppe_list = [d for d in detections if d.class_id in PPE_VIOL_IDS | PPE_OK_IDS]

        # Agrupar EPP en personas detectadas; si no hay detector auxiliar,
        # usamos el propio EPP para construir "personas virtuales".
        virtual_persons: list[BBox] = list(persons or [])
        person_ppe: dict[int, dict[str, BBox]] = {}

        for idx in range(len(virtual_persons)):
            person_ppe[idx] = {}

        for ppe in ppe_list:
            matched_v = -1
            for v_idx, v_box in enumerate(virtual_persons):
                # Si hay solapamiento, expandir el bounding box de la persona virtual
                if self._inter(ppe, v_box) > 0 or self._iou(ppe, v_box) > 0.01:
                    matched_v = v_idx
                    v_box.x1 = min(v_box.x1, ppe.x1)
                    v_box.y1 = min(v_box.y1, ppe.y1)
                    v_box.x2 = max(v_box.x2, ppe.x2)
                    v_box.y2 = max(v_box.y2, ppe.y2)
                    break
            
            if matched_v == -1:
                matched_v = len(virtual_persons)
                # Crear caja virtual base
                vp = BBox(ppe.x1, ppe.y1, ppe.x2, ppe.y2, ppe.confidence, -1, "persona")
                virtual_persons.append(vp)
                person_ppe[matched_v] = {}
            
            n = ppe.class_name
            if n not in person_ppe[matched_v] or ppe.confidence > person_ppe[matched_v][n].confidence:
                person_ppe[matched_v][n] = ppe

        # Evaluar reglas por cada persona virtual
        for p_idx, matched in person_ppe.items():
            person_box = virtual_persons[p_idx]
            
            # Cubrebocas
            v_mask = "no_mask" in matched or "mask" not in matched
            b_mask = matched.get("no_mask") or person_box
            self._handle_anomaly(f"ppe_{p_idx}_no_mask", v_mask, "epp_violation", "no_mask",
                                 PPE_VIOL_ES["no_mask"], b_mask.confidence, b_mask.centroid, p_idx, None, anomalies)

            # Casco/Cofia
            v_hat = "no_helmet" in matched or "helmet" not in matched
            b_hat = matched.get("no_helmet") or person_box
            self._handle_anomaly(f"ppe_{p_idx}_no_helmet", v_hat, "epp_violation", "no_helmet",
                                 PPE_VIOL_ES["no_helmet"], b_hat.confidence, b_hat.centroid, p_idx, None, anomalies)

            # Guantes
            v_glove = "no_glove" in matched or "glove" not in matched
            b_glove = matched.get("no_glove") or person_box
            self._handle_anomaly(f"ppe_{p_idx}_no_glove", v_glove, "epp_violation", "no_glove",
                                 PPE_VIOL_ES["no_glove"], b_glove.confidence, b_glove.centroid, p_idx, None, anomalies)

            # Goggles
            v_goggles = "no_goggles" in matched or "goggles" not in matched
            b_goggles = matched.get("no_goggles") or person_box
            self._handle_anomaly(f"ppe_{p_idx}_no_goggles", v_goggles, "epp_violation", "no_goggles",
                                 PPE_VIOL_ES["no_goggles"], b_goggles.confidence, b_goggles.centroid, p_idx, None, anomalies)

        return anomalies


    # ── Analisis EPP modelo CUSTOM ─────────────────────────────────────────
    def _analyze_custom(self, detections: list[BBox], shape: tuple) -> list[Anomaly]:
        anomalies: list[Anomaly] = []
        persons  = [d for d in detections if d.class_id == CUSTOM_PERSON_ID]
        epp_list = [d for d in detections if d.class_id in CUSTOM_EPP_VIOL | CUSTOM_EPP_OK]
        hyg_list = [d for d in detections if d.class_id in CUSTOM_HYGIENE]

        person_ppe: dict[int, dict[str, BBox]] = {i: {} for i in range(len(persons))}
        used_ppe = set()

        for i, person in enumerate(persons):
            for epp in epp_list:
                if (self._contain(epp, person) >= 0.15 or self._iou(epp, person) >= 0.05):
                    n = epp.class_name
                    if n not in person_ppe[i] or epp.confidence > person_ppe[i][n].confidence:
                        person_ppe[i][n] = epp
                        used_ppe.add(id(epp))

        virtual_persons = list(persons)
        for epp in epp_list:
            if id(epp) not in used_ppe:
                matched_v = -1
                for v_idx in range(len(persons), len(virtual_persons)):
                    if self._inter(epp, virtual_persons[v_idx]) > 0:
                        matched_v = v_idx
                        break
                if matched_v == -1:
                    matched_v = len(virtual_persons)
                    virtual_persons.append(epp)
                    person_ppe[matched_v] = {}
                n = epp.class_name
                if n not in person_ppe[matched_v] or epp.confidence > person_ppe[matched_v][n].confidence:
                    person_ppe[matched_v][n] = epp

        for p_idx, matched in person_ppe.items():
            person_box = virtual_persons[p_idx]
            viols = {k: v for k, v in matched.items() if v.class_id in CUSTOM_EPP_VIOL}
            oks   = {k: v for k, v in matched.items() if v.class_id in CUSTOM_EPP_OK}
            
            # Cubrebocas
            v_cub = "sin_cubrebocas" in viols or "cubrebocas_ok" not in oks
            b_cub = viols.get("sin_cubrebocas") or person_box
            self._handle_anomaly(f"custom_{p_idx}_sin_cubrebocas", v_cub, "epp_violation", "sin_cubrebocas",
                                 "Sin Cubrebocas", b_cub.confidence, b_cub.centroid, p_idx, None, anomalies)

            # Cofia
            v_cofia = "sin_cofia" in viols or "cofia_ok" not in oks
            b_cof = viols.get("sin_cofia") or person_box
            self._handle_anomaly(f"custom_{p_idx}_sin_cofia", v_cofia, "epp_violation", "sin_cofia",
                                 "Sin Cofia", b_cof.confidence, b_cof.centroid, p_idx, None, anomalies)

        for h_idx, hbox in enumerate(hyg_list):
            cx, cy = hbox.centroid
            in_r = self._in_roi(cx, cy, shape)
            self._handle_anomaly(f"hyg_{h_idx}_{hbox.class_name}", in_r, "hygiene_issue", hbox.class_name,
                                 hbox.class_name.replace("_", " ").title(), hbox.confidence, (cx, cy), None, True, anomalies)

        return anomalies

    # ── Anotacion visual ───────────────────────────────────────────────────
    def _annotate(
        self,
        frame: np.ndarray,
        detections: list[BBox],
        anomalies: list[Anomaly],
    ) -> np.ndarray:
        out = frame.copy()

        # ROI
        if self.roi_polygon is not None:
            ov = out.copy()
            cv2.fillPoly(ov, [self.roi_polygon], C_ROI)
            cv2.addWeighted(ov, 0.12, out, 0.88, 0, out)
            cv2.polylines(out, [self.roi_polygon], True, C_ROI, 2, cv2.LINE_AA)

        # Bounding boxes
        for det in detections:
            if self._mode == self.MODE_PPE:
                color    = (C_PERSON if det.class_name == "persona"
                            else C_DANGER if det.class_id in PPE_VIOL_IDS
                            else C_OK)
                label_es = ("Persona" if det.class_name == "persona" else
                            PPE_VIOL_ES.get(det.class_name)
                            or PPE_OK_ES.get(det.class_name)
                            or det.class_name)
                lbl = f"{label_es} {det.confidence:.0%}"
            elif self._mode == self.MODE_CUSTOM:
                color = (C_PERSON if det.class_id == CUSTOM_PERSON_ID
                         else C_DANGER if det.class_id in CUSTOM_EPP_VIOL
                         else C_OK    if det.class_id in CUSTOM_EPP_OK
                         else C_WARN)
                lbl = f"{det.class_name} {det.confidence:.0%}"
            else:
                color = C_PERSON if det.class_name == "persona" else C_WARN
                lbl   = f"{det.class_name} {det.confidence:.0%}"

            cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), color, 2, cv2.LINE_AA)
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.rectangle(out, (det.x1, det.y1 - th - 6), (det.x1 + tw + 4, det.y1), color, -1)
            cv2.putText(out, lbl, (det.x1 + 2, det.y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_WHITE, 1, cv2.LINE_AA)

        # Panel alertas activas
        if anomalies:
            panel_h = min(len(anomalies) * 30 + 24, 250)
            ov2     = out[:panel_h, :360].copy()
            cv2.rectangle(out, (0, 0), (360, panel_h), (12, 12, 12), -1)
            cv2.addWeighted(ov2, 0.2, out[:panel_h, :360], 0.8, 0, out[:panel_h, :360])
            cv2.putText(out, "ALERTAS DE BIOSEGURIDAD", (8, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_DANGER, 1, cv2.LINE_AA)
            for i, a in enumerate(anomalies[:7]):
                pid = f"[P{a.person_id}] " if a.person_id is not None else ""
                txt = f"{pid}{a.message_es}  ({a.confidence:.0%})"
                cv2.putText(out, txt, (8, 38 + i * 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, C_DANGER, 1, cv2.LINE_AA)

        # Timestamp + modo (esquina inferior derecha)
        h_f, w_f = out.shape[:2]
        mode_lbl = {"custom":"BPM Custom","ppe":"PPE Detector",
                    "coco":"COCO Generic","demo":"DEMO"}.get(self._mode, self._mode)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(out, f"{mode_lbl}  {ts}", (w_f - 320, h_f - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_GRAY, 1, cv2.LINE_AA)
        return out

    def _demo_frame(self, frame: np.ndarray) -> np.ndarray:
        out  = frame.copy()
        h, w = out.shape[:2]
        cv2.putText(out, "Sin modelo YOLO disponible",
                    (w // 2 - 220, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_WARN, 2, cv2.LINE_AA)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(out, ts, (w - 200, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_GRAY, 1, cv2.LINE_AA)
        return out

    # ── Serializacion ─────────────────────────────────────────────────────
    def _serialize(self, detections: list[BBox]) -> list[dict]:
        out = []
        for d in detections:
            if self._mode == self.MODE_PPE:
                cat     = ("person" if d.class_id == PPE_PERSON_ID
                           else "epp_violation" if d.class_id in PPE_VIOL_IDS
                           else "epp_ok")
                name_es = ("Persona" if d.class_name == "persona" else
                           PPE_VIOL_ES.get(d.class_name)
                           or PPE_OK_ES.get(d.class_name)
                           or d.class_name)
            elif self._mode == self.MODE_CUSTOM:
                cat     = ("person"        if d.class_id == CUSTOM_PERSON_ID
                           else "epp_violation" if d.class_id in CUSTOM_EPP_VIOL
                           else "epp_ok"        if d.class_id in CUSTOM_EPP_OK
                           else "hygiene")
                name_es = d.class_name
            else:
                cat     = "person" if d.class_name == "persona" else "object"
                name_es = d.class_name

            out.append({
                "class_id":   d.class_id,
                "class_name": d.class_name,
                "name_es":    name_es,
                "confidence": round(d.confidence, 3),
                "bbox":       [d.x1, d.y1, d.x2, d.y2],
                "centroid":   list(d.centroid),
                "category":   cat,
            })
        return out

    # ── Generador principal (headless) ─────────────────────────────────────
    def stream(self) -> Generator[tuple[bytes, list[dict]], None, None]:
        """
        Generador MJPEG headless.
        Yields: (jpeg_bytes, anomalies_list)
        """
        self._running  = True
        enc            = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        raw_conf       = (0.35 if self._mode in (self.MODE_PPE, self.MODE_COCO)
                          else min(CONF_EPP, CONF_HYGIENE, CONF_PERSON))

        while self._running:
            # Abrir camara
            self._cap = self._open_capture()
            if not self._cap.isOpened():
                logger.warning("Reintentando en %.1fs...", self.reconnect_delay)
                offline = self._offline_frame(
                    "Sin camara disponible" if self.source == "auto"
                    else f"No se pudo abrir la fuente: {self.source}"
                )
                _, buf = cv2.imencode(".jpg", offline, enc)
                yield buf.tobytes(), self.last_anomalies
                time.sleep(self.reconnect_delay)
                continue

            logger.info("Stream iniciado. Modo: %s", self._mode)

            while self._running:
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    logger.warning("Frame perdido. Reconectando...")
                    break

                self._fc += 1
                frame     = cv2.flip(frame, 1)   # efecto espejo

                # Frames no procesados (solo para fluidez visual)
                if self._fc % self._frame_skip != 0:
                    _, buf = cv2.imencode(".jpg", frame, enc)
                    yield buf.tobytes(), self.last_anomalies
                    continue

                # ── Inferencia ─────────────────────────────────────
                anomalies: list[Anomaly] = []

                if self._mode != self.MODE_DEMO and self._model is not None:
                    try:
                        results    = self._model.predict(frame, verbose=False, conf=raw_conf)
                        detections = self._parse(results)
                    except Exception as e:
                        logger.error("Error en inferencia: %s", e)
                        detections = []

                    # Analisis segun modo
                    if self._mode == self.MODE_PPE:
                        persons = self._detect_persons(frame)
                        detections = persons + detections
                        anomalies = self._analyze_ppe(detections, persons)
                    elif self._mode == self.MODE_CUSTOM:
                        anomalies = self._analyze_custom(detections, frame.shape)
                    # COCO: sin analisis PPE (no tiene esas clases)

                    annotated            = self._annotate(frame, detections, anomalies)
                    self.last_detections = self._serialize(detections)

                else:
                    detections           = []
                    self.last_detections = []
                    annotated            = self._demo_frame(frame)

                # Serializar anomalias
                self.last_anomalies = [
                    {
                        "type":       a.type,
                        "label":      a.label,
                        "message_es": a.message_es,
                        "confidence": round(a.confidence, 3),
                        "centroid":   list(a.centroid),
                        "person_id":  a.person_id,
                        "in_roi":     a.in_roi,
                        "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                    for a in anomalies
                ]

                if self.last_anomalies:
                    logger.info("Anomalias validadas: %d", len(self.last_anomalies))

                _, buf = cv2.imencode(".jpg", annotated, enc)
                yield buf.tobytes(), self.last_anomalies

            if self._cap:
                self._cap.release()
                logger.info("Capture liberado.")
            time.sleep(self.reconnect_delay)

    def stop(self) -> None:
        """Detiene el stream y libera recursos."""
        self._running = False
        if self._cap and self._cap.isOpened():
            self._cap.release()
        logger.info("Engine detenido.")
