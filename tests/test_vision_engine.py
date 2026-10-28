"""
Tests de la logica pura del motor: geometria, debounce, ROI y reglas de EPP.

No necesitan camara ni modelo: el engine se construye sin cargar YOLO y las
detecciones se pasan como BBox ya armadas.
"""
from __future__ import annotations

import pytest

from vision_engine import (
    CUSTOM_PERSON_ID,
    BBox,
    BiosecurityVisionEngine,
)

DEBOUNCE = 3
SHAPE = (720, 1280, 3)


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setattr(BiosecurityVisionEngine, "_try_load_model", lambda self: None)
    return BiosecurityVisionEngine(source=0, debounce_frames=DEBOUNCE)


def box(x1, y1, x2, y2, cid=0, name="persona", conf=0.9) -> BBox:
    return BBox(x1, y1, x2, y2, confidence=conf, class_id=cid, class_name=name)


def labels(anomalies) -> set[str]:
    return {a.label for a in anomalies}


# ── Geometria ─────────────────────────────────────────────────────────────
class TestGeometria:
    def test_area_y_centroide(self):
        b = box(0, 0, 10, 20)
        assert b.area == 200
        assert b.centroid == (5, 10)

    def test_area_de_caja_invertida_es_cero(self):
        assert box(10, 10, 0, 0).area == 0

    def test_interseccion_parcial(self):
        assert BiosecurityVisionEngine._inter(box(0, 0, 10, 10), box(5, 5, 15, 15)) == 25

    def test_sin_interseccion(self):
        assert BiosecurityVisionEngine._inter(box(0, 0, 10, 10), box(20, 20, 30, 30)) == 0

    def test_cajas_que_solo_se_tocan_no_intersectan(self):
        assert BiosecurityVisionEngine._inter(box(0, 0, 10, 10), box(10, 0, 20, 10)) == 0

    def test_iou_identicas_es_uno(self):
        a = box(0, 0, 10, 10)
        assert BiosecurityVisionEngine._iou(a, a) == pytest.approx(1.0)

    def test_iou_parcial(self):
        # interseccion 25, union 100 + 100 - 25 = 175
        iou = BiosecurityVisionEngine._iou(box(0, 0, 10, 10), box(5, 5, 15, 15))
        assert iou == pytest.approx(25 / 175)

    def test_iou_con_cajas_vacias_no_divide_entre_cero(self):
        assert BiosecurityVisionEngine._iou(box(0, 0, 0, 0), box(0, 0, 0, 0)) == 0.0

    def test_contencion_total(self):
        assert BiosecurityVisionEngine._contain(box(2, 2, 4, 4), box(0, 0, 10, 10)) == 1.0

    def test_contencion_de_caja_vacia_es_cero(self):
        assert BiosecurityVisionEngine._contain(box(5, 5, 5, 5), box(0, 0, 10, 10)) == 0.0


# ── Debounce ──────────────────────────────────────────────────────────────
class TestDebounce:
    def test_no_dispara_antes_de_n_frames(self, engine):
        results = [engine._debounce("k", True) for _ in range(DEBOUNCE)]
        assert results == [False] * (DEBOUNCE - 1) + [True]

    def test_un_frame_limpio_reinicia_la_alerta(self, engine):
        for _ in range(DEBOUNCE):
            engine._debounce("k", True)
        assert engine._debounce("k", False) is False
        assert engine._debounce("k", True) is False

    def test_las_claves_son_independientes(self, engine):
        for _ in range(DEBOUNCE):
            engine._debounce("a", True)
        assert engine._debounce("b", True) is False


# ── ROI ───────────────────────────────────────────────────────────────────
class TestROI:
    def test_sin_roi_todo_el_frame_cuenta(self, engine):
        assert engine._in_roi(5000, 5000, SHAPE) is True

    def test_punto_dentro_y_fuera(self, engine):
        engine.roi_polygon = __import__("numpy").array(
            [(100, 100), (300, 100), (300, 300), (100, 300)], dtype="int32"
        )
        assert engine._in_roi(200, 200, SHAPE) is True
        assert engine._in_roi(500, 500, SHAPE) is False

    def test_borde_cuenta_como_dentro(self, engine):
        engine.roi_polygon = __import__("numpy").array(
            [(100, 100), (300, 100), (300, 300), (100, 300)], dtype="int32"
        )
        assert engine._in_roi(100, 200, SHAPE) is True


# ── Reglas modelo custom ──────────────────────────────────────────────────
def frames(fn, n=DEBOUNCE):
    """Ejecuta el analisis n veces (simula n frames) y devuelve el ultimo."""
    out = []
    for _ in range(n):
        out = fn()
    return out


class TestReglasCustom:
    persona = staticmethod(lambda: box(100, 100, 300, 500, CUSTOM_PERSON_ID, "persona"))

    def test_persona_con_todo_el_epp_no_genera_alertas(self, engine):
        dets = lambda: [
            self.persona(),
            box(150, 120, 250, 160, 3, "cofia_ok"),
            box(160, 180, 240, 220, 1, "cubrebocas_ok"),
        ]
        assert frames(lambda: engine._analyze_custom(dets(), SHAPE)) == []

    def test_sin_cubrebocas_explicito_genera_alerta(self, engine):
        dets = lambda: [
            self.persona(),
            box(150, 120, 250, 160, 3, "cofia_ok"),
            box(160, 180, 240, 220, 2, "sin_cubrebocas"),
        ]
        assert labels(frames(lambda: engine._analyze_custom(dets(), SHAPE))) == {"sin_cubrebocas"}

    def test_ausencia_de_epp_cuenta_como_violacion(self, engine):
        # Persona sin ninguna deteccion de EPP: no hay evidencia de que lo use.
        dets = lambda: [self.persona()]
        assert labels(frames(lambda: engine._analyze_custom(dets(), SHAPE))) == {
            "sin_cubrebocas", "sin_cofia",
        }

    def test_la_alerta_espera_al_debounce(self, engine):
        dets = lambda: [self.persona()]
        assert frames(lambda: engine._analyze_custom(dets(), SHAPE), n=DEBOUNCE - 1) == []

    def test_epp_lejos_de_la_persona_no_se_le_asigna(self, engine):
        dets = lambda: [
            self.persona(),
            box(900, 100, 1000, 140, 3, "cofia_ok"),
            box(900, 160, 1000, 200, 1, "cubrebocas_ok"),
        ]
        anomalies = frames(lambda: engine._analyze_custom(dets(), SHAPE))
        persona_0 = {a.label for a in anomalies if a.person_id == 0}
        assert persona_0 == {"sin_cubrebocas", "sin_cofia"}

    def test_suciedad_fuera_del_roi_se_ignora(self, engine):
        import numpy as np
        engine.roi_polygon = np.array([(0, 0), (200, 0), (200, 200), (0, 200)], dtype=np.int32)
        dets = lambda: [box(800, 600, 900, 700, 5, "basura_suelo")]
        assert frames(lambda: engine._analyze_custom(dets(), SHAPE)) == []

    def test_suciedad_dentro_del_roi_genera_alerta(self, engine):
        dets = lambda: [box(800, 600, 900, 700, 6, "derrame_suciedad")]
        anomalies = frames(lambda: engine._analyze_custom(dets(), SHAPE))
        assert [(a.type, a.label) for a in anomalies] == [("hygiene_issue", "derrame_suciedad")]


# ── Reglas modelo PPE ─────────────────────────────────────────────────────
class TestReglasPPE:
    def test_persona_con_todo_el_epp_no_genera_alertas(self, engine):
        dets = lambda: [
            box(150, 100, 250, 150, 2, "helmet"),
            box(160, 170, 240, 210, 3, "mask"),
            box(140, 150, 260, 190, 1, "goggles"),
            box(120, 300, 280, 350, 0, "glove"),
        ]
        persons = lambda: [box(100, 100, 300, 500, -1, "persona")]
        assert frames(lambda: engine._analyze_ppe(dets(), persons())) == []

    def test_persona_sin_epp_detectado_genera_todas_las_alertas(self, engine):
        persons = lambda: [box(100, 100, 300, 500, -1, "persona")]
        assert labels(frames(lambda: engine._analyze_ppe([], persons()))) == {
            "no_mask", "no_helmet", "no_glove", "no_goggles",
        }

    def test_sin_detector_de_personas_agrupa_el_epp_en_persona_virtual(self, engine):
        dets = lambda: [
            box(150, 100, 250, 150, 2, "helmet"),
            box(160, 140, 240, 210, 7, "no_mask"),
        ]
        anomalies = frames(lambda: engine._analyze_ppe(dets(), None))
        assert {a.person_id for a in anomalies} == {0}
        assert "no_mask" in labels(anomalies)
        assert "no_helmet" not in labels(anomalies)

    @pytest.mark.xfail(reason="_analyze_ppe expande las cajas de persona que recibe (muta la entrada)")
    def test_no_modifica_las_cajas_de_persona_recibidas(self, engine):
        persona = box(100, 100, 300, 500, -1, "persona")
        engine._analyze_ppe([box(50, 50, 150, 150, 2, "helmet")], [persona])
        assert (persona.x1, persona.y1) == (100, 100)
