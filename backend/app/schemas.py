from pydantic import BaseModel, Field
from typing import List
from datetime import datetime

# ==========================================
# 1. Contrato de Ingesta (n8n -> FastAPI)
# ==========================================

class ClimaExterior(BaseModel):
    temperatura_c: float = Field(..., description="Temperatura externa en grados Celsius")
    humedad_pct: float = Field(..., description="Porcentaje de humedad relativa externa")

class SensorInterno(BaseModel):
    zona_id: str = Field(..., description="Identificador único de la zona (ej. cocina_principal, almacen_seco)")
    temp_c: float = Field(..., description="Temperatura interna en grados Celsius")
    hum_pct: float = Field(..., description="Porcentaje de humedad relativa interna")

class IngestPayload(BaseModel):
    timestamp: datetime = Field(..., description="Fecha y hora de la lectura de los datos")
    clima_exterior: ClimaExterior = Field(..., description="Variables climáticas externas")
    sensores_internos: List[SensorInterno] = Field(..., description="Lista de lecturas de sensores internos por zona")


# ==========================================
# 2. Contrato de Alerta Externa (FastAPI -> n8n)
# ==========================================

class AlertaExternaPayload(BaseModel):
    alerta_id: str = Field(..., description="Identificador único autogenerado de la alerta")
    zona_id: str = Field(..., description="Zona afectada por la alerta")
    nivel_riesgo: str = Field(..., description="Nivel de riesgo detectado (ALTO o CRÍTICO)")
    accion_nudge: str = Field(..., description="Instrucción de acción física recomendada para el usuario")
    timestamp: datetime = Field(..., description="Fecha y hora en que se disparó la alerta")


# ==========================================
# 3. Contrato de Actualización UI (FastAPI -> Angular)
# ==========================================

class ZonaEstado(BaseModel):
    zona_id: str = Field(..., description="Identificador único de la zona")
    color_hex: str = Field(..., description="Color en formato hexadecimal para pintar el mapa SVG")
    status: str = Field(..., description="Nivel de riesgo determinado (SEGURO, BAJO, MEDIO, ALTO, CRÍTICO)")

class UIUpdatePayload(BaseModel):
    mapa_estado: List[ZonaEstado] = Field(..., description="Listado del estado actual de todas las zonas")
