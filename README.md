# PREVIA · Vigilante de Inocuidad BPM

![tests](https://github.com/Julio-Cuellar/previa/actions/workflows/tests.yml/badge.svg)

Motor de visión por computadora que vigila en tiempo real si el personal de una cocina o planta de alimentos cumple las **Buenas Prácticas de Manufactura (BPM)**: cubrebocas, cofia, guantes y gafas, además de basura o derrames en zonas limpias. Toma el video de una webcam o cámara IP, detecta las faltas y las publica en un dashboard web y en una API JSON.

> 📸 *Agrega aquí un GIF o captura de la detección en vivo.*

## El problema

En la industria alimentaria, una persona sin cubrebocas o sin cofia en la zona de producción es un riesgo de contaminación y una falta en auditorías sanitarias. Hoy eso se revisa a ojo y de forma esporádica. PREVIA lo hace de forma continua con la cámara que ya existe en el lugar.

## Cómo funciona

```
cámara (USB / RTSP)
      │  reconexión automática, salto de frames
      ▼
detector YOLOv8 ──► asignación de EPP a cada persona (intersección / IoU)
      │                         │
      │                         ▼
      │               reglas BPM por persona  +  zona limpia (ROI)
      │                         │
      │                         ▼
      │               debounce: la falta debe persistir N frames
      ▼                         ▼
stream MJPEG anotado      anomalías validadas
      └──────────── Flask: dashboard web + API JSON
```

Decisiones de diseño:

- **El EPP se asigna a una persona, no se cuenta suelto.** Cada cubrebocas o cofia detectado se asocia a la persona con la que se solapa (proporción de contención o IoU). Si no hay detector de personas, el motor agrupa el EPP cercano en "personas virtuales".
- **La ausencia de evidencia cuenta como falta.** Si a una persona no se le detecta cubrebocas, se reporta "sin cubrebocas" aunque el modelo no haya visto la clase negativa. Es la postura conservadora que exige una auditoría.
- **Debounce contra falsos positivos.** Una alerta solo se dispara si la falta aparece en N frames seguidos (5 por defecto). Un frame limpio la reinicia. Esto elimina el parpadeo de alertas por detecciones aisladas.
- **Zona limpia configurable.** Las faltas de higiene (basura, derrames) solo cuentan dentro de un polígono ROI que se actualiza en caliente desde la API, sin reiniciar el servicio.
- **Degradación en cascada.** El motor elige el mejor modelo disponible y sigue funcionando si falta alguno:

  | Modo | Modelo | Qué detecta |
  |---|---|---|
  | `custom` | `best.pt` propio | Persona, cubrebocas, cofia (OK / falta), basura, derrames |
  | `ppe` | YOLOv8 de EPP (Hugging Face) + YOLOv8n para personas | Cubrebocas, casco/cofia, guantes, gafas (OK / falta) |
  | `coco` | YOLOv8n genérico | 80 clases COCO, sin reglas BPM |
  | `demo` | ninguno | Solo el stream con un aviso |

- **Operación sin supervisión.** Si la cámara se desconecta, el motor muestra un frame de aviso y se reconecta solo. El salto de frames (`frame_skip`) controla el costo de inferencia en CPU.

## Ejecutarlo

Requiere Python 3.10+ y una cámara.

```bash
pip install -r requirements.txt
python app.py
```

Abre http://localhost:5000. Los pesos de YOLOv8n se descargan solos la primera vez. Si tienes un modelo entrenado con la taxonomía BPM, colócalo como `best.pt` en la raíz.

Para elegir la cámara usa la variable `ENGINE_SOURCE`: `auto` (prueba los índices 0 a 5), un índice como `1`, o una URL `rtsp://...` de cámara IP.

## API

| Método | Ruta | Respuesta |
|---|---|---|
| `GET` | `/` | Dashboard web |
| `GET` | `/video_feed` | Stream MJPEG con las detecciones dibujadas |
| `GET` | `/api/anomalies` | Faltas validadas (ya pasaron el debounce) |
| `GET` | `/api/detections` | Todas las detecciones del último frame |
| `GET` | `/api/status` | Modo del modelo, estado de la cámara, configuración |
| `POST` | `/api/roi` | Define la zona limpia: `{"polygon": [[x,y], ...]}` o `{"polygon": null}` para desactivarla |

## Tests

La lógica de decisión (geometría, debounce, ROI y reglas de EPP) se prueba sin cámara ni modelo:

```bash
pip install -r requirements-dev.txt
pytest
```

Corren en GitHub Actions en cada push.

## Estructura

```
├── app.py              Servidor Flask y API
├── vision_engine.py    Motor: carga de modelos, captura, reglas BPM, anotación
├── templates/
│   └── index.html      Dashboard
└── tests/              Pruebas de la lógica del motor
```

## Siguientes pasos

- Medir precisión y recall por clase sobre un set de imágenes etiquetadas, y FPS en CPU.
- Entrenar el modelo `custom` con imágenes de cocinas reales.
- Enviar alertas a WhatsApp o correo cuando una falta persista.
- Seguimiento de personas entre frames para no depender del índice de detección.
