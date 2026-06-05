# 📷 Cámara Web – Live Stream

Stream de cámara web en tiempo real usando **Python + Flask + OpenCV**.

## Estructura

```
Previa/
├── app.py              ← Servidor Flask + captura de cámara
├── requirements.txt    ← Dependencias
└── templates/
    └── index.html      ← Interfaz web
```

## Instalación y ejecución

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar el servidor
python app.py
```

Luego abre http://localhost:5000 en tu navegador.

## Características

- Stream MJPEG de baja latencia desde la cámara 0
- Resolución 1280 × 720 @ 30 fps
- Efecto espejo (flip horizontal)
- Captura de fotografías desde el navegador
- Modo pantalla completa
- Contador de tiempo activo
- Indicador de estado en tiempo real
