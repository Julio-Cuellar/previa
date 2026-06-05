# PREVIA (Fase 1 - Entregable Base)

PREVIA es un ecosistema IoT predictivo y un dashboard reactivo diseñado para predecir y prevenir riesgos de infestación de plagas en establecimientos comerciales.

---

## Estructura del Ecosistema

El proyecto está estructurado de la siguiente forma:

* **/backend**: Servidor FastAPI (Python 3.11) con motor de reglas y soporte para WebSockets en tiempo real.
* **/frontend**: Aplicación en Angular 17 para el dashboard web reactivo y el mapa SVG interactivo.
* **/n8n**: Configuración de flujos de automatización (Ingesta de sensores y Webhooks de alerta).

---

## Cómo Ejecutar el Proyecto

Este proyecto está preparado para correr de forma local utilizando **Docker** y **Docker Compose**, lo que evita la necesidad de configurar Node.js, Python o Nginx localmente en tu sistema operativo.

### Requisitos Previos

* Tener instalado [Docker](https://www.docker.com/) y que el daemon esté corriendo en tu sistema.

### Instrucciones de Inicio

1. Clona el repositorio y navega a la raíz del proyecto.
2. Ejecuta el siguiente comando para compilar y levantar todos los servicios:
   ```bash
   docker compose up --build
   ```
3. Una vez finalice la construcción, los servicios estarán disponibles en los siguientes puertos:
   * **Dashboard de Angular (Frontend):** `http://localhost:80`
   * **Servidor FastAPI (Backend):** `http://localhost:8000` (documentación interactiva disponible en `http://localhost:8000/docs`)
   * **n8n (Orquestador):** `http://localhost:5678`

---

## Desarrollo Paso a Paso

Las clases, componentes y flujos lógicos serán creados y detallados de forma incremental. El estado actual del proyecto contiene la configuración del entorno y los esqueletos iniciales de los componentes.
