# PREVIA - Guía de Pasos de Desarrollo Incremental

Esta guía detalla los pasos ordenados para implementar las clases y componentes del ecosistema **PREVIA**. Iremos desarrollando cada paso detalladamente de forma conjunta.

---

## 📋 Lista de Pasos y Roadmap de Desarrollo

### [Paso 1] Configuración y Modelos de Datos (Backend) ─── ✅ COMPLETADO
* [x] **Configuración global (`config.py`):** Configurar la carga de variables de entorno (como la URL del webhook de n8n) y puertos.
* [x] **Esquemas de datos (`schemas.py`):** Implementar la validación con Pydantic para los payloads de ingesta, alertas salientes y actualizaciones de la interfaz.

---

### [Paso 2] Motor de Evaluación de Riesgos (Backend) ─── 🟥 PENDIENTE
* [ ] **Lógica de reglas (`rules.py`):**
  * Crear la lógica para evaluar el nivel de riesgo (`SEGURO`, `BAJO`, `MEDIO`, `ALTO`, `CRÍTICO`) a partir de la temperatura/humedad interna de cada zona en relación al clima externo.
  * Asignar colores hexadecimales correspondientes a cada estado de riesgo (ej. verde para seguro, rojo para crítico).
  * Redactar las recomendaciones físicas ("Nudge Action") específicas de prevención de plagas para riesgos elevados.

---

### [Paso 3] Servidor API y Difusión por WebSockets (Backend) ─── 🟥 PENDIENTE
* [ ] **Controlador principal (`main.py`):**
  * Configurar la app FastAPI con políticas de CORS.
  * Implementar un administrador de conexiones de WebSockets (`ConnectionManager`) para registrar las conexiones activas del dashboard Angular.
  * Crear el endpoint `POST /api/v1/evaluate-risk`.
  * Vincular el endpoint al motor de reglas para evaluar cada zona ingresada.
  * Transmitir la actualización en tiempo real a todos los clientes WebSocket conectados.
  * Disparar una petición HTTP externa al webhook de n8n cuando se detecte un riesgo `ALTO` o `CRÍTICO`.

---

### [Paso 4] Comunicación en Tiempo Real (Frontend) ─── 🟥 PENDIENTE
* [ ] **Servicio de WebSockets (`websocket.service.ts`):**
  * Crear el servicio Angular para establecer y mantener la conexión WebSocket con el backend.
  * Exponer el flujo de datos usando programación reactiva (RxJS) para permitir la suscripción de múltiples componentes de la aplicación.

---

### [Paso 5] Plano SVG Interactivo del Establecimiento (Frontend) ─── 🟥 PENDIENTE
* [ ] **Componente del plano (`floor-plan/`):**
  * Diseñar la estructura visual SVG detallada del establecimiento (mostrando cocina, almacén, comedor, etc.).
  * Configurar directivas dinámicas de Angular vinculadas a los IDs de zona (`cocina_principal`, `almacen_seco`) para cambiar sus colores de relleno según el estado recibido de WebSocket con transiciones CSS suaves.

---

### [Paso 6] Métricas y Centro de Notificaciones (Frontend) ─── 🟥 PENDIENTE
* [ ] **Componente Dashboard (`dashboard/`) y Alertas (`nudges/`):**
  * Crear las tarjetas de métricas del clima exterior y sensores internos en una cuadrícula con diseño oscuro premium.
  * Implementar el registro del historial de alertas y las recomendaciones físicas ("Nudges") vigentes enviadas al personal.

---

### [Paso 7] Orquestación y Simulación (n8n) ─── 🟥 PENDIENTE
* [ ] **Flujo de Ingesta:** Programar en n8n la lectura periódica (cron), la consulta del clima exterior e inyección de datos simulados a FastAPI.
* [ ] **Flujo de Alertas:** Configurar el recibo del webhook del backend y el envío de notificaciones automáticas a Telegram, Slack o Email.

---

### [Paso 8] Pruebas de Latencia e Integración de Extremo a Extremo ─── 🟥 PENDIENTE
* [ ] Ejecutar todo el ecosistema mediante Docker Compose.
* [ ] Verificar que el ciclo completo (de n8n a FastAPI, cálculo del motor, transmisión por WebSocket y repintado de la zona en Angular) tome menos de 2 segundos.
