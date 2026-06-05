import os
from dotenv import load_dotenv

# Cargar variables de entorno desde un archivo .env si estuviera presente
load_dotenv()

class Settings:
    PROJECT_NAME: str = "PREVIA - Ecosistema IoT Predictivo"
    
    # URL del Webhook de n8n para despachar las Nudge-Alerts
    # Por defecto apunta al contenedor n8n en la red interna de Docker Compose
    N8N_WEBHOOK_URL: str = os.getenv(
        "N8N_WEBHOOK_URL", 
        "http://n8n:5678/webhook/alert"
    )
    
    # Puerto e información de entorno
    ENV: str = os.getenv("ENV", "development")
    API_PORT: int = int(os.getenv("API_PORT", 8000))

settings = Settings()
