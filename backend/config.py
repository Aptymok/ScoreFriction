import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key-change-in-prod')
    GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
    GROQ_MODEL = 'llama-3.3-70b-versatile'
    DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///instance/friction.db')
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'

    # ── Orquestador Proactivo ──────────────────────────────────────
    GOOGLE_SERVICE_ACCOUNT_JSON = os.environ.get('GOOGLE_SERVICE_ACCOUNT_JSON')
    GOOGLE_CREDENTIALS_FILE     = os.environ.get('GOOGLE_CREDENTIALS_FILE', 'credentials/service_account.json')
    GOOGLE_CALENDAR_ID          = os.environ.get('GOOGLE_CALENDAR_ID', 'primary')
    GOOGLE_DRIVE_FOLDER_ID      = os.environ.get('GOOGLE_DRIVE_FOLDER_ID', '')
    TELEGRAM_BOT_TOKEN          = os.environ.get('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID            = os.environ.get('TELEGRAM_CHAT_ID')
    ORCHESTRATOR_POLL_SECONDS   = int(os.environ.get('ORCHESTRATOR_POLL_SECONDS', 300))
    MIDI_OUTPUT_DIR             = os.environ.get('MIDI_OUTPUT_DIR', 'instance/midi_output')
    TRACK_KEYWORD               = os.environ.get('TRACK_KEYWORD', 'generación de pista')
    HOOK_THRESHOLD              = float(os.environ.get('HOOK_THRESHOLD', 0.7))