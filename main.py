import os
import sys

# 1. Agregamos la carpeta backend al path de Python para que los imports funcionen
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# 2. Importamos la app de Flask que está dentro de backend/app.py
from app import app

# 3. Railway usará este archivo para arrancar
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)