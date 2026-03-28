"""
main.py – Entry point para Railway/gunicorn.
Railway busca 'main:app' desde la raíz del proyecto.
Este archivo agrega backend/ al path y re-exporta la app Flask.
"""
import sys
import os

# Asegura que backend/ esté en el path de Python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from app import app  # noqa: F401 – re-export para gunicorn

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
