import os
from flask import Flask, jsonify, send_from_directory

app = Flask(__name__)

@app.route('/')
def index():
    # Sirve el index.html desde la carpeta principal
    return send_from_directory(os.path.dirname(os.path.dirname(__file__)), 'index.html')

@app.route('/health')
def health():
    return jsonify({"status": "ok", "message": "Backend funcionando"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)