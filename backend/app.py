import os
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ==================================================
# 1. CREAR LA APLICACIÓN (NO TOCAR)
# ==================================================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ==================================================
# 2. CARPETAS INTERNAS (PARA QUE RAILWAY FUNCIONE)
# ==================================================
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_INSTANCE_DIR = os.path.join(_BACKEND_DIR, 'instance')
_MIDI_DIR = os.path.join(_INSTANCE_DIR, 'midi_output')
os.makedirs(_INSTANCE_DIR, exist_ok=True)
os.makedirs(_MIDI_DIR, exist_ok=True)

# ==================================================
# 3. SIMULADOR DEL MOTOR MIHM (PARA QUE NO FALTE NADA)
# ==================================================
class MIHM:
    def __init__(self):
        self.state = {'ihg': -0.62, 'nti': 0.351, 'r': 0.45, 'cff': 0.0}
        self.irc = 0.38
        self.history = []
    def cost_function(self): return 0.42
    def monte_carlo_projection(self): return [0.1, 0.2, 0.3]
    def apply_delta(self, delta, action):
        for k, v in delta.items():
            if k in self.state:
                self.state[k] += v
        return 0.0, self.cost_function()
    def process_delayed_updates(self): pass
    def meta_control(self): pass

mihm = MIHM()

# ==================================================
# 4. TODOS LOS ENDPOINTS QUE NECESITA TU FRONTEND
# ==================================================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'mihm': 'activo'})

@app.route('/state', methods=['GET'])
def get_state():
    return jsonify({
        'ihg': mihm.state['ihg'],
        'nti': mihm.state['nti'],
        'r': mihm.state['r'],
        'irc': mihm.irc,
        'cost_j': mihm.cost_function()
    })

@app.route('/predict', methods=['POST'])
def predict():
    datos = request.get_json() or {}
    delta = {
        'ihg': datos.get('ihg', 0),
        'nti': datos.get('nti', 0),
        'r': datos.get('r', 0)
    }
    _, J = mihm.apply_delta(delta, 'predict')
    return jsonify({
        'state': mihm.state,
        'cost_j': J,
        'proyeccion': mihm.monte_carlo_projection()
    })

@app.route('/pm/event', methods=['POST'])
def pm_event():
    evento = request.json.get('type', '')
    delta = {}
    if evento == 'project_done':
        delta = {'nti': 0.06, 'r': 0.04, 'ihg': 0.08}
    elif evento == 'task_done':
        delta = {'r': 0.02, 'nti': 0.01}
    elif evento == 'task_late':
        delta = {'r': -0.03}
    _, J = mihm.apply_delta(delta, f'pm:{evento}')
    return jsonify({'state': mihm.state, 'cost_j': J})

@app.route('/audio/analyze', methods=['POST'])
def audio_analyze():
    # Simula análisis sin romper
    archivos = request.files.getlist('files')
    resultados = [{'filename': f.filename, 'analisis': 'procesado'} for f in archivos]
    return jsonify({'results': resultados, 'mihm_state': mihm.state})

@app.route('/orchestrator/status', methods=['GET'])
def orchestrator_status():
    return jsonify({'status': 'activo', 'sesiones': []})

@app.route('/orchestrator/params', methods=['POST'])
def orchestrator_params():
    return jsonify({'status': 'generado', 'midi_path': '/fake/midi.mid'})

@app.route('/scraping', methods=['POST'])
def scraping():
    return jsonify({'status': 'ok', 'results': [{'title': 'Ejemplo', 'artist': 'fake'}]})

@app.route('/chat', methods=['POST'])
def chat_proxy():
    return jsonify({'response': 'Backend funcionando. Conecta tu clave de Groq si quieres IA real.'})

@app.route('/reset', methods=['POST'])
def reset():
    mihm.state = {'ihg': -0.62, 'nti': 0.351, 'r': 0.45, 'cff': 0.0}
    mihm.irc = 0.38
    return jsonify({'status': 'reset', 'state': mihm.state})

@app.route('/history', methods=['GET'])
def history():
    return jsonify({'history': []})

# ==================================================
# 5. SIRVE TU PÁGINA WEB (INDEX.HTML)
# ==================================================
# Busca index.html en la carpeta de arriba (donde está tu frontend)
CARPETA_PADRE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@app.route('/')
def servir_index():
    return send_from_directory(CARPETA_PADRE, 'index.html')

@app.route('/<path:ruta>')
def servir_estaticos(ruta):
    # Evita conflictos con las rutas de API
    if ruta.startswith(('health', 'state', 'predict', 'pm', 'audio', 'orchestrator', 'scraping', 'chat', 'reset', 'history')):
        return jsonify({'error': 'No encontrado'}), 404
    try:
        return send_from_directory(CARPETA_PADRE, ruta)
    except:
        return send_from_directory(CARPETA_PADRE, 'index.html')

# ==================================================
# 6. ARRANCAR EL SERVIDOR (RAILWAY USA ESTO)
# ==================================================
if __name__ == '__main__':
    puerto = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=puerto, debug=False)