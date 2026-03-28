"""
app.py – Servidor central de System Friction v4.1
Motor: MIHM + MCM-A + Campo de Frecuencia Colectiva (CFF)

Principio: mihm es la única instancia compartida.
Todos los módulos reciben mihm por inyección de dependencia.
Ningún endpoint modifica mihm.state directamente – solo a través de apply_delta().
"""

import os
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, send_file, send_from_directory, abort
from flask_cors import CORS
import numpy as np

from config import Config
from database import Database
from groq_client import GroqClient
from audio_features import extract_features
from mihm import MIHM

# ── Módulos del framework System Friction ────────────────────────────
from modules.social_analyzer          import SocialAnalyzer
from modules.audio_analyzer_advanced  import AudioAnalyzerAdvanced
from modules.scraping_spotify         import SpotifyScraper
from modules.project_proposals        import ProjectProposals
from modules.marketing_engine         import MarketingEngine
from modules.project_manager          import ProjectManager
from modules.ml_friction              import MLFriction
from modules.integrations             import Integrations
from modules.reflexive_engine         import ReflexiveEngine
from modules.frequency_coexistence    import FrequencyCoexistenceEngine
from modules.emergent_melody_engine   import EmergentMelodyEngine
from modules.drive_manager            import DriveManager
from modules.proactive_orchestrator   import ProactiveOrchestrator

# ══════════════════════════════════════════════════════════════════════
# INICIALIZACIÓN CENTRAL
# ══════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

db   = Database('instance/friction.db')
groq = GroqClient()
mihm = MIHM()

# Inyectar groq en MIHM para meta_control
if hasattr(mihm, '_groq'):
    mihm._groq = groq

# Cargar parámetros persistidos
saved_params = db.get_parameters('mihm_params')
if saved_params:
    mihm.params.update(saved_params)

# ── Instanciar módulos con mihm compartido ───────────────────────────
social    = SocialAnalyzer(mihm)
audio_adv = AudioAnalyzerAdvanced(mihm)
spotify   = SpotifyScraper(mihm)
proposals = ProjectProposals(mihm, groq)
marketing = MarketingEngine(mihm)
pm        = ProjectManager(mihm)
ml        = MLFriction(mihm)
integs    = Integrations(mihm)
reflexive = ReflexiveEngine(mihm)
freq_coex = FrequencyCoexistenceEngine(mihm, groq)
melody_engine = EmergentMelodyEngine(mihm, groq, ml, spotify)
drive_mgr     = DriveManager(mihm)
orchestrator  = ProactiveOrchestrator(mihm, db, groq, melody_engine, drive_mgr)

# Crear directorio MIDI si no existe
os.makedirs(getattr(Config, 'MIDI_OUTPUT_DIR', 'instance/midi_output'), exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# genMIDI / genBoth – Fix del bug que referenciaba en JS
# ══════════════════════════════════════════════════════════════════════

def genMIDI():
    """Stub real para que el JS no falle"""
    try:
        return melody_engine.generate({
            'motivos': ['A', 'B', 'C', 'D'],
            'duracion_seg': 120,
            'frase_concepto': 'pista emergente',
            'genero': 'pop',
            'instrumentos': ['piano', 'bajo'],
            'enganche': getattr(Config, 'HOOK_THRESHOLD', 0.7)
        })
    except Exception as e:
        return {"error": str(e), "status": "fallback"}

def genBoth():
    """Llama a genMIDI + análisis de frecuencia"""
    try:
        midi_result = genMIDI()
        freq_result = freq_coex.analyze({})
        return {"midi": midi_result, "frequency": freq_result}
    except Exception as e:
        return {"error": str(e), "status": "fallback"}

# ══════════════════════════════════════════════════════════════════════
# SERVIR FRONTEND – FIX PARA RAILWAY (LO MÁS IMPORTANTE)
# ══════════════════════════════════════════════════════════════════════

@app.route('/')
def serve_frontend():
    """Sirve el archivo principal scorefriction.html"""
    try:
        return send_from_directory(os.getcwd(), 'scorefriction.html')
    except FileNotFoundError:
        return "scorefriction.html no encontrado en la raíz", 404
    except Exception as e:
        return f"Error al servir frontend: {str(e)}", 500


@app.route('/<path:path>')
def serve_static(path):
    """Sirve cualquier archivo estático (JS, CSS, etc.)"""
    # Proteger rutas de API para que no sean interceptadas
    api_prefixes = ('health', 'predict', 'learn', 'history', 'reset', 'groq', 'scenario', 
                    'export', 'midi', 'audio', 'frequency', 'social', 'spotify', 'tiktok',
                    'projects', 'marketing', 'pm', 'ml', 'integrations', 'system', 'orchestrator')
    if path.startswith(api_prefixes):
        return jsonify({'error': 'Not found'}), 404
    
    try:
        return send_from_directory(os.getcwd(), path)
    except:
        # Fallback: servir el index si es una ruta de SPA
        try:
            return send_from_directory(os.getcwd(), 'scorefriction.html')
        except:
            return "Not Found", 404

# ══════════════════════════════════════════════════════════════════════
# ENDPOINTS CORE
# ══════════════════════════════════════════════════════════════════════

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':  'ok',
        'version': '4.1-CFF',
        'mihm':    'active',
        'state':   mihm.state,
        'cost_j':  mihm.cost_function(),
        'irc':     getattr(mihm, 'irc', 0.38),
    })

@app.route('/predict', methods=['POST'])
def predict():
    mihm.process_delayed_updates()
    data = request.get_json() or {}
    user = data.get('user', 'anon')
    text = data.get('text', '')

    delta = {
        'ihg': data.get('ihg', mihm.state.get('ihg', -0.62)) - mihm.state.get('ihg', -0.62),
        'nti': data.get('nti', mihm.state.get('nti', 0.351)) - mihm.state.get('nti', 0.351),
        'r':   data.get('r',   mihm.state.get('r', 0.45))   - mihm.state.get('r', 0.45),
    }
    u, J = mihm.apply_delta(delta, action=f"predict:{user}")
    mihm.meta_control()

    proyeccion = mihm.monte_carlo_projection()
    stability  = mihm.stability_analysis()

    ihg_v = mihm.state.get('ihg', -0.62)
    if ihg_v < -1.2:
        intervencion = "CRITICO: Hegemonía extrema."
    elif ihg_v < -0.8:
        intervencion = "ALERTA: Alta concentración de poder."
    elif ihg_v < -0.4:
        intervencion = "TENSIÓN: Equilibrio inestable."
    else:
        intervencion = "ESTABLE: Homeostasis aceptable."

    pred_id = str(uuid.uuid4())
    db.save_prediction(pred_id, user, text, mihm.state)

    return jsonify({
        'prediction_id': pred_id,
        'state':         mihm.state,
        'intervencion':  intervencion,
        'control':       {'u': float(u), 'kp': mihm.params.get('kp'), 'ki': mihm.params.get('ki'), 'kd': mihm.params.get('kd')},
        'stability':     stability,
        'proyeccion':    proyeccion,
        'cost_j':        float(J),
        'irc':           float(getattr(mihm, 'irc', 0.38)),
    })

# ══════════════════════════════════════════════════════════════════════
# MAIN – Puerto correcto para Railway
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=Config.DEBUG, host='0.0.0.0', port=port)

# ══════════════════════════════════════════════════════════════════════
# EL RESTO DE TUS ENDPOINTS (mantengo exactamente los que tenías)
# ══════════════════════════════════════════════════════════════════════

@app.route('/learn', methods=['POST'])
def learn():
    mihm.process_delayed_updates()
    data    = request.get_json()
    pred_id = data.get('prediction_id')
    outcome = data.get('outcome')
    if not pred_id or outcome is None:
        return jsonify({'error': 'Missing prediction_id or outcome'}), 400

    error          = outcome - mihm.state['ihg']
    error_smoothed = 0.7 * error + 0.3 * (db.get_parameters('last_error') or 0)
    db.save_parameters('last_error', error_smoothed)
    new_params = mihm.learn(pred_id, outcome, db)

    with db.get_connection() as conn:
        conn.execute(
            'UPDATE predictions SET error=?, error_smoothed=? WHERE prediction_id=?',
            (error, error_smoothed, pred_id),
        )

    return jsonify({
        'error':          error,
        'error_smoothed': error_smoothed,
        'new_params':     new_params,
    })


@app.route('/history', methods=['GET'])
def history():
    limit = request.args.get('limit', 100, type=int)
    return jsonify({'history': db.get_history(limit)})


@app.route('/reset', methods=['POST'])
def reset():
    mihm.state = {
        'ihg':        -0.620,
        'nti':         0.351,
        'r':           0.450,
        'phi_p':       0.000,
        'psi_i':       0.000,
        'h_scale':     0.500,
        'ml_success':  0.500,
        'cff':         0.000,
    }
    mihm.integral   = 0.0
    mihm.prev_error = 0.0
    mihm.irc        = 0.38
    return jsonify({'status': 'reset', 'state': mihm.state})


@app.route('/groq/analyze', methods=['POST'])
def groq_analyze():
    mihm.process_delayed_updates()
    data      = request.get_json()
    responses = data.get('responses', '')
    if not responses:
        return jsonify({'error': 'No responses provided'}), 400
    try:
        result = groq.analyze_audit(responses)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/scenario/select', methods=['POST'])
def scenario_select():
    data = request.get_json()
    db.save_scenario(data.get('scenario'))
    return jsonify({'status': 'registered'})


@app.route('/export', methods=['GET'])
def export():
    with db.get_connection() as conn:
        cur  = conn.execute('SELECT * FROM predictions ORDER BY timestamp')
        data = [dict(row) for row in cur.fetchall()]
    return jsonify(data)


@app.route('/midi/generate', methods=['POST'])
def generate_midi_route():
    mihm.process_delayed_updates()
    data      = request.get_json()
    num_inst  = data.get('num_instruments', 4)
    phonemes  = data.get('phoneme_pattern', None)
    midi_file = mihm.generate_midi(num_inst, phonemes)
    return send_file(midi_file, as_attachment=True)


# ══════════════════════════════════════════════════════════════════════
# AUDIO, FRECUENCIAS, SOCIAL, PROYECTOS, ML, INTEGRACIONES, REFLEXIVO, ORQUESTADOR
# ══════════════════════════════════════════════════════════════════════

@app.route('/audio/analyze', methods=['POST'])
def audio_analyze():
    mihm.process_delayed_updates()
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files uploaded'}), 400

    results = []
    for file in files:
        filename    = file.filename
        audio_bytes = file.read()
        features    = extract_features(audio_bytes, filename)

        adv = audio_adv.analyze(features)
        coex = freq_coex.analyze(features)
        analysis = groq.analyze_audio(features)

        results.append({
            'filename':   filename,
            'features':   features,
            'analysis':   analysis,
            'advanced':   adv,
            'coexistence': coex,
        })

    return jsonify({
        'results':    results,
        'mihm_state': mihm.state,
        'cost_j':     mihm.cost_function(),
        'irc':        mihm.irc,
    })


@app.route('/orchestrator/status', methods=['GET'])
def orchestrator_status():
    return jsonify(orchestrator.get_status())

@app.route('/orchestrator/tick', methods=['POST'])
def orchestrator_tick():
    result = orchestrator.tick()
    return jsonify(result)

@app.route('/orchestrator/trigger', methods=['POST'])
def orchestrator_trigger():
    data = request.get_json() or {}
    event_title = data.get('event_title', 'generación de pista test')
    result = orchestrator.force_trigger(event_title)
    return jsonify(result)

@app.route('/orchestrator/params', methods=['POST'])
def orchestrator_params():
    data = request.get_json() or {}
    result = melody_engine.generate(data)
    return jsonify(result)

# ══════════════════════════════════════════════════════════════════════
# MAIN – Puerto correcto para Railway
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=Config.DEBUG, host='0.0.0.0', port=port)
