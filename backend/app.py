import os
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np

from config import Config
from database import Database
from groq_client import GroqClient
from audio_features import extract_features
import mihm as mihm_

# Módulos del framework System Friction
from modules.social_analyzer import SocialAnalyzer
from modules.audio_analyzer_advanced import AudioAnalyzerAdvanced
from modules.scraping_spotify import SpotifyScraper
from modules.project_proposals import ProjectProposals
from modules.marketing_engine import MarketingEngine
from modules.project_manager import ProjectManager
from modules.ml_friction import MLFriction
from modules.integrations import Integrations
from modules.reflexive_engine import ReflexiveEngine

# ─────────────────────────────────────────────
# Inicialización central
# ─────────────────────────────────────────────

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

db     = Database('instance/friction.db')
groq   = GroqClient()
mihm   = mihm_.MIHM()

# Inyectar groq en mihm para propose_new_rule_via_groq
mihm._groq = groq

# Cargar parámetros guardados
saved_params = db.get_parameters('mihm_params')
if saved_params:
    mihm.params.update(saved_params)

# Instanciar todos los módulos con el mismo mihm (inyección de dependencia)
social    = SocialAnalyzer(mihm)
audio_adv = AudioAnalyzerAdvanced(mihm)
spotify   = SpotifyScraper(mihm)
proposals = ProjectProposals(mihm, groq)
marketing = MarketingEngine(mihm)
pm        = ProjectManager(mihm)
ml        = MLFriction(mihm)
integs    = Integrations(mihm)
reflexive = ReflexiveEngine(mihm)


def gen_id():
    return str(uuid.uuid4())


# ─────────────────────────────────────────────
# ENDPOINTS LEGACY (compatibilidad total)
# ─────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':  'ok',
        'version': '4.0-MIHM-MCM-A',
        'mihm':    'active',
        'state':   mihm.state,
        'cost_j':  mihm.cost_function(),
        'irc':     mihm.irc,
    })


@app.route('/predict', methods=['POST'])
def predict():
    mihm.process_delayed_updates()
    data = request.get_json()
    user = data.get('user', 'anon')
    text = data.get('text', '')

    ihg = data.get('ihg', mihm.state['ihg'])
    nti = data.get('nti', mihm.state['nti'])
    r   = data.get('r',   mihm.state['r'])

    # Aplicar mediante apply_delta (regla de oro)
    delta = {
        'ihg': ihg - mihm.state['ihg'],
        'nti': nti - mihm.state['nti'],
        'r':   r   - mihm.state['r'],
    }
    u, J = mihm.apply_delta(delta, action=f"predict:{user}")
    mihm.meta_control()

    proyeccion = mihm.monte_carlo_projection()
    stability  = mihm.stability_analysis()

    if mihm.state['ihg'] < -1.2:
        intervencion = "CRITICO: Hegemonía extrema. Se requiere redistribución de decisiones."
    elif mihm.state['ihg'] < -0.8:
        intervencion = "ALERTA: Alta concentración de poder. Revisar roles."
    elif mihm.state['ihg'] < -0.4:
        intervencion = "TENSION: Equilibrio inestable. Monitorear flujo de decisiones."
    else:
        intervencion = "ESTABLE: La homeostasis está dentro de rangos aceptables."

    pred_id = gen_id()
    db.save_prediction(pred_id, user, text, mihm.state)
    db.save_state(mihm.state, mihm.irc, f"predict:{user}", J)

    return jsonify({
        'prediction_id': pred_id,
        'state':         mihm.state,
        'intervencion':  intervencion,
        'control': {
            'u':  u,
            'kp': mihm.params['kp'],
            'ki': mihm.params['ki'],
            'kd': mihm.params['kd'],
        },
        'stability':  stability,
        'proyeccion': proyeccion,
        'cost_j':     J,
        'irc':        mihm.irc,
    })


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
            (error, error_smoothed, pred_id)
        )

    return jsonify({
        'error':         error,
        'error_smoothed': error_smoothed,
        'new_params':    new_params,
    })


@app.route('/history', methods=['GET'])
def history():
    limit = request.args.get('limit', 100, type=int)
    rows  = db.get_history(limit)
    return jsonify({'history': rows})


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
    data     = request.get_json()
    scenario = data.get('scenario')
    db.save_scenario(scenario)
    return jsonify({'status': 'registered'})


@app.route('/export', methods=['GET'])
def export():
    with db.get_connection() as conn:
        cur  = conn.execute('SELECT * FROM predictions ORDER BY timestamp')
        rows = cur.fetchall()
        data = [dict(row) for row in rows]
    return jsonify(data)


@app.route('/midi/generate', methods=['POST'])
def generate_midi_route():
    mihm.process_delayed_updates()
    data      = request.get_json()
    num_inst  = data.get('num_instruments', 4)
    phonemes  = data.get('phoneme_pattern', None)
    midi_file = mihm.generate_midi(num_inst, phonemes)
    return send_file(midi_file, as_attachment=True)


# ─────────────────────────────────────────────
# AUDIO
# ─────────────────────────────────────────────

@app.route('/audio/analyze', methods=['POST'])
def audio_analyze():
    mihm.process_delayed_updates()
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files uploaded'}), 400

    results = []
    for file in files:
        filename   = file.filename
        audio_bytes = file.read()
        features   = extract_features(audio_bytes, filename)

        # Análisis avanzado con acoplamiento al MIHM
        adv = audio_adv.analyze(features)

        try:
            analysis = groq.analyze_audio(features)
        except Exception:
            analysis = "No se pudo analizar con Groq."

        results.append({
            'filename':  filename,
            'features':  features,
            'analysis':  analysis,
            'advanced':  adv,
        })

    return jsonify({
        'results':   results,
        'mihm_state': mihm.state,
        'cost_j':    mihm.cost_function(),
        'irc':       mihm.irc,
    })


# ─────────────────────────────────────────────
# SOCIAL
# ─────────────────────────────────────────────

@app.route('/social/analyze', methods=['POST'])
def analyze_social():
    mihm.process_delayed_updates()
    data  = request.get_json()
    query = data.get('query', '')
    if not query:
        return jsonify({'error': 'Missing query'}), 400
    result = social.analyze_social(query)
    return jsonify(result)


# ─────────────────────────────────────────────
# SPOTIFY
# ─────────────────────────────────────────────

@app.route('/spotify/trends', methods=['GET'])
def spotify_trends():
    mihm.process_delayed_updates()
    genre = request.args.get('genre', 'reggaeton')
    limit = request.args.get('limit', 20, type=int)
    result = spotify.analyze_trends(genre, limit)
    return jsonify(result)


# ─────────────────────────────────────────────
# TIKTOK (legacy + nuevo acoplamiento)
# ─────────────────────────────────────────────

@app.route('/tiktok/scrape', methods=['GET'])
def tiktok_scrape():
    mihm.process_delayed_updates()
    query  = request.args.get('query', 'viral')
    result = social.analyze_social(query)
    return jsonify(result)


# ─────────────────────────────────────────────
# PROPUESTAS DE PROYECTOS
# ─────────────────────────────────────────────

@app.route('/projects/propose', methods=['POST'])
def projects_propose():
    mihm.process_delayed_updates()
    data    = request.get_json() or {}
    result  = proposals.generate(data)
    return jsonify(result)


# ─────────────────────────────────────────────
# MARKETING
# ─────────────────────────────────────────────

@app.route('/marketing/campaign', methods=['POST'])
def marketing_campaign():
    mihm.process_delayed_updates()
    data         = request.get_json()
    release_name = data.get('release_name', 'Untitled')
    budget       = float(data.get('budget', 1000.0))
    channels     = data.get('channels', ['tiktok', 'instagram', 'spotify'])
    result       = marketing.plan_campaign(release_name, budget, channels)
    return jsonify(result)


# ─────────────────────────────────────────────
# PROJECT MANAGER
# ─────────────────────────────────────────────

@app.route('/pm/project', methods=['POST'])
def pm_create():
    mihm.process_delayed_updates()
    data         = request.get_json()
    name         = data.get('name', 'Proyecto sin nombre')
    members      = data.get('members', [])
    deadline     = int(data.get('deadline_days', 30))
    result       = pm.create_project(name, members, deadline)
    return jsonify(result)


@app.route('/pm/task', methods=['POST'])
def pm_task():
    mihm.process_delayed_updates()
    data       = request.get_json()
    project_id = data.get('project_id', '')
    task       = data.get('task', '')
    done       = bool(data.get('done', False))
    result     = pm.update_task(project_id, task, done)
    return jsonify(result)


@app.route('/pm/projects', methods=['GET'])
def pm_list():
    return jsonify({'projects': pm.list_projects()})


# ─────────────────────────────────────────────
# ML FRICTION
# ─────────────────────────────────────────────

@app.route('/ml/predict', methods=['POST'])
def ml_predict():
    mihm.process_delayed_updates()
    data     = request.get_json()
    features = data.get('features', {})
    result   = ml.predict_success(features)
    return jsonify(result)


@app.route('/ml/train', methods=['POST'])
def ml_train():
    mihm.process_delayed_updates()
    data         = request.get_json()
    features     = data.get('features', {})
    true_outcome = float(data.get('true_outcome', 0.5))
    result       = ml.train(features, true_outcome)
    return jsonify(result)


# ─────────────────────────────────────────────
# INTEGRATIONS
# ─────────────────────────────────────────────

@app.route('/integrations/youtube', methods=['POST'])
def int_youtube():
    mihm.process_delayed_updates()
    data     = request.get_json()
    video_id = data.get('video_id', '')
    metrics  = data.get('metrics', {})
    result   = integs.ingest_youtube_analytics(video_id, metrics)
    return jsonify(result)


@app.route('/integrations/soundcloud', methods=['POST'])
def int_soundcloud():
    mihm.process_delayed_updates()
    data     = request.get_json()
    track_id = data.get('track_id', '')
    plays    = int(data.get('plays', 0))
    reposts  = int(data.get('reposts', 0))
    result   = integs.ingest_soundcloud(track_id, plays, reposts)
    return jsonify(result)


@app.route('/integrations/generic', methods=['POST'])
def int_generic():
    mihm.process_delayed_updates()
    data        = request.get_json()
    platform    = data.get('platform', 'unknown')
    signal_name = data.get('signal_name', 'generic')
    value       = float(data.get('value', 0.5))
    weight      = float(data.get('weight', 0.1))
    result      = integs.ingest_generic(platform, signal_name, value, weight)
    return jsonify(result)


# ─────────────────────────────────────────────
# SISTEMA REFLEXIVO (MCM-A)
# ─────────────────────────────────────────────

@app.route('/system/health', methods=['GET'])
def system_health():
    """Estado reflexivo completo del sistema."""
    mihm.process_delayed_updates()
    mihm.meta_control()
    result = reflexive.evaluate_system_health()
    # Persistir en DB
    db.save_state(mihm.state, mihm.irc, "system_health_check", mihm.cost_function())
    return jsonify(result)


@app.route('/system/meta_control', methods=['POST'])
def force_meta_control():
    """Fuerza meta_control independientemente del contador de historial."""
    mihm.process_delayed_updates()
    result = reflexive.force_meta_control()
    db.save_state(mihm.state, mihm.irc, "forced_meta_control", mihm.cost_function())
    return jsonify(result)


@app.route('/system/history', methods=['GET'])
def state_history():
    """Historial de estados del sistema."""
    limit  = request.args.get('limit', 200, type=int)
    rows   = db.get_state_history(limit)
    return jsonify({'state_history': rows, 'count': len(rows)})


@app.route('/system/rules', methods=['GET'])
def reflexive_rules():
    """Reglas auto-descubiertas por el sistema."""
    rows = db.get_reflexive_rules(50)
    return jsonify({
        'rules':      rows,
        'live_rules': mihm.reflexive_rules[-10:],
    })


@app.route('/system/state', methods=['GET'])
def system_state():
    """Estado completo del MIHM en tiempo real."""
    mihm.process_delayed_updates()
    return jsonify({
        'state':           mihm.state,
        'irc':             mihm.irc,
        'meta_j':          mihm.compute_meta_j(),
        'cost_j':          mihm.cost_function(),
        'params':          mihm.params,
        'history_size':    len(mihm.history),
        'delayed_queue':   len(mihm.delayed_updates),
        'reflexive_rules': len(mihm.reflexive_rules),
        'timestamp':       datetime.utcnow().isoformat(),
    })


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)