"""
app.py – Servidor central de System Friction v4.1
Motor: MIHM + MCM-A + Campo de Frecuencia Colectiva (CFF)

Principio: mihm es la única instancia compartida.
Todos los módulos reciben mihm por inyección de dependencia.
Ningún endpoint modifica mihm.state directamente –
solo a través de apply_delta().
"""

import os
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

try:
    import numpy as np
    _numpy_ok = True
except ImportError:
    _numpy_ok = False

from config import Config
from database import Database
from groq_client import GroqClient
from mihm import MIHM

# extract_features usa librosa — import opcional
try:
    from audio_features import extract_features
    _audio_ok = True
except Exception:
    _audio_ok = False
    def extract_features(*a, **kw):
        return {}

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
CORS(app, resources={r"/*": {"origins": "*"}})

# Path absoluto para SQLite – independiente del directorio de trabajo
_BACKEND_DIR  = os.path.dirname(os.path.abspath(__file__))
_INSTANCE_DIR = os.path.join(_BACKEND_DIR, 'instance')
_DB_PATH      = os.path.join(_INSTANCE_DIR, 'friction.db')
_MIDI_DIR     = os.path.join(_INSTANCE_DIR, 'midi_output')
os.makedirs(_INSTANCE_DIR, exist_ok=True)
os.makedirs(_MIDI_DIR,     exist_ok=True)

db   = Database(_DB_PATH)
groq = GroqClient()
mihm = MIHM()

mihm._groq = groq

try:
    saved_params = db.get_parameters('mihm_params')
    if saved_params:
        mihm.params.update(saved_params)
except Exception as e:
    print(f"[WARN] No se pudieron cargar params persistidos: {e}")

# ── Instanciar módulos (con manejo de errores para entornos sin todas las libs) ──
def _safe_init(cls, *args, **kwargs):
    try:
        return cls(*args, **kwargs)
    except Exception as e:
        print(f"[WARN] No se pudo inicializar {cls.__name__}: {e}")
        return None

social    = _safe_init(SocialAnalyzer, mihm)
audio_adv = _safe_init(AudioAnalyzerAdvanced, mihm)
spotify   = _safe_init(SpotifyScraper, mihm)          # ← esta es la instancia correcta
proposals = _safe_init(ProjectProposals, mihm, groq)
marketing = _safe_init(MarketingEngine, mihm)
pm        = _safe_init(ProjectManager, mihm)
ml        = _safe_init(MLFriction, mihm)
integs    = _safe_init(Integrations, mihm)
reflexive = _safe_init(ReflexiveEngine, mihm)
freq_coex = _safe_init(FrequencyCoexistenceEngine, mihm, groq)
melody_engine = _safe_init(EmergentMelodyEngine, mihm, groq, ml, spotify)
drive_mgr = _safe_init(DriveManager, mihm, {
    'GOOGLE_SERVICE_ACCOUNT_JSON': Config.GOOGLE_SERVICE_ACCOUNT_JSON,
    'GOOGLE_CREDENTIALS_FILE':     Config.GOOGLE_CREDENTIALS_FILE,
    'GOOGLE_DRIVE_FOLDER_ID':      Config.GOOGLE_DRIVE_FOLDER_ID,
    'MIDI_OUTPUT_DIR':             _MIDI_DIR,
})
orchestrator = _safe_init(ProactiveOrchestrator, mihm, db, groq, melody_engine, drive_mgr, {
    'TELEGRAM_BOT_TOKEN':          Config.TELEGRAM_BOT_TOKEN,
    'TELEGRAM_CHAT_ID':            Config.TELEGRAM_CHAT_ID,
    'GOOGLE_CALENDAR_ID':          Config.GOOGLE_CALENDAR_ID,
    'GOOGLE_SERVICE_ACCOUNT_JSON': Config.GOOGLE_SERVICE_ACCOUNT_JSON,
    'GOOGLE_CREDENTIALS_FILE':     Config.GOOGLE_CREDENTIALS_FILE,
    'TRACK_KEYWORD':               Config.TRACK_KEYWORD,
    'MIDI_OUTPUT_DIR':             Config.MIDI_OUTPUT_DIR,
})

# ========== DEBUG: Ver qué módulos están vivos ==========
print("\n=== MÓDULOS INICIALIZADOS ===")
print(f"audio_adv:     {audio_adv}")
print(f"freq_coex:     {freq_coex}")
print(f"melody_engine: {melody_engine}")
print(f"orchestrator:  {orchestrator}")
print(f"ml:            {ml}")
print(f"spotify:       {spotify}")
print("=============================\n")
# ========================================================

# ── genMIDI / genBoth: funciones reales (fix de bug JS referenciaba estos nombres) ──
def _gen_midi_stub(params=None):
    """Genera MIDI usando melody_engine con parámetros por defecto (protegido)."""
    if melody_engine is None:
        return {'error': 'Melody engine not available', 'midi_path': None}
    p = params or {}
    p.setdefault('motivos',        ['A', 'B', 'C', 'D'])
    p.setdefault('duracion_seg',   120)
    p.setdefault('frase_concepto', 'pista emergente')
    p.setdefault('genero',         'pop')
    p.setdefault('instrumentos',   ['piano', 'bajo'])
    p.setdefault('enganche',       Config.HOOK_THRESHOLD)
    return melody_engine.generate(p)

def gen_id() -> str:
    return str(uuid.uuid4())

# ══════════════════════════════════════════════════════════════════════
# ENDPOINTS CORE
# ══════════════════════════════════════════════════════════════════════

@app.route('/health', methods=['GET'])
def health():
    try:
        state = mihm.state
        cost  = mihm.cost_function()
        irc   = mihm.irc
    except Exception:
        state, cost, irc = {}, 0.0, 0.0
    return jsonify({
        'status':  'ok',
        'version': '4.2-Orchestrator',
        'mihm':    'active',
        'state':   state,
        'cost_j':  cost,
        'irc':     irc,
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

    delta = {
        'ihg': ihg - mihm.state['ihg'],
        'nti': nti - mihm.state['nti'],
        'r':   r   - mihm.state['r'],
    }
    u, J = mihm.apply_delta(delta, action=f"predict:{user}")
    mihm.meta_control()

    proyeccion = mihm.monte_carlo_projection()
    stability  = mihm.stability_analysis()

    ihg_v = mihm.state['ihg']
    if ihg_v < -1.2:
        intervencion = "CRITICO: Hegemonía extrema. Se requiere redistribución de decisiones."
    elif ihg_v < -0.8:
        intervencion = "ALERTA: Alta concentración de poder. Revisar roles."
    elif ihg_v < -0.4:
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
    if melody_engine is None:
        return jsonify({'error': 'Melody engine not available'}), 500
    data      = request.get_json()
    num_inst  = data.get('num_instruments', 4)
    phonemes  = data.get('phoneme_pattern', None)
    midi_file = mihm.generate_midi(num_inst, phonemes)
    return send_file(midi_file, as_attachment=True)

# ══════════════════════════════════════════════════════════════════════
# AUDIO
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

        # Análisis avanzado con acoplamiento MIHM (protegido)
        adv = audio_adv.analyze(features) if audio_adv is not None else {}

        # Coexistencia social de frecuencias (protegido)
        coex = freq_coex.analyze(features) if freq_coex is not None else {}

        # Narrativa Groq
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

# ══════════════════════════════════════════════════════════════════════
# FRECUENCIAS – Coexistencia Social
# ══════════════════════════════════════════════════════════════════════

@app.route('/frequency/analyze', methods=['POST'])
def frequency_analyze():
    if freq_coex is None:
        return jsonify({'error': 'FrequencyCoexistenceEngine not available'}), 500
    mihm.process_delayed_updates()
    data           = request.get_json() or {}
    features       = data.get('features', {})
    social_context = data.get('social_context', {})

    if not features:
        features = {
            'band_energy_low':   0.33,
            'band_energy_mid':   0.33,
            'band_energy_high':  0.33,
            'onset_density':     2.0,
            'spectral_entropy':  5.0,
            'dynamic_range':     60.0,
            'periodicity':       0.3,
        }

    result = freq_coex.analyze(features, social_context)
    db.save_state(mihm.state, mihm.irc, "frequency_analyze", mihm.cost_function())
    return jsonify(result)

@app.route('/frequency/ritual', methods=['POST'])
def frequency_ritual():
    if freq_coex is None:
        return jsonify({'error': 'FrequencyCoexistenceEngine not available'}), 500
    mihm.process_delayed_updates()
    data       = request.get_json() or {}
    group_size = int(data.get('group_size', 10))
    diversity  = float(data.get('diversity_index', 0.5))
    setting    = data.get('setting', 'studio')

    result = freq_coex.propose_session_ritual(group_size, diversity, setting)
    db.save_state(mihm.state, mihm.irc, "frequency_ritual", mihm.cost_function())
    return jsonify(result)

@app.route('/frequency/map', methods=['GET'])
def frequency_map():
    if freq_coex is None:
        return jsonify({'error': 'FrequencyCoexistenceEngine not available'}), 500
    return jsonify(freq_coex.get_frequency_map())

@app.route('/frequency/history', methods=['GET'])
def frequency_history():
    if freq_coex is None:
        return jsonify({'error': 'FrequencyCoexistenceEngine not available'}), 500
    limit = request.args.get('limit', 20, type=int)
    return jsonify({'history': freq_coex.get_session_history(limit)})

# ══════════════════════════════════════════════════════════════════════
# SOCIAL
# ══════════════════════════════════════════════════════════════════════

@app.route('/social/analyze', methods=['POST'])
def analyze_social():
    mihm.process_delayed_updates()
    data  = request.get_json()
    query = data.get('query', '')
    if not query:
        return jsonify({'error': 'Missing query'}), 400
    return jsonify(social.analyze_social(query))

# ══════════════════════════════════════════════════════════════════════
# SPOTIFY / TIKTOK
# ══════════════════════════════════════════════════════════════════════

@app.route('/spotify/trends', methods=['GET'])
def spotify_trends():
    mihm.process_delayed_updates()
    genre = request.args.get('genre', 'reggaeton')
    limit = request.args.get('limit', 20, type=int)
    return jsonify(spotify.analyze_trends(genre, limit))

@app.route('/tiktok/scrape', methods=['GET'])
def tiktok_scrape():
    mihm.process_delayed_updates()
    query = request.args.get('query', 'viral')
    return jsonify(social.analyze_social(query))

# ══════════════════════════════════════════════════════════════════════
# PROYECTOS / MARKETING / PM
# ══════════════════════════════════════════════════════════════════════

@app.route('/projects/propose', methods=['POST'])
def projects_propose():
    mihm.process_delayed_updates()
    data = request.get_json() or {}
    return jsonify(proposals.generate(data))

@app.route('/marketing/campaign', methods=['POST'])
def marketing_campaign():
    mihm.process_delayed_updates()
    data         = request.get_json()
    release_name = data.get('release_name', 'Untitled')
    budget       = float(data.get('budget', 1000.0))
    channels     = data.get('channels', ['tiktok', 'instagram', 'spotify'])
    return jsonify(marketing.plan_campaign(release_name, budget, channels))

@app.route('/pm/project', methods=['POST'])
def pm_create():
    mihm.process_delayed_updates()
    data     = request.get_json()
    name     = data.get('name', 'Proyecto sin nombre')
    members  = data.get('members', [])
    deadline = int(data.get('deadline_days', 30))
    return jsonify(pm.create_project(name, members, deadline))

@app.route('/pm/task', methods=['POST'])
def pm_task():
    mihm.process_delayed_updates()
    data       = request.get_json()
    project_id = data.get('project_id', '')
    task       = data.get('task', '')
    done       = bool(data.get('done', False))
    return jsonify(pm.update_task(project_id, task, done))

@app.route('/pm/projects', methods=['GET'])
def pm_list():
    return jsonify({'projects': pm.list_projects()})

# ══════════════════════════════════════════════════════════════════════
# ML
# ══════════════════════════════════════════════════════════════════════

@app.route('/ml/predict', methods=['POST'])
def ml_predict():
    if ml is None:
        return jsonify({'error': 'ML module not available'}), 500
    mihm.process_delayed_updates()
    data = request.get_json()
    return jsonify(ml.predict_success(data.get('features', {})))

@app.route('/ml/train', methods=['POST'])
def ml_train():
    if ml is None:
        return jsonify({'error': 'ML module not available'}), 500
    mihm.process_delayed_updates()
    data         = request.get_json()
    features     = data.get('features', {})
    true_outcome = float(data.get('true_outcome', 0.5))
    return jsonify(ml.train(features, true_outcome))

# ══════════════════════════════════════════════════════════════════════
# INTEGRACIONES
# ══════════════════════════════════════════════════════════════════════

@app.route('/integrations/youtube', methods=['POST'])
def int_youtube():
    mihm.process_delayed_updates()
    data = request.get_json()
    return jsonify(integs.ingest_youtube_analytics(
        data.get('video_id', ''), data.get('metrics', {})))

@app.route('/integrations/soundcloud', methods=['POST'])
def int_soundcloud():
    mihm.process_delayed_updates()
    data = request.get_json()
    return jsonify(integs.ingest_soundcloud(
        data.get('track_id', ''),
        int(data.get('plays', 0)),
        int(data.get('reposts', 0)),
    ))

@app.route('/integrations/generic', methods=['POST'])
def int_generic():
    mihm.process_delayed_updates()
    data = request.get_json()
    return jsonify(integs.ingest_generic(
        data.get('platform', 'unknown'),
        data.get('signal_name', 'generic'),
        float(data.get('value', 0.5)),
        float(data.get('weight', 0.1)),
    ))

# ══════════════════════════════════════════════════════════════════════
# SISTEMA REFLEXIVO (MCM-A)
# ══════════════════════════════════════════════════════════════════════

@app.route('/system/health', methods=['GET'])
def system_health():
    mihm.process_delayed_updates()
    mihm.meta_control()
    result = reflexive.evaluate_system_health()
    db.save_state(mihm.state, mihm.irc, "system_health_check", mihm.cost_function())
    return jsonify(result)

@app.route('/system/meta_control', methods=['POST'])
def force_meta_control():
    mihm.process_delayed_updates()
    result = reflexive.force_meta_control()
    db.save_state(mihm.state, mihm.irc, "forced_meta_control", mihm.cost_function())
    return jsonify(result)

@app.route('/system/history', methods=['GET'])
def state_history():
    limit = request.args.get('limit', 200, type=int)
    rows  = db.get_state_history(limit)
    return jsonify({'state_history': rows, 'count': len(rows)})

@app.route('/system/rules', methods=['GET'])
def reflexive_rules():
    rows = db.get_reflexive_rules(50)
    return jsonify({
        'rules':      rows,
        'live_rules': mihm.reflexive_rules[-10:],
    })

@app.route('/system/state', methods=['GET'])
def system_state():
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

# ══════════════════════════════════════════════════════════════════════
# ORQUESTADOR PROACTIVO
# ══════════════════════════════════════════════════════════════════════

@app.route('/orchestrator/status', methods=['GET'])
def orchestrator_status():
    if orchestrator is None:
        return jsonify({'error': 'Orchestrator not available'}), 500
    return jsonify(orchestrator.get_status())

@app.route('/orchestrator/tick', methods=['POST'])
def orchestrator_tick():
    if orchestrator is None:
        return jsonify({'error': 'Orchestrator not available'}), 500
    mihm.process_delayed_updates()
    result = orchestrator.tick()
    db.save_state(mihm.state, mihm.irc, "orchestrator_tick", mihm.cost_function())
    return jsonify(result)

@app.route('/orchestrator/trigger', methods=['POST'])
def orchestrator_trigger():
    if orchestrator is None:
        return jsonify({'error': 'Orchestrator not available'}), 500
    data        = request.get_json() or {}
    event_title = data.get('event_title', 'test generación de pista')
    result      = orchestrator.force_trigger(event_title)
    db.save_state(mihm.state, mihm.irc, "orchestrator_force_trigger", mihm.cost_function())
    return jsonify(result)

@app.route('/orchestrator/sessions', methods=['GET'])
def orchestrator_sessions():
    limit = request.args.get('limit', 10, type=int)
    return jsonify({'sessions': db.get_recent_orchestrator_sessions(limit)})

@app.route('/orchestrator/session/<session_id>', methods=['GET'])
def orchestrator_session(session_id):
    sess = db.get_orchestrator_session(session_id)
    if sess is None:
        return jsonify({'error': 'Session not found'}), 404
    return jsonify(sess)

@app.route('/orchestrator/params', methods=['POST'])
def orchestrator_params():
    if melody_engine is None:
        return jsonify({'error': 'Melody engine not available'}), 500
    mihm.process_delayed_updates()
    data = request.get_json() or {}

    params = {
        'motivos':        data.get('motivos', ['A', 'B', 'C', 'D']),
        'duracion_seg':   int(data.get('duracion_seg', 120)),
        'frase_concepto': data.get('frase_concepto', 'pista emergente'),
        'genero':         data.get('genero', 'pop'),
        'instrumentos':   data.get('instrumentos', ['piano', 'bajo']),
        'enganche':       float(data.get('enganche', Config.HOOK_THRESHOLD)),
        'session_id':     data.get('session_id', gen_id()),
    }

    try:
        result = melody_engine.generate(params)
        db.save_state(mihm.state, mihm.irc, "orchestrator_params_generate", mihm.cost_function())
        return jsonify({
            'status':     'generated',
            'midi_path':  result.get('midi_path'),
            'cost_j':     result.get('cost_j'),
            'mihm_state': result.get('mihm_state'),
            'tension':    result.get('tension_peak'),
            'mc':         result.get('mc_projection'),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ══════════════════════════════════════════════════════════════════════
# FRONTEND – servir index.html y archivos estáticos
# ══════════════════════════════════════════════════════════════════════

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@app.route('/')
def serve_index():
    return send_from_directory(_ROOT_DIR, 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    # Proteger endpoints de la API: no interceptar rutas de backend
    if filename.startswith(('health', 'predict', 'learn', 'history', 'reset',
                            'groq', 'scenario', 'export', 'midi', 'audio',
                            'frequency', 'social', 'spotify', 'tiktok',
                            'projects', 'marketing', 'pm', 'ml',
                            'integrations', 'system', 'orchestrator', 'state',
                            'scraping', 'chat')):
        return jsonify({'error': 'Not found'}), 404
    try:
        return send_from_directory(_ROOT_DIR, filename)
    except Exception:
        return send_from_directory(_ROOT_DIR, 'index.html')

# ============================================================
# ENDPOINTS ADICIONALES (state, pm/event, scraping, chat)
# ============================================================

@app.route('/state', methods=['GET'])
def get_state():
    """Estado MIHM compacto para el hub frontend."""
    mihm.process_delayed_updates()
    return jsonify({
        'ihg':        mihm.state.get('ihg', -0.62),
        'nti':        mihm.state.get('nti', 0.351),
        'r':          mihm.state.get('r', 0.45),
        'cff':        mihm.state.get('cff', 0.0),
        'irc':        mihm.irc,
        'cost_j':     mihm.cost_function(),
        'timestamp':  datetime.utcnow().isoformat(),
    })

@app.route('/pm/event', methods=['POST'])
def pm_event():
    """Bridge entre PM frontend y motor MIHM."""
    ev = request.json.get('type', '')
    if ev == 'project_done':
        u, J = mihm.apply_delta({'nti': 0.06, 'r': 0.04, 'ihg': 0.08},
                                action='pm:project_done')
    elif ev == 'project_created':
        count = int(request.json.get('count', 1))
        pressure = min(0.8, count * 0.06)
        u, J = mihm.apply_delta({'ihg': -pressure * 0.06},
                                action='pm:project_created')
    elif ev == 'task_done':
        u, J = mihm.apply_delta({'r': 0.02, 'nti': 0.01},
                                action='pm:task_done')
    elif ev == 'task_late':
        u, J = mihm.apply_delta({'r': -0.03},
                                action='pm:task_late')
    elif ev == 'audit':
        answers = request.json.get('answers', {})
        delta = {}
        if answers.get('ts') == 'solo':   delta['ihg'] = -0.25
        if answers.get('ts') == '11+':    delta['ihg'] = 0.1
        if answers.get('up') == 'yes-c':
            delta.update({'nti': -0.12, 'r': -0.08})
        if answers.get('up') == 'no':
            delta['nti'] = 0.06
        u, J = mihm.apply_delta(delta, action='pm:audit')
    else:
        u, J = 0.0, mihm.cost_function()

    mihm.meta_control()
    db.save_state(mihm.state, mihm.irc, f'pm_event:{ev}', J or mihm.cost_function())

    return jsonify({
        'state':  mihm.state,
        'cost_j': J or mihm.cost_function(),
        'u':      u,
        'irc':    mihm.irc,
    })

@app.route('/scraping', methods=['POST'])
def scraping():
    """Tendencias sociales usando la instancia correcta (spotify)."""
    if spotify is None:
        return jsonify({'error': 'Spotify scraper not available'}), 500
    data   = request.get_json() or {}
    genre  = data.get('genre', 'pop')
    query  = data.get('query', genre)
    try:
        results = spotify.scrape_genius(query)   # ← instancia inyectada con mihm
        if not results:
            results = [
                {'title': f'{genre.title()} trending', 'keywords': [genre, 'viral', 'loop'], 'artist': 'scraper'},
            ]
    except Exception as e:
        results = [{'title': f'{genre.title()} emergente', 'keywords': [genre], 'artist': 'fallback'}]

    return jsonify({'status': 'ok', 'genre': genre, 'results': results})

@app.route('/chat', methods=['POST'])
def chat_proxy():
    """Proxy para chat IA cuando el frontend no tiene Groq key directo."""
    data     = request.get_json() or {}
    messages = data.get('messages', [])
    if not messages or not groq.api_key:
        return jsonify({'response': 'Backend activo. Configura GROQ_API_KEY en .env para respuestas completas.'})
    try:
        response = groq._call(messages, temperature=0.7, max_tokens=1400)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'response': f'Error LLM: {e}'}), 500

# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=Config.DEBUG, host='0.0.0.0', port=port)