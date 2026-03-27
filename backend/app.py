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
from mihm import MIHM

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

db = Database('instance/friction.db')
groq = GroqClient()
mihm = MIHM()

# Cargar parámetros guardados
saved_params = db.get_parameters('mihm_params')
if saved_params:
    mihm.params.update(saved_params)

def generate_prediction_id():
    return str(uuid.uuid4())

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'version': '3.3', 'mihm': 'active'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user = data.get('user', 'anon')
    text = data.get('text', '')

    # Simular extracción de inputs desde el texto (o usar Groq)
    # Por ahora, usamos valores de ejemplo o intentamos extraer con Groq
    # En una versión completa, se llamaría a Groq para obtener métricas del texto.
    # Pero para simplificar, podemos usar el estado actual del MIHM.
    # El frontend ya llama a /groq/analyze antes de /predict.
    # Así que aquí simplemente actualizamos el estado con valores del texto?
    # En realidad, el texto ya contiene las métricas? No, el frontend envía el texto crudo.
    # Para integrar correctamente, en el frontend después de obtener el análisis de Groq,
    # se debería llamar a /predict con las métricas ya calculadas.
    # Rediseñemos: /predict recibe un objeto con las métricas (ihg, nti, r) y actualiza.
    # O podemos hacer que /predict llame a Groq internamente. Pero para mantener separación,
    # dejaremos que el frontend llame a /groq/analyze y luego envíe el resultado a /predict.

    # Por ahora, si no vienen métricas, usamos las del MIHM
    ihg = data.get('ihg', mihm.state['ihg'])
    nti = data.get('nti', mihm.state['nti'])
    r = data.get('r', mihm.state['r'])

    # Actualizar estado
    mihm.state.update({'ihg': ihg, 'nti': nti, 'r': r})

    # Calcular señal de control PID
    u = mihm.pid_control()
    # Proyección Monte Carlo
    proyeccion = mihm.monte_carlo_projection()
    # Estabilidad
    stability = mihm.stability_analysis()

    # Generar intervención basada en IHG
    if mihm.state['ihg'] < -1.2:
        intervencion = "🔴 CRÍTICO: Hegemonía extrema. Se requiere redistribución de decisiones."
    elif mihm.state['ihg'] < -0.8:
        intervencion = "🟠 ALERTA: Alta concentración de poder. Revisar roles."
    elif mihm.state['ihg'] < -0.4:
        intervencion = "🟡 TENSIÓN: Equilibrio inestable. Monitorear flujo de decisiones."
    else:
        intervencion = "🟢 ESTABLE: La homeostasis está dentro de rangos aceptables."

    pred_id = generate_prediction_id()
    # Guardar en BD (sin error por ahora)
    db.save_prediction(pred_id, user, text, mihm.state)

    return jsonify({
        'prediction_id': pred_id,
        'state': mihm.state,
        'intervencion': intervencion,
        'control': {
            'u': u,
            'kp': mihm.params['kp'],
            'ki': mihm.params['ki'],
            'kd': mihm.params['kd']
        },
        'stability': stability,
        'proyeccion': proyeccion
    })

@app.route('/learn', methods=['POST'])
def learn():
    data = request.get_json()
    pred_id = data.get('prediction_id')
    outcome = data.get('outcome')
    if not pred_id or outcome is None:
        return jsonify({'error': 'Missing prediction_id or outcome'}), 400

    # Obtener predicción anterior (simplificado, aquí solo actualizamos MIHM)
    # En producción, deberías obtener el valor predicho desde la BD
    # Simulamos error
    error = outcome - mihm.state['ihg']  # error absoluto
    error_smoothed = 0.7 * error + 0.3 * (db.get_parameters('last_error') or 0)
    db.save_parameters('last_error', error_smoothed)

    new_params = mihm.learn(pred_id, outcome, db)

    # Actualizar error en la BD (asumiendo que tenemos registro de esa predicción)
    # Guardar error
    with db.get_connection() as conn:
        conn.execute('UPDATE predictions SET error=?, error_smoothed=? WHERE prediction_id=?',
                     (error, error_smoothed, pred_id))

    return jsonify({
        'error': error,
        'error_smoothed': error_smoothed,
        'new_params': new_params
    })

@app.route('/history', methods=['GET'])
def history():
    limit = request.args.get('limit', 100, type=int)
    rows = db.get_history(limit)
    return jsonify({'history': rows})

@app.route('/reset', methods=['POST'])
def reset():
    # Reiniciar estado del MIHM
    mihm.state = {'ihg': -0.5, 'nti': 0.5, 'r': 0.5}
    mihm.integral = 0
    mihm.prev_error = 0
    return jsonify({'status': 'reset'})

@app.route('/groq/analyze', methods=['POST'])
def groq_analyze():
    data = request.get_json()
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
    scenario = data.get('scenario')
    db.save_scenario(scenario)
    return jsonify({'status': 'registered'})

@app.route('/export', methods=['GET'])
def export():
    # Exportar todas las predicciones
    with db.get_connection() as conn:
        cur = conn.execute('SELECT * FROM predictions ORDER BY timestamp')
        rows = cur.fetchall()
        data = [dict(row) for row in rows]
    return jsonify(data)

@app.route('/audio/analyze', methods=['POST'])
def audio_analyze():
    """Recibe uno o varios archivos de audio y devuelve características + análisis."""
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files uploaded'}), 400
    
@app.route('/tiktok/scrape', methods=['GET'])
async def tiktok_scrape():
    trends = await scrape_tiktok_trends()
    # Actualizar MIHM con forzamiento
    external = {'viral_phoneme': len(trends['phonetic_vector']), 'trend_density': trends['viral_score']}
    mihm.update_state(external)
    return jsonify(trends)

@app.route('/midi/generate', methods=['POST'])
def generate_midi_route():
    data = request.get_json()
    num_inst = data.get('num_instruments', 4)
    phonemes = data.get('phoneme_pattern', None)
    midi_file = mihm.generate_midi(num_inst, phonemes)
    return send_file(midi_file, as_attachment=True)

    results = []
    for file in files:
        filename = file.filename
        audio_bytes = file.read()
        features = extract_features(audio_bytes, filename)
        # Opcional: usar Groq para interpretar
        try:
            analysis = groq.analyze_audio(features)
        except:
            analysis = "No se pudo analizar con Groq."
        results.append({
            'filename': filename,
            'features': features,
            'analysis': analysis
        })
        # Actualizar estado MIHM con los features? Podríamos promediar varios.
        # Ejemplo: usar la entropía espectral como H_freq, densidad de onsets como H_time, etc.
        # Esto sería para actualizar el estado del sistema.
        # Por ahora, solo devolvemos los resultados.
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=Config.DEBUG, host='0.0.0.0', port=5000)