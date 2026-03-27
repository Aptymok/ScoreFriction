import os
import json
import uuid
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from database import Database
from audio_analyzer import analyze_audio
from groq_client import GroqClient

app = Flask(__name__)
CORS(app)
db = Database()
groq = GroqClient()

kp, ki, kd = 0.8, 0.2, 0.1
integral_error = 0.0
last_error = 0.0
last_ihg = -0.4
error_history = []

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'version': '3.3'})

@app.route('/reset', methods=['POST'])
def reset():
    global kp, ki, kd, integral_error, last_error, last_ihg
    kp, ki, kd = 0.8, 0.2, 0.1
    integral_error = 0.0
    last_error = 0.0
    last_ihg = -0.4
    return jsonify({'status': 'reset'})

@app.route('/predict', methods=['POST'])
def predict():
    global kp, ki, kd, integral_error, last_error, last_ihg, error_history
    data = request.json
    user = data.get('user', 'system')
    text = data.get('text', '')

    ihg = last_ihg + (np.random.rand() - 0.5) * 0.03
    ihg = max(-2.0, min(0.5, ihg))

    error = -0.4 - ihg
    integral_error += error
    derivative = error - last_error
    u = kp * error + ki * integral_error + kd * derivative
    u = np.tanh(u)

    last_error = error
    last_ihg = ihg

    nti = 0.3 + 0.4 * (1 - abs(ihg + 0.4))
    r = 0.5 + 0.3 * (1 - abs(ihg + 0.8))

    prediction_id = str(uuid.uuid4())
    db.save_prediction(user, text, ihg, nti, r, prediction_id)

    return jsonify({
        'prediction_id': prediction_id,
        'state': {'ihg': ihg, 'nti': nti, 'r': r},
        'control': {'u': u, 'kp': kp, 'ki': ki, 'kd': kd},
        'stability': {'stability': 'estable' if abs(error) < 0.1 else 'inestable', 'iad_approx': 0.05},
        'intervencion': 'Ajustar dinámica' if abs(error) > 0.2 else 'Monitoreo',
        'scorecard': {
            'narrativa': f'Estado actual: IHG={ihg:.2f}, NTI={nti:.2f}, R={r:.2f}',
            'bemoles': [],
            'sostenidos': [],
            'tareas_dia': []
        }
    })

@app.route('/learn', methods=['POST'])
def learn():
    global kp, ki, kd, integral_error, last_error, error_history
    data = request.json
    prediction_id = data.get('prediction_id')
    outcome = float(data.get('outcome'))

    with db.get_connection() as conn:
        cur = conn.execute('SELECT ihg, nti, r FROM predictions WHERE prediction_id = ?', (prediction_id,))
        row = cur.fetchone()
        if not row:
            return jsonify({'error': 'Prediction not found'}), 404
        ihg_pred, nti_pred, r_pred = row

    error = abs(outcome - ihg_pred)
    db.save_learning(prediction_id, outcome, error)

    if len(error_history) > 10:
        mean_error = np.mean(error_history[-10:])
        if mean_error > 0.2:
            kp *= 1.05
            ki *= 1.02
        elif mean_error < 0.05:
            kp *= 0.98
            ki *= 0.99

    db.save_params(kp, ki, kd)
    error_history.append(error)
    if len(error_history) > 100:
        error_history.pop(0)

    return jsonify({
        'new_params': {'kp': kp, 'ki': ki, 'kd': kd},
        'error': error,
        'status': 'learned'
    })

@app.route('/history', methods=['GET'])
def history():
    limit = request.args.get('limit', 100, type=int)
    hist = db.get_history(limit)
    errors = [h['error'] for h in hist if h['error'] is not None]
    smoothed = []
    if errors:
        smoothed.append(errors[0])
        for i in range(1, len(errors)):
            smoothed.append(0.7 * errors[i] + 0.3 * smoothed[-1])
    for i, h in enumerate(hist):
        if i < len(smoothed):
            h['error_smoothed'] = smoothed[i]
    return jsonify({'history': hist})

@app.route('/groq/analyze', methods=['POST'])
def groq_analyze():
    data = request.json
    responses = data.get('responses', '')
    result = groq.analyze(responses)
    return jsonify(result)

@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        file_bytes = file.read()
        analysis = analyze_audio(file_bytes)
        return jsonify(analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/scenario/select', methods=['POST'])
def select_scenario():
    data = request.json
    scenario = data.get('scenario')
    return jsonify({'status': 'selected', 'scenario': scenario})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)