import os
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/health')
def health():
    return jsonify({"status": "ok", "message": "Backend funcionando correctamente"})

@app.route('/state')
def state():
    return jsonify({
        "ihg": -0.62,
        "nti": 0.351,
        "r": 0.45,
        "cost_j": 0.42,
        "irc": 0.38
    })

@app.route('/pm/event', methods=['POST'])
def pm_event():
    return jsonify({"status": "ok", "cost_j": 0.42})

@app.route('/audio/analyze', methods=['POST'])
def audio_analyze():
    return jsonify({"results": [], "mihm_state": {"ihg": -0.62}})

@app.route('/predict', methods=['POST'])
def predict():
    return jsonify({"state": {"ihg": -0.62, "nti": 0.351, "r": 0.45}, "cost_j": 0.42})

@app.route('/orchestrator/status', methods=['GET'])
def orchestrator_status():
    return jsonify({"status": "activo"})

@app.route('/chat', methods=['POST'])
def chat():
    return jsonify({"response": "Backend funcionando. Conecta tu clave de Groq para respuestas IA."})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)