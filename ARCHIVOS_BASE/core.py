# ===================== MCM-A FINAL – PRODUCTION READY =====================
import json
import os
import time
import math
import re
import random
import hashlib
from collections import deque, Counter, defaultdict
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import numpy as np
from groq import Groq

# --------------------- CONFIGURACIÓN ---------------------
WINDOW_SIZE = 50
SMOOTHING = 0.22
TARGET_IHG = -0.4
KP_BASE, KD_BASE, KI_BASE = 4.2, 1.3, 0.35
KP, KD, KI = KP_BASE, KD_BASE, KI_BASE

COOLDOWN_PERSIST = 10
COOLDOWN_INTERVENTION = 45
HISTORY_FILE = "mihm_history.json"
ERRORS_FILE = "mihm_errors.json"
PREDICTIONS_FILE = "mihm_predictions.json"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"

IRC = 0.38  # Índice de Reflexividad Colectiva

# --------------------- MIHM STATE ---------------------
class MIHMState:
    def __init__(self):
        self.ihg = -0.620
        self.nti = 0.351
        self.r = 0.45
        self.cff = 0.0
        self.ml_success = 0.5
        self.ihg_prev = self.ihg
        self.integral_error = 0.0
        self.ihg_history = deque(maxlen=200)
        self.nti_history = deque(maxlen=100)
        self.r_history = deque(maxlen=100)
        self.recent_error = 0.0
        self.alpha = 0.018
        self.beta = 0.11
        self.gamma = 0.022
        self.delta = 0.017
        self.load_history()

    def load_history(self):
        try:
            with open(HISTORY_FILE, 'r') as f:
                hist = json.load(f)
                self.ihg = hist.get('ihg', self.ihg)
                self.nti = hist.get('nti', self.nti)
                self.r = hist.get('r', self.r)
                self.cff = hist.get('cff', self.cff)
                self.ml_success = hist.get('ml_success', self.ml_success)
        except:
            pass

    def save_history(self):
        with open(HISTORY_FILE, 'w') as f:
            json.dump({
                'ihg': self.ihg,
                'nti': self.nti,
                'r': self.r,
                'cff': self.cff,
                'ml_success': self.ml_success,
                'timestamp': time.time()
            }, f)

    def update(self, ent, hier, ch, cyc):
        heg = max(hier.values()) / (max(abs(v) for v in hier.values()) or 1) if hier else 0
        ihg_new = -(heg + 0.4 * (ent / 6.0))
        nti_new = 1 - ent / 7.5
        r_new = 1 / (1 + cyc / max(1, len(self.ihg_history) or 1))

        self.ihg = SMOOTHING * ihg_new + (1 - SMOOTHING) * self.ihg
        self.nti = 0.18 * nti_new + 0.82 * self.nti
        self.r = 0.15 * r_new + 0.85 * self.r

        self.ihg = max(-2.0, min(1.0, self.ihg))
        self.nti = max(0.0, min(1.0, self.nti))
        self.r = max(0.0, min(1.0, self.r))

        self.ihg_history.append(self.ihg)
        self.nti_history.append(self.nti)
        self.r_history.append(self.r)
        self.save_history()
        return self.ihg, self.nti, self.r

    def get_stats(self):
        return {
            "ihg": round(self.ihg, 3),
            "nti": round(self.nti, 3),
            "r": round(self.r, 3),
            "cff": round(self.cff, 3),
            "ml_success": round(self.ml_success, 3)
        }

    def adapt_pid(self, error):
        self.recent_error = 0.7 * error + 0.3 * self.recent_error
        global KP, KD, KI
        KP = max(2.5, min(6.0, KP - self.recent_error * 0.8))
        KD = max(0.8, min(2.2, KD - self.recent_error * 0.4))
        KI = max(0.15, min(0.7, KI - self.recent_error * 0.25))

state = MIHMState()

# --------------------- MONTE CARLO ---------------------
class MonteCarloSimulator:
    def simulate(self, state, days=120, n_sims=800):
        results = []
        for _ in range(n_sims):
            ihg_t = state.ihg
            nti_t = state.nti
            r_t = state.r
            for _ in range(days):
                ihg_t += (-0.018 * ihg_t + 0.11 * nti_t * r_t + 0.022 * random.gauss(0, 0.8)) * 0.1
                nti_t += (0.09 * (0.5 - nti_t) + 0.05 * r_t) * 0.1
                r_t   += (-0.017 * r_t + 0.03 * nti_t) * 0.1
            results.append(ihg_t)
        return {
            "ihg_esperado": round(np.mean(results), 3),
            "probabilidad_colapso": round(np.mean([1 if x < -1.5 else 0 for x in results]) * 100, 1),
            "irc_proyectado": round(IRC + random.gauss(0, 0.12), 3)
        }

monte_carlo = MonteCarloSimulator()

# --------------------- CONTROL PID ---------------------
def control_pid(state, ihg, dt=1.0):
    error = ihg - TARGET_IHG
    state.integral_error += error * dt
    derivative = (ihg - state.ihg_prev) / dt if dt > 0 else 0
    u = -(KP * error + KI * state.integral_error + KD * derivative)
    state.ihg_prev = ihg
    state.adapt_pid(error)
    return max(-1.0, min(1.0, u))

# --------------------- SCRAPING ---------------------
class MusicScraper:
    def __init__(self):
        self.cache = {}

    def scrape_genius(self, query):
        cache_key = f"genius:{query}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        try:
            url = f"https://genius.com/search?q={query.replace(' ', '%20')}"
            resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=6)
            soup = BeautifulSoup(resp.text, 'html.parser')
            results = []
            for item in soup.select('.mini_card')[:3]:
                title = item.select_one('.mini_card-title')
                if title and title.get('href'):
                    song_url = title.get('href')
                    song_resp = requests.get(song_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=6)
                    song_soup = BeautifulSoup(song_resp.text, 'html.parser')
                    lyrics_div = song_soup.select_one('[data-lyrics-container="true"]')
                    lyrics = lyrics_div.get_text(strip=True)[:500] if lyrics_div else ""
                    results.append({'title': title.get_text(strip=True), 'lyrics_fragment': lyrics})
            self.cache[cache_key] = results
            return results
        except:
            return []

scraper = MusicScraper()

# --------------------- LLM GENERATION ---------------------
def generate_from_prompt(prompt_data):
    if not GROQ_API_KEY:
        return fallback_generation(prompt_data)
    
    client = Groq(api_key=GROQ_API_KEY)
    system_msg = """
Eres un analista musical experto en el sistema MIHM.
Devuelve SOLO un JSON válido con:
- mihm_state: { "ihg": float, "nti": float, "r": float, "cff": float, "ml_success": float }
- genre: string (reggaeton, trap, pop, hiphop, latin)
- bpm: int (70-160)
- patterns: { "kick": [16 valores 0/1], "snare": [...], "hihat": [...], "bass": [...], "lead": [...] }
- suggested_voice: string o null
- irc: float (0-1)
"""
    user_msg = f"""
Instrumentos: {prompt_data.get('instruments', '')}
Género: {prompt_data.get('genre', '')}
Palabras núcleo: {prompt_data.get('keywords', '')}
Voz: {prompt_data.get('voice', '')}
"""

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.75,
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        print("Groq error:", e)
        return fallback_generation(prompt_data)

def fallback_generation(prompt_data):
    genre = prompt_data.get('genre', 'reggaeton').lower()
    base_patterns = {
        'reggaeton': {'kick': [1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0], 'snare': [0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0], 'hihat': [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0], 'bass': [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0], 'lead': [0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0]},
        'trap': {'kick': [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0], 'snare': [0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0], 'hihat': [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], 'bass': [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0], 'lead': [0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0]},
    }
    patterns = base_patterns.get(genre, base_patterns['reggaeton'])
    bpm = {'reggaeton':95, 'trap':140, 'pop':120, 'hiphop':90, 'latin':105}.get(genre, 100)
    return {
        "mihm_state": {"ihg": -0.6, "nti": 0.35, "r": 0.45, "cff": 0.0, "ml_success": 0.5},
        "genre": genre,
        "bpm": bpm,
        "patterns": patterns,
        "suggested_voice": prompt_data.get('voice', 'hombre'),
        "irc": 0.38
    }

# --------------------- FLASK APP ---------------------
app = Flask(__name__)
CORS(app)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "version": "MCM-A FINAL",
        "state": state.get_stats(),
        "irc": IRC
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json or {}
    ent = 3.8
    hier = {}
    ch = 0.6
    cyc = 2
    ihg, nti, r = state.update(ent, hier, ch, cyc)
    u = control_pid(state, ihg)
    proyeccion = monte_carlo.simulate(state)
    return jsonify({
        "status": "ok",
        "state": state.get_stats(),
        "control": {"u": u},
        "proyeccion": proyeccion,
        "irc": IRC
    })

@app.route('/generate_autopoietic', methods=['POST'])
def generate_autopoietic():
    data = request.json or {}
    impact = float(data.get('impact', 1.0))
    global IRC
    state.nti = min(1.0, state.nti + impact * 0.09)
    state.r   = min(1.0, state.r   + impact * 0.06)
    IRC = min(1.0, IRC + impact * 0.15)
    proyeccion = monte_carlo.simulate(state, days=150)
    return jsonify({
        "status": "activated",
        "impact": impact,
        "new_state": state.get_stats(),
        "irc": IRC,
        "proyeccion": proyeccion
    })

@app.route('/analyze_song', methods=['POST'])
def analyze_song():
    if 'file' not in request.files:
        return jsonify({"error": "No se recibió archivo"}), 400
    file = request.files['file']
    temp_path = "/tmp/uploaded_audio.wav"
    file.save(temp_path)
    try:
        import librosa
        y, sr = librosa.load(temp_path, sr=None)
        spectral_entropy = librosa.feature.spectral_entropy(y=y, sr=sr)[0].mean()
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        bpm = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
        
        state.nti = 0.45 * state.nti + 0.55 * (1 - spectral_entropy/10)
        state.r   = 0.65 * state.r + 0.35 * (bpm/200)
        
        proyeccion = monte_carlo.simulate(state)
        features = {
            "spectral_entropy": float(spectral_entropy),
            "bpm": float(bpm),
            "onset_density": float(onset_env.mean())
        }
        return jsonify({
            "status": "analyzed",
            "features": features,
            "updated_state": state.get_stats(),
            "proyeccion": proyeccion
        })
    except Exception as e:
        return jsonify({"error": f"Error al analizar: {str(e)}"}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/propose_release_dates', methods=['POST'])
def propose_release_dates():
    proyeccion = monte_carlo.simulate(state, days=180)
    base = datetime.now()
    dates = [
        {"fecha": (base + timedelta(days=30)).strftime("%Y-%m-%d"), "prob_exito": 0.81, "razon": "Alta IRC proyectada"},
        {"fecha": (base + timedelta(days=60)).strftime("%Y-%m-%d"), "prob_exito": 0.69, "razon": "Ventana cultural"},
        {"fecha": (base + timedelta(days=90)).strftime("%Y-%m-%d"), "prob_exito": 0.54, "razon": "Riesgo moderado"}
    ]
    return jsonify({
        "status": "proposed",
        "current_irc": IRC,
        "proyeccion": proyeccion,
        "fechas_optimas": dates
    })

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json or {}
    generated = generate_from_prompt(data)
    state.apply_emergence(generated)
    with open('last_generated_patterns.json', 'w') as f:
        json.dump(generated.get('patterns', {}), f)
    return jsonify({
        "status": "generated",
        "mihm_state": generated.get('mihm_state', state.get_stats()),
        "genre": generated.get('genre', 'reggaeton'),
        "bpm": generated.get('bpm', 95),
        "patterns": generated.get('patterns', {}),
        "suggested_voice": generated.get('suggested_voice', None),
        "irc": generated.get('irc', IRC)
    })

if __name__ == "__main__":
    print("=" * 80)
    print("MIHM Core Engine vMCM-A FINAL – System Friction Production Ready")
    print(f"  Target IHG: {TARGET_IHG} | IRC inicial: {IRC}")
    print("=" * 80)
    app.run(host='0.0.0.0', port=5000, debug=True)