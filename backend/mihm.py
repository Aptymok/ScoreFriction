import numpy as np
from scipy.integrate import odeint
from mido import MidiFile, MidiTrack, Message
import random
import random

class MIHM:
    def __init__(self, params=None):
        self.params = params or {
            'kp': 1.2,   # PID proporcional
            'ki': 0.1,   # integral
            'kd': 0.5,   # derivativo
            'w1': 0.4,   # pesos para IHG
            'w2': 0.3,
            'w3': 0.3,
            'w4': 0.2,   # pesos para NTI
            'w5': 0.5,
            'w6': 0.3,
            'sigma': 0.15  # incertidumbre para Monte Carlo
        }
        self.state = {'ihg': -0.5, 'nti': 0.5, 'r': 0.5}
        self.integral = 0
        self.prev_error = 0

    def update_state(self, external_inputs):
        """
        external_inputs: dict con claves posibles:
        - Heg_audio: dominancia de capas (0..1)
        - H_freq: entropía espectral
        - H_time: entropía rítmica
        - dE_dt: derivada de energía
        - lambda_rhythm: inestabilidad rítmica
        - Psi: presión del entorno (0..1)
        - C_loop: periodicidad
        """
        # IHG
        Heg = external_inputs.get('Heg_audio', 0.5)
        Hf = external_inputs.get('H_freq', 0.5)
        Ht = external_inputs.get('H_time', 0.5)
        ihg = - (self.params['w1'] * Heg + self.params['w2'] * Hf + self.params['w3'] * Ht)
        # NTI
        dE = external_inputs.get('dE_dt', 0)
        lam = external_inputs.get('lambda_rhythm', 0)
        psi = external_inputs.get('Psi', 0)
        nti = self.params['w4'] * dE + self.params['w5'] * lam + self.params['w6'] * psi
        # R
        Cloop = external_inputs.get('C_loop', 0.3)
        r = 1 / (1 + Cloop)

        # Suavizado
        self.state['ihg'] = 0.7 * ihg + 0.3 * self.state['ihg']
        self.state['nti'] = 0.7 * nti + 0.3 * self.state['nti']
        self.state['r'] = 0.7 * r + 0.3 * self.state['r']
        return self.state

    def pid_control(self, target_ihg=-0.4):
        """Calcula señal de control u basada en error de IHG."""
        error = target_ihg - self.state['ihg']
        self.integral += error
        derivative = error - self.prev_error
        u = (self.params['kp'] * error +
             self.params['ki'] * self.integral +
             self.params['kd'] * derivative)
        # saturación tanh
        u = np.tanh(u)
        self.prev_error = error
        return u

    def monte_carlo_projection(self, horizon_days=90, n_sims=1000):
        """Proyecta IHG futuro usando modelo simple con ruido."""
        ihg0 = self.state['ihg']
        mu = 0.0  # tendencia neutral
        sigma = self.params['sigma']
        dt = 1/30  # paso diario aproximado
        n_steps = horizon_days
        final_ihg = []
        for _ in range(n_sims):
            ihg = ihg0
            for _ in range(n_steps):
                dW = np.random.normal(0, np.sqrt(dt))
                ihg += mu * dt + sigma * dW
                # límites realistas
                ihg = max(-2.0, min(0.5, ihg))
            final_ihg.append(ihg)
        mean_ihg = np.mean(final_ihg)
        prob_collapse = np.mean([1 if ihg < -1.5 else 0 for ihg in final_ihg])
        return {
            'ihg_esperado': mean_ihg,
            'probabilidad_colapso': prob_collapse * 100,
            'sigma_usado': sigma
        }

    def stability_analysis(self):
        """Análisis de estabilidad basado en jacobiano aproximado (para fines ilustrativos)."""
        # Simulación de valores propios (placeholder)
        # En realidad debería depender de ecuaciones diferenciales
        # Aquí usamos un criterio heurístico
        if self.state['ihg'] < -1.2:
            stability = 'inestable'
            iad = 0.02
        elif self.state['ihg'] < -0.8:
            stability = 'marginal'
            iad = 0.05
        else:
            stability = 'estable'
            iad = 0.12
        return {
            'stability': stability,
            'iad_approx': iad,
            'eigenvalues': [0.5, -0.3]  # placeholder
        }

    def learn(self, prediction_id, outcome, db):
        """Actualiza parámetros PID basado en error."""
        # Obtener predicción anterior
        # (simplificado: aquí usaríamos db para obtener el error)
        error = outcome - self.state['ihg']  # error real vs esperado
        # Ajustar parámetros (ejemplo: simple gradiente)
        self.params['kp'] += 0.01 * error * self.prev_error
        self.params['ki'] += 0.01 * error * self.integral
        self.params['kd'] += 0.01 * error * (error - self.prev_error)
        # Limitar rangos
        for k in ['kp', 'ki', 'kd']:
            self.params[k] = max(0.01, min(3.0, self.params[k]))
        # Guardar en DB
        db.save_parameters('mihm_params', self.params)
        return self.params
    
    
def generate_midi(self, num_instruments=4, phoneme_pattern=None, duration=30):
    mid = MidiFile()
    tracks = []
    instruments = ['drums', 'bass', 'melody', 'vocals'][:num_instruments]
    
    for i, inst in enumerate(instruments):
        track = MidiTrack()
        mid.tracks.append(track)
        track.append(Message('program_change', program=i*10, time=0))  # patch básico
        
        # Inyectar fonemas como notas repetidas (ej. "ah" → nota C4 repetida)
        if phoneme_pattern and inst == 'vocals':
            for p in phoneme_pattern * 8:  # repetir
                note = 60 + random.randint(-5, 5)  # C4 base
                track.append(Message('note_on', note=note, velocity=80 + random.randint(-20,20), time=120))
                track.append(Message('note_off', note=note, velocity=0, time=240))
        else:
            # Patrón genérico
            for _ in range(32):
                note = 48 + i*12 + random.randint(0, 12)
                track.append(Message('note_on', note=note, velocity=70, time=180))
                track.append(Message('note_off', note=note, velocity=0, time=180))
    
    filename = f"generated_{num_instruments}inst_{datetime.now().strftime('%Y%m%d')}.mid"
    mid.save(filename)
    return filename