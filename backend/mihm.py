import numpy as np
import random
from collections import deque
from datetime import datetime, timedelta
from mido import MidiFile, MidiTrack, Message


class MIHM:
    """
    Motor de Inercia Hegemónica Musical (MIHM) – Núcleo del framework System Friction.

    Principios:
    - Todo módulo externo DEBE llamar apply_delta() – nunca actualiza estado directamente.
    - compute_control() es el único decisor de u(t).
    - meta_control() es el nivel superior que ajusta sus propios parámetros.
    - La función de costo J es el eje de decisión global.
    """

    def __init__(self, params=None):
        self.params = params or {
            'kp': 1.2,
            'ki': 0.1,
            'kd': 0.5,
            'w1': 1.0,   # peso IHG en J
            'w2': 0.8,   # peso NTI en J
            'w3': 0.6,   # peso R en J
            'sigma': 0.15
        }

        # Vector de estado extendido:
        # x = [IHG, NTI, R, Phi_p, Psi_i, H_scale, ML_success, IRC, Meta_J]
        self.state = {
            'ihg':        -0.620,
            'nti':         0.351,
            'r':           0.450,
            'phi_p':       0.000,
            'psi_i':       0.000,
            'h_scale':     0.500,
            'ml_success':  0.500,
        }

        # Memoria temporal (deque de dicts con snapshot de estado)
        self.history = deque(maxlen=1000)

        # Buffer de actualizaciones retardadas: list of (trigger_time, delta_dict, action)
        self.delayed_updates = []

        # PID internals
        self.integral = 0.0
        self.prev_error = 0.0

        # Capa reflexiva (MCM-A)
        self.irc = 0.38                      # Índice de Reflexividad Colectiva
        self.meta_j_history = deque(maxlen=200)
        self.reflexive_rules = []            # reglas auto-descubiertas

        # Groq client (se inyecta desde app.py para avoid import circular)
        self._groq = None

    # ------------------------------------------------------------------
    # FUNCIÓN DE COSTO GLOBAL (J)
    # ------------------------------------------------------------------

    def cost_function(self):
        """
        J es el eje de decisión global explícito.
        Minimizar J significa: IHG → –0.38, NTI → 1, R → 1.
        """
        w1 = self.params['w1']
        w2 = self.params['w2']
        w3 = self.params['w3']
        J = (w1 * (self.state['ihg'] + 0.38) ** 2 +
             w2 * (1.0 - self.state['nti']) ** 2 +
             w3 * (1.0 - self.state['r']) ** 2)
        return float(J)

    # ------------------------------------------------------------------
    # MEMORIA TEMPORAL
    # ------------------------------------------------------------------

    def record_state(self, action=""):
        """Guarda snapshot del estado actual en la memoria temporal."""
        snap = {
            'timestamp':  datetime.utcnow().isoformat(),
            'ihg':        self.state['ihg'],
            'nti':        self.state['nti'],
            'r':          self.state['r'],
            'ml_success': self.state['ml_success'],
            'irc':        self.irc,
            'cost_j':     self.cost_function(),
            'action':     action,
        }
        self.history.append(snap)

    # ------------------------------------------------------------------
    # ACOPLAMIENTO OBLIGATORIO – apply_delta
    # ------------------------------------------------------------------

    def apply_delta(self, delta_dict, delay_seconds=0, action=""):
        """
        Método único de acoplamiento.
        TODO módulo externo DEBE llamar este método – nunca escribir state directamente.

        Si delay_seconds > 0 el delta se encola en delayed_updates.
        Retorna (u, J) cuando se aplica inmediatamente, None cuando se encola.
        """
        if delay_seconds > 0:
            trigger = datetime.utcnow() + timedelta(seconds=delay_seconds)
            self.delayed_updates.append((trigger, delta_dict, action))
            return None, None

        for key, delta in delta_dict.items():
            if key in self.state:
                self.state[key] = float(np.clip(self.state[key] + delta, -2.0, 1.0))

        self.record_state(action)
        u = self.compute_control({"delta": delta_dict, "action": action})
        J = self.cost_function()

        # Guardar Meta_J en historia reflexiva
        self.meta_j_history.append(J)

        return u, J

    # ------------------------------------------------------------------
    # CONTROLADOR UNIFICADO (Nivel 1 – rápido)
    # ------------------------------------------------------------------

    def compute_control(self, context):
        """
        El único método que decide u(t).
        Los módulos solo proveen contexto; este método calcula la acción.
        """
        target_ihg = -0.38
        error = target_ihg - self.state['ihg']
        self.integral += error
        derivative = error - self.prev_error
        u = (self.params['kp'] * error +
             self.params['ki'] * self.integral +
             self.params['kd'] * derivative)
        u = float(np.tanh(u))   # saturación suave
        self.prev_error = error
        return u

    # ------------------------------------------------------------------
    # RETARDOS – procesar actualizaciones encoladas
    # ------------------------------------------------------------------

    def process_delayed_updates(self):
        """Procesa actualizaciones cuyo tiempo ya venció."""
        now = datetime.utcnow()
        remaining = []
        for trigger, delta, action in self.delayed_updates:
            if trigger <= now:
                self.apply_delta(delta, delay_seconds=0, action=action + " (delayed)")
            else:
                remaining.append((trigger, delta, action))
        self.delayed_updates = remaining

    # ------------------------------------------------------------------
    # CAPA REFLEXIVA (MCM-A) – meta_control (Nivel 2 – lento)
    # ------------------------------------------------------------------

    def compute_meta_j(self):
        """
        Meta_J: evalúa la calidad de la propia función de costo.
        Bajo = buena función de costo (estable y convergente).
        """
        recent = list(self.history)[-50:]
        costs = [h['cost_j'] for h in recent if 'cost_j' in h]
        if len(costs) < 10:
            return 1.0
        xs = np.arange(len(costs), dtype=float)
        trend = float(np.polyfit(xs, costs, 1)[0])
        return float(abs(trend) + np.std(costs))

    def update_irc(self):
        """
        IRC – Índice de Reflexividad Colectiva.
        Mide la correlación entre el índice temporal y la evolución de J.
        Un IRC negativo indica que J está disminuyendo (sistema mejora).
        """
        if len(self.history) < 20:
            return
        costs = [h['cost_j'] for h in self.history if 'cost_j' in h]
        if len(costs) < 10:
            return
        corr = np.corrcoef(np.arange(len(costs), dtype=float), costs)[0, 1]
        if np.isnan(corr):
            corr = 0.0
        self.irc = float(0.7 * self.irc + 0.3 * corr)

    def meta_control(self):
        """
        Nivel superior (Nivel 2 – lento).
        Se ejecuta cada 100 entradas en historial (o forzado externamente).
        Ajusta los gains del PID o pesos de J según IRC y Meta_J.
        """
        if len(self.history) == 0:
            return
        # Solo actuar cada 100 pasos (mod check sobre tamaño actual)
        if len(self.history) % 100 != 0:
            return

        meta_j = self.compute_meta_j()
        self.update_irc()

        if self.irc > 0.65 and meta_j < 0.15:
            # Sistema bien alineado → relajar integral
            self.params['ki'] = float(np.clip(self.params['ki'] * 0.95, 0.001, 3.0))
        elif self.irc < 0.45:
            # Sistema desconectado → más reactividad
            self.params['kd'] = float(np.clip(self.params['kd'] * 1.08, 0.001, 3.0))

        # Con probabilidad 15% pedir nueva regla a Groq
        if random.random() < 0.15 and self._groq is not None:
            self.propose_new_rule_via_groq()

    def propose_new_rule_via_groq(self):
        """
        El sistema genera y evalúa sus propias reglas a través de Groq.
        Sólo se aplica si mejora J.
        """
        if self._groq is None:
            return
        history_summary = str(list(self.history)[-30:])[:800]
        prompt = (
            f"Analiza esta historia de estado MIHM y propón UNA sola regla nueva de control "
            f"(en formato delta_dict). Historia resumida: {history_summary}. "
            f"Estado actual: IHG={self.state['ihg']:.3f}, NTI={self.state['nti']:.3f}, "
            f"IRC={self.irc:.3f}. "
            f'Responde SOLO con un JSON: {{"rule": "descripción breve", "delta": {{"key": delta_value}}}}'
        )
        try:
            import json
            raw = self._groq.raw_completion(prompt, max_tokens=200)
            data = json.loads(raw)
            rule_name = data.get('rule', 'auto_rule')
            delta = data.get('delta', {})
            if not delta:
                return
            # Solo aplicar si mejora J
            j_before = self.cost_function()
            self.apply_delta(delta, action=f"meta_rule:{rule_name}")
            j_after = self.cost_function()
            if j_after > j_before:
                # Revertir: aplicar delta negativo
                revert = {k: -v for k, v in delta.items()}
                self.apply_delta(revert, action=f"revert_meta_rule:{rule_name}")
            else:
                self.reflexive_rules.append({
                    'rule': rule_name,
                    'delta': delta,
                    'j_improvement': j_before - j_after,
                    'timestamp': datetime.utcnow().isoformat()
                })
        except Exception:
            pass

    # ------------------------------------------------------------------
    # LEGACY – mantener compatibilidad con app.py existente
    # ------------------------------------------------------------------

    def update_state(self, external_inputs):
        """Compatibilidad: traduce inputs legacy a apply_delta."""
        w1 = self.params.get('w1', 1.0)
        w2 = self.params.get('w2', 0.8)
        w3 = self.params.get('w3', 0.6)
        Heg = external_inputs.get('Heg_audio', 0.5)
        Hf = external_inputs.get('H_freq', 0.5)
        Ht = external_inputs.get('H_time', 0.5)
        new_ihg = -(0.4 * Heg + 0.3 * Hf + 0.3 * Ht)
        dE = external_inputs.get('dE_dt', 0)
        lam = external_inputs.get('lambda_rhythm', 0)
        psi = external_inputs.get('Psi', 0)
        new_nti = 0.2 * dE + 0.5 * lam + 0.3 * psi
        Cloop = external_inputs.get('C_loop', 0.3)
        new_r = 1.0 / (1.0 + Cloop)
        delta = {
            'ihg': new_ihg - self.state['ihg'],
            'nti': new_nti - self.state['nti'],
            'r':   new_r - self.state['r'],
        }
        self.apply_delta(delta, action="update_state_legacy")
        return self.state

    def pid_control(self, target_ihg=-0.38):
        """Compatibilidad: alias de compute_control con target explícito."""
        old_target = -0.38
        error = target_ihg - self.state['ihg']
        self.integral += error
        derivative = error - self.prev_error
        u = (self.params['kp'] * error +
             self.params['ki'] * self.integral +
             self.params['kd'] * derivative)
        u = float(np.tanh(u))
        self.prev_error = error
        return u

    def monte_carlo_projection(self, horizon_days=90, n_sims=1000):
        ihg0 = self.state['ihg']
        sigma = self.params.get('sigma', 0.15)
        dt = 1.0 / 30.0
        final_ihg = []
        rng = np.random.default_rng()
        for _ in range(n_sims):
            ihg = ihg0
            noise = rng.normal(0, np.sqrt(dt), horizon_days)
            for dW in noise:
                ihg += sigma * dW
                ihg = float(np.clip(ihg, -2.0, 0.5))
            final_ihg.append(ihg)
        arr = np.array(final_ihg)
        return {
            'ihg_esperado': float(np.mean(arr)),
            'probabilidad_colapso': float(np.mean(arr < -1.5) * 100),
            'sigma_usado': sigma
        }

    def stability_analysis(self):
        ihg = self.state['ihg']
        if ihg < -1.2:
            stability, iad = 'inestable', 0.02
        elif ihg < -0.8:
            stability, iad = 'marginal', 0.05
        else:
            stability, iad = 'estable', 0.12
        return {
            'stability': stability,
            'iad_approx': iad,
            'eigenvalues': [0.5, -0.3]
        }

    def learn(self, prediction_id, outcome, db):
        error = outcome - self.state['ihg']
        self.params['kp'] = float(np.clip(self.params['kp'] + 0.01 * error * self.prev_error, 0.01, 3.0))
        self.params['ki'] = float(np.clip(self.params['ki'] + 0.01 * error * self.integral, 0.01, 3.0))
        self.params['kd'] = float(np.clip(self.params['kd'] + 0.01 * error * (error - self.prev_error), 0.01, 3.0))
        db.save_parameters('mihm_params', self.params)
        return self.params

    # ------------------------------------------------------------------
    # GENERACIÓN MIDI (legacy – mantenida)
    # ------------------------------------------------------------------

    def generate_midi(self, num_instruments=4, phoneme_pattern=None, duration=30):
        mid = MidiFile()
        instruments = ['drums', 'bass', 'melody', 'vocals'][:num_instruments]
        for i, inst in enumerate(instruments):
            track = MidiTrack()
            mid.tracks.append(track)
            track.append(Message('program_change', program=i * 10, time=0))
            if phoneme_pattern and inst == 'vocals':
                for p in phoneme_pattern * 8:
                    note = 60 + random.randint(-5, 5)
                    track.append(Message('note_on', note=note, velocity=80 + random.randint(-20, 20), time=120))
                    track.append(Message('note_off', note=note, velocity=0, time=240))
            else:
                for _ in range(32):
                    note = 48 + i * 12 + random.randint(0, 12)
                    track.append(Message('note_on', note=note, velocity=70, time=180))
                    track.append(Message('note_off', note=note, velocity=0, time=180))
        filename = f"generated_{num_instruments}inst_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.mid"
        mid.save(filename)
        return filename
