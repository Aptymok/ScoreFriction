import numpy as np
from collections import deque
from datetime import datetime, timedelta

class MIHM:
    def __init__(self, params=None):
        self.params = params or {'kp':1.2, 'ki':0.1, 'kd':0.5}
        self.state = {'ihg': -0.620, 'nti': 0.351, 'r': 0.45, 'ml_success': 0.5}
        self.history = deque(maxlen=1000)   # memoria temporal
        self.delayed_updates = []           # buffer de retardos
        self.integral = 0
        self.prev_error = 0

    def record_state(self, action=""):
        self.history.append({
            'timestamp': datetime.utcnow(),
            'ihg': self.state['ihg'],
            'nti': self.state['nti'],
            'r': self.state['r'],
            'ml_success': self.state['ml_success'],
            'action': action
        })

    def cost_function(self):
        """Función de costo global explícita"""
        w1, w2, w3 = 1.0, 0.8, 0.6
        J = (w1 * abs(self.state['ihg'] + 0.38)**2 +
             w2 * (1 - self.state['nti'])**2 +
             w3 * (1 - self.state['r'])**2)
        return J

    def apply_delta(self, delta_dict, delay_seconds=0, action=""):
        """Método único de acoplamiento. TODO módulo debe usarlo."""
        if delay_seconds > 0:
            future_time = datetime.utcnow() + timedelta(seconds=delay_seconds)
            self.delayed_updates.append((future_time, delta_dict, action))
            return

        for key, delta in delta_dict.items():
            if key in self.state:
                self.state[key] = max(-2.0, min(1.0, self.state[key] + delta))

        self.record_state(action)
        u = self.compute_control({"delta": delta_dict, "action": action})
        return u, self.cost_function()

    def compute_control(self, context):
        """Controlador unificado. Solo este método decide u(t)"""
        error = -0.38 - self.state['ihg']
        self.integral += error
        derivative = error - self.prev_error
        u = (self.params['kp'] * error +
             self.params['ki'] * self.integral +
             self.params['kd'] * derivative)
        u = np.tanh(u)   # saturación
        self.prev_error = error
        return u

    def process_delayed_updates(self):
        now = datetime.utcnow()
        remaining = []
        for t, delta, action in self.delayed_updates:
            if t <= now:
                self.apply_delta(delta, delay_seconds=0, action=action + " (delayed)")
            else:
                remaining.append((t, delta, action))
        self.delayed_updates = remaining