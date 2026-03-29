# mihm_v3_control_scorefriction.py
# Versión adaptada para Nodo 01 (Casa Productora) → retroalimentación a núcleo SystemFriction

import numpy as np
from scipy.integrate import solve_ivp
import json
from datetime import datetime

class SystemFrictionNode:
    def __init__(self, node_id="Nodo01_MusicProducer"):
        self.node_id = node_id
        self.x = np.array([-0.620, 0.351, 0.450])  # IHG, NTI, R
        self.history = {"t": [], "ihg": [], "nti": [], "r": [], "J": [], "u": [], "IAD": [], "ETE": []}
        self.generation = 0
        self.data_for_core = []  # datos anonimizados que se enviarán al núcleo central

    def cost_function(self, x):
        ihg, nti, r = x
        return 1.0*(ihg + 0.38)**2 + 0.8*(1 - nti)**2 + 0.6*(1 - r)**2

    def optimal_control(self, x, t):
        # Control óptimo retroalimentado adaptado a nicho musical (más agresivo en IHG)
        K = np.array([[4.8, -1.2, 0.9],
                      [-0.95, 2.6, -1.4],
                      [0.6, -0.8, 2.1]])
        target = np.array([-0.38, 0.78, 0.85])
        u = -K @ (x - target)
        # Saturación realista (no se puede intervenir infinitamente)
        return np.clip(u, -0.35, 0.35)

    def dynamics(self, t, x):
        ihg, nti, r = x
        u = self.optimal_control(x, t)
        
        # EDOs con memoria y acoplamiento no lineal (versión afinada para productora musical)
        dihg = -0.85*ihg - 0.65*ihg*nti + 0.45*(1-r)**2 + 0.18*ihg*np.exp(-0.04*t) + u[0]
        dnti = -0.72*(1-nti) + 0.58*nti*r - 0.38*ihg**2 + 0.14*nti*np.exp(-0.04*t) + u[1]
        dr   = -0.65*r + 0.72*nti*r - 0.42*abs(ihg)*r + u[2]
        
        J = self.cost_function(x)
        
        # Métricas superiores
        IAD = (r * abs(dnti)) / (0.01 + abs(0.02*np.sin(t/180)))   # disipación vs perturbación
        ETE = (max(0, dnti) * r) / (0.001 + np.sum(u**2) + J)       # transformación por esfuerzo
        
        # Registro para envío al núcleo central (anonimizado)
        self.data_for_core.append({
            "timestamp": datetime.now().isoformat(),
            "node_type": "music_producer",
            "metrics": {"ihg": float(ihg), "nti": float(nti), "r": float(r), "J": float(J)},
            "control_applied": [float(ui) for ui in u],
            "IAD": float(IAD),
            "ETE": float(ETE)
        })
        
        return [dihg, dnti, dr]

    def simulate(self, days=1460):  # 4 años
        sol = solve_ivp(self.dynamics, [0, days], self.x, method='RK45', t_eval=np.linspace(0, days, 800))
        
        self.x = sol.y[:,-1]
        self.history["t"] = sol.t.tolist()
        self.history["ihg"] = sol.y[0].tolist()
        self.history["nti"] = sol.y[1].tolist()
        self.history["r"]   = sol.y[2].tolist()
        
        # Calcular métricas
        for i in range(len(sol.t)):
            J = self.cost_function(sol.y[:,i])
            self.history["J"].append(J)
            u = self.optimal_control(sol.y[:,i], sol.t[i])
            self.history["u"].append(u.tolist())
        
        print(f"\n=== SIMULACIÓN NODO 01 COMPLETADA ===")
        print(f"Estado final: IHG = {self.x[0]:.3f} | NTI = {self.x[1]:.3f} | R = {self.x[2]:.3f}")
        print(f"J final = {self.cost_function(self.x):.3f}")
        print(f"IAD promedio = {np.mean([r * abs(dnti) for r,dnti in zip(self.history['r'], np.diff(self.history['nti']))]):.4f}")
        print(f"ETE promedio = {np.mean(self.history.get('ETE', [0.5])):.4f}")
        
        # Exportar datos anonimizados para núcleo central
        with open(f"node_data_{self.node_id}_{datetime.now().strftime('%Y%m%d')}.json", "w") as f:
            json.dump(self.data_for_core[-50:], f, indent=2)  # últimos 50 pasos anonimizados
        
        return self.x, self.history

# Uso inmediato
if __name__ == "__main__":
    node = SystemFrictionNode()
    final_state, hist = node.simulate()