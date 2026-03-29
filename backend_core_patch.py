# ============================================================
# PATCH PARA backend/core.py (MIHMState class)
# Agregar el método apply_emergence dentro de la clase MIHMState
# justo después del método save_history()
# ============================================================

    def apply_emergence(self, generated):
        """Acopla el estado generado por LLM/procedural de vuelta al MIHM.
        Usa smoothing exponencial (0.3 nuevo / 0.7 actual) para no
        colapsar el estado con cada generación.
        """
        ms = generated.get('mihm_state', {})
        if not isinstance(ms, dict):
            return
        if 'ihg' in ms and ms['ihg'] is not None:
            self.ihg = float(0.3 * ms['ihg'] + 0.7 * self.ihg)
        if 'nti' in ms and ms['nti'] is not None:
            self.nti = float(0.3 * ms['nti'] + 0.7 * self.nti)
        if 'r' in ms and ms['r'] is not None:
            self.r   = float(0.3 * ms['r']   + 0.7 * self.r)
        if 'cff' in ms and ms['cff'] is not None:
            self.cff = float(0.3 * ms['cff'] + 0.7 * self.cff)
        # Clip
        self.ihg = max(-2.0, min(0.0, self.ihg))
        self.nti = max(0.0,  min(1.0, self.nti))
        self.r   = max(0.0,  min(1.0, self.r))
        self.cff = max(-1.0, min(1.0, self.cff))
        self.save_history()

    def get_momentum(self):
        """Traduce función de costo J → Momentum 0-100 (lo que ve el usuario)."""
        J = self.cost_function()
        return max(0, min(100, round(100 - J * 32)))

    def cost_function(self):
        """J homeostático. Objetivo: IHG→-0.38, NTI→1, R→1, CFF→0."""
        return (1.0 * (self.ihg + 0.38)**2 +
                0.8 * (1 - self.nti)**2 +
                0.6 * (1 - self.r)**2 +
                0.4 * self.cff**2)
