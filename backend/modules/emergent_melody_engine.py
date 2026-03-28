# emergent_melody_engine.py – Orquestador Proactivo · Paso 4
# Scraping + Monte Carlo MIHM + ML + pico tensión narrativa → MIDI
# Delta MIHM: {'ml_success': +0.08, 'phi_p': 0.1*enganche, 'nti': +0.05}

import os
import json
import math
import random
from datetime import datetime


class EmergentMelodyEngine:
    def __init__(self, mihm, groq=None, ml_module=None, spotify_module=None):
        self.mihm          = mihm
        self.groq          = groq
        self.ml_module     = ml_module
        self.spotify_module = spotify_module

        out_dir = getattr(mihm, 'config', {}).get('MIDI_OUTPUT_DIR', 'instance/midi_output') \
                  if hasattr(mihm, 'config') else 'instance/midi_output'
        self.output_dir = os.environ.get('MIDI_OUTPUT_DIR', out_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    # ── public ──────────────────────────────────────────────────────

    def generate(self, params: dict) -> dict:
        """
        params = {
          'motivos':       list[str],   # 4 motivos
          'duracion_seg':  int,
          'frase_concepto': str,
          'genero':        str,
          'instrumentos':  list[str],
          'enganche':      float,       # ~0.7
        }
        Retorna dict completo con midi_path, tension_peak, mc_projection, etc.
        """
        genero   = params.get('genero', 'pop')
        enganche = float(params.get('enganche', 0.7))

        # 1. Scraping / tendencias sociales
        social_trends = self._get_social_trends(genero)

        # 2. Proyección Monte Carlo
        mc_projection = self._monte_carlo_projection()

        # 3. Predicción ML de éxito
        ml_prediction = self._predict_success(params, mc_projection, social_trends)

        # 4. Pico de tensión narrativa
        tension_peak = self._compute_tension_peak(params.get('frase_concepto', ''), mc_projection)

        # 5. Generación MIDI con pico
        midi_path = self._generate_midi_with_peak(params, tension_peak)

        # 6. Análisis del MIDI generado
        midi_analysis = {
            'tension_beat':      tension_peak.get('beat', 32),
            'tension_intensity': tension_peak.get('intensity', 0.8),
            'phoneme_key':       tension_peak.get('phoneme_key', ''),
            'duration_beats':    self._sec_to_beats(params.get('duracion_seg', 120), params.get('bpm', 100)),
            'instruments':       params.get('instrumentos', []),
            'genre':             genero,
            'midi_path':         midi_path,
        }

        # 7. Delta MIHM
        delta = {
            'ml_success': 0.08,
            'phi_p':      0.1 * enganche,
            'nti':        0.05,
        }
        u, J = self.mihm.apply_delta(delta, action='melody_engine_generate')
        self.mihm.meta_control()

        return {
            'midi_path':     midi_path,
            'tension_peak':  tension_peak,
            'mc_projection': mc_projection,
            'ml_prediction': ml_prediction,
            'social_trends': social_trends,
            'midi_analysis': midi_analysis,
            'mihm_state':    dict(self.mihm.state),
            'cost_j':        J,
            'u':             u,
        }

    # ── private helpers ─────────────────────────────────────────────

    def _get_social_trends(self, genero: str) -> dict:
        """Analiza tendencias sociales para el género dado."""
        if self.spotify_module and hasattr(self.spotify_module, 'analyze_trends'):
            try:
                return self.spotify_module.analyze_trends(genero)
            except Exception:
                pass

        # Fallback: tendencias sintéticas basadas en estado MIHM
        cff = self.mihm.state.get('cff', 0.0)
        return {
            'genero':          genero,
            'popularidad':     0.6 + 0.2 * cff,
            'tendencia':       'ascendente' if cff > 0 else 'descendente',
            'bpm_promedio':    {'reggaeton': 95, 'pop': 120, 'latin': 100,
                                'electronica': 128, 'trap': 140}.get(genero.lower(), 110),
            'engagement_ref':  0.65,
            'source':          'synthetic_cff',
        }

    def _monte_carlo_projection(self, horizon_days: int = 30, n_sims: int = 500) -> dict:
        """Monte Carlo sobre el estado MIHM proyectado a horizon_days."""
        if hasattr(self.mihm, 'monte_carlo_projection'):
            try:
                return self.mihm.monte_carlo_projection(horizon_days=horizon_days)
            except Exception:
                pass

        # Fallback manual
        state = self.mihm.state
        ihg0  = state.get('ihg', -0.6)
        nti0  = state.get('nti', 0.35)
        r0    = state.get('r', 0.45)

        resultados = []
        for _ in range(n_sims):
            ihg = ihg0 + random.gauss(0.05 * horizon_days, 0.02 * math.sqrt(horizon_days))
            nti = min(1.0, max(0.0, nti0 + random.gauss(0.03, 0.01)))
            r   = min(1.0, max(0.0, r0   + random.gauss(0.02, 0.01)))
            exito = (ihg > -0.3) and (nti > 0.5) and (r > 0.5)
            resultados.append({'ihg': ihg, 'nti': nti, 'r': r, 'exito': exito})

        ihg_vals  = [x['ihg'] for x in resultados]
        prob      = sum(1 for x in resultados if x['exito']) / n_sims
        ihg_prom  = sum(ihg_vals) / len(ihg_vals)

        # Fecha óptima: día en que IHG cruza -0.3 (estimación lineal)
        dias_opt  = max(1, int(((-0.3 - ihg0) / 0.05))) if ihg0 < -0.3 else 7

        return {
            'ihg_esperado':   round(ihg_prom, 4),
            'prob_exito':     round(prob, 3),
            'fecha_optima':   dias_opt,
            'horizon_days':   horizon_days,
            'n_sims':         n_sims,
            'source':         'fallback_mc',
        }

    def _predict_success(self, params: dict, mc: dict, trends: dict) -> dict:
        """Predicción ML de éxito del track."""
        if self.ml_module and hasattr(self.ml_module, 'predict_success'):
            features = {
                'ihg_esperado': mc.get('ihg_esperado', -0.5),
                'prob_mc':      mc.get('prob_exito', 0.5),
                'popularidad':  trends.get('popularidad', 0.6),
                'enganche':     params.get('enganche', 0.7),
                'duracion_seg': params.get('duracion_seg', 120),
            }
            try:
                return self.ml_module.predict_success(features)
            except Exception:
                pass

        # Heurística simple
        score = (mc.get('prob_exito', 0.5) * 0.4
                 + trends.get('popularidad', 0.6) * 0.3
                 + params.get('enganche', 0.7) * 0.3)
        return {
            'score':      round(score, 3),
            'confidence': 0.6,
            'label':      'probable_exito' if score > 0.6 else 'incierto',
            'source':     'heuristic',
        }

    def _compute_tension_peak(self, frase_concepto: str, mc: dict) -> dict:
        """
        Detecta el punto de máxima tensión narrativa.
        - Longitud de palabras → sílabas → momento de impacto
        - Combinado con mc['ihg_esperado'] para escalar intensidad
        """
        words = frase_concepto.split() if frase_concepto else ['tension']

        # Contar sílabas estimadas (heurística: vocal count)
        def _count_syllables(word):
            vowels = 'aeiouáéíóúü'
            return max(1, sum(1 for c in word.lower() if c in vowels))

        syllables_per_word = [_count_syllables(w) for w in words]
        total_syllables    = sum(syllables_per_word)

        # El pico está en la palabra con más sílabas (mayor peso narrativo)
        peak_word_idx  = syllables_per_word.index(max(syllables_per_word))
        cumulative_pct = sum(syllables_per_word[:peak_word_idx + 1]) / max(total_syllables, 1)

        # Beat del pico: mapear al 60%-80% de la canción (golden section narrativa)
        # Asumimos 4/4 a ~100bpm → 1 beat ≈ 0.6s
        duracion_beats = 128  # default base
        beat = int(duracion_beats * max(0.55, min(0.85, cumulative_pct)))

        # Intensidad escalada por IHG esperado (más negativo → más tensión)
        ihg_factor = max(0.0, min(1.0, 1.0 - (mc.get('ihg_esperado', -0.5) + 1.0)))
        base_intensity = 0.7 + 0.2 * params_enganche if (params_enganche := 0.7) else 0.7
        intensity = round(min(1.0, base_intensity + 0.1 * ihg_factor), 3)

        # Fonema clave: primera vocal de la palabra pico
        phoneme_key = words[peak_word_idx][0].lower() if words else 'a'

        return {
            'beat':        beat,
            'intensity':   intensity,
            'phoneme_key': phoneme_key,
            'peak_word':   words[peak_word_idx] if words else '',
            'total_syllables': total_syllables,
        }

    def _generate_midi_with_peak(self, params: dict, tension_peak: dict) -> str:
        """
        Genera archivo MIDI con pico de tensión en el beat indicado.
        Usa mido si está disponible; fallback a archivo MIDI mínimo sintético.
        """
        midi_dir   = self.output_dir
        timestamp  = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        session_id = params.get('session_id', 'session')[:8]
        filename   = f'track_{session_id}_{timestamp}.mid'
        filepath   = os.path.join(midi_dir, filename)

        try:
            import mido
            self._generate_with_mido(filepath, params, tension_peak)
        except ImportError:
            self._generate_minimal_midi(filepath, params, tension_peak)

        return filepath

    def _generate_with_mido(self, filepath: str, params: dict, tension_peak: dict):
        """Genera MIDI con mido."""
        import mido

        genero        = params.get('genero', 'pop')
        duracion_seg  = int(params.get('duracion_seg', 120))
        enganche      = float(params.get('enganche', 0.7))
        instrumentos  = params.get('instrumentos', ['piano'])
        motivos       = params.get('motivos', ['A', 'B', 'C', 'D'])

        bpm        = self._bpm_for_genre(genero)
        ticks_ppq  = 480
        sec_per_beat = 60.0 / bpm
        tempo      = mido.bpm2tempo(bpm)
        total_beats = int(duracion_seg / sec_per_beat)
        peak_beat   = tension_peak.get('beat', total_beats // 2)
        peak_int    = tension_peak.get('intensity', 0.8)

        mid = mido.MidiFile(ticks_per_beat=ticks_ppq)

        # Pista de tempo
        tempo_track = mido.MidiTrack()
        mid.tracks.append(tempo_track)
        tempo_track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
        tempo_track.append(mido.MetaMessage('track_name', name=f'ScoreFriction_{genero}', time=0))

        # Escala base según género
        scale_map = {
            'reggaeton':  [0, 2, 3, 5, 7, 8, 10],
            'pop':        [0, 2, 4, 5, 7, 9, 11],
            'latin':      [0, 2, 3, 5, 7, 8, 10],
            'electronica': [0, 2, 4, 6, 7, 9, 11],
            'trap':       [0, 2, 3, 5, 7, 8, 10],
        }
        scale = scale_map.get(genero.lower(), [0, 2, 4, 5, 7, 9, 11])

        # General MIDI program numbers para instrumentos comunes
        program_map = {
            'piano': 0, 'guitarra': 25, 'bajo': 33, 'batería': 0,
            'sintetizador': 80, 'violín': 40, 'trompeta': 56,
            'flauta': 73, 'órgano': 16, 'clave': 6,
        }

        for ch_idx, instr in enumerate(instrumentos[:8]):  # max 8 canales
            channel = ch_idx % 9  # evitar canal 9 (percusión) salvo batería
            if 'bater' in instr.lower() or 'drum' in instr.lower():
                channel = 9

            track = mido.MidiTrack()
            mid.tracks.append(track)
            track.append(mido.MetaMessage('track_name', name=instr, time=0))

            if channel != 9:
                prog = program_map.get(instr.lower().split()[0], 80)
                track.append(mido.Message('program_change', channel=channel, program=prog, time=0))

            # Generar notas
            beat = 0
            root = 60  # C4
            while beat < total_beats:
                # Motivo base: ciclo de 4 motivos
                motivo_idx = (beat // 8) % len(motivos)
                pitch_offset = (motivo_idx * 2) % len(scale)
                note = root + scale[pitch_offset % len(scale)]
                if ch_idx % 2 == 0:
                    note += 12  # octava más alta para melodía

                # Velocidad sube en el pico de tensión
                base_vel = int(60 + 30 * enganche)
                if abs(beat - peak_beat) < 4:
                    velocity = min(127, int(base_vel + 40 * peak_int))
                else:
                    velocity = base_vel + random.randint(-5, 5)

                duration_ticks = ticks_ppq  # negra

                track.append(mido.Message('note_on',  channel=channel, note=note, velocity=velocity, time=0))
                track.append(mido.Message('note_off', channel=channel, note=note, velocity=0,        time=duration_ticks))

                beat += 1

        mid.save(filepath)

    def _generate_minimal_midi(self, filepath: str, params: dict, tension_peak: dict):
        """Genera un archivo MIDI mínimo válido sin mido (bytes raw)."""
        # MIDI header: MThd + MTrk básicos
        tempo_us = 600000  # 100 bpm

        def var_len(n):
            data = []
            data.append(n & 0x7F)
            n >>= 7
            while n:
                data.append((n & 0x7F) | 0x80)
                n >>= 7
            return bytes(reversed(data))

        # Track con tempo + una nota
        track_events = bytearray()
        # Set tempo
        track_events += b'\x00\xff\x51\x03'
        track_events += tempo_us.to_bytes(3, 'big')
        # Nota on C4
        peak_beat = tension_peak.get('beat', 32)
        delay = var_len(peak_beat * 480)
        track_events += delay + b'\x90\x3c\x64'
        # Nota off
        track_events += var_len(480) + b'\x80\x3c\x00'
        # End of track
        track_events += b'\x00\xff\x2f\x00'

        track_len = len(track_events).to_bytes(4, 'big')

        with open(filepath, 'wb') as f:
            # MThd
            f.write(b'MThd\x00\x00\x00\x06\x00\x01\x00\x01\x01\xe0')
            # MTrk
            f.write(b'MTrk' + track_len + track_events)

    # ── utilities ────────────────────────────────────────────────────

    def _bpm_for_genre(self, genero: str) -> int:
        mapping = {
            'reggaeton': 95, 'pop': 120, 'latin': 100,
            'electronica': 128, 'trap': 140, 'cumbia': 100,
            'salsa': 180, 'rock': 130, 'jazz': 110,
        }
        return mapping.get(genero.lower(), 110)

    def _sec_to_beats(self, seconds: int, bpm: int) -> int:
        return int(seconds * bpm / 60)
