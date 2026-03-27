import requests

class GroqClient:
    # asumo que ya tienes self.base_url, self.api_key, self.model
    # y que existen en __init__

    def _call_groq(self, prompt: str, temperature: float = 0.5, max_tokens: int = 300) -> str:
        """Llamada única a Groq para evitar repetir boilerplate."""
        try:
            response = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": float(temperature),
                    "max_tokens": int(max_tokens),
                },
                timeout=30,
            )
        except Exception as e:
            return f"No se pudo analizar con Groq (error de red): {e}"

        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            # devuelvo un detalle útil por si falla
            try:
                err = response.json()
            except Exception:
                err = response.text
            return f"No se pudo analizar con Groq (HTTP {response.status_code}): {err}"

    def analyze_audio(self, features: dict) -> str:
        """
        Opcional: usar Groq para interpretar características de audio y generar insights.
        `features` viene del extractor (librosa/lo que uses).
        """

        # valores seguros (evita KeyError)
        spectral_entropy = features.get("spectral_entropy", 0.0)
        bel = features.get("band_energy_low", 0.0)
        bem = features.get("band_energy_mid", 0.0)
        beh = features.get("band_energy_high", 0.0)
        onset_density = features.get("onset_density", 0.0)
        dynamic_range = features.get("dynamic_range", 0.0)
        periodicity = features.get("periodicity", 0.0)

        prompt = f"""Eres un productor musical técnico y directo. 
Con base en estas características de audio, genera un análisis breve y recomendaciones prácticas.

MÉTRICAS DE AUDIO:
- Entropía espectral: {spectral_entropy}
- Energía por bandas: baja={bel}, media={bem}, alta={beh}
- Densidad de onsets (ataques por unidad de tiempo): {onset_density}
- Rango dinámico (contraste): {dynamic_range}
- Periodicidad (regularidad/loop feeling): {periodicity}

ENTREGA (máximo 10 bullets):
1) Qué tipo de sonido sugiere (género/estética: lo-fi, techno, corrido, trap, ambient, etc.)
2) Diagnóstico: claridad vs saturación, brillo vs opacidad, punch vs flojo, groove vs rígido
3) Riesgos (ej: harsh en altas, fatiga, masking, dinámica plana)
4) 3 acciones concretas en mezcla/master (EQ/comp/transient/saturación/limit)
5) 1 recomendación de arreglo (estructura) y 1 de hook (5–12 s)

Reglas:
- No inventes datos externos.
- Sé conciso y accionable.
"""
        return self._call_groq(prompt, temperature=0.5, max_tokens=300)

    def song_narrative(self, features: dict, mihm_state: dict) -> str:
        """
        Convierte métricas técnicas + MIHM a narrativa útil para un productor,
        incluyendo recomendación concreta para TikTok.
        """

        # MIHM seguro (evita KeyError)
        ihg = float(mihm_state.get("ihg", 0.0))
        nti = float(mihm_state.get("nti", 0.0))

        # si quieres, puedes meter más campos:
        r = mihm_state.get("r", None)
        pcol = mihm_state.get("p_collapse", None)

        extra_lines = []
        if r is not None:
            extra_lines.append(f"R = {float(r):.3f} (residual / resistencia)")
        if pcol is not None:
            # si viene 0..1, muestro porcentaje
            try:
                extra_lines.append(f"P(colapso) = {float(pcol)*100:.1f}%")
            except Exception:
                extra_lines.append(f"P(colapso) = {pcol}")

        extra_block = "\n".join(extra_lines) if extra_lines else "(sin extras)"

        prompt = f"""Eres un productor experimentado (mezcla + arreglo + viralidad).
Convierte estas métricas técnicas en una narrativa musical entendible y una recomendación de producción real.

MIHM (contexto del sistema):
IHG = {ihg:.3f} → escala [-2..0] (más negativo = más hegemonía/caos controlado)
NTI = {nti:.3f} → escala [0..1] (más alto = más turbulencia / inestabilidad)
Extras: {extra_block}

FEATURES (técnicas):
{features}

ENTREGA (en español, conciso, estilo productor):
A) Traducción a sonido real:
   - ¿Cómo se escucha esto? (ej: “drop con presión tipo 808 saturado a -6dB”, “hi-hats granulares”, “midrange tapado”, etc.)
B) Fortalezas (2–4 bullets) y riesgos (2–4 bullets)
C) Recomendación concreta para TikTok:
   - Duración del hook ideal (en segundos)
   - 1 fonema/palabra clave si hay voz o el equivalente sonoro si no hay voz (ej: “impacto”, “snap”, “reverse”, etc.)
   - Estructura (Intro/Hook/Drop/Break) y dónde meter el “momento meme”
D) 3 acciones de mezcla/master con números aproximados (rango):
   - EQ (frecuencias)
   - Compresión/limit (GR, ceiling)
   - Saturación/transientes (dónde y por qué)

Reglas:
- No inventes instrumentos que no estén implicados por los features.
- Si falta un dato, haz 1 pregunta al final, solo 1.
"""
        return self._call_groq(prompt, temperature=0.6, max_tokens=420)