import requests
import json
from config import Config

class GroqClient:
    def __init__(self):
        self.api_key = Config.GROQ_API_KEY
        self.model = Config.GROQ_MODEL
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"

    def analyze_audit(self, responses_text):
        """Envía el texto de las respuestas de auditoría a Groq para obtener análisis estructurado."""
        prompt = f"""Eres un analista de sistemas musicales. A partir de las siguientes respuestas de auditoría, genera un análisis estructurado en JSON con estos campos:
- ihg: número entre -2 y 0 (concentración de poder, más negativo = más hegemonía)
- nti: número entre 0 y 1 (coherencia, más alto = mejor)
- r: número entre 0 y 1 (resistencia/adaptabilidad)
- ldi_days: número entero (latencia institucional en días)
- dinamica: string, uno de: pianissimo, piano, mezzoforte, forte, fortissimo
- figura_sistema: string, uno de: 𝅝, 𝅗𝅥, ♩, ♪, 𝅘𝅥𝅯, 𝄽
- intervencion_inmediata: string breve (acción prioritaria)
- narrativa: string (resumen en 2-3 líneas)
- bemoles: lista de strings (factores que bajan el tono)
- sostenidos: lista de strings (factores que suben la capacidad)
- proyectos_detectados: lista de objetos con nombre, etapa, vector, figura, ldi, responsable, criticidad
- escenario_a, escenario_b, escenario_c: cada uno con nombre, descripcion, prob_exito (0-100), acciones (lista), ihg_proy, nti_proy, r_proy
- tareas_dia1: lista de strings (3-5 tareas)
- tendencias: lista de objetos con nombre, relevancia (0-1), accion
- duetos: lista de objetos con tipo, descripcion, sinergia (número), justificacion

Responde solo con el JSON, sin texto adicional.

Respuestas de auditoría:
{responses_text}
"""
        response = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "response_format": {"type": "json_object"}
            }
        )
        if response.status_code == 200:
            data = response.json()
            content = data['choices'][0]['message']['content']
            return json.loads(content)
        else:
            raise Exception(f"Groq API error: {response.status_code} - {response.text}")

    def analyze_audio(self, features):
        """Opcional: usar Groq para interpretar características de audio y generar insights."""
        prompt = f"""Dadas las siguientes características de audio, genera un breve análisis musical y recomendaciones:
- Entropía espectral: {features.get('spectral_entropy', 0)}
- Energía por bandas: baja={features.get('band_energy_low', 0)}, media={features.get('band_energy_mid', 0)}, alta={features.get('band_energy_high', 0)}
- Densidad de onsets: {features.get('onset_density', 0)}
- Rango dinámico: {features.get('dynamic_range', 0)}
- Periodicidad: {features.get('periodicity', 0)}

    def song_narrative(self, features, mihm_state):
    prompt = f"""Convierte estas métricas técnicas en narrativa musical entendible y escala real para un productor:
IHG = {mihm_state['ihg']:.3f} → (escala -2 a 0: más negativo = más hegemonía/caos controlado)
NTI = {mihm_state['nti']:.3f}
Features: {features}

Responde en español, estilo productor experimentado:
- Qué significa el número en términos de sonido real (ej. "drop con presión como 808 saturado a -6dB")
- Fortalezas y riesgos
- Recomendación concreta para TikTok (duración hook, fonema clave)"""
Responde con un texto conciso.
"""
        response = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.5,
                "max_tokens": 300
            }
        )
        if response.status_code == 200:
            data = response.json()
            return data['choices'][0]['message']['content']
        else:
            return "No se pudo analizar con Groq."