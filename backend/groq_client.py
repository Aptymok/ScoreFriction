import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class GroqClient:
    def __init__(self):
        self.api_key = os.getenv('GROQ_API_KEY')
        self.client = Groq(api_key=self.api_key) if self.api_key else None

    def analyze(self, responses_text):
        if not self.client:
            return self._fallback_analysis()

        prompt = f"""
        Eres un analista de sistemas musicales. A partir de las siguientes respuestas de auditoría, genera un diagnóstico con:
        - IHG (índice de hegemonía grupal) entre -2.0 y 0.5
        - NTI (nivel de trascendencia institucional) entre 0 y 1
        - R (resistencia) entre 0 y 1
        - bemoles (lista de hasta 5 factores latentes)
        - sostenidos (lista de hasta 5 fortalezas)
        - proyectos_detectados (lista con nombre, etapa, figura, ldi, responsable, criticidad)
        - tareas_dia (lista de hasta 5 tareas concretas)
        - narrativa (texto breve)

        Respuestas:
        {responses_text}

        Devuelve SOLO JSON sin texto adicional.
        """
        try:
            completion = self.client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            result = json.loads(completion.choices[0].message.content)
            result['ihg'] = float(result.get('ihg', -0.5))
            result['nti'] = float(result.get('nti', 0.5))
            result['r'] = float(result.get('r', 0.5))
            return result
        except Exception as e:
            print(f"Groq error: {e}")
            return self._fallback_analysis()

    def _fallback_analysis(self):
        return {
            'ihg': -0.82,
            'nti': 0.38,
            'r': 0.54,
            'bemoles': ['Roles implícitos no declarados', 'Ausencia de calendario maestro'],
            'sostenidos': ['Equipo motivado', 'Proyectos en desarrollo'],
            'proyectos_detectados': [
                {'nombre': 'Proyecto ejemplo', 'etapa': 'producción', 'figura': '𝅗𝅥', 'ldi': 28,
                 'responsable': 'productor', 'criticidad': 'alta'}
            ],
            'tareas_dia': ['Listar proyectos activos', 'Identificar nodo de decisión implícita'],
            'narrativa': 'Diagnóstico en modo local. Conecta Groq para análisis real.'
        }