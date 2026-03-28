# drive_manager.py – Orquestador Proactivo · Paso 3
# Sube 2 Google Docs al Drive; fallback a archivos Markdown locales.
# Delta MIHM: {'nti': +0.03, 'r': +0.02} sin retardo.

import json
import os
import re
from datetime import datetime


class DriveManager:
    SCOPES = [
        'https://www.googleapis.com/auth/drive',
        'https://www.googleapis.com/auth/documents',
    ]

    def __init__(self, mihm, config=None):
        self.mihm   = mihm
        self.config = config or {}
        self._drive_svc = None
        self._docs_svc  = None

    # ── credentials ────────────────────────────────────────────────

    def _get_credentials(self):
        """Carga credenciales de service account (JSON inline o archivo)."""
        try:
            from google.oauth2 import service_account
        except ImportError:
            return None

        # Primero intenta JSON inline en env
        sa_json = self.config.get('GOOGLE_SERVICE_ACCOUNT_JSON') or os.environ.get('GOOGLE_SERVICE_ACCOUNT_JSON')
        if sa_json:
            try:
                info = json.loads(sa_json)
                return service_account.Credentials.from_service_account_info(info, scopes=self.SCOPES)
            except Exception:
                pass

        # Luego intenta archivo
        creds_file = (self.config.get('GOOGLE_CREDENTIALS_FILE')
                      or os.environ.get('GOOGLE_CREDENTIALS_FILE', 'credentials/service_account.json'))
        if os.path.exists(creds_file):
            try:
                return service_account.Credentials.from_service_account_file(creds_file, scopes=self.SCOPES)
            except Exception:
                pass

        return None

    def _drive(self):
        if self._drive_svc is None:
            try:
                from googleapiclient.discovery import build
                creds = self._get_credentials()
                if creds:
                    self._drive_svc = build('drive', 'v3', credentials=creds)
            except Exception:
                pass
        return self._drive_svc

    def _docs(self):
        if self._docs_svc is None:
            try:
                from googleapiclient.discovery import build
                creds = self._get_credentials()
                if creds:
                    self._docs_svc = build('docs', 'v1', credentials=creds)
            except Exception:
                pass
        return self._docs_svc

    # ── helpers ─────────────────────────────────────────────────────

    def _create_doc(self, title, body_markdown):
        """Crea un Google Doc con el contenido dado. Retorna file_id y url."""
        drive = self._drive()
        docs  = self._docs()
        if drive is None or docs is None:
            return None, None

        # Crear documento vacío
        meta = {'name': title, 'mimeType': 'application/vnd.google-apps.document'}
        folder_id = self.config.get('GOOGLE_DRIVE_FOLDER_ID') or os.environ.get('GOOGLE_DRIVE_FOLDER_ID', '')
        if folder_id:
            meta['parents'] = [folder_id]

        file_obj = drive.files().create(body=meta, fields='id').execute()
        file_id  = file_obj.get('id')

        # Escribir contenido con requests de la Docs API
        requests_body = [{'insertText': {'location': {'index': 1}, 'text': body_markdown}}]
        docs.documents().batchUpdate(documentId=file_id, body={'requests': requests_body}).execute()

        url = f'https://docs.google.com/document/d/{file_id}/edit'
        return file_id, url

    def _local_fallback(self, filename, content):
        """Guarda un archivo Markdown local como fallback."""
        out_dir = self.config.get('MIDI_OUTPUT_DIR', 'instance/midi_output')
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return path

    # ── documentos ──────────────────────────────────────────────────

    def create_executive_doc(self, session_id, mihm_state, mc_projection,
                             midi_analysis, groq_narrative, social_proposal_url=''):
        """Crea el Documento 1: Análisis Ejecutivo. Retorna dict con file_id, url, local_path."""
        now   = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
        title = f'[ScoreFriction] Análisis Ejecutivo – {session_id[:8]} – {now}'

        # Construir contenido Markdown
        ihg = mihm_state.get('ihg', 0)
        nti = mihm_state.get('nti', 0)
        r   = mihm_state.get('r', 0)
        irc = mihm_state.get('irc', 0)
        cff = mihm_state.get('cff', 0)
        J   = mihm_state.get('cost_j', self.mihm.cost_function())

        mc_ihg  = mc_projection.get('ihg_esperado', 'N/A')
        mc_prob = mc_projection.get('prob_exito', 'N/A')
        mc_opt  = mc_projection.get('fecha_optima', 'N/A')

        tension_beat = midi_analysis.get('tension_beat', 'N/A')
        tension_int  = midi_analysis.get('tension_intensity', 'N/A')

        content = f"""# Análisis Ejecutivo – Sistema Friction
**Sesión:** {session_id}
**Generado:** {now}

---

## Estado MIHM al momento del análisis
| Indicador | Valor |
|-----------|-------|
| IHG (Inercia Hegemónica Global) | {ihg:.4f} |
| NTI (Índice de Tensión Narrativa) | {nti:.4f} |
| R (Resonancia) | {r:.4f} |
| IRC (Índice Reflexividad Colectiva) | {irc:.4f} |
| CFF (Campo Frecuencia Colectiva) | {cff:.4f} |
| **Función de Costo J** | **{J:.6f}** |

---

## Proyección Monte Carlo (horizonte 30 días)
- **IHG esperado:** {mc_ihg}
- **Probabilidad de éxito:** {mc_prob}
- **Fecha óptima de lanzamiento:** {mc_opt}

---

## Pico de Tensión Narrativa (MIDI)
- **Beat de impacto:** {tension_beat}
- **Intensidad:** {tension_int}

---

## Narrativa Groq
{groq_narrative}

---

## Propuesta Social
{"Ver: " + social_proposal_url if social_proposal_url else "Ver documento adjunto"}

---
*Generado automáticamente por Sistema Friction v4.2 – Motor Orquestador Proactivo*
"""

        file_id, url = self._create_doc(title, content)

        if file_id is None:
            local_path = self._local_fallback(f'executive_{session_id[:8]}.md', content)
            result = {'file_id': None, 'url': None, 'local_path': local_path, 'type': 'executive'}
        else:
            result = {'file_id': file_id, 'url': url, 'local_path': None, 'type': 'executive'}

        # Delta MIHM: la formalización de entregables sube NTI
        u, J_new = self.mihm.apply_delta({'nti': 0.03, 'r': 0.02}, action='drive_upload_executive')
        self.mihm.meta_control()
        result['mihm_u']    = u
        result['mihm_cost'] = J_new
        return result

    def create_social_proposal_doc(self, session_id, params, midi_analysis, mihm_state):
        """Crea el Documento 2: Propuesta Redes Sociales. Retorna dict con file_id, url, local_path."""
        now   = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
        title = f'[ScoreFriction] Propuesta Social – {session_id[:8]} – {now}'

        genero      = params.get('genero', 'N/A')
        motivos     = ', '.join(params.get('motivos', []))
        instrumentos = ', '.join(params.get('instrumentos', []))
        frase       = params.get('frase_concepto', 'N/A')
        enganche    = params.get('enganche', 0.7)
        duracion    = params.get('duracion_seg', 0)
        beat        = midi_analysis.get('tension_beat', 'N/A')

        content = f"""# Propuesta Estrategia Redes Sociales
**Sesión:** {session_id}
**Generado:** {now}

---

## Datos del Track
| Elemento | Valor |
|----------|-------|
| Género | {genero} |
| Motivos | {motivos} |
| Instrumentos | {instrumentos} |
| Frase-concepto | {frase} |
| Duración | {duracion}s |
| Enganche target | {enganche} |

---

## Estrategia de Publicación

### Timing Óptimo de Posts
- **Pre-lanzamiento (D-7):** Teaser con fragmento del beat de tensión (beat {beat}) – máximo 15s
- **Pre-lanzamiento (D-3):** Revelación de instrumentación: {instrumentos}
- **Lanzamiento (D-0):** Track completo + frase-concepto como caption principal
- **Post-lanzamiento (D+2):** Behind-the-scenes del proceso de generación MIHM
- **Post-lanzamiento (D+7):** Métricas de engagement vs. predicción del sistema

### Plataformas Prioritarias
1. **TikTok/Reels:** Fragmento de 15-30s centrado en el pico de tensión (beat {beat})
2. **Spotify:** Lanzamiento con playlist editorial pitching 7 días antes
3. **YouTube Shorts:** Visualización del análisis de frecuencias (CFF={mihm_state.get('cff', 0):.3f})
4. **Twitter/X:** Thread técnico sobre el proceso Monte Carlo + generación MIDI

### Material Visual Recomendado
- Visualizador de frecuencias sincronizado con el track
- Gráfico del costo J durante el ciclo de generación
- Waveform del archivo MIDI con el pico de tensión marcado
- Captura del estado MIHM en tiempo real

### Continuidad y Seguimiento
- Monitoreo de engagement cada 24h durante 2 semanas
- Re-análisis con sistema Friction en D+14 para ajuste de estrategia
- Trigger automático de nuevo ciclo si engagement < 60% del target ({enganche})

---

## KPIs Objetivo
| Métrica | Target |
|---------|--------|
| Plays primera semana | >10,000 |
| Engagement rate | >{enganche*100:.0f}% |
| Save rate | >15% |
| Shares | >500 |

---
*Generado automáticamente por Sistema Friction v4.2 – Motor Orquestador Proactivo*
"""

        file_id, url = self._create_doc(title, content)

        if file_id is None:
            local_path = self._local_fallback(f'social_{session_id[:8]}.md', content)
            result = {'file_id': None, 'url': None, 'local_path': local_path, 'type': 'social'}
        else:
            result = {'file_id': file_id, 'url': url, 'local_path': None, 'type': 'social'}

        # Delta MIHM
        u, J_new = self.mihm.apply_delta({'nti': 0.03, 'r': 0.02}, action='drive_upload_social')
        self.mihm.meta_control()
        result['mihm_u']    = u
        result['mihm_cost'] = J_new
        return result
