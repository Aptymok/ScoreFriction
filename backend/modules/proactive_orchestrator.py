# proactive_orchestrator.py – Orquestador Proactivo · Paso 5
# Google Calendar polling + Telegram + máquina de 5 estados → ciclo completo
# Estados: idle → waiting_telegram → collecting → generating → uploading → done/error

import json
import os
import uuid
from datetime import datetime, timezone, timedelta


class ProactiveOrchestrator:
    STATES = ('idle', 'waiting_telegram', 'collecting', 'generating', 'uploading', 'done', 'error')

    QUESTION_TEXT = (
        "🎵 *Sistema Friction detectó evento: {event_title}*\n\n"
        "Para generar la pista emergente, responde con:\n"
        "1️⃣ 4 motivos (separados por coma)\n"
        "2️⃣ Duración en segundos\n"
        "3️⃣ Frase-concepto (1 frase clave)\n"
        "4️⃣ Género musical\n"
        "5️⃣ Instrumentos (separados por coma)\n"
        "6️⃣ Enganche target (ej: 0.7)\n\n"
        "Ejemplo:\n"
        "amor, ritmo, ciudad, noche | 180 | el amor duele pero suena | reggaeton | bajo, batería, sintetizador | 0.7"
    )

    def __init__(self, mihm, db, groq=None, melody_engine=None, drive_manager=None, config=None):
        self.mihm          = mihm
        self.db            = db
        self.groq          = groq
        self.melody_engine = melody_engine
        self.drive_manager = drive_manager
        self.config        = config or {}

        self._telegram_offset = 0
        self._cal_service     = None

    # ── configuration helpers ────────────────────────────────────────

    def _cfg(self, key, default=None):
        return self.config.get(key) or os.environ.get(key, default)

    @property
    def _bot_token(self):
        return self._cfg('TELEGRAM_BOT_TOKEN')

    @property
    def _chat_id(self):
        return self._cfg('TELEGRAM_CHAT_ID')

    @property
    def _track_keyword(self):
        return self._cfg('TRACK_KEYWORD', 'generación de pista').lower()

    @property
    def _calendar_id(self):
        return self._cfg('GOOGLE_CALENDAR_ID', 'primary')

    # ── Google Calendar ──────────────────────────────────────────────

    def _get_calendar_service(self):
        if self._cal_service:
            return self._cal_service
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build

            sa_json = self._cfg('GOOGLE_SERVICE_ACCOUNT_JSON')
            scopes  = ['https://www.googleapis.com/auth/calendar.readonly']

            if sa_json:
                info  = json.loads(sa_json)
                creds = service_account.Credentials.from_service_account_info(info, scopes=scopes)
            else:
                creds_file = self._cfg('GOOGLE_CREDENTIALS_FILE', 'credentials/service_account.json')
                creds = service_account.Credentials.from_service_account_file(creds_file, scopes=scopes)

            self._cal_service = build('calendar', 'v3', credentials=creds)
        except Exception:
            pass
        return self._cal_service

    def scan_calendar(self) -> list:
        """Retorna eventos próximas 24h cuyo título contenga TRACK_KEYWORD."""
        service = self._get_calendar_service()
        if service is None:
            return []

        try:
            now    = datetime.now(timezone.utc)
            end    = now + timedelta(hours=24)
            result = service.events().list(
                calendarId=self._calendar_id,
                timeMin=now.isoformat(),
                timeMax=end.isoformat(),
                singleEvents=True,
                orderBy='startTime',
            ).execute()

            events = []
            for item in result.get('items', []):
                title = item.get('summary', '')
                if self._track_keyword in title.lower():
                    events.append({
                        'id':    item.get('id'),
                        'title': title,
                        'start': item.get('start', {}).get('dateTime', ''),
                    })
            return events
        except Exception:
            return []

    # ── Telegram ─────────────────────────────────────────────────────

    def send_telegram(self, text: str, parse_mode: str = 'Markdown') -> int:
        """Envía mensaje Telegram. Retorna message_id o -1 si falla."""
        token   = self._bot_token
        chat_id = self._chat_id
        if not token or not chat_id:
            return -1
        try:
            import requests
            url  = f'https://api.telegram.org/bot{token}/sendMessage'
            resp = requests.post(url, json={
                'chat_id':    chat_id,
                'text':       text,
                'parse_mode': parse_mode,
            }, timeout=10)
            data = resp.json()
            if data.get('ok'):
                return data['result']['message_id']
        except Exception:
            pass
        return -1

    def get_telegram_updates(self, offset: int = 0) -> list:
        """Polling manual de updates."""
        token = self._bot_token
        if not token:
            return []
        try:
            import requests
            url  = f'https://api.telegram.org/bot{token}/getUpdates'
            resp = requests.get(url, params={'offset': offset, 'timeout': 5}, timeout=10)
            data = resp.json()
            if data.get('ok'):
                return data.get('result', [])
        except Exception:
            pass
        return []

    def parse_user_params(self, text: str) -> dict | None:
        """
        Parsea respuesta del usuario en formato:
        motivos | duracion | frase | genero | instrumentos | enganche
        Retorna dict validado o None si inválido.
        """
        try:
            parts = [p.strip() for p in text.split('|')]
            if len(parts) < 6:
                return None

            motivos      = [m.strip() for m in parts[0].split(',')]
            duracion_seg = int(parts[1].strip())
            frase        = parts[2].strip()
            genero       = parts[3].strip()
            instrumentos = [i.strip() for i in parts[4].split(',')]
            enganche     = float(parts[5].strip())

            if len(motivos) < 1 or duracion_seg < 10 or not frase or not genero:
                return None
            if not (0.0 <= enganche <= 1.0):
                return None

            return {
                'motivos':        motivos[:4],
                'duracion_seg':   duracion_seg,
                'frase_concepto': frase,
                'genero':         genero,
                'instrumentos':   instrumentos,
                'enganche':       enganche,
            }
        except Exception:
            return None

    # ── ciclo principal ──────────────────────────────────────────────

    def tick(self) -> dict:
        """
        Ejecuta un ciclo del orquestador.
        1. Escanea Calendar → crea sesiones para eventos nuevos
        2. Sesiones waiting_telegram → verifica respuestas
        3. Sesiones collecting → parsea parámetros
        4. Sesiones generating → genera MIDI
        5. Sesiones uploading → sube documentos Drive
        """
        result_sessions = []

        # 1. Procesar delayed updates del MIHM
        if hasattr(self.mihm, 'process_delayed_updates'):
            self.mihm.process_delayed_updates()

        # 2. Escanear Calendar
        cal_events = self.scan_calendar()
        for event in cal_events:
            existing = self.db.get_orchestrator_session(event['id'])
            if existing is None:
                self.db.create_orchestrator_session(event['id'], event['id'], event['title'])
                self.db.update_orchestrator_session(event['id'], status='waiting_telegram')
                msg_id = self.send_telegram(self.QUESTION_TEXT.format(event_title=event['title']))
                self.db.update_orchestrator_session(event['id'], telegram_message_id=msg_id)
                # Delta MIHM por evento detectado
                self.mihm.apply_delta({'nti': 0.05, 'r': 0.03}, action='calendar_event_detected')

        # 3. Obtener updates de Telegram
        updates = self.get_telegram_updates(offset=self._telegram_offset)
        if updates:
            self._telegram_offset = updates[-1]['update_id'] + 1

        # 4. Procesar sesiones activas
        sessions = self.db.get_recent_orchestrator_sessions(limit=20)

        for sess in sessions:
            if sess['status'] in ('done', 'error'):
                result_sessions.append(sess)
                continue

            try:
                sess = self._process_session(sess, updates)
            except Exception as e:
                self.db.update_orchestrator_session(sess['session_id'], status='error', error_msg=str(e))
                sess['status'] = 'error'

            result_sessions.append(sess)

        return {
            'sessions':   result_sessions,
            'mihm_state': dict(self.mihm.state),
            'cost_j':     self.mihm.cost_function(),
            'timestamp':  datetime.utcnow().isoformat(),
        }

    def _process_session(self, sess: dict, updates: list) -> dict:
        """Procesa una sesión individual según su estado."""
        session_id = sess['session_id']
        status     = sess['status']

        if status == 'waiting_telegram':
            # Buscar respuesta del usuario
            for upd in updates:
                msg = upd.get('message', {})
                if msg.get('text'):
                    params = self.parse_user_params(msg['text'])
                    if params:
                        params_json = json.dumps(params)
                        self.db.update_orchestrator_session(
                            session_id, status='generating', user_params_json=params_json
                        )
                        self.mihm.apply_delta({'phi_p': 0.1 * params.get('enganche', 0.7)},
                                              action='params_received')
                        sess['status']           = 'generating'
                        sess['user_params_json'] = params_json
                        break

        elif status == 'collecting':
            # Estado intermedio — redirigir a generating si hay params
            if sess.get('user_params_json'):
                self.db.update_orchestrator_session(session_id, status='generating')
                sess['status'] = 'generating'

        if sess.get('status') == 'generating' and self.melody_engine:
            params = json.loads(sess['user_params_json'] or '{}')
            params['session_id'] = session_id

            result = self.melody_engine.generate(params)

            self.db.update_orchestrator_session(
                session_id,
                status='uploading',
                midi_path=result.get('midi_path', ''),
                mihm_state_snapshot=json.dumps(result.get('mihm_state', {})),
                cost_j_at_trigger=result.get('cost_j'),
            )
            sess['status'] = 'uploading'
            sess['_generate_result'] = result

        if sess.get('status') == 'uploading' and self.drive_manager:
            generate_result = sess.pop('_generate_result', None)

            if generate_result is None:
                # Reconstruir desde DB si no tenemos result en memoria
                generate_result = {
                    'mc_projection': {},
                    'midi_analysis': {'tension_beat': 32, 'tension_intensity': 0.7},
                    'mihm_state':    dict(self.mihm.state),
                    'cost_j':        self.mihm.cost_function(),
                }

            # Narrativa Groq
            groq_narrative = ''
            if self.groq and hasattr(self.groq, 'generate_executive_doc'):
                params = json.loads(sess.get('user_params_json') or '{}')
                groq_narrative = self.groq.generate_executive_doc(
                    generate_result.get('mihm_state', {}),
                    generate_result.get('mc_projection', {}),
                    generate_result.get('midi_analysis', {}),
                    params,
                )

            exec_result = self.drive_manager.create_executive_doc(
                session_id=session_id,
                mihm_state=generate_result.get('mihm_state', {}),
                mc_projection=generate_result.get('mc_projection', {}),
                midi_analysis=generate_result.get('midi_analysis', {}),
                groq_narrative=groq_narrative,
            )
            self.db.save_deliverable(
                session_id, 'executive',
                exec_result.get('file_id', ''),
                exec_result.get('url') or exec_result.get('local_path', ''),
                {'nti': 0.03, 'r': 0.02},
            )

            params = json.loads(sess.get('user_params_json') or '{}')
            social_result = self.drive_manager.create_social_proposal_doc(
                session_id=session_id,
                params=params,
                midi_analysis=generate_result.get('midi_analysis', {}),
                mihm_state=generate_result.get('mihm_state', {}),
            )
            self.db.save_deliverable(
                session_id, 'social',
                social_result.get('file_id', ''),
                social_result.get('url') or social_result.get('local_path', ''),
                {'nti': 0.03, 'r': 0.02},
            )

            self.db.update_orchestrator_session(session_id, status='done')
            sess['status'] = 'done'

        return sess

    def get_status(self) -> dict:
        """Retorna estado actual de todas las sesiones activas + MIHM state."""
        sessions = self.db.get_recent_orchestrator_sessions(limit=10)
        return {
            'sessions':    sessions,
            'mihm_state':  dict(self.mihm.state),
            'cost_j':      self.mihm.cost_function(),
            'active_count': sum(1 for s in sessions if s['status'] not in ('done', 'error')),
            'timestamp':   datetime.utcnow().isoformat(),
        }

    def force_trigger(self, event_title: str = 'test_generation') -> dict:
        """Crea sesión manual sin Calendar. Para testing desde frontend."""
        session_id = str(uuid.uuid4())[:16]
        self.db.create_orchestrator_session(session_id, 'manual', event_title)
        self.db.update_orchestrator_session(session_id, status='waiting_telegram')

        msg_id = self.send_telegram(self.QUESTION_TEXT.format(event_title=event_title))
        self.db.update_orchestrator_session(session_id, telegram_message_id=msg_id)

        # Delta MIHM
        u, J = self.mihm.apply_delta({'nti': 0.05, 'r': 0.03}, action='force_trigger')
        self.mihm.meta_control()

        return {
            'session_id':       session_id,
            'status':           'waiting_telegram',
            'telegram_sent':    msg_id != -1,
            'telegram_msg_id':  msg_id,
            'mihm_u':           u,
            'cost_j':           J,
        }
