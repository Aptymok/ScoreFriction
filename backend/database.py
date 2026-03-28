import sqlite3
import os


class Database:
    def __init__(self, db_path='instance/friction.db'):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_db()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        with self.get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    user TEXT,
                    text TEXT,
                    ihg REAL,
                    nti REAL,
                    r REAL,
                    error REAL,
                    prediction_id TEXT UNIQUE
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS learning (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id TEXT,
                    outcome REAL,
                    error REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS params (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kp REAL,
                    ki REAL,
                    kd REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # Tabla de historial de estado del MIHM
            conn.execute('''
                CREATE TABLE IF NOT EXISTS state_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    ihg REAL,
                    nti REAL,
                    r REAL,
                    ml_success REAL,
                    irc REAL,
                    action TEXT,
                    cost_j REAL
                )
            ''')
            # Tabla de memoria reflexiva (MCM-A)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS reflexive_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    rule TEXT,
                    delta_json TEXT,
                    j_improvement REAL,
                    outcome TEXT
                )
            ''')
            # Tabla de sesiones del Orquestador Proactivo
            conn.execute('''
                CREATE TABLE IF NOT EXISTS orchestrator_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE,
                    created_at TEXT,
                    calendar_event_id TEXT,
                    calendar_event_title TEXT,
                    status TEXT,
                    telegram_message_id INTEGER,
                    user_params_json TEXT,
                    midi_path TEXT,
                    mihm_state_snapshot TEXT,
                    cost_j_at_trigger REAL,
                    error_msg TEXT,
                    updated_at TEXT
                )
            ''')
            # Tabla de entregables (Google Docs generados)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS deliverables (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    deliverable_type TEXT,
                    drive_file_id TEXT,
                    drive_url TEXT,
                    created_at TEXT,
                    mihm_delta_json TEXT
                )
            ''')

    # ------------------------------------------------------------------
    # predictions
    # ------------------------------------------------------------------

    def save_prediction(self, pred_id, user, text, state, error=None,
                        error_smoothed=None, params=None):
        with self.get_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO predictions
                  (prediction_id, timestamp, user, text,
                   ihg, nti, r, error, error_smoothed,
                   params_kp, params_ki, params_kd)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                pred_id, datetime.utcnow().isoformat(), user, text,
                state.get('ihg'), state.get('nti'), state.get('r'),
                error, error_smoothed,
                params.get('kp') if params else None,
                params.get('ki') if params else None,
                params.get('kd') if params else None,
            ))

    def get_history(self, limit=100):
        with self.get_connection() as conn:
            cur = conn.execute('''
                SELECT timestamp, error FROM predictions
                WHERE error IS NOT NULL
                ORDER BY timestamp DESC LIMIT ?
            ''', (limit,))
            rows = cur.fetchall()
            return [{'timestamp': r[0], 'error': r[1], 'error_smoothed': r[2]}
                    for r in rows]

    # ------------------------------------------------------------------
    # parameters
    # ------------------------------------------------------------------

    def save_params(self, kp, ki, kd):
        with self.get_connection() as conn:
            conn.execute('REPLACE INTO parameters (key, value) VALUES (?, ?)',
                         (key, json.dumps(value)))

    def get_parameters(self, key):
        with self.get_connection() as conn:
            cur = conn.execute('SELECT value FROM parameters WHERE key = ?', (key,))
            row = cur.fetchone()
            if row:
                return json.loads(row[0])
            return None

    # ------------------------------------------------------------------
    # scenarios
    # ------------------------------------------------------------------

    def save_scenario(self, scenario):
        with self.get_connection() as conn:
            conn.execute(
                'INSERT INTO scenarios (timestamp, scenario) VALUES (?, ?)',
                (datetime.utcnow().isoformat(), scenario)
            )

    # ------------------------------------------------------------------
    # state_history
    # ------------------------------------------------------------------

    def save_state(self, state, irc, action, cost_j):
        with self.get_connection() as conn:
            conn.execute('''
                INSERT INTO state_history
                  (timestamp, ihg, nti, r, ml_success, irc, action, cost_j)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.utcnow().isoformat(),
                state.get('ihg'), state.get('nti'), state.get('r'),
                state.get('ml_success'), irc, action, cost_j,
            ))

    def get_state_history(self, limit=200):
        with self.get_connection() as conn:
            cur = conn.execute('''
                SELECT timestamp, ihg, nti, r, ml_success, irc, action, cost_j
                FROM state_history
                ORDER BY timestamp DESC LIMIT ?
            ''', (limit,))
            rows = cur.fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # reflexive_memory
    # ------------------------------------------------------------------

    def save_reflexive_rule(self, rule, delta, j_improvement, outcome='pending'):
        with self.get_connection() as conn:
            conn.execute('''
                INSERT INTO reflexive_memory
                  (timestamp, rule, delta_json, j_improvement, outcome)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.utcnow().isoformat(),
                rule, json.dumps(delta), j_improvement, outcome,
            ))

    def get_reflexive_rules(self, limit=50):
        with self.get_connection() as conn:
            cur = conn.execute('''
                SELECT timestamp, rule, delta_json, j_improvement, outcome
                FROM reflexive_memory
                ORDER BY j_improvement DESC LIMIT ?
            ''', (limit,))
            rows = cur.fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # orchestrator_sessions
    # ------------------------------------------------------------------

    def create_orchestrator_session(self, session_id, event_id, event_title):
        now = datetime.utcnow().isoformat()
        with self.get_connection() as conn:
            conn.execute('''
                INSERT OR IGNORE INTO orchestrator_sessions
                  (session_id, created_at, calendar_event_id,
                   calendar_event_title, status, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (session_id, now, event_id, event_title, 'idle', now))

    def update_orchestrator_session(self, session_id, **fields):
        if not fields:
            return
        fields['updated_at'] = datetime.utcnow().isoformat()
        set_clause = ', '.join(f'{k} = ?' for k in fields)
        values = list(fields.values()) + [session_id]
        with self.get_connection() as conn:
            conn.execute(
                f'UPDATE orchestrator_sessions SET {set_clause} WHERE session_id = ?',
                values,
            )

    def get_orchestrator_session(self, session_id):
        with self.get_connection() as conn:
            cur = conn.execute(
                'SELECT * FROM orchestrator_sessions WHERE session_id = ?',
                (session_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def get_recent_orchestrator_sessions(self, limit=10):
        with self.get_connection() as conn:
            cur = conn.execute('''
                SELECT * FROM orchestrator_sessions
                ORDER BY created_at DESC LIMIT ?
            ''', (limit,))
            rows = cur.fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # deliverables
    # ------------------------------------------------------------------

    def save_deliverable(self, session_id, deliverable_type, drive_file_id,
                         drive_url, mihm_delta):
        with self.get_connection() as conn:
            conn.execute('''
                INSERT INTO deliverables
                  (session_id, deliverable_type, drive_file_id,
                   drive_url, created_at, mihm_delta_json)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                session_id, deliverable_type, drive_file_id, drive_url,
                datetime.utcnow().isoformat(), json.dumps(mihm_delta),
            ))
